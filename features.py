import re
from urllib.parse import urlsplit
from email.utils import parseaddr
from bs4 import BeautifulSoup
import numpy as np

URL_RE = re.compile(r'https?://[^\s<>"\']+', re.I)
SHORTENERS = {'bit.ly','tinyurl.com','t.co','is.gd','cutt.ly','tiny.cc','rb.gy'}
KEYWORDS = ['verify','urgent','suspended','password','click here','account','security alert','immediately','confirm','login','payment']
FEATURE_NAMES = ['url_count','max_url_length','ip_host_count','shortener_count','at_sign_urls','subdomain_count','urgent_term_count','exclamation_count','html_link_count','html_form_count','sender_reply_mismatch','sender_missing']

def parse_email(subject='',body='',sender='',reply_to='',html=''):
    soup = BeautifulSoup(html or '', 'html.parser')
    links = [x.get('href','') for x in soup.find_all('a',href=True)]
    visible = soup.get_text(' ', strip=True)
    text = ' '.join([str(subject or ''),str(body or ''),visible]).strip()
    urls = list(dict.fromkeys(URL_RE.findall(text) + [u for u in links if u.lower().startswith(('http://','https://'))]))
    urls = [u.rstrip('.,;!?)') for u in urls]
    return text,urls,soup

def extract_features(subject='',body='',sender='',reply_to='',html=''):
    text, urls, soup = parse_email(subject,body,sender,reply_to,html)
    domains=[]; ip=short=ats=0
    for url in urls:
        try:
            p=urlsplit(url); host=(p.hostname or '').lower();domains.append(host)
            ip+=int(bool(re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}',host)))
            short+=int(host in SHORTENERS)
            ats+=int('@' in p.netloc)
        except ValueError: pass
    sender_domain=parseaddr(sender or '')[1].split('@')[-1].lower()
    reply_domain=parseaddr(reply_to or '')[1].split('@')[-1].lower()
    mismatch=int(bool(sender_domain and reply_domain and sender_domain!=reply_domain))
    lower=text.lower()
    arr=[len(urls),max((len(u) for u in urls),default=0),ip,short,ats,sum(max(0,len(d.split('.'))-2) for d in domains),sum(lower.count(k) for k in KEYWORDS),text.count('!'),len(soup.find_all('a')),len(soup.find_all('form')),mismatch,int(not bool(sender_domain))]
    warnings=[]
    if ip:warnings.append('A URL contains an IP-address host')
    if short:warnings.append('A URL shortening service was detected')
    if mismatch:warnings.append('Sender and Reply-To domains differ')
    if soup.find('form'):warnings.append('HTML form detected in email')
    if urls and max(map(len,urls))>120:warnings.append('An unusually long URL was detected')
    if sum(lower.count(k) for k in KEYWORDS)>2:warnings.append('Several urgency/account-related terms detected')
    for a in soup.find_all('a',href=True):
        displayed=a.get_text(' ',strip=True)
        if displayed.lower().startswith(('https://','http://')):
            try:
                if urlsplit(displayed).hostname != urlsplit(a['href']).hostname:
                    warnings.append('Displayed link and destination have different hosts')
            except ValueError:pass
    return {'text':text,'vector':np.asarray(arr,dtype=float),'urls':urls,'warnings':list(dict.fromkeys(warnings)),'feature_names':FEATURE_NAMES}
