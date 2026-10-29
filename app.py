from pathlib import Path
import time,json
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix,hstack
from joblib import load
from features import extract_features

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='PhishLens | Explainable Email Detection',page_icon='🛡️',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1200px} .hero{background:linear-gradient(110deg,#101f39,#203c60);color:white;padding:28px;border-radius:16px}.hero h1{font-size:2rem;margin:0}.hero p{opacity:.8;margin-bottom:0}</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🛡️ PhishLens</h1><p>A Multimodal, Real-Time and Explainable AI Framework for Phishing Email Detection</p></div>',unsafe_allow_html=True)
st.caption('Research prototype • On-demand local inference • No links are opened')
@st.cache_resource
def model_load():return load(ROOT/'models'/'detector.joblib')
modelpath=ROOT/'models'/'detector.joblib'
if not modelpath.exists():
    st.error('No trained model found. Run `python train_model.py --data data/Phishing_Email.csv` in the project folder, then refresh this page.')
    st.stop()
try:pack=model_load()
except Exception as e:st.error(f'Cannot load model: {e}');st.stop()
with st.sidebar:
    st.header('System status')
    st.success('Model loaded')
    st.write('Model:',pack['mode'].title())
    st.write('Classifier: Logistic Regression')
    st.write('Text features: TF-IDF')
    st.info('Hybrid means text + engineered URL/metadata/HTML features are used by the classifier. Baseline means text-only classification; indicators are separate.')
    met=ROOT/'results'/'metrics.json'
    if met.exists():
        result=json.loads(met.read_text(encoding='utf-8'))
        st.subheader('Held-out evaluation')
        st.caption('Results from the last training run')
        for key in ['accuracy','precision','recall','f1']:
            st.metric(key.upper(),f"{result['metrics'][key]:.3f}")
        st.caption(f"Test samples: {result['test_rows']}")

st.subheader('Analyse an email')
with st.form('email'):
    a,b=st.columns(2)
    with a:subject=st.text_input('Subject',placeholder='Important account notice')
    with b:sender=st.text_input('Sender (optional)',placeholder='alerts@example.com')
    reply=st.text_input('Reply-To (optional)',placeholder='support@example.org')
    body=st.text_area('Email body',height=190,placeholder='Paste email content here...')
    with st.expander('Optional HTML content'):
        html=st.text_area('Raw HTML',height=90)
    submitted=st.form_submit_button('🔍 Analyse Email',use_container_width=True,type='primary')
if submitted:
    if not (subject.strip() or body.strip() or html.strip()):st.warning('Enter a subject, body, or HTML content to analyse.')
    else:
        start=time.perf_counter()
        info=extract_features(subject,body,sender,reply,html)
        xt=pack['tfidf'].transform([info['text']]);x=xt
        if pack['mode']=='hybrid':
            scaled=pack['scaler'].transform(info['vector'].reshape(1,-1));x=hstack([xt,csr_matrix(scaled)],format='csr')
        score=float(pack['model'].predict_proba(x)[0,1]);elapsed=(time.perf_counter()-start)*1000
        phishing=score>=.5
        st.divider()
        if phishing:st.error('⚠️ Potential phishing email')
        else:st.success('✓ Classified as likely legitimate')
        c1,c2,c3=st.columns(3)
        c1.metric('Phishing model score',f'{score*100:.1f}%')
        c2.metric('Inference time',f'{elapsed:.2f} ms')
        c3.metric('URLs found',len(info['urls']))
        st.caption('Model score is not a calibrated real-world probability. This tool does not guarantee an email is safe.')
        st.subheader('Why did the model decide this?')
        terms=np.array(pack['tfidf'].get_feature_names_out())
        weights=pack['model'].coef_[0][:len(terms)]
        row=xt.tocoo();contrib=sorted([(terms[j],float(v*weights[j])) for j,v in zip(row.col,row.data)],key=lambda z:abs(z[1]),reverse=True)
        relevant=[(t,v) for t,v in contrib if (v>0 if phishing else v<0)][:7]
        if relevant:
            for t,v in relevant:st.write(f"• **{t}**: {'towards phishing' if v>0 else 'towards legitimate'} (contribution {v:+.3f})")
        else:st.write('No strong text contributions found for this direction.')
        if pack['mode']=='hybrid':
            st.caption('Structured features also affect this classification; text terms alone do not explain the full score.')
        st.subheader('Additional security indicators')
        if info['warnings']:
            for w in info['warnings']:st.warning(w)
        else:st.info('No predefined structural warning indicators found. This does not establish safety.')
        if info['urls']:
            st.write('**Extracted URLs (not visited):**')
            for url in info['urls'][:12]:st.code(url,language=None)
        st.info('Recommendation: verify unexpected requests through an independent trusted channel. Avoid opening unfamiliar links.')
