import argparse,json,os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
from features import extract_features,FEATURE_NAMES

ROOT=Path(__file__).resolve().parent

def normalize(df):
    colmap={str(c).strip().lower():c for c in df.columns}
    def get(*keys):
        for k in keys:
            if k in colmap:return df[colmap[k]].fillna('').astype(str)
        return pd.Series(['']*len(df),index=df.index)
    labelcol=next((colmap[k] for k in ('email type','email_type','label','type','class','target') if k in colmap),None)
    if labelcol is None:raise ValueError(f'No label column. Columns found: {list(df.columns)}')
    raw=df[labelcol].astype(str).str.strip().str.lower()
    if set(raw.unique()) <= {'0','1'}: labels=raw.astype(int)
    else:
        mapping={'phishing email':1,'phishing':1,'phish':1,'1':1,'safe email':0,'safe':0,'legitimate':0,'legit':0,'benign':0,'0':0}
        labels=raw.map(mapping)
        if labels.isna().any():raise ValueError(f'Unknown labels: {raw[labels.isna()].unique()[:10]}. Please edit normalize() label mapping.')
    out=pd.DataFrame({'subject':get('subject','email_subject'),'body':get('email text','email_text','email content','email_content','body','text','message'),'sender':get('sender','sender_email','from'),'reply_to':get('reply_to','reply-to'),'html':get('html','html_body'),'label':labels.astype(int)})
    out=out[(out['subject'].str.len()+out['body'].str.len()+out['html'].str.len())>5]
    out=out.drop_duplicates(subset=['subject','body','html']).reset_index(drop=True)
    if out['label'].nunique()!=2:raise ValueError('Dataset must have both phishing and legitimate classes.')
    return out

def make_features(df):
    records=[extract_features(r.subject,r.body,r.sender,r.reply_to,r.html) for r in df.itertuples(index=False)]
    return [r['text'] for r in records],np.stack([r['vector'] for r in records])

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/Phishing_Email.csv');parser.add_argument('--max-rows',type=int,default=20000);parser.add_argument('--mode',choices=['baseline','hybrid'],default='baseline');args=parser.parse_args()
    path=Path(args.data);path=path if path.is_absolute() else ROOT/path
    if not path.exists():raise SystemExit(f'Dataset not found: {path}. Read README.md for download instructions.')
    df=normalize(pd.read_csv(path,encoding_errors='replace'))
    if args.max_rows and len(df)>args.max_rows:
        df=df.groupby('label',group_keys=False).sample(frac=min(1,args.max_rows/len(df)),random_state=42)
    train,test=train_test_split(df,test_size=.2,stratify=df.label,random_state=42)
    trtext,trnum=make_features(train);tetext,tenum=make_features(test)
    tfidf=TfidfVectorizer(max_features=12000,ngram_range=(1,2),min_df=1,sublinear_tf=True)
    xtr=tfidf.fit_transform(trtext);xte=tfidf.transform(tetext)
    scaler=None
    if args.mode=='hybrid':
        scaler=StandardScaler();trscaled=scaler.fit_transform(trnum);tescaled=scaler.transform(tenum)
        xtr=hstack([xtr,csr_matrix(trscaled)],format='csr');xte=hstack([xte,csr_matrix(tescaled)],format='csr')
    model=LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)
    model.fit(xtr,train.label)
    pred=model.predict(xte);scores=metrics(test.label,pred)
    report={'mode':args.mode,'source':str(path),'dataset_rows_after_cleaning':len(df),'train_rows':len(train),'test_rows':len(test),'train_class_counts':{str(k):int(v) for k,v in train.label.value_counts().items()},'metrics':scores,'note':'Held-out stratified split; performance may be optimistic due to dataset/source biases. Not independently validated.'}
    (ROOT/'models').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    dump({'model':model,'tfidf':tfidf,'scaler':scaler,'mode':args.mode,'feature_names':FEATURE_NAMES},ROOT/'models'/'detector.joblib')
    (ROOT/'results'/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('\n'+classification_report(test.label,pred,target_names=['Legitimate','Phishing'],zero_division=0))
    print('Saved model to models/detector.joblib')
if __name__=='__main__':main()
