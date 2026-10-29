import sys, json, time, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
src,out=sys.argv[1],sys.argv[2]
df=pd.read_excel(src,dtype=str,keep_default_na=False)
df['asin']=df.chunk_id.str.split('_').str[0]
df['is_spec']=df.chunk_id.str.contains('_spec')
X=df.text_clean.values; y=df.label.values; g=df.asin.values
labs=['negative','neutral','positive']
res={}
# VADER (rule-based, no training)
sia=SentimentIntensityAnalyzer(); t=time.time()
def vp(s):
    c=sia.polarity_scores(s)['compound']; return 'positive' if c>=.05 else 'negative' if c<=-.05 else 'neutral'
pv=np.array([vp(s) for s in X]); res['VADER']=(pv,(time.time()-t)/len(X)*1000)
# TF-IDF + LR, 5-fold CV grouped by product (no product leaks between train/test)
pl=np.empty(len(df),dtype=object); t=time.time()
for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(X,y,g):
    v=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True)
    m=LogisticRegression(max_iter=2000,class_weight='balanced').fit(v.fit_transform(X[tr]),y[tr])
    pl[te]=m.predict(v.transform(X[te]))
res['TFIDF+LR (5-fold, grouped by product)']=(pl,(time.time()-t)/len(X)*1000)
# optional transformer
if len(sys.argv)>3 and sys.argv[3]=='roberta':
    from transformers import pipeline
    p=pipeline('sentiment-analysis',model='cardiffnlp/twitter-roberta-base-sentiment-latest',truncation=True,max_length=512)
    t=time.time(); pr=np.array([r['label'].lower() for r in p(list(X),batch_size=16)])
    res['RoBERTa (cardiffnlp, zero-shot)']=(pr,(time.time()-t)/len(X)*1000)
rows=[];rep={}
for n,(p_,ms) in res.items():
    f=lambda m: f1_score(y[m],p_[m],labels=labs,average='macro',zero_division=0)
    rev=~df.is_spec.values
    rows.append(dict(model=n,accuracy=accuracy_score(y,p_),macro_F1=f(np.ones(len(y),bool)),
        F1_neg=f1_score(y,p_,labels=['negative'],average='macro'),
        macroF1_reviews_only=f(rev),acc_reviews_only=accuracy_score(y[rev],p_[rev]),ms_per_item=ms))
    rep[n]=dict(report=classification_report(y,p_,labels=labs,output_dict=True,zero_division=0),
                cm=confusion_matrix(y,p_,labels=labs).tolist())
    df['pred_'+n.split()[0].lower()]=p_
pd.DataFrame(rows).round(3).to_csv(out+'/summary.csv',index=False)
json.dump(rep,open(out+'/details.json','w'),indent=1)
df.to_csv(out+'/predictions.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string())
for n in rep: print(n,rep[n]['cm'])
