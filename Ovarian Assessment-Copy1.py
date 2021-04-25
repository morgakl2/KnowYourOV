#!/usr/bin/env python
# coding: utf-8

# In[596]:


import pandas as pd


# In[597]:


#https://docs.google.com/spreadsheets/d/1QSl1U-eaC2SxmfKC9rqerwOUBIzBHiUYADdGzNwE8Jg/edit?usp=sharing
# this is the link for the complete data from HA


# In[598]:


#karm-451@know-your-ov-311020.iam.gserviceaccount.com 
# this is my email for this project


# In[599]:


#https://docs.google.com/spreadsheets/d/1QSl1U-eaC2SxmfKC9rqerwOUBIzBHiUYADdGzNwE8Jg/gviz/tq?tqx=out:csv&sheet=joined
# this is the website for the full HA provided data


# In[600]:


df = pd.read_csv('https://docs.google.com/spreadsheets/d/1QSl1U-eaC2SxmfKC9rqerwOUBIzBHiUYADdGzNwE8Jg/gviz/tq?tqx=out:csv&sheet=joined')


# In[602]:


dummy = pd.get_dummies(df[['relation_raw','cancer','fh_ethnicity_raw', 'history_class']], drop_first=True)


# In[603]:


merge = pd.concat([df, dummy], axis='columns')


# In[610]:


merge_new= merge.drop(columns=['gene', 'relation', 'history_class','relation_raw', 'fh_known_brca' , 'fh_cancer_dx_type', 'fh_consent_approval' , 'fh_ethnicity_raw', 'fh_ethnicity'])


# In[643]:


#organize what data I want to focus on, I need to train the machine to look at the data I want it to and use that data to make a prediction as to whether a new person being assessed is at risk of ovarian cancer.
df_new=merge_new.loc[(df['cancer']=='Ovarian') | (df['cancer']=='Breat Cancer')| (df['cancer']=='Prostate Cancer')| (df['cancer']=="Colorectal Cancer") | (df['cancer']=='Pancreatic Cancer')| (df['cancer']=='Colon Cancer')]
done=df_new.fillna({'age':(54.758929)})
done=df_new.fillna({'fh_cancer_dx_age':(47.000000)})
done.head()


# In[644]:


done.replace (to_replace=['Ovarian','Breast Cancer', 'Colorectal Cancer','Prostate Cancer','Pancreatic Cancer','Colon Cancer'], value="1") 


# In[645]:


done.shape


# In[646]:


X = done.loc[:]


# In[647]:


X.shape


# In[648]:


y=done.fh_cancer_dx


# In[649]:


y.shape


# In[650]:


import numpy as np


# In[651]:


#make a classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[652]:


classifier = KNeighborsClassifier()


# In[653]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=10, scoring='accuracy',n_jobs=-1)
scores


# In[654]:



X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)


# In[639]:


from sklearn.preprocessing import StandardScaler
X = [[0, 15],
     [1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)


# In[659]:


#to test new patients data from google sheet survey
test= pd.read_csv('https://docs.google.com/spreadsheets/d/1QeKqrPzSOoE6Cv77e8DgLU4BAxvtqZ9e7RpZon1JTS8/gviz/tq?tqx=out:csv&sheet=Know Your Ov(Responses)')


# In[110]:


#X_new= test.loc[:, feature_cols]


# In[111]:


#X_new.shape


# In[ ]:


#new_pred_class=logreg.predict(X_new)


# In[113]:


#test.patient_id


# In[115]:


#new_pred_class


# In[117]:


#pd.DataFrame({'patient_id': test.patient_id,'fh_cancer_dx':new_pred_class}).set_index('patient_id').to_csv('sub.csv')


# In[154]:


import matplotlib.pyplot as plt


# In[158]:


#plt.figure(figsize=(12,7))
#plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'low')
#plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'medium')
#plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'high')

#plt.title('Ovarian Cancer Risk Assessment',fontsize=20)
#plt.xlabel('Risk Factors',fontsize=16)
#plt.ylabel('fh_cancer_dx',fontsize=16)
#plt.legend(fontsize=16)
#plt.grid(True)
#plt.axhspan(ymin=60,ymax=100,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')
#plt.show()


# In[ ]:




