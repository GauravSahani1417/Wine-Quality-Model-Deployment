#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('datasets_35901_52633_winequalityN.csv')


# In[3]:


df.head()


# In[4]:


df['quality'].unique()


# In[5]:


df.isnull().sum()


# In[6]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["type"]=encoder.fit_transform(df['type'])


# In[7]:


df.head()


# In[8]:


df['fixed acidity']=df['fixed acidity'].fillna(df['fixed acidity'].mean())
df['volatile acidity']=df['volatile acidity'].fillna(df['volatile acidity'].mean())
df['citric acid']=df['citric acid'].fillna(df['citric acid'].mean())
df['residual sugar']=df['residual sugar'].fillna(df['residual sugar'].mean())
df['chlorides']=df['chlorides'].fillna(df['chlorides'].mean())
df['pH']=df['pH'].fillna(df['pH'].mean())
df['sulphates']=df['sulphates'].fillna(df['sulphates'].mean())


# In[9]:


df.dtypes


# In[10]:


df.isnull().sum()


# In[11]:


X=df.iloc[:,0:12]
y=df[['quality']]


# In[12]:


X.head()


# In[13]:


y.head()


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.18,random_state=4)


# In[22]:


from catboost import CatBoostClassifier
CB = CatBoostClassifier(iterations=1200,
                          learning_rate=1.000001,
                          depth=7)
CB.fit(X_train,y_train)


# In[23]:


from sklearn.metrics import accuracy_score, confusion_matrix
pred=CB.predict(X_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(pred,y_test))


# In[ ]:
import pickle
file = open('catboost_classification_model.pkl', 'wb')

pickle.dump(CB, file)



