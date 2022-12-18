#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data = pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\Dataset.csv")
data.head()


# In[2]:


data.columns


# In[3]:


data.isnull().sum()


# In[ ]:





# In[4]:


df=data.copy()


# In[5]:


df.interpolate(inplace=True)


# In[6]:


df.dropna(inplace=True)


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(['encounter_id','patient_id','hospital_id'],axis=1, inplace=True)


# In[9]:


df.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtypes==object:
        df[i]=l.fit_transform(df[i])       


# In[11]:


x=df.drop('hospital_death',axis=1)
y=df['hospital_death']


# In[12]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


# In[13]:


x.shape


# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[15]:


import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(64, input_dim=182))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=25, batch_size=64)


# In[16]:


pred=model.predict(X_test)


# In[17]:


from sklearn.metrics import accuracy_score
a=accuracy_score(pred,y_test)
print('Accuracy : ',a*1000)


# In[ ]:




