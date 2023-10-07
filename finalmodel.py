#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns


# In[4]:


data=pd.read_csv('data.csv')
payment_data = pd.read_csv('payment_data.csv') 
data


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data=data.drop(list(data.filter(regex='^fea',axis=1)),axis=1)
data


# In[ ]:


data.isnull().sum()/len(data)


# In[ ]:


payment_data.dropna(subset=['update_date'], inplace=True)




# In[ ]:


payment_data.dropna(subset=['highest_balance'],inplace=True)


# In[ ]:


payment_data.dropna(subset=['report_date'],inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


payment_data.dropna(subset=['prod_limit'],inplace=True)


# In[ ]:


data.isnull().sum()



# In[ ]:


payment_data['update_date'] = pd.to_datetime(payment_data['update_date'], format='%d/%m/%Y')



# In[ ]:


payment_data['prod_limit'] = pd.to_numeric(payment_data['prod_limit'], errors='coerce')


# In[ ]:


payment_data['new_balance'] = pd.to_numeric(payment_data['prod_limit'], errors='coerce')


# In[ ]:


payment_data['highest_balance'] = pd.to_numeric(payment_data['highest_balance'], errors='coerce')


# In[ ]:


data.info()


# In[ ]:


date=payment_data.set_index(payment_data['report_date'])
date


# In[ ]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[ ]:

merged_data = data.merge(payment_data, on='id', how='inner')

# Split data into features (X) and target labels (y)
y = merged_data['label']
X = merged_data.drop(columns=['label', 'update_date', 'report_date'])

# Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[ ]:


len(X_train)


# In[ ]:


#Logistic regression
model=LogisticRegression()
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy:{accuracy*100:.2f}%")


# In[ ]:


#svc model
model=SVC()
model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:


#decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:


#k-neighborsclassifier model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:





# In[ ]:


# Initialize the Random Forest model with hyperparameters
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:

