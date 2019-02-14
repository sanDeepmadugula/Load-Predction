#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# In[70]:


import os
os.chdir('D:\\python using jupyter')


# In[71]:


data = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
data.head()


# In[72]:


data.info()


# In[73]:


data.isnull().sum()


# In[74]:


# percent of missing "Gender" 
print('Percent of missing "Gender" records is %.2f%%' %((data['Gender'].isnull().sum()/data.shape[0])*100))


# In[75]:


print("Number of people who take a loan group by gender :")
print(data['Gender'].value_counts())
sn.countplot(x='Gender',data=data,palette='Set2')


# In[76]:


# Married missing values
print('Percent of missing "Married" records is %.2f%%' %((data['Married'].isnull().sum()/data.shape[0])*100))


# In[77]:


print("Number of people who take a loan group by marital status :")
print(data['Married'].value_counts())
sn.countplot(x='Married',data=data,palette='Set2')


# In[78]:


# check dependents missing value
print('Percent of missing "Dependents" records is %.2f%%' %((data['Dependents'].isnull().sum()/data.shape[0])*100))


# In[79]:


print("Number of people who take a loan group by dependents :")
print(data['Dependents'].value_counts())
sn.countplot(x='Dependents',data=data,palette='Set2')


# In[80]:


# Self employed missing values
print('Percent of missing "Self employed" records is %.2f%%' %((data['Self_Employed'].isnull().sum()/data.shape[0])*100))


# In[81]:


print("Number of people who take a loan group by self employed :")
print(data['Self_Employed'].value_counts())
sn.countplot(x='Self_Employed',data=data,palette='Set2')


# In[82]:


# Load-Amount missing value
print('Percent of missing "LoanAmount" records is %.2f%%' %((data['LoanAmount'].isnull().sum()/data.shape[0])*100))


# In[83]:


ax = data['LoanAmount'].hist(density=True,stacked=True,color='teal',alpha=0.6)
data['LoanAmount'].plot(kind='density',color='teal')
ax.set(xlabel='Loan Amount')
plt.show()


# In[84]:


# Loan amount term missing value
print('Percent of missing "Loan_Amount_Term" records is %.2f%%' %((data['Loan_Amount_Term'].isnull().sum()/data.shape[0])*100))


# In[85]:


print("Number of people who take a loan group by Loan_Amount_Term :")
print(data['Loan_Amount_Term'].value_counts())
sn.countplot(x='Loan_Amount_Term',data=data,palette='Set2')


# In[86]:


# Credit history missing  values

print('Percent of missing "Credit_History" records is %.2f%%' %((data['Credit_History'].isnull().sum()/data.shape[0])*100))


# In[87]:


print("Number of people who take a loan group by Credit_History :")
print(data['Credit_History'].value_counts())
sn.countplot(x='Credit_History',data=data,palette='Set2')


# # Adjustment to data

# Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:
# 
# 

# If "Gender" is missing for a given row, I'll impute with Male (most common answer).
# 

# If "Married" is missing for a given row, I'll impute with yes (most common answer).
# 

# If "Dependents" is missing for a given row, I'll impute with 0 (most common answer).
# 

# If "Self_Employed" is missing for a given row, I'll impute with no (most common answer).
# 

# If "LoanAmount" is missing for a given row, I'll impute with mean of data.
# 

# If "Loan_Amount_Term" is missing for a given row, I'll impute with 360 (most common answer).
# 

# If "Credit_History" is missing for a given row, I'll impute with 1.0 (most common answer).

# In[88]:


train_data = data.copy()
train_data['Gender'].fillna(train_data['Gender'].value_counts().idxmax(),inplace=True)
train_data['Married'].fillna(train_data['Married'].value_counts().idxmax(),inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].value_counts().idxmax(),inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].value_counts().idxmax(), inplace=True)
train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(skipna=True), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].value_counts().idxmax(), inplace=True)


# In[89]:


train_data.isnull().sum()


# In[90]:


# convert some objects into integer
#Convert some object data type to int64
gender_stat = {'Female': 0, 'Male': 1}
yes_no_stat = {'No' : 0,'Yes' : 1}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}


train_data['Gender'] = train_data['Gender'].replace(gender_stat)
train_data['Married'] = train_data['Married'].replace(yes_no_stat)
train_data['Dependents'] = train_data['Dependents'].replace(dependents_stat)
train_data['Education'] = train_data['Education'].replace(education_stat)
train_data['Property_Area'] = train_data['Property_Area'].replace(property_stat)
train_data['Self_Employed'] = train_data['Self_Employed'].replace(yes_no_stat)


# In[91]:


train_data.head()


# In[92]:


train_data.info()


# # Making Predictions

# In[93]:


#Separate feature and target
x = train_data.iloc[:,1:12]
y = train_data.iloc[:,12]

#make variabel for save the result and to show it
classifier = ('Gradient Boosting','Random Forest','Decision Tree','K-Nearest Neighbor','SVM')
y_pos = np.arange(len(classifier))
score = []


# In[94]:


clf = GradientBoostingClassifier()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuration of classification is %.2f%%' %(scores.mean()*100))


# In[95]:


clf = RandomForestClassifier()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuration of classification is %.2f%%' %(scores.mean()*100))


# In[96]:


clf = DecisionTreeClassifier()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuration of classification is %.2f%%' %(scores.mean()*100))


# In[97]:


clf = KNeighborsClassifier()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuration of classification is %.2f%%' %(scores.mean()*100))


# In[98]:


clf = svm.LinearSVC()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuration of classification is %.2f%%' %(scores.mean()*100))


# # Result

# In[99]:


plt.barh(y_pos, score, align='center', alpha=0.5)
plt.yticks(y_pos, classifier)
plt.xlabel('Score')
plt.title('Classification Performance')
plt.show()


# In[ ]:




