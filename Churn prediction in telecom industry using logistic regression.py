#!/usr/bin/env python
# coding: utf-8

# # CODE CLAUSE INTERNSHIP 
# Task 1: Churn Prediction in telecom industry using logistic regression
# 
# Rotem Cohen

# # Importing required libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.express as px


# # Reading our csv file 

# In[3]:


telecom_dataset = pd.read_csv("D:/internship/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[4]:


telecom_dataset.head()


# In[5]:


telecom_dataset.shape


# In[6]:


telecom_dataset.describe()


# In[10]:


telecom_dataset.notnull().sum()#checking for null values


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


telecom_histogram = px.histogram(telecom_dataset , x ='gender' , color='Churn',marginal='box',color_discrete_sequence =['red','grey'])
telecom_histogram.update_layout(bargap = 0.2)


# In[11]:


plt.bar(telecom_dataset['gender'],telecom_dataset['Churn'])


# In[12]:


telecom_dataset.hist(bins = 30,figsize = (20,15))


# In[13]:


sns.pairplot(telecom_dataset)


# # Cleaning the Data
# 

# In[14]:


#removing gender, customerID,tenure as they are not useful 
col=['gender','customerID','tenure']
telecom_dataset = telecom_dataset.drop(col, axis=1)


# In[15]:


sns.pairplot(telecom_dataset)


# In[16]:


telecom_dataset.head()


# In[17]:


telecom_dataset['TotalCharges'].notnull().sum()


# In[18]:


telecom_dataset['MonthlyCharges'].describe()


# In[19]:


telecom_dataset['TotalCharges'].describe()
#the datatype of the Totalcharges is object so we will have to covert that 


# In[20]:


telecom_dataset['TotalCharges']= telecom_dataset['TotalCharges'].replace(" ",np.nan)
telecom_dataset['TotalCharges']=pd.to_numeric(telecom_dataset['TotalCharges'], errors = 'coerce')
#dropping all the raws in which there us a null value
telecom_dataset= telecom_dataset.dropna(how = "any",axis=0)


# In[21]:


telecom_dataset['TotalCharges'].describe()


# In[22]:


telecom_dataset.notnull().sum()


# In[23]:


telecom_dataset.isnull().sum()


# # Exploratory Data Analysis

# In[24]:


telecom_dataset['Churn'].describe()


# In[26]:


for i, predictor in enumerate(telecom_dataset.drop(columns=['Churn','TotalCharges','MonthlyCharges'])):
    ax= sns.countplot(data = telecom_dataset , x=predictor, hue='Churn')
    if predictor == "PaymentMethod":
        ax.set_xticklables(ax.get_xticklabels(), fontsize=7)
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.show()


# In[28]:


#converting Yes as 1 and No as 0
telecom_dataset["Churn"] = telecom_dataset["Churn"].replace(['Yes','No'],[1,0])


# In[30]:


telecom_dataset


# In[31]:


telecom_dataset_dummies = pd.get_dummies(telecom_dataset)


# In[32]:


telecom_dataset_dummies


# In[33]:


churn_correlation_matrix = telecom_dataset_dummies.corr()


# In[35]:


churn_correlation_matrix['Churn'].sort_values(ascending =False).plot(kind = 'bar' , figsize = (15,10))


# High Churn seen in case of monthly contracts, no online security, no technical support, first year subscription and fiber optic Internet
# Low Churn is seen in the case of long term contracts, subscriptions without internet sservice and customers contracted for more than 5 years
# 
# 

# In[36]:


churn_correlation_matrix['Churn'].sort_values(ascending = False)


# In[37]:


x = telecom_dataset_dummies.drop('Churn', axis=1)


# In[38]:


x


# In[39]:


y = telecom_dataset_dummies['Churn']


# In[40]:


y


# In[41]:


x.shape


# In[42]:


y.shape


# In[43]:


y.value_counts()


# # Variable Imbalancing 

# SMOTE for imbalanced Classification with python 

# In[44]:


from imblearn.over_sampling import SMOTE


# In[45]:


smote= SMOTE(random_state=0)


# In[46]:


x_resampled_smote , y_resampled_smote = smote.fit_resample(x,y)


# In[47]:


y_resampled_smote.value_counts()


# In[48]:


x_resampled_smote


# In[49]:


y_resampled_smote.notnull().sum()


# In[50]:


x_resampled_smote.notnull().sum()


# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42)


# In[53]:


LogisticReg= LogisticRegression(solver='lbfgs',max_iter=400,multi_class='multinomial')


# In[54]:


LogisticReg.fit(x_train,y_train)


# In[55]:


y_prediction = LogisticReg.predict(x_test)


# In[56]:


from sklearn.metrics import accuracy_score


# In[57]:


accuracy_score(y_test,y_prediction)


# In[59]:


x_smote_train,x_smote_test,y_smote_train,y_smote_test = train_test_split(x_resampled_smote,y_resampled_smote,test_size=0.2,random_state=42)


# In[60]:


LogisticReg.fit(x_smote_train,y_smote_train)


# In[61]:


y_smote_prediction = LogisticReg.predict(x_smote_test)


# In[62]:


accuracy_score(y_smote_test,y_smote_prediction)


# In[63]:


from sklearn.preprocessing import StandardScaler


# In[64]:


std = StandardScaler()


# In[65]:


std_train = std.fit_transform(x_smote_train)
std_test = std.transform(x_smote_test)


# In[66]:


LogisticReg.fit(std_train,y_smote_train)


# In[67]:


std_prediction = LogisticReg.predict(std_test)


# In[69]:


accuracy_score(std_prediction,y_smote_test)


# In[70]:


np.where(std_prediction!=y_smote_test)


# In[ ]:




