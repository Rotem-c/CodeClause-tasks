#!/usr/bin/env python
# coding: utf-8

# # Market Basket Analysis in Pthon using Apriori Algorithm 

# Task 2
# Rotem Cohen

# # Loading the packages

# In[2]:


import pandas as pd
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# Loading data: We will ube using the encoding as latin 1 to read the few special characters mentioned in the file

# In[4]:


dataset= pd.read_csv("D:/internship/marketdata.csv",encoding ='latin-1')
dataset.head()


# In[6]:


dataset.shape


# In[7]:


dataset.dtypes


# In[8]:


dataset.describe()


# # Data Cleaning

# In[9]:


dataset['Description'] = dataset['Description'].str.strip()#removing the spaces


# In[11]:


dataset.dropna(axis=0, subset=['InvoiceNo'], inplace=True)#dropping rows that dont have invoice numbers
dataset['InvoiceNo'] = dataset['InvoiceNo'].astype('str')#conerting InvoiceNo column to string
dataset=dataset[~dataset['InvoiceNo'].str.contains('C')]#removing InvoiceNo which contains C which credit transaction


# In[12]:


dataset.shape


# After the cleaning of the data, wewill have to consolidate the items into 1 transaction per row with each product 1 hot encoded. we will be looking at the data of country France

# In[14]:


basket = (dataset[dataset['Country']=="France"]#getting the data with the country france
         .groupby(['InvoiceNo','Description'])['Quantity']#grouping them by Invoice Number and Description
         .sum().unstack().reset_index().fillna(0)#sum the quamtity, unstack them and fill the missing values with zero
         .set_index('InvoiceNo'))#make the InvoiceNo as Index


# In[15]:


basket.head()


# In[16]:


basket.shape


# In[19]:


#function to convert values <0 to 0 and values > 1 to 1
def encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
    #apply function to data using applymap which is looping through the function
basket_sets= basket.applymap(encode_units)


# In[20]:


basket_sets.head()


# In[21]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames= True)


# the final step is generate the asscociation rules with their corresponding support, confidence and lift 

# In[22]:


rules= association_rules(frequent_itemsets, metric ="lift", min_threshold=1)


# In[23]:


rules.head()


# In[24]:


rules[(rules['lift']>=6)&
     (rules['confidence']>=0.8)]


# In[ ]:




