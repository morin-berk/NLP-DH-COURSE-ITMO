#!/usr/bin/env python
# coding: utf-8

# In[6]:


import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random 
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
import re
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[10]:


df=pd.read_csv("reviews.tsv", sep='\t')


# In[11]:


#features = df.iloc[:, 10].values
labels = df.iloc[:, 1].values


# In[12]:


print(labels)


# In[13]:


labels = df['rating']


# In[14]:


labels


# In[19]:


reviews = df.iloc[:, 2].values


# In[20]:


print(reviews)


# In[23]:


reviews = df['review_text']


# In[24]:


reviews


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

