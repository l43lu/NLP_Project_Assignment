#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import shutil
import stopwordsiso as stopwords
import unicodedata as unicode
import random
from numpy.random import randint

from IPython.display import display
from datetime import date, datetime, timedelta
pd.set_option("display.max_rows",180)


# In[2]:


#%tensorflow_version 2.x
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_transform as tft
import tensorflow_text as tftext

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import os
import io
tf.__version__


# In[3]:


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# ## Preprocessing

# In[7]:


article_df = pd.read_csv('./articles.csv', sep=';', names=['sub','text']) 
test       = pd.read_csv('./test.csv', sep=';', names=['sub','text_1','text_2']) 
#train      = pd.read_csv('./train.csv', sep=';', names=['sub','text']) 
train      = pd.read_csv('./train.csv', sep=',') 


# In[8]:


test.fillna('', inplace=True)
train.fillna('', inplace=True)


# In[9]:


train.head()


# #### Merge two columns and delete them

# In[10]:


test['text'] = test['text_1'] + test['text_2']


# In[11]:


test.drop(['text_1','text_2'],inplace=True, axis=1)


# ### Case normalization

# Convert text to the same case

# In[12]:


def lower_case(string_):
    return string_.lower()


# In[13]:


test['text'] = test['text'].apply(lower_case)
train['text'] = train['text'].apply(lower_case)


# ### Remove Stop Words

# In[14]:


de_sw = stopwords.stopwords('de')
def remove_stopwords(text):
    stopwords = list(de_sw)
    querywords = text.split()
    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    
    return result


# In[15]:


train['text'] = train['text'].apply(remove_stopwords)
test['text'] = test['text'].apply(remove_stopwords)


# In[16]:


test.head()


# ### Remove Digits

# In[17]:


def remove_digits_from_string(_string):
    return ''.join([i for i in _string if not i.isdigit()])


# In[18]:


train['text'] = train['text'].apply(remove_digits_from_string)
test['text'] = test['text'].apply(remove_digits_from_string)


# ### Stemming

# In[19]:


nltk.download('punkt')


# In[20]:


def stemming(_string):
    ps = PorterStemmer()
    words = word_tokenize(_string)
    stemmed_string = ""
    for w in words:
        stemmed_string = stemmed_string+" "+ps.stem(w)
        
    return stemmed_string


# In[21]:


train['stemmed_text'] = train['text'].apply(stemming)


# In[25]:


test['stemmed_text']  = test['text'].apply(stemming)


# In[22]:


train.iloc[4]['text']


# In[23]:


train.iloc[4]['stemmed_text']


# In[24]:


train.to_csv('preproccessed_train.csv', sep=',', index=False)


# In[26]:


test.to_csv('preproccessed_test.csv', sep=',', index=False)


# In[ ]:




