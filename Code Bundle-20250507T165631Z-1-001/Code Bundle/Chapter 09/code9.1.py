#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

nltk.download('stopwords')
nltk.download('punkt')
get_ipython().system('pip install datasets')
from datasets import load_dataset

# loading the "fake_news_english" dataset from Huggingface
dataset = load_dataset("fake_news_english")
train_data = dataset["train"]

## Converting  training dataset into a dataframe
df = pd.DataFrame.from_dict(dataset['train'])

df.head()


# In[ ]:




