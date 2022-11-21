#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from myst_nb import glue
#from ipywidgets import widgets
#from ipywidgets import interactive
#from IPython.display import display, Javascript

from matplotlib.colors import LogNorm, Normalize
import bamboolib


# In[2]:


import os
current_directory = os.getcwd()
#print("Current directory : ", current_directory)

aymeric =  "/home/aymeric/python-scripts/espadon/data/" #aymeric
jp = '~/Dropbox/Mac/Desktop/CRD Anses/all3/' # Jean Philippe
jp_index = '~/Dropbox/Mac/Desktop/CRD Anses/code/indexation_results/' # Jean Philippe index

if 'aymeric' in current_directory:
    path_base = aymeric

elif 'Mac' in current_directory:
    path_base = jp
elif 'd:/Projects' in current_directory:
    path_base = "d:/Projects/Medialab/"

#print("Path base : ", path_base)


# In[3]:



dic_id={}
for x in [x for x in pd.read_csv(glob.glob(f'{path_base}sm/*.csv')[0]).columns if 'id' in x]:
    dic_id[x]=str


# In[4]:


df0= pd.read_csv(f"{path_base}corpus_tweets.csv", sep = ",", dtype = dic_id)


# In[5]:


url_checked = pd.read_csv('list_url.csv', sep = ",")


# In[6]:


url_checked


# In[7]:



#users = pd.read_csv('../outcome/user_sm_predicted.csv',dtype=dic_id) #jean-philippe
users = pd.read_csv(f'{path_base}recoded_user_sm_predicted.csv',dtype=dic_id)


# In[ ]:





# In[8]:


df0.columns


# In[9]:


df0 = df0[['query', 'id', 'timestamp_utc', 'local_time',
           'user_screen_name', 'text',  'user_location',  'user_id', 'user_name',
           'domains', 'links', 'media_urls', 'url',
           'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK',
           'BRAF', 'MET', 'RET', 'HER2', 'date', ]]

df0['date'] = pd.to_datetime(pd.to_datetime(df0['date']).dt.date)
df0['Year'] = df0['date'].dt.year


# In[10]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
#df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2012)]


# In[11]:


df.columns


# In[12]:


df1 = df.loc[df['links'].notna()]


# In[13]:


list_tweet_id = []
list_links = []
is_doi = []
list_user_scr_name = []
list_user_name = []
list_date = []

for i, link in enumerate(df1["links"]):
    tweet_id = df1["id"].iloc[i]
    user_scr_name = df1["user_screen_name"].iloc[i]
    user_name = df1["user_name_x"].iloc[i]
    user_status = df1["User_status"].iloc[i]
    date = df1["date"].iloc[i]
    tweet_id = df1["id"].iloc[i]
    list_url = link.split("|")
    for url in list_url:
        if "twitter.com" not in url and "instagram" not in url and "facebook.com" not in url and "youtu" not in url:
            list_tweet_id.append(tweet_id)
            list_links.append(url)
            list_user_scr_name.append(user_scr_name)
            list_user_name.append(user_name)
            list_date.append(date)
            if "doi" in url:
                is_doi.append(True)
            else:
                is_doi.append(False)


# In[14]:


len(list_links)


# In[15]:


data = {"id": list_tweet_id, "urls": list_links, "is_doi": is_doi}
df_url = pd.DataFrame(data)


# In[16]:


df_url1 = df_url.merge(df[["id", "user_screen_name", "date", "User_status"]], on = ["id"],
                       how = "inner")


# In[17]:


df_url1


# In[18]:


list_urls = set(list_links)


# In[19]:


len(list_urls)


# In[20]:


from ural import is_shortened_url
from ural import is_url


# In[21]:


shortened_url_list = []

for x in list(list_urls):
    if is_shortened_url(x) == True:
        shortened_url_list.append(x)


# In[22]:


len(shortened_url_list)


# In[23]:


true_url_list = []

for x in shortened_url_list:
    if is_url(x) == True:
        true_url_list.append(x)


# In[24]:


short_url_list = [x for x in url_checked["short_url"]]


# In[25]:


len(short_url_list)


# In[26]:


true_url_list2 = [x for x in true_url_list if x not in short_url_list]
len(true_url_list2)


# In[27]:


from urllib import request


# In[28]:



r = request.urlopen("http://owl.li/Nxvz30j7XtU")
r.headers


# In[29]:


from tqdm.notebook import tqdm, trange
import time
import csv


# In[ ]:



start_time = time.time()
#print(start_time)
for n, x in  tqdm(enumerate(true_url_list2)):
    #print(n, x)
    try:
        r = request.urlopen(x, timeout =3)
        with open('list_url.csv', 'a', newline='') as csvfile:
            fieldnames = ['short_url', 'long_url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'short_url': x, 'long_url': r.url})
    except:
        with open('list_url.csv', 'a', newline='') as csvfile:
            fieldnames = ['short_url', 'long_url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'short_url': x, 'long_url': 'Failed'})
    


# In[ ]:




