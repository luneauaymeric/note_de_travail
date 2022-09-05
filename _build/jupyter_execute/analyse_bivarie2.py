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
#import bamboolib


# In[2]:


import os
current_directory = os.getcwd()
print("Current directory : ", current_directory)

aymeric =  "/home/aymeric/python-scripts/espadon/data/" #aymeric
jp = '~/Dropbox/Mac/Desktop/CRD Anses/all3/' # Jean Philippe
jp_index = '~/Dropbox/Mac/Desktop/CRD Anses/code/indexation_results/' # Jean Philippe index

if 'aymeric' in current_directory:
    path_base = aymeric

elif 'Mac' in current_directory:
    path_base = jp
elif 'd:/Projects' in current_directory:
    path_base = "d:/Projects/Medialab/"

print("Path base : ", path_base)


# In[3]:



dic_id={}
for x in [x for x in pd.read_csv(glob.glob(f'{path_base}sm/*.csv')[0]).columns if 'id' in x]:
    dic_id[x]=str


# In[4]:


df0 = pd.read_csv(f"{path_base}corpus_tweets.csv", sep = ",", dtype = dic_id)


# In[5]:



#users = pd.read_csv('../outcome/user_sm_predicted.csv',dtype=dic_id) #jean-philippe
users = pd.read_csv(f'{path_base}recoded_user_sm_predicted.csv',dtype=dic_id)


# In[6]:


df0 = df0[['query', 'id', 'timestamp_utc', 'local_time',
           'user_screen_name', 'text',  'user_location',  'user_id', 'user_name',
           'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK',
           'BRAF', 'MET', 'RET', 'HER2', 'date']]


# In[7]:


df0['date'] = pd.to_datetime(pd.to_datetime(df0['date']).dt.date)
df0['Year'] = df0['date'].dt.year


# In[8]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2012)]


# # Le rôle des différents groupes dans la publicisation des biomarqueurs

# In[9]:


def time_series(data, time_length, x, year_base):

    data_user = data.drop_duplicates(subset = [time_length, "user_id"])
    time_author = data_user.groupby([time_length, x]).agg(nb_user_by_group = ("user_id", "count")).reset_index()
    timeseries = data.groupby([time_length, x]).agg(nb_tweets_by_group = ("id", "count")).reset_index()

    datadate = data.groupby([time_length]).agg(nb_tweets = ("id", "count")).reset_index()
    datadate2 = data_user.groupby([time_length]).agg(nb_users = ("user_id", "count")).reset_index()
    datadate = datadate.merge(datadate2, on = [time_length], how = "left")
    timeseries = timeseries.merge(datadate, on = [time_length], how = "left")
    timeseries = timeseries.merge(time_author, on = [time_length, x], how = "left")
    timeseries["prop_tweets"] = timeseries["nb_tweets_by_group"]/timeseries["nb_tweets"]*100
    timeseries["prop_users"] = timeseries["nb_user_by_group"]/timeseries["nb_users"]*100
    timeseries["average_publication"] = timeseries["nb_tweets_by_group"]/timeseries["nb_user_by_group"]
    timeseries.index = timeseries[time_length]
    timeseries = timeseries.drop(columns = [time_length])
    timeseries['tx_var_tweets'] = (timeseries.groupby([x]).nb_tweets_by_group.pct_change())*100
    timeseries['tx_var_users'] = (timeseries.groupby([x]).nb_user_by_group.pct_change())*100
    timeseries['CM_tweets'] = (timeseries['tx_var_tweets'] / 100)+1
    timeseries['CM_users'] =  (timeseries['tx_var_users'] / 100)+1
    
    datatime = timeseries.copy()
    datatime[time_length] = datatime.index
    list_year = []
    list_role = []
    list_base_100_tweet = []
    list_base_100_user = []
    for n in datatime[x].unique():
        datatemp = datatime.loc[datatime[x] == n]
        first_year = np.min(datatemp.index)

        datatemp = datatemp.loc[datatemp[time_length] == year_base]
        datatemp = datatemp.drop_duplicates()
        
        #print(len(datatemp))
        if len(datatemp) == 1:
            ref_value_tweet = datatemp["nb_tweets_by_group"].unique()[0]
            ref_value_user = datatemp["nb_user_by_group"].unique()[0]
        else:
            ref_value_tweet = 0
            ref_value_user = 0
        #print(ref_value_tweet, ref_value_user)

        list_year.append(first_year)
        list_role.append(n)
        list_base_100_tweet.append(ref_value_tweet)
        list_base_100_user.append(ref_value_user)

    
    data = {x : list_role, "ref_value_tweet" : list_base_100_tweet,
           "ref_value_user" : list_base_100_user}
    data_base = pd.DataFrame(data)
    datatime = datatime.drop_duplicates()
    datatemp = datatime.merge(data_base, on = [x], how = "left")
    datatemp["base_100_tweets"] = (datatemp["nb_tweets_by_group"]/datatemp["ref_value_tweet"])*100
    datatemp["base_100_users"] = (datatemp["nb_user_by_group"]/datatemp["ref_value_user"])*100
    datatemp

    
    
    return(datatemp)


# In[10]:


time_length = "Year"
x = "User_status"
what = "id"


# In[11]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["ALK"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users" :
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur ALK<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[12]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["ROS1"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant<br>le biomarqueur ROS1 en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[13]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["HER2"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur HER2<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[14]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["BRAF"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur BRAF<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[15]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["EXON"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur EXON<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[16]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["EGFR"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur EGFR<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[17]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["KRAS"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur KRAS<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[18]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["NTRK"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur NTRK<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[19]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["MET"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur MET<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[20]:



df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012) &
           (df["RET"]>=1)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users":
            fig1.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig1.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig1.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer in fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur RET<br>en fonction des rôles (variable : {role_variable})',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()


# In[21]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2012)]

df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]


# In[22]:


def time_series4(data, time_length, year_base, bio):
    
    dfdate = df.groupby([time_length]).agg(nb_tweet = ("id", "count")).reset_index()
    datadate = data.loc[(data[bio] == 1)& (data["Year"] >= 2012)]
    datadate = datadate.groupby([time_length, bio]).agg(nb_mentionned_biom = ("id", "count")).reset_index()
    datadate = datadate.merge(dfdate, on = [time_length], how = "left")
    
    datadate['tx_var_mentions'] = datadate.nb_mentionned_biom.pct_change()*100
    datadate['CM_mentions'] = (datadate['tx_var_mentions'] / 100)+1
    
    if len(datadate["nb_mentionned_biom"].loc[datadate[time_length] == year_base]) >=1:
        datadate["mentions_year_ref"] = datadate["nb_mentionned_biom"].loc[datadate[time_length] == year_base].unique()[0]
    else:
        datadate["mentions_year_ref"] = 0
    
    datadate["base_100_mentions"] = (datadate["nb_mentionned_biom"]/datadate["mentions_year_ref"])*100
    datadate["prop_of_biom"] = datadate["nb_mentionned_biom"]/datadate["nb_tweet"]*100
    
    datadate["biomarker"] = bio
    datadate = datadate.drop(columns = [bio])
    
    return(datadate)

    
    
    


# In[23]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#


# In[24]:


biom=['ALK','EXON', "MET"]#

df_stat = df.loc[df["User_status"] == "Oncologist"]

for i,bio in enumerate(biom):
    df_tmp = time_series4(data = df_stat, time_length = "Year", year_base = 2018, bio = bio)
    
    #sns.lineplot( x = 'Year', y = "prop_of_biom", data = df_tmp)
    
    if i==0:
        biomm=df_tmp
    else:
        biomm=pd.concat([biomm,df_tmp])

time_length = "Year"


list_of_compute = ['nb_mentionned_biom', 'prop_of_biom', #'average_publication',
       'base_100_mentions', 'tx_var_mentions']


fig4 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = biomm.copy()
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1["biomarker"].unique()):
        if col == "prop_of_biom":# or col == "nb_mentionned_biom":
            fig4.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1["biomarker"]==n)],
                    y= df1[col].loc[(df1["biomarker"]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig4.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1["biomarker"]==n)],
                y= df1[col].loc[(df1["biomarker"]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig4.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_mentionned_biom" in fig4.data[k].meta[0]:
        fig4.update_traces(visible=True, selector = k)
    else:
        fig4.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):
    list_to_display = []

    for i, trace in enumerate(fig4.data):
        #print(fig.data[i].meta[1])
        if customer in fig4.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True,
                        }])




fig4.update_layout(
    title= f'Evolution annuelle du nombre de tweets publiés par les oncologues mentionnant<br>les biomarqueurs ALK, EXON et MET',
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(list_of_compute)],
            x= 1.2,
            y= 1.2
        ),

    ]
)


#print(fig.layout.updatemenus)




fig4.show()

