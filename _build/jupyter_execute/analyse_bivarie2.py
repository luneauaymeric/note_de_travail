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
# 
# Les graphiques ci-dessous visent à compléter l'analyse présentée à la page précédente en regardant pour chaque biomarqueur la place occupée par les différentes catégories d'acteurs dans sa diffusion. Cette diffusion est appréhendée à travers le nombre de tweets mentionnant au moins une fois un biomarqueur. 
# 
# Outre les valeurs absolues, plusieurs procédure de normalisation ont été appliquée. La première, toute simple, consiste à calculer pour chaque année la proportion de tweets contenant le biomarqueur *x* publiée par chacune des catégories sur l'ensemble des tweets contenant ce biomarqueur. 
# 
# La deuxième donne la proportion de tweets contenant le biomarqueur *x* publiés par une catégorie sur l'ensemble des tweets de cette catégorie.
# 
# La troisème rapporte le nombre de tweets contenant au moins une référence au biomarqueur *x* sur l'ensemble des tweets contenant un biomarqueur publiés par chacune des catégories d'acteurs.

# In[9]:


def time_series(data, time_length, x, year_base, biom):
    """
    data= data framne
    time_lenght = time variable (year, month, day)
    x = la variale utilisée pour regrouper les acteurs en statuts (User_status, User_status_1, etc.)
    year_base = année de référence pour le calcule des indices en base 100
    biom : le biomarqueur à considérer
    """

    data_user = data.drop_duplicates(subset = [time_length, "user_id"])
    time_author = data_user.groupby([time_length, x]).agg(nb_user_by_group = ("user_id", "count")).reset_index()
    timeseries = data.groupby([time_length, x]).agg(nb_tweets_by_group = ("id", "count")).reset_index()
    
    data_biom = data.loc[data[biom] == 1]
    data_user_biom = data_biom.drop_duplicates(subset = [time_length, "user_id"])
    
    on_biom = data.loc[data["somme_biom"] >= 1]
    timeseries2 = on_biom.groupby([time_length, x]).agg(nb_tweets_on_marker_by_group = ("id", "count")).reset_index()
    data_user2 = on_biom.drop_duplicates(subset = [time_length, "user_id"])
    time_author2 = data_user2.groupby([time_length, x]).agg(nb_user_on_marker_by_group = ("user_id", "count")).reset_index()
    
    
    
    time_author_biom = data_user_biom.groupby([time_length, x]).agg(nb_user_by_group_biom = ("user_id", "count")).reset_index()
    timeseries_biom = data_biom.groupby([time_length, x]).agg(nb_tweets_by_group_biom = ("id", "count")).reset_index()
    
    datadate = data.groupby([time_length]).agg(nb_tweets = ("id", "count")).reset_index()
    datadate2 = data_user.groupby([time_length]).agg(nb_users = ("user_id", "count")).reset_index()
    
    datadate3 = data_biom.groupby([time_length]).agg(nb_tweets_biom = ("id", "count")).reset_index()
    datadate4 = data_user_biom.groupby([time_length]).agg(nb_users_biom = ("id", "count")).reset_index()
    
    datadate = datadate.merge(datadate2, on = [time_length], how = "left")     .merge(datadate3, on = [time_length], how = "left").merge(datadate4, on = [time_length], how = "left")
    
    timeseries = timeseries    .merge(datadate, on = [time_length], how = "left")
    
    timeseries = timeseries.merge(timeseries2, on = [time_length, x], how = "left")    .merge(timeseries_biom, on = [time_length, x], how = "left")    .merge(time_author, on = [time_length, x], how = "left")    .merge(time_author2, on = [time_length, x], how = "left")    .merge(time_author_biom, on = [time_length, x], how = "left")
    
    timeseries["prop_tweets"] = timeseries["nb_tweets_by_group"]/timeseries["nb_tweets"]*100
    timeseries["prop_users"] = timeseries["nb_user_by_group"]/timeseries["nb_users"]*100
    
    timeseries["prop_tweets_biom_in_group"] = timeseries["nb_tweets_by_group_biom"]/timeseries["nb_tweets_by_group"]*100
    timeseries["prop_users_biom_in_group"] = timeseries["nb_user_by_group_biom"]/timeseries["nb_user_by_group"]*100
    
    timeseries["prop_tweets_biom_by_group"] = timeseries["nb_tweets_by_group_biom"]/timeseries["nb_tweets_biom"]*100
    timeseries["prop_users_biom_by_group"] = timeseries["nb_user_by_group_biom"]/timeseries["nb_users_biom"]*100
    
    timeseries["prop_tweets_on_biom_by_group"] = timeseries["nb_tweets_by_group_biom"]/timeseries["nb_tweets_on_marker_by_group"]*100
    timeseries["prop_users_on_biom_by_group"] = timeseries["nb_user_by_group_biom"]/timeseries["nb_user_on_marker_by_group"]*100
    
    
    timeseries["average_publication"] = timeseries["nb_tweets_by_group"]/timeseries["nb_user_by_group"]
    timeseries.index = timeseries[time_length]
    
    
    timeseries['tx_var_tweets'] = (timeseries.groupby([x]).nb_tweets_by_group_biom.pct_change())*100
    timeseries.loc[timeseries['Year'] == 2012, 'tx_var_tweets'] = 0
    timeseries['tx_var_users'] = (timeseries.groupby([x]).nb_user_by_group_biom.pct_change())*100
    timeseries.loc[timeseries['Year'] == 2012,'tx_var_users'] = 0
    
    timeseries['CM_tweets'] = (timeseries['tx_var_tweets'] / 100)+1
    timeseries.loc[timeseries['Year']==2012, 'CM_tweets'] = 1
    timeseries['CM_users'] =  (timeseries['tx_var_users'] / 100)+1
    timeseries.loc[timeseries['Year'] == 2012, 'CM_users'] = 1
    timeseries['nb_tweet_biom_reel'] =  timeseries["nb_tweets_by_group_biom"]/timeseries['CM_users']
    valeur_depart = timeseries["nb_tweets_by_group_biom"]/timeseries['CM_tweets']
    timeseries['tx_var_tweets_reel'] = (timeseries['nb_tweet_biom_reel'] - valeur_depart)/valeur_depart*100
    timeseries = timeseries.drop(columns = [time_length])
    
    
    datatime = timeseries.copy()
    datatime[time_length] = datatime.index
    list_year = []
    list_role = []
    list_base_100_tweet = []
    list_base_100_user = []
    list_base_100_tweet_biom = []
    list_base_100_user_biom = []
    for n in datatime[x].unique():
        datatemp = datatime.loc[datatime[x] == n]
        first_year = np.min(datatemp.index)

        datatemp = datatemp.loc[datatemp[time_length] == year_base]
        datatemp = datatemp.drop_duplicates()
        
        if len(datatemp) == 1:
            ref_value_tweet_biom = datatemp["nb_tweets_by_group_biom"].unique()[0]
            ref_value_user_biom = datatemp["nb_user_by_group_biom"].unique()[0]
            ref_value_tweet = datatemp["nb_tweets_by_group"].unique()[0]
            ref_value_user = datatemp["nb_user_by_group"].unique()[0]
        else:
            ref_value_tweet_biom = 0
            ref_value_user_biom = 0
            ref_value_tweet = 0
            ref_value_user = 0
        #print(ref_value_tweet, ref_value_user)

        list_year.append(first_year)
        list_role.append(n)
        list_base_100_tweet_biom.append(ref_value_tweet_biom)
        list_base_100_user_biom.append(ref_value_user_biom)
        list_base_100_tweet.append(ref_value_tweet)
        list_base_100_user.append(ref_value_user)

    
    data = {x : list_role, "ref_value_tweet" : list_base_100_tweet,
           "ref_value_user" : list_base_100_user,
           "ref_value_tweet_biom" : list_base_100_tweet_biom,
           "ref_value_user_biom" : list_base_100_user_biom}
    
    data_base = pd.DataFrame(data)
    datatime = datatime.drop_duplicates()
    datatemp = datatime.merge(data_base, on = [x], how = "left")
    datatemp["base_100_tweets_biom"] = (datatemp["nb_tweets_by_group_biom"]/datatemp["ref_value_tweet_biom"])*100
    datatemp["base_100_users_biom"] = (datatemp["nb_user_by_group_biom"]/datatemp["ref_value_user_biom"])*100
    
    datatemp["base_100_tweets"] = (datatemp["nb_tweets_by_group"]/datatemp["ref_value_tweet"])*100
    datatemp["base_100_users"] = (datatemp["nb_user_by_group"]/datatemp["ref_value_user"])*100
    
    #datatemp["valeur_corrigee"] = datatemp["nb_tweets_by_group_biom"]*(datatemp["base_100_users"]/100)
    #datatemp["base_100_tweets_corrigee"] =  (((datatemp["valeur_corrigee"] - datatemp["ref_value_tweet"])/datatemp["ref_value_tweet"])+1)*100
    
    
    return(datatemp)


# In[10]:


time_length = "Year"
x = "User_status"
year_base = 2018

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = [# 'nb_user_by_group', 
    'nb_tweets_by_group', 'nb_user_by_group',
    "prop_tweets", "prop_users",
    "base_100_tweets", "base_100_users"# "base_100_tweets_corrigee"#'average_publication',
       ]


# In[11]:


biomarker = "ALK"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]



role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets' or col == 'prop_users' :
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

    if "nb_tweets" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets <br>en fonction des rôles (variable : {role_variable})',
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


# ````{margin}
# ```{note}
# "nb_tweets" : Nombre de tweets par groupe.
# 
# "nb_users" : Nombre d'utilisateurs par groupe.
# 
# "prop_tweets" : Part des tweets publiés par un groupe sur l'ensemble des tweets du corpus
# 
# "prop_users" : Part des utilisateurs d'un groupe sur l'ensemble des utilisateurs
# 
# "base_100_tweets" : Evolution en base 100 du nombre de tweets dans le corpus 
# 
# "base_100_users" : Evolution en base 100 du nombre d'utilisateurs. 
# 
# "nb_tweets_by_group_biom" : Nombre de tweets par groupe contenant le biomarqueur *x*.
# 
# "nb_user_by_group_biom" : Nombre d'utilisateur par groupe ayant publié au moins un tweet contenant le biomarqueur *x*.
# 
# "prop_tweets_biom_by_group" : Part des tweets publiés par un groupe sur l'ensemble des tweets contenant le biomarqueur *x*.
# 
# "prop_users_biom_by_group" : Part des utilisateurs d'un groupe sur l'ensemble des utilisateurs ayant publié au moins un tweet contenant le biomarqueur *x*.
# 
# "prop_tweets_biom_in_group" : Part des tweets contenant le biomarqueur *x* sur l'ensemble des tweets  publiés par un groupe.
# 
# "prop_users_biom_in_group" : Part des utilisateurs ayant publié au moins un tweet contenant le biomarqueur *x* sur l'ensemble des utilisateurs du groupe.
# 
# "prop_tweets_on_biom_by_group" : Part des tweets contenant le biomarqueur *x* sur l'ensemble des tweets contenant un biomarqueur publiés par un groupe.
# 
# "prop_users_on_biom_by_group" : Part des utilisateurs ayant publié au moins un tweet contenant le biomarqueur *x* sur l'ensemble des utilisateurs du groupe ayant mentionné au moins une fois un biomarqueur dans leurs tweets.
# 
# "base_100_tweets_biom" : Evolution en base 100 du nombre de tweets contenant le biomarqueur *x*. 
# 
# "base_100_users_biom" : Evolution en base 100 du nombre d'utilisateurs ayant publié au moins un tweet contenant le biomarqueur *x*. 
# 
# ```
# ````

# In[12]:


time_length = "Year"
x = "User_status"
year_base = 2018

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = [# 'nb_user_by_group', 
    'nb_tweets_by_group_biom', 'nb_user_by_group_biom',
    "prop_tweets_biom_by_group", "prop_users_biom_by_group",
    'prop_tweets_biom_in_group', 'prop_users_biom_in_group',
    "prop_tweets_on_biom_by_group", "prop_users_on_biom_by_group",
    'base_100_tweets_biom', 'base_100_users_biom' # "base_100_tweets_corrigee"#'average_publication',
       ]


# In[13]:


biomarker = "ALK"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2013)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


# Dans le cas du biomarqueur ALK par exemple, 23 advocacy patients ont publiés en 2018 plus de 1000 tweets y faisant référence. Ils représentent ainsi 14% des comptes ayant parlés au moins une fois de ALK et sont à l'origine de 52% des tweets sur ce biomarqueur en 2018. Cette même année, 58 oncologues ont twitté 220 messages sur le même biomarqueur. Ils représentaient donc 36% des comptes publiant sur ALK en 2018, mais n'ont été à l'origine que de 11% des tweets.
# 
# Ramené au nombre total de tweets publiés par ces deux catégories, on constate qu'en 2018 moins de 1% des tweets des oncologues contenaient une référence à ALK, contre 5% pour les advocacy patients. En termes de compte, 10% des oncologues et 17% des advocacy patients ont publié au moins un tweet parlant de ALK.
# 
# Enfin, 67% des tweets mentionnant un biomarqueur publié par les advocacy patients en 2018 concernent ALK. Dans le cas des oncologues cette proportion passe à 19%.
# 
# 

# In[14]:


biomarker = "ROS1"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2013)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


biomarker = "HER2"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


biomarker = "BRAF"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


biomarker = "EXON"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


biomarker = "EGFR"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


biomarker = "KRAS"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


biomarker = "NTRK"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2013)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


biomarker = "MET"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


# In[22]:


biomarker = "RET"
df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]

df = df.loc[(df["User_status"] != "Other") & 
            (df["User_status"] != "Undefined") & 
            (df["Year"] >= 2012)]


role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = year_base, biom = biomarker)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == 'prop_tweets_biom_by_group' or col == 'prop_users_biom_by_group':
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

    if "nb_tweets_by_group_biom" in fig1.data[k].meta[0]:
        fig1.update_traces(visible=True, selector = k)
    else:
        fig1.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig1.data):
        #print(fig.data[i].meta[1])
        if customer == fig1.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig1.update_layout(
    title= f'Evolution du nombre de comptes et de tweets mentionnant le biomarqueur {biomarker}<br>en fonction des rôles (variable : {role_variable})',
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


# In[23]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2012)]

df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]


# In[24]:


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

    
    
    


# In[25]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#


# In[26]:


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

