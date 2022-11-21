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


# In[9]:


len(df0)


# In[10]:


df1 = df0[['query', 'id', 'timestamp_utc', 'local_time',
           'user_screen_name', 'text',  'user_location',  'user_id', 'user_name',
           'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK',
           'BRAF', 'MET', 'RET', 'HER2', 'date', "nb_of_biomarker"]]


# In[20]:


df1_user = df1.groupby(["user_id"]).agg(sum_of_biomarker=("nb_of_biomarker", "sum"))


# In[23]:


df2_user= users.merge(df1_user,on=['user_id'], how = "left")#how = inner by default
len(df2_user)


# # Le rôle des différents groupes dans la publicisation des biomarqueurs
# 
# Les graphiques ci-dessous visent à compléter les analyses présentées dans les pages précédentes en ajoutant une dimension temporelle aux relations entre catégories d'acteurs et biomarqueurs. Il s'agit de cette manière de saisir le rôle de ces différents acteurs dans la diffusion de chacun des biomarqueurs à travers le nombre de tweets qu'ils publient.
# 
# Outre les valeurs absolues, plusieurs procédures de normalisation ont été appliquées. La première, toute simple, consiste à calculer pour chaque année la proportion de tweets contenant le biomarqueur *x* publiée par chacune des catégories sur l'ensemble des tweets contenant ce biomarqueur. 
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
    "prop_tweets", "prop_users",'tx_var_tweets',
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
    "prop_tweets_on_biom_by_group", "prop_users_on_biom_by_group", 'CM_tweets', 'CM_users',
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


# Par exemple, 23 advocacy patients ont publiés en 2018 plus de 1000 tweets faisant référence à ALK. Ils représentent ainsi 14% des comptes ayant parlés au moins une fois de ALK et sont à l'origine de 52% des tweets sur ce biomarqueur en 2018. Cette même année, 58 oncologues ont twitté 220 messages sur le même biomarqueur. Ils représentaient donc 36% des comptes publiant sur ALK en 2018, mais n'ont été à l'origine que de 11% des tweets. Ramené au nombre total de tweets publiés par ces deux catégories, on constate qu'en 2018 moins de 1% des tweets des oncologues contenaient une référence à ALK, contre 5% pour les advocacy patients. En termes de compte, 10% des oncologues et 17% des advocacy patients ont publié au moins un tweet parlant de ALK. Enfin, 67% des tweets mentionnant un biomarqueur publié par les advocacy patients en 2018 concernent ALK. La proportion des oncologues  passe quant à elle à 19%.
# 
# Dans le cas des advocacy patients, c'est justement en 2018 que l'attention portée à ALK atteint son maximum. Le nombre de tweets mentionnant ce biomarqueur est multipliés par 15 entre 2017 et 2018, pandis que le nombre d'utilisateurs à l'origine de ces messages est multiplié par 2 : on dénombre 10 comptes d'advocacy patients en 2017 et 23 en 2018. On constate en revanche que le nombre de tweets publiés par des oncologues mentionnant ALK continue de progresser jusqu'en 2019. Le nombre de tweets décroit à partir de 2020 et atteint en 2021 un volume inférieur à celui de 2018. Il est intéressant de noter que la croissance du nombre d'oncologues publiant des tweets sur ALK ralentit dès 2017, là où l'augmentation des tweets eux-mêmes s'essoufle à partir de 2018.
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


# Si ROS1 semble lui-aussi être un biomarqueur fortement investi par les advocacy patients, sa dynamique présente quelques différences par rapport à ALK, notamment si on compare les tweets des advocacy patients avec ceux des oncologues. On observe toujours une première phase de latence où les volumes de tweets publiés par ces deux catégories d'acteurs sont proches. Jusqu'en 2016, le plus gros écart observé est en 2015 où on compte 39 tweets faisant référence à ROS1 postés par quatre oncologues pour 19 tweets envoyés par deux advocacy patients.On observe ensuite un décalage. La mobilisation des advocacy patients s'intensifient à partir de 2017, là où celle des oncologues s'accroit à partir de 2018. Elles atteignent toutes les deux leurs acmées en 2019. Si l'engagement des oncologues reste relativement en retrait, elle semble suivre malgré tout la mobilisation des advocacy patients au contraire de ALK.

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


# HER2 a également un profil intéressant. On a vu que les médias ont un lien particulier avec ce marqueur. Depuis, 2016, ils sont à l'origine de plus d'un tiers des tweets y faisant référence. Entre 2016 et 2021, les tweets sur HER2 ont représenté entre 50 et 40% de tous les messages publiés par des médias contenant un biomarqueur. À titre de comparaison, le taux des tweets concernant ALK sur l'ensemble des tweets des médias contenant un biomarqueur varient entre 20 et 5% sur cette même période. Les organisations non-lucratives (NPO) ont aussi contribué à la publicisation de HER2. Bien qu'en 2021, les NPO n'occupent que le troisième rang volume de tweets, derrière les médias et les oncologues, elles passent premières une fois ce volume ramenée à l'ensemble des tweets publiées par les NPO : sur 9627 tweets publiés en 2021, 351 contenaient le terme "HER2" (soit 3,6%).
# 
# En revanche, HER2 n'est pas le biomarqueur sur lequel les advocacy patients se sont mobilisés le plus. Mis à part en 2015 où 46%  des tweets sur HER2 ont été publiés par des advocacy patients, leur "contribution" n'a pas dépassé les 10%. On notera également que les messages sur HER2 représentent au mieux 1,4% de l'ensemble des tweets publiés par les advocacy patients. En 2021, ils ont été à l'origine de 2% des tweets évoquant HER2. Un volume de tweets qui représentait cette année-là (2021) 0,5% de l'ensemble de leur publication.

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


# In[27]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2012)]
df = df.sort_values(["Year"])
variable = "User_status"

for annee in df["Year"].unique():
    biom = ['ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']
    dfy = df.loc[df["Year"] == annee]
    df_status = dfy[[variable, "Year", "id", 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]



    for i, bio in enumerate(biom):
        df_tmp = df_status.loc[df_status[bio]==1].groupby([variable, bio]).agg(bio = ("id", "count")).reset_index()
        df_tmp = df_tmp[[variable,"bio"]].rename(columns = {"bio": bio})

        if i==0:
            pivot_table1 = df_tmp
        else:
            pivot_table1 = pivot_table1.merge(df_tmp, how = "left", on = [variable]) 

    pivot_table1["somme_ligne"] = pivot_table1[biom].sum(axis=1)


    df_tmp = pivot_table1.copy()
    biom.append("somme_ligne")
    for i, status in enumerate(pivot_table1[variable]):
        for i, bio in enumerate(biom):
            df_tmp[bio] = pivot_table1[bio]#/pivot_table["somme_ligne"]*100

    pivot_table1 = df_tmp[[variable, 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2', 'somme_ligne']]
    pivot_table1.index = pivot_table1[variable]
    pivot_table1 = pivot_table1.drop(columns = [variable])
    pivot_table1.loc['Column_Total']= pivot_table1.sum(numeric_only=True, axis=0)
    pivot_table1 = pivot_table1.fillna(0)
    
    list_status = []
    pivot_table1 = pivot_table1.reset_index()
    #col = biom.append("User_status")
    #dft = pd.DataFrame(columns = [biom])
    dict_score = {}
    
    biom = ['ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']
    
    
    for bio in biom:
        list_score = []
        for i, x in enumerate(pivot_table1[bio]):
            n = pivot_table1["somme_ligne"].iloc[i]
            m = pivot_table1[bio].iloc[-1]
            total = pivot_table1["somme_ligne"].iloc[-1]
            a = x
            b = n-x
            c= m-x
            d = (total-n)-(c)
            numerateur = (a*d)-(b*c)
            denominateur = np.sqrt(n*m*(total-n)*(total-m))
            np.seterr(divide='ignore', invalid='ignore')
            phi_score = np.divide(numerateur, denominateur)
            #phi_score2 = ((a)) / np.sqrt(n*m)
            chi_square_value = total*np.square(phi_score)
            normalised_score = x/(n*m)

            if chi_square_value > 6.6349:
                list_score.append(phi_score)
            else:
                list_score.append(np.nan)

            dict_score[bio] = list_score

    
    for x in pivot_table1["User_status"]:
        list_status.append(x)
        dict_score["User_status"] = list_status
    
    dft = pd.DataFrame(dict_score)
    dft.index = dft ["User_status"]
    dft = dft.drop(columns=["User_status"])
    dft = dft.drop(labels=["Column_Total"])
    dft["Year"] = annee
    
    if a ==0:
        dft1 = dft.copy()
    else:
        dft1= pd.concat([dft1, dft], axis = 0)



role_variable = "User_status"
dft1 = dft1.reset_index().sort_values([role_variable, "Year"])

biom = ['ALK', 'ROS1' , 'HER2', 'EGFR', 'EXON', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'EXON']

fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
for z, col in enumerate(biom):
    for i, n in enumerate(dft1[role_variable].unique()):
        fig1.append_trace(
        go.Scatter(
            x= dft1["Year"].loc[(dft1[role_variable]==n)],
            y= dft1[col].loc[(dft1[role_variable]== n)],
            name = n,
            meta= [col]),
        1,1)

#
Ld=len(fig1.data)
Lc =len(dft1.columns[0:9])



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "ROS1" in fig1.data[k].meta[0]:
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
    title= f"Evolution des relations entre catégories d'acteurs et biomarqueurs (coefficient de phi)",
    updatemenus=[
        go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(biom)],
            x= 1.2,
            y= 1.8
        ),

    ]
)


#print(fig.layout.updatemenus)




fig1.show()

    


# Le dernier diagramme ci-dessus rend compte de l'évolution des relations mesurées à l'aide du coefficient de *Phi* (voir page précédente) entre catégories d'acteurs et biomarqueurs. Comme pour la heatmap affiché sur la page précédente, seuls les coefficients de *Phi* significatifs (seuil à 0,01) sont dessinés, ce qui explique les points isolés. On espère grâce à ce diagramme donné une vision plus synthétique des dynamiques entre catégories d'acteurs et biomarqueurs. Si on prend HER2 décrit plus haut, on observe une relation significativement positive entre les médias et ce marqueur depuis 2016. Au contraire, depuis 2017, cette relation est plutôt négative dans le cas des advocacy patients. Il est également de s'arrêter sur les oncologues qui, jusqu'en 2017, semblent peu mobilisés sur HER2.

# In[ ]:




