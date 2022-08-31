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


df0= pd.read_csv(f"{path_base}corpus_tweets.csv", sep = ",", dtype = dic_id)


# In[55]:



#users = pd.read_csv('../outcome/user_sm_predicted.csv',dtype=dic_id) #jean-philippe
users = pd.read_csv(f'{path_base}recoded_user_sm_predicted.csv',dtype=dic_id)


# In[56]:


df0 = df0[['query', 'id', 'timestamp_utc', 'local_time',
           'user_screen_name', 'text',  'user_location',  'user_id', 'user_name',
           'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK',
           'BRAF', 'MET', 'RET', 'HER2', 'date']]


# In[57]:


df0['date'] = pd.to_datetime(pd.to_datetime(df0['date']).dt.date)
df0['Year'] = df0['date'].dt.year


# In[58]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2012)]


# In[59]:


def time_series2(data, time_length, year_base):

    data_user = data.drop_duplicates(subset = [time_length, "user_id"])
    datadate = data.groupby([time_length]).agg(nb_tweets = ("id", "count")).reset_index()
    datadate2 = data_user.groupby([time_length]).agg(nb_users = ("user_id", "count")).reset_index()
    datadate = datadate.merge(datadate2, on = [time_length], how = "left")
    datadate["average_publication"] = datadate["nb_tweets"]/datadate["nb_users"]
    datadate['tx_var_tweets'] = datadate.nb_tweets.pct_change()*100
    datadate['tx_var_users'] = datadate.nb_users.pct_change()*100
    datadate['CM_tweets'] = (datadate['tx_var_tweets'] / 100)+1
    datadate['CM_users'] =  (datadate['tx_var_users'] / 100)+1
    tweets_year_ref = datadate["nb_tweets"].loc[datadate[time_length] == year_base].unique()[0]
    users_year_ref = datadate["nb_users"].loc[datadate[time_length] == year_base].unique()[0]

    datadate["tweets_year_ref"] = tweets_year_ref
    datadate["users_year_ref"] = users_year_ref
    datadate["base_100_tweets"] = (datadate["nb_tweets"]/datadate["tweets_year_ref"])*100
    datadate["base_100_users"] = (datadate["nb_users"]/datadate["users_year_ref"])*100
    
    
    


    
    
    return(datadate)


# In[60]:


time_length = "Year"
year_base = 2018

dftime = time_series2(data = df, time_length = time_length, year_base = 2018)



# # Quelques graphiques
# 
# Les différents graphiques qui suivent donne un aperçu de la distribution dans le temps des rôles et des références aux différents biomarqueurs.

# In[61]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2011)]

time_length = "Year"
year_base = 2018

#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

list_of_compute = ['nb_tweets','nb_users', 'tx_var_tweets', #'average_publication',
       'base_100_tweets', 'base_100_users']


fig0 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series2(data = df, time_length = time_length, year_base = 2018)
for z, col in enumerate(list_of_compute):
    if col ==  "nb_tweets" or col ==  "nb_users":
        fig0.append_trace(
            go.Scatter(
                x= df1[time_length],
                y= df1[col],
                stackgroup='one',
                name = col,
            meta=  [col]),
            1,1)
    else :
        fig0.append_trace(
        go.Scatter(
            x= df1[time_length],
            y= df1[col],
            name = col,
            meta= [col]),
        1,1)

#
Ld=len(fig0.data)
Lc =len(list_of_compute)


#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets" in fig0.data[k].meta[0]:
        fig0.update_traces(visible=True, selector = k)
    else:
        fig0.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):
    #print("button ",k, customer)

    #print(len(fig.layout.updatemenus))
    #coef = len(df1[role_variable].unique())
    visibility= [False]*Lc
    list_to_display = []

    for i, trace in enumerate(fig0.data):
        #print(fig.data[i].meta[1])
        if customer in fig0.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])


fig0.update_layout(
    title= f'Evolution annuelle globale du nombre de comptes et de tweets)',
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




fig0.show()


# In[62]:


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
        
        ref_value_tweet = datatemp["nb_tweets_by_group"].unique()[0]
        ref_value_user = datatemp["nb_user_by_group"].unique()[0]

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


# 
# 
# ## Des professionnels de plus en plus présents
# 
# 
# L'analyse de la distribution des rôles montre la présence croissante des professionnels, en particulier des oncologues (*oncologists*) et, dans une moindre mesure, des chercheurs.

# In[63]:


time_length = "Year"
x = "User_status"
what = "id"


dftime = time_series(data = df, time_length = time_length, x = x, year_base= 2018)
dftime = time_series2(data = df, time_length = time_length, year_base= 2018)


# In[64]:


dftime = time_series2(data = df, time_length = time_length, year_base= 2018)

list_role = ["Global"]
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

dftemp = dftime.copy()
dftemp1 = dftemp.loc[dftemp["Year"] < 2018]
dftemp2 = dftemp.loc[dftemp["Year"] >= 2018]

Cum_CM = np.cumprod(dftemp["CM_tweets"][1:len(dftemp)])
CM_glob = Cum_CM.iloc[-1]
CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
list_CM_glob.append(CM_glob)
list_CM_moy.append(CM_moy)

Cum_CM1 = np.cumprod(dftemp1["CM_tweets"][1:len(dftemp1)])
CM_glob1 = Cum_CM1.iloc[-1]
CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
list_CM_glob1.append(CM_glob1)
list_CM_moy1.append(CM_moy1)

Cum_CM2 = np.cumprod(dftemp2["CM_tweets"][1:len(dftemp2)])
CM_glob2 = Cum_CM2.iloc[-1]
CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
list_CM_moy2.append(CM_moy2)
list_CM_glob2.append(CM_glob2)

data = {"role" : list_role,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_global_tweets = pd.DataFrame(data)


# In[65]:


dftime = time_series(data = df, time_length = time_length, x = x, year_base= 2018)

list_role = []
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

for role in dftime[x].unique():
    list_role.append(role)
    
    dftemp = dftime.loc[dftime[x]== role]
    dftemp1 = dftemp.loc[dftemp["Year"] < 2018]
    dftemp2 = dftemp.loc[dftemp["Year"] >= 2018]
    Cum_CM = np.cumprod(dftemp["CM_tweets"][1:len(dftemp)])
    CM_glob = Cum_CM.iloc[-1]
    CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
    list_CM_glob.append(CM_glob)
    list_CM_moy.append(CM_moy)
    
    Cum_CM1 = np.cumprod(dftemp1["CM_tweets"][1:len(dftemp1)])
    CM_glob1 = Cum_CM1.iloc[-1]
    CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
    list_CM_glob1.append(CM_glob1)
    list_CM_moy1.append(CM_moy1)
    
    Cum_CM2 = np.cumprod(dftemp2["CM_tweets"][1:len(dftemp2)])
    CM_glob2 = Cum_CM2.iloc[-1]
    CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
    list_CM_moy2.append(CM_moy2)
    list_CM_glob2.append(CM_glob2)

    
data = {"role" : list_role,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_tweets = pd.DataFrame(data)
evol_global_tweets2 = pd.concat([evol_global_tweets, evol_tweets])
evol_global_tweets2.index = evol_global_tweets2["role"]
evol_global_tweets2 = evol_global_tweets2.drop(columns = ["role"])

evol_global_tweets2.columns = pd.MultiIndex.from_product([['Ensemble de la période (2011-2021)', 'Période 2011-2017', 'Période 2018-2021'],
                                    ["Coefficient global","Coefficient annuel moyen"]])
evol_global_tweets2_style = evol_global_tweets2.style.format(precision=2, na_rep='').set_caption("Evolution du nombre de tweets par catégorie d'acteur (selon la variable User_status)") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)
evol_global_tweets2_style


# Le tableau ci-dessus montre que le nombre de tweets a été multiplié en moyenne par 1,21 chaque année entre 2011 et 2021. Cette croissance s'observe principalement sur la période 2011-2017. On observe ensuite un diminution progressive sur la période 2018-2021. Au cours de cette seconde période, la baisse la plus importante concerne les tweets publiés par les "Advocacy Patients" : le nombre de tweets en 2021 est divisé par deux par rapport à 2018 et a été divisé par 1,25 en moyenne les trois dernière années. À l'inverse la plus forte croissance est observée parmi les médias, dont le nombre de tweets a été multiplié par 1,12 en moyenne chaque année.

# In[66]:


dftime = time_series2(data = df, time_length = time_length, year_base= 2018)

list_role = ["Global"]
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

dftemp = dftime.copy()
dftemp1 = dftemp.loc[dftemp["Year"] < 2018]
dftemp2 = dftemp.loc[dftemp["Year"] >= 2018]

Cum_CM = np.cumprod(dftemp["CM_users"][1:len(dftemp)])
CM_glob = Cum_CM.iloc[-1]
CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
list_CM_glob.append(CM_glob)
list_CM_moy.append(CM_moy)

Cum_CM1 = np.cumprod(dftemp1["CM_users"][1:len(dftemp1)])
CM_glob1 = Cum_CM1.iloc[-1]
CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
list_CM_glob1.append(CM_glob1)
list_CM_moy1.append(CM_moy1)

Cum_CM2 = np.cumprod(dftemp2["CM_users"][1:len(dftemp2)])
CM_glob2 = Cum_CM2.iloc[-1]
CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
list_CM_moy2.append(CM_moy2)
list_CM_glob2.append(CM_glob2)

data = {"role" : list_role,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_global_users = pd.DataFrame(data)


# In[67]:


dftime = time_series(data = df, time_length = time_length, x = x, year_base= 2018)

list_role = []
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

for role in dftime[x].unique():
    list_role.append(role)
    
    dftemp = dftime.loc[dftime[x]== role]
    dftemp1 = dftemp.loc[dftemp["Year"] < 2018]
    dftemp2 = dftemp.loc[dftemp["Year"] >= 2018]
    Cum_CM = np.cumprod(dftemp["CM_users"][1:len(dftemp)])
    CM_glob = Cum_CM.iloc[-1]
    CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
    list_CM_glob.append(CM_glob)
    list_CM_moy.append(CM_moy)
    
    Cum_CM1 = np.cumprod(dftemp1["CM_users"][1:len(dftemp1)])
    CM_glob1 = Cum_CM1.iloc[-1]
    CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
    list_CM_glob1.append(CM_glob1)
    list_CM_moy1.append(CM_moy1)
    
    Cum_CM2 = np.cumprod(dftemp2["CM_users"][1:len(dftemp2)])
    CM_glob2 = Cum_CM2.iloc[-1]
    CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
    list_CM_moy2.append(CM_moy2)
    list_CM_glob2.append(CM_glob2)

    
data = {"role" : list_role,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_users = pd.DataFrame(data)
evol_global_users2 = pd.concat([evol_global_users, evol_users])
evol_global_users2.index = evol_global_users2["role"]
evol_global_users2 = evol_global_users2.drop(columns = ["role"])

evol_global_users2.columns = pd.MultiIndex.from_product([['Ensemble de la période (2011-2021)', 'Période 2011-2017', 'Période 2018-2021'],
                                    ["Coefficient global","Coefficient annuel moyen"]])
evol_global_users2_style = evol_global_users2.style.format(precision=2, na_rep='').set_caption("Evolution du nombre de comptes par catégorie d'acteur (selon la variable *User_status*)") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)
evol_global_users2_style


# Si on s'intéresse cette fois au nombre de comptes indépendamment du statut (voir "Global"), on observe une croissance annuelle moyenne de 27 % entre 2011 et 2021. On constate également une première phase de croissance globale entre 2011 et 2017, puis une relative stagnation entre 2018 et 2021. La diminution du nombre de tweets publiés par les "Advocacy patients" s'accompagne d'une baisse du nombre de comptes liés à cette  catégorie. En revanche, le nombre de compte des oncologues et les chercheurs ont continué de croître de 6 et 4 % par an en moyenne. Enfin, le nombre de média a diminiué de 3% par an entre 2018 et 2021. Rapporté à l'augmentation du nombre de tweets publiés par cette même catégorie, peut-on supposer l'émergence de médias spécialisés ?

# ````{margin}
# ```{note}
# La variable *User_status* est une "réduction" de la variable *User_role3*. La modalité *Advocacy* rassemble tous les comptes considérés comme jouant un rôle d’advocacy, sauf si ce sont des professionnels de santé ou des patients. Dans le cas des professionnels de santé, ceux qui joue un rôle d’advocacy sont placés avec les *Health professional*. Dans le cas des patients, on a distingué les patients qui font un travail de plaidoyer des autres patients. Par ailleurs, qu’il s’agisse des *Patients* ou des *Advocacy patients*, ces deux modalités réunissent en leur sein les survivors et les cancer patients.
# ```
# ````

# Les diagrammes ci-dessous représentent l'évolution du nombre de compte et de tweets publiés en fonction des différents recodages effectués.

# In[100]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2011)]


#df1 = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

variable = ["User_role3", "User_status", "User_status2", "User_status3"]
list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', #'average_publication',
       'base_100_tweets', 'base_100_users']


role_variable = variable[0]



fig = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users" or col ==  "nb_tweets_by_group":
            fig.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig.data[k].meta[0]:
        fig.update_traces(visible=True, selector = k)
    else:
        fig.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):
    #print("button ",k, customer)

    #print(len(fig.layout.updatemenus))
    coef = len(df1[role_variable].unique())
    visibility= [False]*coef*Lc
    list_to_display = []

    for i, trace in enumerate(fig.data):
        #print(fig.data[i].meta[1])
        if customer in fig.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])


fig.update_layout(
    title= f'Evolution du nombre de comptes et de tweets par role (variable : {role_variable})',
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




fig.show()


# In[ ]:





role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users" or col ==  "nb_tweets_by_group":
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
    title= f'Evolution du nombre de comptes et de tweets par role (variable : {role_variable})',
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


# In[ ]:



role_variable = variable[2]



fig2 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users" or col ==  "nb_tweets_by_group":
            fig2.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig2.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig2.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig2.data[k].meta[0]:
        fig2.update_traces(visible=True, selector = k)
    else:
        fig2.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):

    list_to_display = []

    for i, trace in enumerate(fig2.data):
        #print(fig.data[i].meta[1])
        if customer in fig2.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])



fig2.update_layout(
    title= f'Evolution du nombre de comptes et de tweets par role (variable : {role_variable})',
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




fig2.show()


# In[ ]:



role_variable = variable[3]



fig3 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2018)
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[role_variable].unique()):
        if col == "prop_tweets" or col == "prop_users" or col ==  "nb_tweets_by_group":
            fig3.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[role_variable]==n)],
                    y= df1[col].loc[(df1[role_variable]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig3.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[role_variable]==n)],
                y= df1[col].loc[(df1[role_variable]== n)],
                name = n,
                meta= [col]),
            1,1)

#
Ld=len(fig3.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_tweets_by_group" in fig3.data[k].meta[0]:
        fig3.update_traces(visible=True, selector = k)
    else:
        fig3.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):
    list_to_display = []

    for i, trace in enumerate(fig3.data):
        #print(fig.data[i].meta[1])
        if customer in fig3.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])




fig3.update_layout(
    title= f'Evolution du nombre de comptes et de tweets par role (variable : {role_variable})',
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




fig3.show()


# In[ ]:


#sns.lineplot(x = 'Year', y = "prop", data = dftime, hue = x)


# In[ ]:


import seaborn as sns
dftime = time_series(data = df, time_length = time_length, x = x, year_base = 2018)

g = sns.FacetGrid(dftime, row= x, hue = x,
                  height=1.7, aspect=6,)

g.map(sns.lineplot, "Year", "base_100_tweets")
g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
g.fig.suptitle('Evolution du nombre de tweets par statuts et par an (base 100 en 2018)')


# In[ ]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2012)]

df["somme_biom"] = df['ROS1'] +df['ALK']+df['EXON']+df['EGFR']+df['KRAS']+df['NTRK']+df['BRAF']+df["MET"]+df['RET']+df["HER2"]


# In[ ]:


def time_series3(data, time_length, year_base):

    datadate = data.loc[data["somme_biom"] >= 1]
    datadate = datadate.groupby([time_length]).agg(nb_mentionned_biom = ("id", "count")).reset_index()
    datadate['tx_var_mentions'] = datadate.nb_mentionned_biom.pct_change()*100
    datadate['CM_mentions'] = (datadate['tx_var_mentions'] / 100)+1

    datadate["mentions_year_ref"] = datadate["nb_mentionned_biom"].loc[datadate[time_length] == year_base].unique()[0]
    datadate["base_100_mentions"] = (datadate["nb_mentionned_biom"]/datadate["mentions_year_ref"])*100
    
    return(datadate)


# In[ ]:


def time_series4(data, time_length, year_base, bio):
    
    dfdate = df.groupby([time_length]).agg(nb_tweet = ("id", "count")).reset_index()
    datadate = data.loc[(data[bio] == 1)& (data["Year"] >= 2012)]
    datadate = datadate.groupby([time_length, bio]).agg(nb_mentionned_biom = ("id", "count")).reset_index()
    datadate = datadate.merge(dfdate, on = [time_length], how = "left")
    
    datadate['tx_var_mentions'] = datadate.nb_mentionned_biom.pct_change()*100
    datadate['CM_mentions'] = (datadate['tx_var_mentions'] / 100)+1

    datadate["mentions_year_ref"] = datadate["nb_mentionned_biom"].loc[datadate[time_length] == year_base].unique()[0]
    datadate["base_100_mentions"] = (datadate["nb_mentionned_biom"]/datadate["mentions_year_ref"])*100
    datadate["prop_of_biom"] = datadate["nb_mentionned_biom"]/datadate["nb_tweet"]*100
    
    datadate["biomarker"] = bio
    datadate = datadate.drop(columns = [bio])
    
    return(datadate)

    
    
    


# In[ ]:



df_biom = time_series3(data = df, time_length = "Year", year_base = 2018)


# In[ ]:


time_length = "Year"
list_of_compute = ['nb_mentionned_biom', 'tx_var_mentions', #'average_publication',
       'base_100_mentions']


fig01 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = df_biom = time_series3(data = df, time_length = "Year", year_base = 2018)
for z, col in enumerate(list_of_compute):
    if col ==  "nb_mentionned_biom":
        fig01.append_trace(
            go.Scatter(
                x= df1[time_length],
                y= df1[col],
                stackgroup='one',
                name = col,
            meta=  [col]),
            1,1)
    else :
        fig01.append_trace(
        go.Scatter(
            x= df1[time_length],
            y= df1[col],
            name = col,
            meta= [col]),
        1,1)

#
Ld=len(fig01.data)
Lc =len(list_of_compute)


#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_mentionned_biom" in fig01.data[k].meta[0]:
        fig01.update_traces(visible=True, selector = k)
    else:
        fig01.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):
    #print("button ",k, customer)

    #print(len(fig.layout.updatemenus))
    #coef = len(df1[role_variable].unique())
    visibility= [False]*Lc
    list_to_display = []

    for i, trace in enumerate(fig01.data):
        #print(fig.data[i].meta[1])
        if customer in fig01.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True}])


fig01.update_layout(
    title= f'Evolution annuelle globale du nombre de biomarqueurs mentionnés',
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




fig01.show()


# In[ ]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#

for i,bio in enumerate(biom):
    df_tmp = time_series4(data = df, time_length = "Year", year_base = 2018, bio = bio)
    
    #sns.lineplot( x = 'Year', y = "prop_of_biom", data = df_tmp)
    
    if i==0:
        biomm=df_tmp
    else:
        biomm=pd.concat([biomm,df_tmp])


# In[ ]:


list_of_compute = ['nb_mentionned_biom', 'prop_of_biom', #'average_publication',
       'base_100_mentions', 'tx_var_mentions']


fig4 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = biomm.copy()
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1["biomarker"].unique()):
        if col == "prop_of_biom" or col == "nb_mentionned_biom":
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
    title= f'Evolution annuelle du nombre de tweets mentionnant chacun des biomarqueurs',
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


# In[ ]:



g = sns.FacetGrid(biomm, row= 'biomarker', hue = 'biomarker',
                  height=1.7, aspect=6)

g.map(sns.lineplot, "Year", "base_100_mentions")
g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
g.fig.suptitle('Evolution annuelle du nombre de tweets mentionnant chacun des biomarqueurs')


# In[ ]:


df['month_year'] = df['date'].dt.to_period('M')
biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#

for i,bio in enumerate(biom):
    df_tmp = time_series4(data = df, time_length = "month_year", year_base = "2018-01", bio = bio)
    
    #sns.lineplot( x = 'Year', y = "prop_of_biom", data = df_tmp)
    
    if i==0:
        biomm=df_tmp
    else:
        biomm=pd.concat([biomm,df_tmp])


# In[ ]:


biomm["month"] = biomm["month_year"].dt.strftime('%Y-%m')
time_length = "month"

list_of_compute = ['nb_mentionned_biom', 'prop_of_biom',
       'base_100_mentions', 'tx_var_mentions']


fig5 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = biomm.copy()
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1["biomarker"].unique()):
        if col == "prop_of_biom" or col == "nb_mentionned_biom":
            fig5.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1["biomarker"]==n)],
                    y= df1[col].loc[(df1["biomarker"]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig5.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1["biomarker"]==n)],
                y= df1[col].loc[(df1["biomarker"]== n)],
                name = n,
                meta= [col]),
            1,1)


#
Ld=len(fig5.data)
Lc =len(list_of_compute)



#print(fig)
for k in range(0, Ld):
    #print(fig.data[k].meta[0])
    #print(role_variable)

    if "nb_mentionned_biom" in fig5.data[k].meta[0]:
        fig5.update_traces(visible=True, selector = k)
    else:
        fig5.update_traces(visible=False, selector = k)



def create_layout_button(k, customer):
    list_to_display = []

    for i, trace in enumerate(fig5.data):
        #print(fig.data[i].meta[1])
        if customer in fig5.data[i].meta[0] :
            list_to_display.append(True)
        else:
            list_to_display.append(False)
    return dict(label = customer,
                method = 'update',
                args = [{'visible': list_to_display,
                         'title': customer,
                         'showlegend': True,
                        }])




fig5.update_layout(
    title= f'Evolution mensuel nombre de tweets mentionnant chacun des biomarqueurs',
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




fig5.show()


# In[ ]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#

list_bio = ["Global"]
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

dftime = time_series3(data = df, time_length = "Year", year_base = 2018)


dftemp = dftime.copy()
dftemp1 = dftemp.loc[dftemp["Year"] < 2018]
dftemp2 = dftemp.loc[dftemp["Year"] >= 2018]

Cum_CM = np.cumprod(dftemp["CM_mentions"][1:len(dftemp)])
CM_glob = Cum_CM.iloc[-1]
CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
list_CM_glob.append(CM_glob)
list_CM_moy.append(CM_moy)

Cum_CM1 = np.cumprod(dftemp1["CM_mentions"][1:len(dftemp1)])
CM_glob1 = Cum_CM1.iloc[-1]
CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
list_CM_glob1.append(CM_glob1)
list_CM_moy1.append(CM_moy1)

Cum_CM2 = np.cumprod(dftemp2["CM_mentions"][1:len(dftemp2)])
CM_glob2 = Cum_CM2.iloc[-1]
CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
list_CM_moy2.append(CM_moy2)
list_CM_glob2.append(CM_glob2)

data = {"Biomarqueurs" : list_bio,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_global_mentions = pd.DataFrame(data)


# In[ ]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#

list_bio = []
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

for i,bio in enumerate(biom):
    list_bio.append(bio)
    dftime = time_series4(data = df, time_length = "Year", year_base = 2018, bio = bio)


    dftemp = dftime.copy()
    dftemp1 = dftemp.loc[dftemp["Year"] < 2018]
    dftemp2 = dftemp.loc[dftemp["Year"] >= 2018]

    Cum_CM = np.cumprod(dftemp["CM_mentions"][1:len(dftemp)])
    CM_glob = Cum_CM.iloc[-1]
    CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
    list_CM_glob.append(CM_glob)
    list_CM_moy.append(CM_moy)

    Cum_CM1 = np.cumprod(dftemp1["CM_mentions"][1:len(dftemp1)])
    CM_glob1 = Cum_CM1.iloc[-1]
    CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
    list_CM_glob1.append(CM_glob1)
    list_CM_moy1.append(CM_moy1)

    Cum_CM2 = np.cumprod(dftemp2["CM_mentions"][1:len(dftemp2)])
    CM_glob2 = Cum_CM2.iloc[-1]
    CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
    list_CM_moy2.append(CM_moy2)
    list_CM_glob2.append(CM_glob2)

data = {"Biomarqueurs" : list_bio,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_mentions = pd.DataFrame(data)

evol_global_mentions2 = pd.concat([evol_global_mentions, evol_mentions])
evol_global_mentions2.index = evol_global_mentions2["Biomarqueurs"]
evol_global_mentions2 = evol_global_mentions2.drop(columns = ["Biomarqueurs"])

evol_global_mentions2.columns = pd.MultiIndex.from_product([['Ensemble de la période (2012-2021)', 
                                                             'Période 2012-2017', 'Période 2018-2021'],
                                    ["Coefficient global","Coefficient annuel moyen"]])
evol_global_mentions2_style = evol_global_mentions2.style.format(precision=2, na_rep='').set_caption("Evolution annuelle du nombre de tweets par biomarqueur") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)

evol_global_mentions2_style


# L'évolution du nombre de biomarqueurs mentionnés dans les tweets a connu une croissance globale de 62% par an et de 4% par mois entre 2012 et 2021 (voir tableau ci-dessous). Cette croissance est surtout marquée dans les premières années. De 2012 à 2017, le nombre de tweets faisant au moins une référence à un biomarqueur est multiplié par deux en moyenne chaque année. Puis, entre 2018 et 2021, on observe une croissance annuelle moyenne de 7% environ.
# 
# Toutefois, les biomarqueurs ne connaissent pas la même évolution. On remarque par exemple que le nombre de tweets faisant au moins une référence au biomarqueur KRAS double en moyenne chaque année entre 2018 et 2021, tandis que les références au biomarqueur ROS1 ont diminué annuellement de 30% en moyenne. La forte croissance du biomarqueur EGFR doit par contre être interprétée avec précaution. Comme l'indique les graphiques ci-dessus, la croissance la plus forte se situe entre l'année 2012 et 2013 et est liée au très faible nombre de mentions en 2012 (seulement 2).

# In[101]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#


list_bio = ["Global"]
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

df2 = df.loc[(df["Year"] >= 2012)]
dftime = time_series3(data = df2, time_length = "month_year", year_base = "2018-06")

dftime["Year"] = dftime['month_year'].dt.year

dftemp = dftime.copy()
dftemp1 = dftemp.loc[dftemp["Year"] <= 2018]
dftemp2 = dftemp.loc[dftemp["Year"] > 2018]

Cum_CM = np.cumprod(dftemp["CM_mentions"][1:len(dftemp)])
CM_glob = Cum_CM.iloc[-1]
CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
list_CM_glob.append(CM_glob)
list_CM_moy.append(CM_moy)

Cum_CM1 = np.cumprod(dftemp1["CM_mentions"][1:len(dftemp1)])
CM_glob1 = Cum_CM1.iloc[-1]
CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
list_CM_glob1.append(CM_glob1)
list_CM_moy1.append(CM_moy1)

Cum_CM2 = np.cumprod(dftemp2["CM_mentions"][1:len(dftemp2)])
CM_glob2 = Cum_CM2.iloc[-1]
CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
list_CM_moy2.append(CM_moy2)
list_CM_glob2.append(CM_glob2)

data = {"Biomarqueurs" : list_bio,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_global_mentions = pd.DataFrame(data)


# In[ ]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#

list_bio = []
list_CM_glob = []
list_CM_glob1 = []
list_CM_glob2 = []

list_CM_moy = []
list_CM_moy1 = []
list_CM_moy2 = []

for i,bio in enumerate(biom):
    list_bio.append(bio)
    df2 = df.loc[(df["Year"] >= 2012)]
    dftime = time_series4(data = df2, time_length = "month_year", year_base = "2018-01", bio = bio)

    dftime["Year"] = dftime['month_year'].dt.year
    dftemp = dftime.copy()
    dftemp1 = dftemp.loc[dftemp["Year"] <= 2018]
    dftemp2 = dftemp.loc[dftemp["Year"] > 2018]

    Cum_CM = np.cumprod(dftemp["CM_mentions"][1:len(dftemp)])
    CM_glob = Cum_CM.iloc[-1]
    CM_moy = np.power(CM_glob, 1/(len(Cum_CM)))
    list_CM_glob.append(CM_glob)
    list_CM_moy.append(CM_moy)

    Cum_CM1 = np.cumprod(dftemp1["CM_mentions"][1:len(dftemp1)])
    CM_glob1 = Cum_CM1.iloc[-1]
    CM_moy1 = np.power(CM_glob1, 1/(len(Cum_CM1)))
    list_CM_glob1.append(CM_glob1)
    list_CM_moy1.append(CM_moy1)

    Cum_CM2 = np.cumprod(dftemp2["CM_mentions"][1:len(dftemp2)])
    CM_glob2 = Cum_CM2.iloc[-1]
    CM_moy2 = np.power(CM_glob2, 1/(len(Cum_CM2)))
    list_CM_moy2.append(CM_moy2)
    list_CM_glob2.append(CM_glob2)

data = {"Biomarqueurs" : list_bio,
        "Coef global 2011-2022" : list_CM_glob,
        "Coef moyenne 2011-2022" : list_CM_moy,
        "Coef global 2011-2017" : list_CM_glob1,
        "Coef moyenne 2011-2017" : list_CM_moy1,
        "Coef global 2018-2022" : list_CM_glob2,
        "Coef moyenne 2018-2022" : list_CM_moy2}

evol_mentions = pd.DataFrame(data)

evol_global_mentions2 = pd.concat([evol_global_mentions, evol_mentions])
evol_global_mentions2.index = evol_global_mentions2["Biomarqueurs"]
evol_global_mentions2 = evol_global_mentions2.drop(columns = ["Biomarqueurs"])

evol_global_mentions2.columns = pd.MultiIndex.from_product([['Ensemble de la période (2012-2021)', 
                                                             'Période 2012-2017', 'Période 2018-2021'],
                                    ["Coefficient mensuel global","Coefficient mensuel moyen"]])
evol_mensuelle_mentions2_style = evol_global_mentions2.style.format(precision=2, na_rep='').set_caption("Evolution mensuelle du nombre de tweets par biomarqueur") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)

evol_mensuelle_mentions2_style


# ## Qui parle de quel marqueur ?
# 
# Enfin, la matrice ci-après indique la part occupée par les différents biomarqueurs dans les tweets qui en mentionnent au moins un en fonction du statut de leurs auteurs. Par exemple, 30% des 2548 biomarqueurs cités par les "défenseurs de cause" (*advocacy*) concernent le marqueur EGFR. 

# In[102]:


df_status = df[["User_status", "id", 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]
df_tmp = df_status.loc[df_status["ALK"]==1].groupby(["User_status", "ALK"]).agg(ALK_c = ("id", "count"))


# In[103]:


biom = ['ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']
variable = "User_status"
df_status = df[[variable, "id", 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]


for i, bio in enumerate(biom):
    df_tmp = df_status.loc[df_status[bio]==1].groupby([variable, bio]).agg(bio = ("id", "count")).reset_index()
    df_tmp = df_tmp[[variable,"bio"]].rename(columns = {"bio": bio})
    
    if i==0:
        pivot_table = df_tmp
    else:
        pivot_table = pivot_table.merge(df_tmp, how = "left", on = [variable]) 

pivot_table["somme_ligne"] = pivot_table[biom].sum(axis=1)
pivot_table.loc['Column_Total']= pivot_table.sum(numeric_only=True, axis=0)
pivot_table = pivot_table.fillna("Total")

df_tmp = pivot_table.copy()
biom.append("somme_ligne")
for i, status in enumerate(pivot_table[variable]):
    for i, bio in enumerate(biom):
        df_tmp[bio] = pivot_table[bio]/pivot_table["somme_ligne"]*100
    
pivot_table = df_tmp[[variable, 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]
pivot_table.index = pivot_table[variable]
pivot_table = pivot_table.drop(columns = [variable])


# In[104]:


biom = ['ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']
variable = "User_status"
df_status = df[[variable, "id", 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]


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

pivot_table1_style = pivot_table1.style.format(precision=0, na_rep='').set_caption("Les nombre de références aux biomarqueurs en fonction du statut des auteurs") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top ; color: black ; font-size : 14pt'
 }], overwrite=False)

pivot_table1_style


# In[105]:


fig = px.imshow(pivot_table,color_continuous_scale='reds', title = "Proportion des références aux différents biomarqueurs en fonction du statut des auteurs")
fig.show()


# In[106]:




fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

cpalette = sns.color_palette("GnBu_d")
res = sns.heatmap(pivot_table, annot=True, linewidths=.5, fmt='.2f',  cmap="Reds")

for t in res.texts: t.set_text(t.get_text() + " %")
#plt.savefig('biomarkers_dist.pdf')
res.set(title ="Proportion des références aux biomarqueurs en fonction du statut des auteurs (% en ligne)")


# In[107]:


biom = ['ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']
variable = "User_status"
df_status = df[[variable, "id", 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]


for i, bio in enumerate(biom):
    df_tmp = df_status.loc[df_status[bio]==1].groupby([variable, bio]).agg(bio = ("id", "count")).reset_index()
    df_tmp = df_tmp[[variable,"bio"]].rename(columns = {"bio": bio})
    
    if i==0:
        pivot_table = df_tmp
    else:
        pivot_table = pivot_table.merge(df_tmp, how = "left", on = [variable]) 

pivot_table["Row_Total"] = pivot_table[biom].sum(axis=1)
pivot_table.loc['Column_Total']= pivot_table.sum(numeric_only=True, axis=0)
pivot_table = pivot_table.fillna("Total")

df_tmp = pivot_table.copy()

df_tmp.index = df_tmp[variable]
df_tmp = df_tmp.drop(columns = [variable])

pivot_table = pd.DataFrame.transpose(df_tmp).reset_index()
pivot_table
df_tmp = pivot_table.copy()

for i, status in enumerate(pivot_table["index"]):
    for i, bio in enumerate(pivot_table.columns[1:]):
        df_tmp[bio] = pivot_table[bio]/pivot_table["Total"]*100

pivot_table = df_tmp.drop(columns = ["Total"])
pivot_table.index = pivot_table["index"]
pivot_table = pivot_table.drop(columns = ["index"])

pivot_table2 = pivot_table.transpose()


# In[108]:


fig = px.imshow(pivot_table2,color_continuous_scale='reds', title = "Part des statuts en fonction des références aux différents biomarqueurs (% en colonne)")
fig.show()


# In[109]:




fig = plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')

cpalette = sns.color_palette("GnBu_d")
res = sns.heatmap(pivot_table2, annot=True, linewidths=.5, fmt='.2f',  cmap="Reds")

for t in res.texts: t.set_text(t.get_text() + " %")
#plt.savefig('biomarkers_dist.pdf')
res.set(title ="Part des statuts en fonction des références aux différents biomarqueurs (% en colonne)")


# 
# ```{note}
# Lecture de la matrice : 40% des références au biomarqueur MET sont le fait des oncologues, tandis que 27% des références au biomarqueur ALK proviennent des "Advocacy patiens".
# ```
# 

# In[110]:


pivot_table1 = pivot_table1.reset_index()


# In[111]:


biom = ['ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']
list_status = []
#col = biom.append("User_status")
#dft = pd.DataFrame(columns = [biom])
dict_score = {}

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
        #print(a, b, c, d, n, m, total)
        phi_score = ((a*d)-(b*c)) / np.sqrt(n*m*(total-n)*(total-m))
        chi_square_value = total*np.square(phi_score)
        normalised_score = x/(n*m)
        #print(phi_score, chi_square_value)
        list_score.append(phi_score)
        #if chi_square_value > 6.6349:
        #    list_score.append(phi_score)
        #else:
        #    list_score.append(np.nan)
        dict_score[bio] = list_score

for x in pivot_table1["User_status"]:
        list_status.append(x)
        dict_score["User_status"] = list_status


# In[112]:


dft = pd.DataFrame(dict_score)
dft.index = dft ["User_status"]
dft = dft.drop(columns=["User_status"])
dft = dft.drop(labels=["Column_Total"])


# In[113]:


fig = px.imshow(dft,color_continuous_scale='Balance', title = "Corrélation entre biomarqueurs et statuts (coefficient de Phi)")
fig.show()

#fig.savefig('relation_biomarqueurs_status.png')


# In[ ]:




