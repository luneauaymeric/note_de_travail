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
df = df.loc[(df["User_status"] != "Other") & (df["User_status"] != "Undefined") & (df["Year"] >= 2011)]


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


# # Quelques graphiques
# 
# Les différents graphiques qui suivent donne un aperçu de la distribution dans le temps des rôles et des références aux différents biomarqueurs.
# 
# ## Des professionnels de plus en plus présents
# 
# L'analyse de la distribution des rôles montre la présence croissante des professionnels, en particulier des oncologues (*oncologists*) et, dans une moindre, des chercheurs. Alors que les oncologues représentent un peu moins de 5% des comptes en 2012, ils constituent environ 27% des comptes en 2020. La part des professionnel de santé (*health professional*) semble quant à elle relativement stable dans le temps. La proportion de comptes jouant les rôles de patients, c'est-à-dire les *survivors* et les *cancer patient*, ne dépasse pas les 10% sur toutes la périodes. On observe également une faible représentation des "défenseurs de causes" (*advocacy*). Rappelons toutefois que cette catégorie comprend les comptes qui ne jouent pas d'autres rôles. Par exemple, un "survivant" qui a été annoté également comme un défenseur de cause (*advocacy*) sera compté parmi les patients et non les *advocacy*.
# 

# In[10]:


time_length = "Year"
x = "User_status"
what = "id"

if x == "User_status":
    Y= 'nb_user_by_group'
elif x == "id":
    Y= "nb_tweets_by_group"

dftime = time_series(data = df, time_length = time_length, x = x, year_base= 2015)
#plot_time_serie = px.line(dftime, x=time_length, y= Y, color= x)
area_time_serie = px.line(dftime, x=dftime[time_length], y="base_100_users", color= x, #pattern_shape= x,
                         title = "Evolution du nombre de comptes et de rôle par rôle")#, line_group="country")
#area_time_serie


# In[11]:



list_of_compute = ['nb_tweets_by_group','nb_user_by_group', 'prop_tweets', 'prop_users', 'average_publication',
       'tx_var_tweets', 'tx_var_users',
       'ref_value_user', 'base_100_tweets','base_100_users']


area_time_serie = px.line(dftime, x=dftime[time_length], y="nb_tweets_by_group", color= x, #pattern_shape= x,
                         title = "Evolution du nombre de comptes et de rôle par rôle")#, line_group="country")

#area_time_serie


# In[12]:


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
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2016)
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


# In[13]:





role_variable = variable[1]



fig1 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2016)
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


# In[14]:



role_variable = variable[2]



fig2 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2016)
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


# In[15]:



role_variable = variable[3]



fig3 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = time_series(data = df, time_length = time_length, x = role_variable, year_base = 2016)
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


# Les deux diagrammes ci-dessous représentent respectivement la proportion de comptes et de tweets par statut et par an. Les "statuts" correspondent aux modalités de la variable *User_status*. On voit ainsi que les médias constituent moins de 10% des comptes en 2021, mais sont à l'origine de plus de 25% des tweets à la même époque.

# ````{margin}
# ```{note}
# La variable *User_status* est une "réduction" de la variable *User_role2*. Ainsi, la modalité "Health professionals" regroupe les rôles d'oncologues, de chercheurs et de professionnels de la santé (hors médecins).
# ```
# ````

# In[16]:


#sns.lineplot(x = 'Year', y = "prop", data = dftime, hue = x)


# In[17]:


import seaborn as sns
dftime = time_series(data = df, time_length = time_length, x = x, year_base = 2016)

g = sns.FacetGrid(dftime, row= x, hue = x,
                  height=1.7, aspect=6,)

g.map(sns.lineplot, "Year", "base_100_tweets")
g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
g.fig.suptitle('Evolution du nombre de tweets par statuts et par an (base 100 en 2016)')


# In[18]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#
for i,bio in enumerate(biom):
    
    df_tmp = df.loc[(df[bio] == 1)& (df["Year"] >= 2012)]
    #print(i, bio, len(df_tmp))
    df_tmp = df_tmp.groupby(["Year", bio]).agg(nb_tweet_on_biomarker = ("id", "count")).reset_index()
    dfdate = df.groupby(["Year"]).agg(nb_tweet = ("id", "count")).reset_index()
    df_tmp = df_tmp.merge(dfdate, on = ["Year"], how = "left")
    df_tmp["biomarker"] = bio
    df_tmp = df_tmp.drop(columns = [bio])
    df_tmp["prop_of_biom"] = df_tmp["nb_tweet_on_biomarker"]/df_tmp["nb_tweet"]*100
    
    #sns.lineplot( x = 'Year', y = "prop_of_biom", data = df_tmp)
    
    if i==0:
        biomm=df_tmp
    else:
        biomm=pd.concat([biomm,df_tmp])
 


# In[19]:


time_length = "Year"
x = "biomarker"
year_base = 2016

datatime = biomm.copy()
#datatime[time_length] = datatime.index
list_year = []
list_role = []
list_base_100_tweet = []
list_base_100_user = []
for n in datatime[x].unique():
    datatemp = datatime.loc[datatime[x] == n]
    #first_year = np.min(datatemp.index)

    datatemp = datatemp.loc[datatemp[time_length] == year_base]
    datatemp = datatemp.drop_duplicates()

    ref_value_tweet = datatemp["nb_tweet_on_biomarker"].unique()[0]
    list_role.append(n)
    list_base_100_tweet.append(ref_value_tweet)
    
    data = {x : list_role, "ref_value_tweet" : list_base_100_tweet}
    data_base = pd.DataFrame(data)
    datatime = datatime.drop_duplicates()
    datatemp = datatime.merge(data_base, on = [x], how = "left")
    datatemp["base_100_tweets"] = (datatemp["nb_tweet_on_biomarker"]/datatemp["ref_value_tweet"])*100
    datatemp

datatemp['tx_var_tweets'] = (datatemp.groupby([x]).nb_tweet_on_biomarker.pct_change())*100


# ## La dynamique des biomarqueurs
# 
# Les diagrammes ci-dessous donnent la proportion de tweets contenant chacun des biomarqueurs par an et par mois respectivement, sachant qu'un tweet peut contenir plusieurs biomarqueurs. Ainsi, en 2018, le biomarqueur ALK était présent dans près 1,5% des tweets. Puis en janvier 2021, le biomarqueur EGFR était présent dans 3% des tweets.

# In[20]:


list_of_compute = ['nb_tweet_on_biomarker', 'prop_of_biom', #'average_publication',
       'base_100_tweets', 'tx_var_tweets']


fig4 = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

#for j, v in enumerate(variable):
df1 = datatemp
for z, col in enumerate(list_of_compute):
    for i, n in enumerate(df1[x].unique()):
        if col == "prop_of_biom" or col == "nb_tweet_on_biomarker":
            fig4.append_trace(
                go.Scatter(
                    x= df1[time_length].loc[(df1[x]==n)],
                    y= df1[col].loc[(df1[x]== n)],
                    stackgroup='one',
                    name = n,
                meta=  [col]),
                1,1)
        else :
            fig4.append_trace(
            go.Scatter(
                x= df1[time_length].loc[(df1[x]==n)],
                y= df1[col].loc[(df1[x]== n)],
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

    if "nb_tweet_on_biomarker" in fig4.data[k].meta[0]:
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
    title= f'Evolution du nombre de comptes et de tweets par role',
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


# In[21]:



g = sns.FacetGrid(biomm, row= 'biomarker', hue = 'biomarker',
                  height=1.7, aspect=6)

g.map(sns.lineplot, "Year", "prop_of_biom")
g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
g.fig.suptitle('Proportion de tweets par an contenant chacun des biomarqueurs')


# In[22]:


dfc=df.copy()
dfc.index=df['date']#.resample
dfc=dfc[dfc.date>'01-01-2012']


# In[23]:




tr='1m'
total=dfc.resample(tr)['id'].count()

for i,bio in enumerate(biom):
    collective_permonth=(dfc[dfc[bio]==1].resample(tr)['id'].count()/total*100).reset_index()
    collective_permonth['category']=bio

    if i==0:
        biomm=collective_permonth
    else:
        biomm=pd.concat([biomm,collective_permonth])


import plotly.express as px
fig = px.area(biomm, x="date", y="id",color='category',pattern_shape="category", 
              pattern_shape_sequence=[".", "x", "+"],
             title="Proportion de tweets contenant chacun des biomarqueurs par mois")#, line_group="country")
#fig.layout.yaxis.tickformat = ',.0%'
fig.write_html("biomarkers_time.html")



fig.show()


# ## Qui parle de quel marqueur ?
# 
# Enfin, la matrice ci-après indique la part occupée par les différents biomarqueurs dans les tweets qui en mentionnent au moins un en fonction du statut de leurs auteurs. Par exemple, 30% des 2548 biomarqueurs cités par les "défenseurs de cause" (*advocacy*) concernent le marqueur EGFR. 

# In[24]:


df_status = df[["User_status", "id", 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]
df_tmp = df_status.loc[df_status["ALK"]==1].groupby(["User_status", "ALK"]).agg(ALK_c = ("id", "count"))


# In[25]:


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


# In[26]:


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


# In[27]:


fig = px.imshow(pivot_table,color_continuous_scale='reds', title = "Proportion des références aux différents biomarqueurs en fonction du statut des auteurs")
fig.show()


# In[28]:




fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

cpalette = sns.color_palette("GnBu_d")
res = sns.heatmap(pivot_table, annot=True, linewidths=.5, fmt='.2f',  cmap="Reds")

for t in res.texts: t.set_text(t.get_text() + " %")
#plt.savefig('biomarkers_dist.pdf')
res.set(title ="Proportion des références aux différents biomarqueurs en fonction du statut des auteurs")


# In[29]:


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

pivot_table


# In[30]:


fig = px.imshow(pivot_table,color_continuous_scale='reds', title = "Proportion des références aux différents biomarqueurs en fonction du statut des auteurs")
fig.show()


# In[31]:




fig = plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')

cpalette = sns.color_palette("GnBu_d")
res = sns.heatmap(pivot_table, annot=True, linewidths=.5, fmt='.2f',  cmap="Reds")

for t in res.texts: t.set_text(t.get_text() + " %")
#plt.savefig('biomarkers_dist.pdf')
res.set(title ="Proportion des références aux différents biomarqueurs en fonction du statut des auteurs")


# 
# ```{note}
# Lecture de la matrice : 40% des références au biomarqueur MET sont le fait des oncologues, tandis que 27% des références au biomarqueur ALK proviennent des "Advocacy patiens".
# ```
# 

# In[ ]:




