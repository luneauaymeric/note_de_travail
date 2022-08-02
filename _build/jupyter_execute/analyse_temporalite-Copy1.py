#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
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


#df0= pd.read_csv(f"{path_base}corpus_tweets.csv", sep = ",", dtype = dic_id)


# In[5]:



#users = pd.read_csv('../outcome/user_sm_predicted.csv',dtype=dic_id) #jean-philippe
users = pd.read_csv(f'{path_base}user_sm_predicted.csv',dtype=dic_id)


# In[6]:


roles = ['advocacy', 'cancer patient', 'collective',
       'corpus', 'female', 'health center', 'health professional', 'industry',
       'male', 'media', 'npo', 'oncologist', 'other', 'research', 'survivor']


# In[7]:


for r in roles:
    try:
        users[users.loc[users[r]>1]]=1
        print(r)
    except:
        pass


# In[8]:


def binarize(x):
    if x>1:
        return 1
    else: 
        return int(x)
for r in roles:
    users[r]=users[r].apply(binarize)


# # Regroupement des rôles
# 
# S'agissant des rôles et suivant les suggestions de la réunion précédente, plusieurs niveaux de regroupements ont été expérimentés. J'ai d'abord distingué les rôles caractérisant essentiellement des "personnes" (*advocacy, survivor, cancer patient, oncologist, health professional* et *research*)  de ceux définissant des "collectifs" (*media, npo, industry, health center*). Toutefois, comme on le verra plus loin, il existe des collectifs codés comme "patients" ou "oncologues" et, réciproquement, des "personnes" codées comme "media" ou "npo".
# 
# Je me suis ensuite appuyer sur une analyse des "similarités" entre les rôles pour définir les rapprochements pertinent. Par "similarité", j'entends le fait que 2 rôles ou plus sont associés à un même compte. Par exemple, on a 235 comptes qui ont été annoté à la fois comme oncologues (*oncologist*) et chercheur (*research*).
# 
# La figure X indique pour chaque couple de rôles le nombre d'individus qu'ils ont en commun (matrice de co-occurrence) et leur degré de similarité (matrice de similarité). On observe une certaine "similarité" entre les "défenseurs de cause" (*advocacy*), les "survivants" (*survivor*) et les "patients" (*cancer patient*) d'une part, les "chercheurs" (*research*) et "oncologues" (*oncologist*) d'autre part qui justifient leur rapprochement au sein d'une même catégorie. En revanche, on peut s'interroger sur l'intérêt de regrouper les "professionnels de santé" (*health professional*) avec les "oncologues" et les "chercheurs", étant donnée leur faible similarité. De même, le faible nombre de chevauchements entre les rôles collectifs plaide pour que ces derniers continuent d'être distinguer.
# 
# 
# ![images/all_categories_plot.png](images/all_categories_plot.png)
# 
# Le travail de recodage a alors été divisé en plusieurs étapes afin de construire de nouvelles variables permettant d'attribuer chaque compte à une catégorie unique tout en conservant les rôles définis lors de l'annotation. De cette manière, il est possible ensuite de tester différents niveaux de regroupement.
# 
# - Etape 1 : j'ai crée une première variable intutilée *User_role*. Elle comprend à la fois les rôles "purs", c'est-à-dire les comptes jouant un seul rôle, et l'ensemble des associations de rôles observés existantes. Si un compte a uniquement été classé comme *cancer patient*, il conserve cette valeur. En revanche, si un compte est à la fois classé comme *cancer patient* et *advocacy*, alors il prendra la valeur *Cancer patient & advocacy*.
# 
# 

# In[9]:


# Type
users.loc[users["collective"] == 0, "User_type"] = "Person"
users.loc[users["collective"] > 0, "User_type"] = "Collective"

# Gender
users.loc[(users["User_type"] == "Person") & ((users["male"] == 1) &  (users["female"] == 0)), "Genre"] = "Male"
users.loc[(users["User_type"] == "Person") & ((users["male"] == 0) &  (users["female"] == 1)), "Genre"] = "Female"
users.loc[(users["User_type"] == "Person") & ((users["male"] == 1) &  (users["female"] == 1)), "Genre"] = "FeMale"
users.loc[(users["User_type"] == "Person") & ((users["male"] == 0) &  (users["female"] == 0)), "Genre"] = "Asexual"


roles = ['advocacy', 'cancer patient', 'health center', 'health professional', 'industry',
        'media', 'npo', 'oncologist', 'other', 'research', 'survivor']

users["Somme_role"] = users[roles].sum(axis = 1)


# In[10]:


## Unique role

# Individuals
## Survivor
users.loc[((users["Somme_role"] == 1) &
          (users["survivor"] == 1)),  "User_role"]= "Survivor"


## Cancer patients
users.loc[((users["Somme_role"] == 1) &
          (users["cancer patient"] == 1)),  "User_role"]= "Cancer patient"


## Oncologists
users.loc[((users["Somme_role"] == 1) &
           (users["oncologist"] == 1)) , "User_role"]= "Oncologist"


## Researcher
users.loc[((users["Somme_role"] == 1) &
          (users["research"] == 1)), "User_role"]= "Researcher"


## Health professional      
users.loc[((users["Somme_role"] == 1) &
          (users["health professional"] == 1)), "User_role"]= "Health professional"

### Other
users.loc[((users["other"] == 1) & 
          (users["Somme_role"] == 1)), "User_role"]= "Other"


### Advocacy
users.loc[((users["advocacy"]==1) & 
          (users["Somme_role"] == 1)), "User_role"]= "Advocacy"



## Collective
### Health center
users.loc[((users["Somme_role"] == 1) & 
           (users["health center"] == 1)), "User_role"]= "Health center"

users.loc[((users["Somme_role"] == 1) & 
          (users["industry"] == 1)), "User_role"]= "Industry"

users.loc[((users["Somme_role"] == 1) &
          (users["npo"] == 1)), "User_role"]= "NPO"





### Other
users.loc[((users["Somme_role"] == 1) &
          (users["other"] == 1)), "User_role"]= "Other"

### Media
users.loc[((users["Somme_role"] == 1) &
          (users["media"]==1)) , "User_role"]= "Media"



users.loc[users["Somme_role"] == 0, "User_role"]= "Undefined"

## Double roles

# Individuals
## Survivor
users.loc[(users["Somme_role"] == 2) &
          ((users["survivor"] == 1) &
          (users["cancer patient"] == 1)),  "User_role"]= "Survivor & cancer patient"

users.loc[(users["Somme_role"] == 2) &
          ((users["survivor"] == 1) &
          (users["advocacy"] == 1)),  "User_role"]= "Survivor & advocacy"

users.loc[(users["Somme_role"] == 2) &
          ((users["survivor"] == 1) &
          (users["oncologist"] == 1)),  "User_role"]= "Survivor & oncologist"

users.loc[(users["Somme_role"] == 2) &
          ((users["survivor"] == 1) &
          (users["research"] == 1)),  "User_role"]= "Survivor & research"

users.loc[(users["Somme_role"] == 2) &
          ((users["survivor"] == 1) &
          (users["health professional"] == 1)),  "User_role"]= "Survivor & health professional"

users.loc[(users["Somme_role"] == 2) &
          ((users["survivor"] == 1) &
          (users["media"] == 1)),  "User_role"]= "Survivor & media"


users.loc[(users["Somme_role"] == 2) &
          ((users["survivor"] == 1) &
          (users["npo"] == 1)),  "User_role"]= "Survivor & npo"


## Cancer patients
users.loc[(users["Somme_role"] == 2) &
          ((users["cancer patient"] == 1) &
          (users["advocacy"] == 1)),  "User_role"]= "Cancer patient & advocacy"

users.loc[(users["Somme_role"] == 2) &
          ((users["cancer patient"] == 1) &
          (users["health professional"] == 1)),  "User_role"]= "Cancer patient & health professional"

users.loc[(users["Somme_role"] == 2) &
          ((users["cancer patient"] == 1) &
          (users["npo"] == 1)),  "User_role"]= "Cancer patient & npo"

users.loc[(users["Somme_role"] == 2) &
          ((users["cancer patient"] == 1) &
          (users["other"] == 1)),  "User_role"]= "Cancer patient & other"


### Advocacy
users.loc[(users["Somme_role"] == 2) &
          ((users["advocacy"] == 1)&
          (users["health professional"] == 1)),  "User_role"]= "Advocacy & health professional"


users.loc[(users["Somme_role"] == 2) &
          ((users["advocacy"] == 1)&
          (users["oncologist"] == 1)),  "User_role"]= "Advocacy & oncologist"

users.loc[(users["Somme_role"] == 2) &
          ((users["advocacy"] == 1)&
          (users["research"] == 1)),  "User_role"]= "Advocacy & research"

users.loc[(users["Somme_role"] == 2) &
          ((users["advocacy"] == 1)&
          (users["media"] == 1)),  "User_role"]= "Advocacy & media"


users.loc[(users["Somme_role"] == 2) &
          ((users["advocacy"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Advocacy & npo"

## Oncologists
users.loc[(users["Somme_role"] == 2) &
          ((users["oncologist"] == 1)&
          (users["health professional"] == 1)),  "User_role"]= "Oncologist & health professional"

users.loc[(users["Somme_role"] == 2) &
          ((users["oncologist"] == 1)&
          (users["research"] == 1)),  "User_role"]= "Oncologist & research"

users.loc[(users["Somme_role"] == 2) &
          ((users["oncologist"] == 1)&
          (users["industry"] == 1)),  "User_role"]= "Oncologist & industry"

users.loc[(users["Somme_role"] == 2) &
          ((users["oncologist"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Oncologist & npo"


## Researcher
users.loc[(users["Somme_role"] == 2) &
          ((users["research"] == 1)&
          (users["health professional"] == 1)),  "User_role"]= "Research & health professional"

users.loc[(users["Somme_role"] == 2) &
          ((users["research"] == 1)&
          (users["health center"] == 1)),  "User_role"]= "Research & health center"

users.loc[(users["Somme_role"] == 2) &
          ((users["research"] == 1)&
          (users["industry"] == 1)),  "User_role"]= "Research & industry"

users.loc[(users["Somme_role"] == 2) &
          ((users["research"] == 1)&
          (users["media"] == 1)),  "User_role"]= "Research & media"

users.loc[(users["Somme_role"] == 2) &
          ((users["research"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Research & npo"


## Health professional      
users.loc[(users["Somme_role"] == 2) &
          ((users["health professional"] == 1)&
          (users["industry"] == 1)),  "User_role"]= "Health professional & industry"

users.loc[(users["Somme_role"] == 2) &
          ((users["health professional"] == 1)&
          (users["media"] == 1)),  "User_role"]= "Health professional & media"

users.loc[(users["Somme_role"] == 2) &
          ((users["health professional"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Health professional & npo"

### Other
users.loc[(users["Somme_role"] == 2) &
          ((users["other"] == 1)&
          (users["industry"] == 1)),  "User_role"]= "Other & industry"


users.loc[(users["Somme_role"] == 2) &
          ((users["other"] == 1)&
          (users["media"] == 1)),  "User_role"]= "Other & media"

users.loc[(users["Somme_role"] == 2) &
          ((users["other"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Other & npo"

### Health center
users.loc[(users["Somme_role"] == 2) & 
           ((users["health center"] == 1)&
           (users["industry"] == 1)), "User_role"]= "Health center & industry"

users.loc[(users["Somme_role"] == 2) & 
           ((users["health center"] == 1)&
           (users["media"] == 1)), "User_role"]= "Health center & media"


### Industry
users.loc[(users["Somme_role"] == 2) & 
          ((users["industry"] == 1)&
          (users["media"] == 1)), "User_role"]= "Industry & media"

users.loc[(users["Somme_role"] == 2) & 
          ((users["industry"] == 1)&
          (users["npo"] == 1)), "User_role"]= "Industry & npo"


## Triple role

### Survivor & cancer patient & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["survivor"] == 1) &
          (users["cancer patient"] == 1)&
          (users["advocacy"] == 1)),  "User_role"]= "Survivor & cancer patient & advocacy"

### Survivor & cancer patient & npo
users.loc[(users["Somme_role"] == 3) &
          ((users["survivor"] == 1) &
          (users["cancer patient"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Survivor & cancer patient & npo"

### Survivor & npo & media
users.loc[(users["Somme_role"] == 3) &
          ((users["survivor"] == 1) &
          (users["media"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Survivor & npo & media"

### Survivor & research & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["survivor"] == 1) &
          (users["research"] == 1)&
          (users["advocacy"] == 1)),  "User_role"]= "Survivor & research & advocacy"

### Survivor & research & health professional
users.loc[(users["Somme_role"] == 3) &
          ((users["survivor"] == 1) &
          (users["research"] == 1)&
          (users["health professional"] == 1)),  "User_role"]= "Survivor & research & health professional"

### Survivor & research & advocacy professional
users.loc[(users["Somme_role"] == 3) &
          ((users["survivor"] == 1) &
          (users["advocacy"] == 1)&
          (users["health professional"] == 1)),  "User_role"]= "Survivor & research & health professional"


### Survivor & media & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["survivor"] == 1) &
          (users["media"] == 1)&
          (users["advocacy"] == 1)),  "User_role"]= "Survivor & media & advocacy"

### Cancer patient & npo & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["cancer patient"] == 1) &
          (users["npo"] == 1)&
          (users["advocacy"] == 1)),  "User_role"]= "Cancer patient & npo & advocacy"

### Oncologist & research & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["oncologist"] == 1)&
          (users["research"] == 1)&
          (users["advocacy"] == 1)),  "User_role"]= "Oncologist & research & advocacy"

### Oncologist & research & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["oncologist"] == 1)&
          (users["research"] == 1)&
          (users["health professional"] == 1)),  "User_role"]= "Oncologist & research & health professional"

### Oncologist & research & industry
users.loc[(users["Somme_role"] == 3) &
          ((users["oncologist"] == 1)&
          (users["research"] == 1)&
          (users["industry"] == 1)),  "User_role"]= "Oncologist & research & industry"

### Oncologist & research & npo
users.loc[(users["Somme_role"] == 3) &
          ((users["oncologist"] == 1)&
          (users["research"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Oncologist & research & npo"

### Health professional & npo & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["advocacy"] == 1)&
          (users["health professional"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Health professional & npo & advocacy"

### Research & npo & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["research"] == 1)&
          (users["advocacy"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Research & npo & advocacy"

### Research & health professional & advocacy
users.loc[(users["Somme_role"] == 3) &
          ((users["research"] == 1)&
          (users["health professional"] == 1)&
          (users["advocacy"] == 1)),  "User_role"]= "Research & health professional & advocacy"

### Media & npo & other
users.loc[(users["Somme_role"] == 3) &
          ((users["media"] == 1)&
          (users["npo"] == 1)&
          (users["other"] == 1)),  "User_role"]= "Media & npo & other"


## Quadruple role

### Survivor & cancer patient & advocacy & npo
users.loc[(users["Somme_role"] == 4) &
          ((users["survivor"] == 1) &
          (users["cancer patient"] == 1)&
          (users["advocacy"] == 1)&
          (users["npo"] == 1)),  "User_role"]= "Survivor & cancer patient & advocacy & npo"


# 
# -  Etape 2 : les modalités de la variable *User_role* sont regroupées à l'aide d'une nouvelle variable appelée *User_role2*. Il s'agit ici d'opérer une première réduction des rôles. J'ai fait par exemple le choix de regrouper toutes les modalités contenant le terme "survivor" dans une méta-catégorie appelée *Survivor* (avec S majuscule). De même toutes les modalités contenant le terme "cancer patient" dans la méta-catégorie *Cancer patient*, sauf celles contenant aussi le terme "survivor". 
# 

# In[11]:


users["User_role2"] = users["User_role"]
# individuals
## Survivor

users.loc[(users['User_type'].isin(['Person'])) &
          (users['User_role'].str.contains('survivor', case = False)), "User_role2"] = "Survivor"

## Cancer patients
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('cancer', case = False)) &
          (users['survivor'] == 0)), "User_role2"] = "Cancer patient"

## Oncologists
users.loc[(users['User_type'].isin(['Person'])) &
          (users['User_role'].str.contains('oncologist', case = False)&
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)), "User_role2"] = "Oncologist"


## Researchers
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('research', case = False)) &
          (users['oncologist'] == 0)&
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)), "User_role2"] = "Researcher"

## Health professionnals
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('health', case = False)) &
          (users['oncologist'] == 0) &
          (users['research'] == 0) &
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)), "User_role2"] = "Health professional"

## Advocacy
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('advocacy', case = False)) &
          (users['oncologist'] == 0) &
          (users['research'] == 0) &
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)&
          (users['health professional'] == 0)), "User_role2"] = "Advocacy"

## Other
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('other', case = False)) &
          (users['oncologist'] == 0) &
          (users['research'] == 0) &
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)&
          (users['health professional'] == 0)&
          (users['advocacy'] == 0)), "User_role2"] = "Other"

## Medias
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('media', case = False)) &
          (users['oncologist'] == 0) &
          (users['research'] == 0) &
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)&
          (users['health professional'] == 0)&
          (users['advocacy'] == 0)&
          (users['other'] == 0)), "User_role2"] = "Media"

# collective
## media

## Advocacy

users.loc[(users['User_type'].isin(['Collective'])) &
          ((users['User_role'].str.contains('research', case = False))&
          (users['advocacy'] == 1) ), "User_role2"] = "Researcher"

users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('cancer', case = False)), "User_role2"] = "Cancer patient"

users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('survivor', case = False)), "User_role2"] = "Survivor"

users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('media', case = False))&
          (users['health center'] == 0), "User_role2"] = "Media"

users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('npo', case = False))&
          (users['media'] == 0)&
          (users['industry'] == 0), "User_role2"] = "NPO"


users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('advocacy', case = False))&
          (users['media'] == 0) &
          (users['research'] == 0) &
          (users['cancer patient'] == 0) &
          (users['survivor'] == 0) &
          (users['npo'] == 0), "User_role2"] = "Advocacy"

users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('health center', case = False))&
          (users['npo'] == 0) &
          (users['industry'] == 0), "User_role2"] = "Health center"

users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('industry', case = False))&
          (users['media'] == 0), "User_role2"] = "Industry"


# - Etape 3 : La variable *User_status* réunit à son tour les modalités de la variable *User_role2*. Dans cette dernière variable, les modalités *Cancer patient* et *Survivor* sont réunies au sein de la catégorie générale *Patients*. Tandis que la catégorie *Health professionals* rassemblent les "oncologues", les "professionnels de santé" et les "chercheurs". En ce qui concerne les collectifs, j'ai regroupés les "centres de santé" et les "industries" dans une catégorie appelée "Health organisations". J'ai par ailleurs fait le choix de conserver une catégorie "média" et "NPO" dès lors que le comptes n'appartiennent pas à la classe des "industries" et des "centres de santé".
# 
# 
# 

# In[12]:



#Individuals
## Patients
users["User_status"] = users["User_role2"]
users.loc[(users["User_role2"] == "Cancer patient") | 
          (users["User_role2"] == "Survivor"), "User_status" ]= "Patients"


# Professionals

users.loc[(users["User_role2"] == "Oncologist") | 
          (users["User_role2"] == "Researcher") |
          (users["User_role2"] == "Health professional"), "User_status" ]= "Health professionals"

users.loc[(users["User_role2"] == "Health center") | 
          (users["User_role2"] == "Industry") , "User_status" ]= "Health organisations"


# In[13]:


role_tree = users.groupby(["User_type", "User_status", "User_role2","User_role"]).size()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(role_tree)


# In[14]:


users = users[['user_creat', 'user_descr', 'user_follo',
       'user_frien', 'user_id', 'user_image', 'user_likes', 'user_lists',
       'user_locat', 'user_name', 'user_numbe', 'user_scree', 'user_tweet',
       'user_url', 'user_id_id', 
       'comment_text', 'training_set', 'validation_set',
       'predicted_categories_ML', 'advocacy', 'cancer patient', 'collective',
       'corpus', 'female', 'health center', 'health professional', 'industry',
       'male', 'media', 'npo', 'oncologist', 'other', 'research', 'survivor',
               'User_type', 'Genre', 'User_status','User_role', 'User_role2']]

users.to_csv(f"{path_base}recoded_user_sm_predicted.csv", sep = ',')


# In[15]:


df0 = df0[['query', 'id', 'timestamp_utc', 'local_time',
           'user_screen_name', 'text',  'user_location',  'user_id', 'user_name',
           'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK',
           'BRAF', 'MET', 'RET', 'HER2', 'date']]


# In[ ]:


df0['date'] = pd.to_datetime(pd.to_datetime(df0['date']).dt.date)
df0['Year'] = df0['date'].dt.year


# In[ ]:


df=df0.merge(users,on=['user_id'], how = "inner")#how = inner by default


# In[ ]:


def time_series(time_length, x, what):
    if what == "user_id":
        df_user = df.drop_duplicates(subset = [time_length, "user_id"])
        timeseries = df_user.groupby([time_length, x]).agg(nb_user_by_group = ("user_id", "count")).reset_index()
        dfdate = df_user.groupby([time_length]).agg(nb_user = ("user_id", "count")).reset_index()
        timeseries = timeseries.merge(dfdate, on = [time_length], how = "left")
        timeseries["prop"] = timeseries["nb_user_by_group"]/timeseries["nb_user"]*100
    elif what == "id":
        df_user = df.drop_duplicates(subset = [time_length, "user_id"])
        time_author = df_user.groupby([time_length, x]).agg(nb_user_by_group = ("user_id", "count")).reset_index()
        timeseries = df.groupby([time_length, x]).agg(nb_tweets_by_group = ("id", "count")).reset_index()
        
        dfdate = df.groupby([time_length]).agg(nb_tweets = ("id", "count")).reset_index()
        timeseries = timeseries.merge(dfdate, on = [time_length], how = "left")
        timeseries = timeseries.merge(time_author.drop(columns = [x]), on = [time_length], how = "left")
        timeseries["prop"] = timeseries["nb_tweets_by_group"]/timeseries["nb_tweets"]*100
        timeseries["average_publication"] = timeseries["nb_tweets_by_group"]/timeseries["nb_user_by_group"]
        
    
    
    return(timeseries)


# In[ ]:


time_length = "Year"
x = "User_status"
what = "user_id"

dftime = time_series(time_length, x, what)
plot_time_serie = px.line(dftime, x=time_length, y='nb_user_by_group', color= x)
area_time_serie = px.area(dftime, x=time_length, y='prop', color= x, pattern_shape= x)#, line_group="country")
plot_time_serie


# In[ ]:


import seaborn as sns

sns.lineplot( x = 'date', y = 'nb_tweets_by_group', data = dftime, hue = "User_role")


# In[ ]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#
for i,bio in enumerate(biom):
    
    df_tmp = df.loc[(df[bio] == 1)& (df["Year"] >= 2012)]
    print(i, bio, len(df_tmp))
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
 


# In[ ]:


biomm


# In[ ]:


fig = px.line(biomm, x="Year", y="prop_of_biom", color='biomarker',
             title="Proportion de tweets contenant chacun des biomarqueurs")#, line_group="country")

#fig.layout.yaxis.tickformat = ',.0%'
fig.write_html("biomarkers_time.html")
fig


# In[ ]:


dfc=df.copy()
dfc.index=df['date']#.resample
dfc=dfc[dfc.date>'01-01-2012']


# In[ ]:




tr='1m'
total=dfc.resample(tr)['id'].count()

for i,bio in enumerate(biom):
    collective_permonth=(dfc[dfc[bio]==1].resample(tr)['id'].count()/total).reset_index()
    collective_permonth['category']=bio

    if i==0:
        biomm=collective_permonth
    else:
        biomm=pd.concat([biomm,collective_permonth])


import plotly.express as px
fig = px.area(biomm, x="date", y="id",color='category',pattern_shape="category", 
              pattern_shape_sequence=[".", "x", "+"],
             title="Proportion de tweets contenant chacun des biomarqueurs")#, line_group="country")
fig.layout.yaxis.tickformat = ',.0%'
fig.write_html("biomarkers_time.html")



fig.show()


# In[ ]:


df_status = df[["User_status", "id", 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]
df_tmp = df_status.loc[df_status["ALK"]==1].groupby(["User_status", "ALK"]).agg(ALK_c = ("id", "count"))
df_tmp


# In[ ]:


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
pivot_table

df_tmp = pivot_table.copy()
biom.append("somme_ligne")
for i, status in enumerate(pivot_table[variable]):
    print(status)
    for i, bio in enumerate(biom):
        df_tmp[bio] = pivot_table[bio]/pivot_table["somme_ligne"]*100
    
pivot_table = df_tmp[[variable, 'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK', 'BRAF', 'MET', 'RET', 'HER2']]
pivot_table.index = pivot_table[variable]
pivot_table = pivot_table.drop(columns = [variable])


# In[ ]:


fig = px.imshow(pivot_table,color_continuous_scale='reds')
fig.show()


# In[ ]:


import seaborn as sns 
from matplotlib.colors import LogNorm, Normalize

fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

cpalette = sns.color_palette("Paired")
res = sns.heatmap(pivot_table, annot=True, linewidths=.5, fmt='.2f')

for t in res.texts: t.set_text(t.get_text() + " %")
plt.savefig('biomarkers_dist.pdf')


# In[ ]:




