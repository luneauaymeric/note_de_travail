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



#users = pd.read_csv('../outcome/user_sm_predicted.csv',dtype=dic_id) #jean-philippe
users = pd.read_csv(f'{path_base}user_sm_predicted.csv',dtype=dic_id)


# In[5]:


roles = ['advocacy', 'cancer patient', 'collective',
       'corpus', 'female', 'health center', 'health professional', 'industry',
       'male', 'media', 'npo', 'oncologist', 'other', 'research', 'survivor']


# In[6]:


for r in roles:
    try:
        users[users.loc[users[r]>1]]=1
        print(r)
    except:
        pass


# In[7]:


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
# Je me suis ensuite appuyer sur une analyse des "similarités" entre les rôles pour définir les rapprochements pertinent. Par "similarité", j'entends le fait que 2 rôles ou plus soient associés à un même compte. Par exemple, on a 235 comptes qui ont été annotés à la fois comme oncologues (*oncologist*) et chercheurs (*research*).
# 
# La figure X indique pour chaque couple de rôles le nombre d'individus qu'ils ont en commun (matrice de co-occurrence) et leur degré de similarité (matrice de similarité). On observe une certaine "similarité" entre les "défenseurs de cause" (*advocacy*), les "survivants" (*survivor*) et les "patients" (*cancer patient*) d'une part, les "chercheurs" (*research*) et "oncologues" (*oncologist*) d'autre part qui justifient leur rapprochement au sein d'une même catégorie. En revanche, on peut s'interroger sur l'intérêt de regrouper les "professionnels de santé" (*health professional*) avec les "oncologues" et les "chercheurs", étant donnée leur faible similarité. De même, le faible nombre de chevauchements entre les rôles collectifs plaide pour que ces derniers continuent d'être distinguer.
# 
# 
# ![all_categories_plot.png](images/all_categories_plot.png)
# 
# Le travail de recodage a alors été divisé en plusieurs étapes afin de construire de nouvelles variables permettant d'attribuer chaque compte à une catégorie unique tout en conservant les rôles définis lors de l'annotation. De cette manière, il est possible ensuite de tester différents niveaux de regroupement.
# 
# - Etape 1 : j'ai crée une première variable intutilée *User_role*. Elle comprend à la fois les rôles "purs", c'est-à-dire les comptes jouant un seul rôle, et l'ensemble des associations de rôles observés existantes. Si un compte a uniquement été classé comme *cancer patient*, il conserve cette valeur. En revanche, si un compte est à la fois classé comme *cancer patient* et *advocacy*, alors il prendra la valeur *Cancer patient & advocacy*.
# 
# 

# In[8]:


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


# In[9]:


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

# In[10]:


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

# In[11]:



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


# In[12]:


import plotly.express as px
fig = px.treemap(users, path=['User_type', 'User_status', 'User_role2', 'User_role'], color='User_status')
fig


# In[13]:


role_tree = users.groupby(["User_status", "User_role2","User_role"]).agg(n=('user_id', 'count'))
role_tree_style = individual_role_tree.style.format(precision=0, na_rep='')
#role_tree_style.to_html()


# <style type="text/css"></style><table id="T_f5498_">  <thead>    <tr>      <th class="blank" >&nbsp;</th>      <th class="blank" >&nbsp;</th>      <th class="blank level0" >&nbsp;</th>      <th class="col_heading level0 col0" >n</th>    </tr>    <tr>      <th class="index_name level0" >User_status</th>      <th class="index_name level1" >User_role2</th>      <th class="index_name level2" >User_role</th>      <th class="blank col0" >&nbsp;</th>    </tr>  </thead>  <tbody>    <tr>      <th id="T_f5498_level0_row0" class="row_heading level0 row0" rowspan="3">Advocacy</th>      <th id="T_f5498_level1_row0" class="row_heading level1 row0" rowspan="3">Advocacy</th>      <th id="T_f5498_level2_row0" class="row_heading level2 row0" >Advocacy</th>      <td id="T_f5498_row0_col0" class="data row0 col0" >262</td>    </tr>    <tr>      <th id="T_f5498_level2_row1" class="row_heading level2 row1" >Advocacy & media</th>      <td id="T_f5498_row1_col0" class="data row1 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row2" class="row_heading level2 row2" >Advocacy & npo</th>      <td id="T_f5498_row2_col0" class="data row2 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level0_row3" class="row_heading level0 row3" rowspan="8">Health organisations</th>      <th id="T_f5498_level1_row3" class="row_heading level1 row3" rowspan="3">Health center</th>      <th id="T_f5498_level2_row3" class="row_heading level2 row3" >Health center</th>      <td id="T_f5498_row3_col0" class="data row3 col0" >188</td>    </tr>    <tr>      <th id="T_f5498_level2_row4" class="row_heading level2 row4" >Health center & media</th>      <td id="T_f5498_row4_col0" class="data row4 col0" >21</td>    </tr>    <tr>      <th id="T_f5498_level2_row5" class="row_heading level2 row5" >Research & health center</th>      <td id="T_f5498_row5_col0" class="data row5 col0" >68</td>    </tr>    <tr>      <th id="T_f5498_level1_row6" class="row_heading level1 row6" rowspan="5">Industry</th>      <th id="T_f5498_level2_row6" class="row_heading level2 row6" >Health center & industry</th>      <td id="T_f5498_row6_col0" class="data row6 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row7" class="row_heading level2 row7" >Industry</th>      <td id="T_f5498_row7_col0" class="data row7 col0" >600</td>    </tr>    <tr>      <th id="T_f5498_level2_row8" class="row_heading level2 row8" >Industry & npo</th>      <td id="T_f5498_row8_col0" class="data row8 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row9" class="row_heading level2 row9" >Other & industry</th>      <td id="T_f5498_row9_col0" class="data row9 col0" >3</td>    </tr>    <tr>      <th id="T_f5498_level2_row10" class="row_heading level2 row10" >Research & industry</th>      <td id="T_f5498_row10_col0" class="data row10 col0" >8</td>    </tr>    <tr>      <th id="T_f5498_level0_row11" class="row_heading level0 row11" rowspan="19">Health professionals</th>      <th id="T_f5498_level1_row11" class="row_heading level1 row11" rowspan="5">Health professional</th>      <th id="T_f5498_level2_row11" class="row_heading level2 row11" >Advocacy & health professional</th>      <td id="T_f5498_row11_col0" class="data row11 col0" >10</td>    </tr>    <tr>      <th id="T_f5498_level2_row12" class="row_heading level2 row12" >Health professional</th>      <td id="T_f5498_row12_col0" class="data row12 col0" >362</td>    </tr>    <tr>      <th id="T_f5498_level2_row13" class="row_heading level2 row13" >Health professional & industry</th>      <td id="T_f5498_row13_col0" class="data row13 col0" >9</td>    </tr>    <tr>      <th id="T_f5498_level2_row14" class="row_heading level2 row14" >Health professional & media</th>      <td id="T_f5498_row14_col0" class="data row14 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row15" class="row_heading level2 row15" >Health professional & npo</th>      <td id="T_f5498_row15_col0" class="data row15 col0" >3</td>    </tr>    <tr>      <th id="T_f5498_level1_row16" class="row_heading level1 row16" rowspan="8">Oncologist</th>      <th id="T_f5498_level2_row16" class="row_heading level2 row16" >Advocacy & oncologist</th>      <td id="T_f5498_row16_col0" class="data row16 col0" >3</td>    </tr>    <tr>      <th id="T_f5498_level2_row17" class="row_heading level2 row17" >Oncologist</th>      <td id="T_f5498_row17_col0" class="data row17 col0" >768</td>    </tr>    <tr>      <th id="T_f5498_level2_row18" class="row_heading level2 row18" >Oncologist & health professional</th>      <td id="T_f5498_row18_col0" class="data row18 col0" >6</td>    </tr>    <tr>      <th id="T_f5498_level2_row19" class="row_heading level2 row19" >Oncologist & npo</th>      <td id="T_f5498_row19_col0" class="data row19 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row20" class="row_heading level2 row20" >Oncologist & research</th>      <td id="T_f5498_row20_col0" class="data row20 col0" >229</td>    </tr>    <tr>      <th id="T_f5498_level2_row21" class="row_heading level2 row21" >Oncologist & research & advocacy</th>      <td id="T_f5498_row21_col0" class="data row21 col0" >3</td>    </tr>    <tr>      <th id="T_f5498_level2_row22" class="row_heading level2 row22" >Oncologist & research & health professional</th>      <td id="T_f5498_row22_col0" class="data row22 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row23" class="row_heading level2 row23" >Oncologist & research & industry</th>      <td id="T_f5498_row23_col0" class="data row23 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level1_row24" class="row_heading level1 row24" rowspan="6">Researcher</th>      <th id="T_f5498_level2_row24" class="row_heading level2 row24" >Advocacy & research</th>      <td id="T_f5498_row24_col0" class="data row24 col0" >13</td>    </tr>    <tr>      <th id="T_f5498_level2_row25" class="row_heading level2 row25" >Research & health professional</th>      <td id="T_f5498_row25_col0" class="data row25 col0" >19</td>    </tr>    <tr>      <th id="T_f5498_level2_row26" class="row_heading level2 row26" >Research & industry</th>      <td id="T_f5498_row26_col0" class="data row26 col0" >4</td>    </tr>    <tr>      <th id="T_f5498_level2_row27" class="row_heading level2 row27" >Research & media</th>      <td id="T_f5498_row27_col0" class="data row27 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row28" class="row_heading level2 row28" >Research & npo</th>      <td id="T_f5498_row28_col0" class="data row28 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row29" class="row_heading level2 row29" >Researcher</th>      <td id="T_f5498_row29_col0" class="data row29 col0" >224</td>    </tr>    <tr>      <th id="T_f5498_level0_row30" class="row_heading level0 row30" rowspan="8">Media</th>      <th id="T_f5498_level1_row30" class="row_heading level1 row30" rowspan="8">Media</th>      <th id="T_f5498_level2_row30" class="row_heading level2 row30" >Advocacy & media</th>      <td id="T_f5498_row30_col0" class="data row30 col0" >3</td>    </tr>    <tr>      <th id="T_f5498_level2_row31" class="row_heading level2 row31" >Industry & media</th>      <td id="T_f5498_row31_col0" class="data row31 col0" >36</td>    </tr>    <tr>      <th id="T_f5498_level2_row32" class="row_heading level2 row32" >Media</th>      <td id="T_f5498_row32_col0" class="data row32 col0" >893</td>    </tr>    <tr>      <th id="T_f5498_level2_row33" class="row_heading level2 row33" >Media & npo & other</th>      <td id="T_f5498_row33_col0" class="data row33 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row34" class="row_heading level2 row34" >Other & media</th>      <td id="T_f5498_row34_col0" class="data row34 col0" >20</td>    </tr>    <tr>      <th id="T_f5498_level2_row35" class="row_heading level2 row35" >Research & media</th>      <td id="T_f5498_row35_col0" class="data row35 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row36" class="row_heading level2 row36" >Survivor & media</th>      <td id="T_f5498_row36_col0" class="data row36 col0" >6</td>    </tr>    <tr>      <th id="T_f5498_level2_row37" class="row_heading level2 row37" >Survivor & npo & media</th>      <td id="T_f5498_row37_col0" class="data row37 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level0_row38" class="row_heading level0 row38" rowspan="13">NPO</th>      <th id="T_f5498_level1_row38" class="row_heading level1 row38" rowspan="13">NPO</th>      <th id="T_f5498_level2_row38" class="row_heading level2 row38" >Advocacy & npo</th>      <td id="T_f5498_row38_col0" class="data row38 col0" >49</td>    </tr>    <tr>      <th id="T_f5498_level2_row39" class="row_heading level2 row39" >Cancer patient & npo</th>      <td id="T_f5498_row39_col0" class="data row39 col0" >14</td>    </tr>    <tr>      <th id="T_f5498_level2_row40" class="row_heading level2 row40" >Cancer patient & npo & advocacy</th>      <td id="T_f5498_row40_col0" class="data row40 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row41" class="row_heading level2 row41" >Health professional & npo</th>      <td id="T_f5498_row41_col0" class="data row41 col0" >13</td>    </tr>    <tr>      <th id="T_f5498_level2_row42" class="row_heading level2 row42" >Health professional & npo & advocacy</th>      <td id="T_f5498_row42_col0" class="data row42 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row43" class="row_heading level2 row43" >NPO</th>      <td id="T_f5498_row43_col0" class="data row43 col0" >443</td>    </tr>    <tr>      <th id="T_f5498_level2_row44" class="row_heading level2 row44" >Oncologist & npo</th>      <td id="T_f5498_row44_col0" class="data row44 col0" >8</td>    </tr>    <tr>      <th id="T_f5498_level2_row45" class="row_heading level2 row45" >Oncologist & research & npo</th>      <td id="T_f5498_row45_col0" class="data row45 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row46" class="row_heading level2 row46" >Research & npo</th>      <td id="T_f5498_row46_col0" class="data row46 col0" >22</td>    </tr>    <tr>      <th id="T_f5498_level2_row47" class="row_heading level2 row47" >Research & npo & advocacy</th>      <td id="T_f5498_row47_col0" class="data row47 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row48" class="row_heading level2 row48" >Survivor & cancer patient & advocacy & npo</th>      <td id="T_f5498_row48_col0" class="data row48 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row49" class="row_heading level2 row49" >Survivor & cancer patient & npo</th>      <td id="T_f5498_row49_col0" class="data row49 col0" >4</td>    </tr>    <tr>      <th id="T_f5498_level2_row50" class="row_heading level2 row50" >Survivor & npo</th>      <td id="T_f5498_row50_col0" class="data row50 col0" >5</td>    </tr>    <tr>      <th id="T_f5498_level0_row51" class="row_heading level0 row51" rowspan="3">Other</th>      <th id="T_f5498_level1_row51" class="row_heading level1 row51" rowspan="3">Other</th>      <th id="T_f5498_level2_row51" class="row_heading level2 row51" >Other</th>      <td id="T_f5498_row51_col0" class="data row51 col0" >1479</td>    </tr>    <tr>      <th id="T_f5498_level2_row52" class="row_heading level2 row52" >Other & industry</th>      <td id="T_f5498_row52_col0" class="data row52 col0" >11</td>    </tr>    <tr>      <th id="T_f5498_level2_row53" class="row_heading level2 row53" >Other & media</th>      <td id="T_f5498_row53_col0" class="data row53 col0" >22</td>    </tr>    <tr>      <th id="T_f5498_level0_row54" class="row_heading level0 row54" rowspan="19">Patients</th>      <th id="T_f5498_level1_row54" class="row_heading level1 row54" rowspan="6">Cancer patient</th>      <th id="T_f5498_level2_row54" class="row_heading level2 row54" >Cancer patient</th>      <td id="T_f5498_row54_col0" class="data row54 col0" >256</td>    </tr>    <tr>      <th id="T_f5498_level2_row55" class="row_heading level2 row55" >Cancer patient & advocacy</th>      <td id="T_f5498_row55_col0" class="data row55 col0" >109</td>    </tr>    <tr>      <th id="T_f5498_level2_row56" class="row_heading level2 row56" >Cancer patient & health professional</th>      <td id="T_f5498_row56_col0" class="data row56 col0" >11</td>    </tr>    <tr>      <th id="T_f5498_level2_row57" class="row_heading level2 row57" >Cancer patient & npo</th>      <td id="T_f5498_row57_col0" class="data row57 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row58" class="row_heading level2 row58" >Cancer patient & npo & advocacy</th>      <td id="T_f5498_row58_col0" class="data row58 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row59" class="row_heading level2 row59" >Cancer patient & other</th>      <td id="T_f5498_row59_col0" class="data row59 col0" >8</td>    </tr>    <tr>      <th id="T_f5498_level1_row60" class="row_heading level1 row60" rowspan="13">Survivor</th>      <th id="T_f5498_level2_row60" class="row_heading level2 row60" >Survivor</th>      <td id="T_f5498_row60_col0" class="data row60 col0" >180</td>    </tr>    <tr>      <th id="T_f5498_level2_row61" class="row_heading level2 row61" >Survivor & advocacy</th>      <td id="T_f5498_row61_col0" class="data row61 col0" >96</td>    </tr>    <tr>      <th id="T_f5498_level2_row62" class="row_heading level2 row62" >Survivor & cancer patient</th>      <td id="T_f5498_row62_col0" class="data row62 col0" >7</td>    </tr>    <tr>      <th id="T_f5498_level2_row63" class="row_heading level2 row63" >Survivor & cancer patient & advocacy</th>      <td id="T_f5498_row63_col0" class="data row63 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row64" class="row_heading level2 row64" >Survivor & cancer patient & npo</th>      <td id="T_f5498_row64_col0" class="data row64 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row65" class="row_heading level2 row65" >Survivor & health professional</th>      <td id="T_f5498_row65_col0" class="data row65 col0" >6</td>    </tr>    <tr>      <th id="T_f5498_level2_row66" class="row_heading level2 row66" >Survivor & media</th>      <td id="T_f5498_row66_col0" class="data row66 col0" >3</td>    </tr>    <tr>      <th id="T_f5498_level2_row67" class="row_heading level2 row67" >Survivor & media & advocacy</th>      <td id="T_f5498_row67_col0" class="data row67 col0" >1</td>    </tr>    <tr>      <th id="T_f5498_level2_row68" class="row_heading level2 row68" >Survivor & npo</th>      <td id="T_f5498_row68_col0" class="data row68 col0" >3</td>    </tr>    <tr>      <th id="T_f5498_level2_row69" class="row_heading level2 row69" >Survivor & oncologist</th>      <td id="T_f5498_row69_col0" class="data row69 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level2_row70" class="row_heading level2 row70" >Survivor & research</th>      <td id="T_f5498_row70_col0" class="data row70 col0" >11</td>    </tr>    <tr>      <th id="T_f5498_level2_row71" class="row_heading level2 row71" >Survivor & research & advocacy</th>      <td id="T_f5498_row71_col0" class="data row71 col0" >6</td>    </tr>    <tr>      <th id="T_f5498_level2_row72" class="row_heading level2 row72" >Survivor & research & health professional</th>      <td id="T_f5498_row72_col0" class="data row72 col0" >2</td>    </tr>    <tr>      <th id="T_f5498_level0_row73" class="row_heading level0 row73" >Undefined</th>      <th id="T_f5498_level1_row73" class="row_heading level1 row73" >Undefined</th>      <th id="T_f5498_level2_row73" class="row_heading level2 row73" >Undefined</th>      <td id="T_f5498_row73_col0" class="data row73 col0" >514</td>    </tr>  </tbody></table>
