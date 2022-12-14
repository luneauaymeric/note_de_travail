#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns 
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


df0 = df0[['query', 'id', 'timestamp_utc', 'local_time',
           'user_screen_name', 'text',  'user_location',  'user_id', 'user_name',
           'ROS1', 'ALK', 'EXON', 'EGFR', 'KRAS', 'NTRK',
           'BRAF', 'MET', 'RET', 'HER2', 'date']]
df0['date'] = pd.to_datetime(pd.to_datetime(df0['date']).dt.date)
df0['Year'] = df0['date'].dt.year


# In[6]:



#users = pd.read_csv('../outcome/user_sm_predicted.csv',dtype=dic_id) #jean-philippe
users = pd.read_csv(f'{path_base}user_sm_predicted.csv',dtype=dic_id)


# In[7]:


roles = ['advocacy', 'cancer patient', 'collective',
       'corpus', 'female', 'health center', 'health professional', 'industry',
       'male', 'media', 'npo', 'oncologist', 'other', 'research', 'survivor']


# In[8]:


df1 = df0.groupby("user_id").agg(nb = ("user_screen_name", "size"))


# In[9]:


df1 = df0.groupby(["user_id", "user_screen_name"]).agg(nb = ("id", "size")).reset_index()


# In[10]:


users2 = users.merge(df1, on = ["user_id"], how = "left")


# In[11]:


for r in roles:
    try:
        users[users.loc[users[r]>1]]=1
        print(r)
    except:
        pass


# In[12]:


def binarize(x):
    if x>1:
        return 1
    else: 
        return int(x)
for r in roles:
    users[r]=users[r].apply(binarize)


# # Travail sur les cat??gories
# 
# Pour faciliter l'analyse, on cr??e de nouvelles variables ?? partir des des cat??gories issues du travail d'annotation : *male* or *female*, *collective*, *advocacy*, *cancer patient*, *oncologist*, etc. ?? l'origine, ces cat??gories prennent la forme de variables binaires. Par exemple, les cat??gories *female* et *male* forment deux variables qui prennent comme valeurs "1" lorsque le compte est une femme (ou un homme) et 0 si s'en est pas une (ou un). Th??oriquement, comme l'illustre {numref}`le tableau %s <male-female>`, un compte annot?? comme ??tant un "homme" devrait ??tre d??fini comme "non-femme", c'est-??-dire prendre la valeur 1 pour la variable *male* et 0 pour la variable *female*.
# 
# Le travail de recodage consiste alors ?? regrouper ces variables binaires au sein d'une m??me et seule variable. Les variables *male* et *female* deviennent ainsi les modalit??s de la variable intitul??e *Gender*.
# 
# ```{table} Exemple de deux variables binaires
# :name: male-female
# 
# 
# | User_id | Type_user | male | female | Gender |
# |---|---|---|---|---|
# | 1 | Person | 1 | 0 | Male |
# | 2 | Person | 0 | 1 | Female |
# | 3 | Collective | 0 | 0 | NA |
# 
# ```
# 
# Le travail d'annotation repose sur quinze variables qui sont : *collective*, *male*, *female*, *advocacy*, *cancer patient*, *corpus*, *health center*, *health professional*, *industry*,  *media*, *npo*, *oncologist*, *other*, *research* et *survivor*. Nous les avons regroup??s en trois classes correspondant au type d'entit??, au genre et au r??le.
# 
# ## Type d'entit??
# 
# Par *type d'entit??*, on distingue les utlisatrices et utilisateurs qui agissent en leur "nom propre" de celles et ceux qui twittent au nom d'une organisation. Cette information est donn??e par la variable *collective*.
# 

# In[13]:


# Type
users.loc[users["collective"] == 0, "User_type"] = "Person"
users.loc[users["collective"] > 0, "User_type"] = "Collective"



# In[14]:


entity = users[["user_id", "User_type"]]
df_e=df0.merge(entity,on=['user_id'], how = "inner")#how = inner by default


# In[15]:



u_entity = entity.groupby(["User_type"]).agg(Frequency =('user_id', 'count'))
u_entity["Percentage"] = u_entity["Frequency"] / np.sum(u_entity["Frequency"]) * 100
t_entity = df_e.groupby(["User_type"]).agg(Frequency =('id', 'count'))
t_entity["Percentage"] = t_entity["Frequency"] / np.sum(t_entity["Frequency"]) * 100
tu_entity = u_entity.merge(t_entity, how = "left", on = ["User_type"])
tu_entity.loc["Total"] = tu_entity.sum(numeric_only=True, axis=0)
#df.columns = pd.MultiIndex.from_product([["new_label"], df.columns])
tu_entity.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])
tu_entity_style = tu_entity.style.format(precision=0, na_rep='')
tu_entity_style


# ## Le genre
# 
# Le genre est donn?? par la variable *Gender*. Le genre Certains comptes ont ??t?? annot??s ?? la fois comme "homme" et "femme", ou que leur genre n'a pu ??tre d??termin??. On a trois modalit??s : *Male*, *Female*, *Asexual* pour les comptes dont le genre n'a pu ??tre d??termin??.  On retiendra par ailleurs que les cat??gories de genre ne sont appliqu??es qu'aux "personnes".
# 
# 

# In[16]:


#R??attribution manuelle du genre aux deux comptes ayant ??t?? class?? ?? la fois comme male et female
users.loc[users["user_name"] == "Malin Hultcrantz MD PhD", "male"] = 0
users.loc[users["user_name"] == "Malin Hultcrantz MD PhD", "female"] = 1
users.loc[users["user_scree"] == "chano_py", "male"] = 1
users.loc[users["user_scree"] == "chano_py", "female"] = 0


# In[17]:


# Gender


users.loc[(users["User_type"] == "Person") & ((users["male"] == 1) &  (users["female"] == 0)), "Gender"] = "Male"
users.loc[(users["User_type"] == "Person") & ((users["male"] == 0) &  (users["female"] == 1)), "Gender"] = "Female"
users.loc[(users["User_type"] == "Person") & ((users["male"] == 1) &  (users["female"] == 1)), "Gender"] = "FemMale"
users.loc[(users["User_type"] == "Person") & ((users["male"] == 0) &  (users["female"] == 0)), "Gender"] = "Asexual"
users.loc[(users["User_type"] == "Collective"), "Gender"] = np.nan


# In[18]:


gender = users[["user_id", "Gender"]]
df_g=df0.merge(gender,on=['user_id'], how = "inner")#how = inner by default


# In[19]:


u_gender = gender.groupby(["Gender"]).agg(account_frequency =('user_id', 'count'))
u_gender["account_percentage"] = u_gender["account_frequency"] / np.sum(u_gender["account_frequency"]) * 100
t_gender = df_g.groupby(["Gender"]).agg(tweet_frequency =('id', 'count'))
t_gender["tweet_percentage"] = t_gender["tweet_frequency"] / np.sum(t_gender["tweet_frequency"]) * 100

tu_gender = u_gender.merge(t_gender, how = "left", on = ["Gender"])

# creation of figure and axes
fig, axes = plt.subplots(2,2, figsize=(10,10))


# loop for plotting each column
for i, col in enumerate(tu_gender.columns):
    if i < 2 :
        j = 0
    else :
        j = 1
        i = i-2
    sns.barplot(x=tu_gender.index, y=tu_gender[col], ax=axes[j,i]).set_title(col)

fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.suptitle('Nombre de comptes et de tweets par cat??gorie de genre')

#sns.barplot(x = "Gender", y="account_frequency", data = tu_gender)


# ## Regroupement des r??les
# 
# Le "r??le" vise ?? d??terminer si les utilisatrices et utilisateurs s'expriment en tant que patients, oncologues, professionnels de sant??, etc. Les variables consid??r??es comme des "r??les" sont les onze suivantes : *advocacy*, *cancer patient*, *health center*, *health professional*, *industry*, *media*, *npo*, *oncologist*, *other*, *research*, *survivor*. ?? la diff??rence du type d'entit?? ou du genre, un compte peut  ??tre class?? dans plusieurs r??les. Par exemple, il peut ??tre ?? la fois un "oncologue" et un "patient" ou un "m??dia" et une "organisation ?? but non-lucratif" (*non-profit organisation*). Compte tenu des multiples imbrications, le regroupement des diff??rents r??les au sein d'une m??me variable a d??s lors impliqu?? plusieurs op??rations de recodage 
# 
# Le travail de recodage a alors ??t?? divis?? en plusieurs ??tapes afin de construire de nouvelles variables permettant d'attribuer chaque compte ?? une cat??gorie unique tout en conservant les r??les d??finis lors de l'annotation. Ces r??les sont directement li??s ?? la question qui nous int??resse qui est de comprendre les rapports qu'entretiennent les patients avec les professionnels de sant?? et l'influence qu'ils peuvent avoir sur l'??volution des pratiques m??dicales, des traitements ou de la recherche. Partant de cette question, l'objectif est de construire des variables qui permettent de distinguer les patients des professionnels de sant??, les patients engag??s dans la d??fense d'une cause de ceux qui ne le sont pas ou encore les oncologues des autres professions de sant?? et des chercheurs non-oncologues.
# 
# ### Etape 1
# 
# On a cr???? une premi??re variable, appel??e *User_role*, qui regroupe toutes les combinaisons de r??le observ??es ?? partir de matrices de co-occurrence. Sur la {numref}`figure %s <role-matrice>`, on voit par exemple qu'on a 569 comptes class??s uniques comme *advocacy* et 115 qui sont consid??r??s ?? la fois comme *advocacy* et *cancer patient*.
# 
# 
# ```{figure} images/all_categories_plot.png
# :name: role-matrice
# 
# Les liens entre les r??les
# ```

# In[20]:


roles = ['advocacy', 'cancer patient', 'health center', 'health professional', 'industry',
        'media', 'npo', 'oncologist', 'other', 'research', 'survivor']

users["Somme_role"] = users[roles].sum(axis = 1)


# In[21]:


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


# In[22]:


role = users[["user_id", "User_type", "User_role"]]
df_r=df0.merge(role,on=['user_id'], how = "inner")#how = inner by default


# In[23]:


u_role = role.groupby(["User_role"]).agg(Accounts =('user_id', 'count'))
u_role["Account frequency"] = u_role["Accounts"]/np.sum(u_role["Accounts"])*100
t_role = df_r.groupby(["User_role"]).agg(Tweets =('id', 'count'))
t_role["Tweet frequency"] = t_role["Tweets"]/np.sum(t_role["Tweets"])*100

tu_role = u_role.merge(t_role,on=['User_role'], how = "inner")#how = inner by default
tu_role = tu_role.sort_values(by = ["User_role"], ascending = True)
tu_role.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])

tu_role_style = tu_role.style.format(precision=2, na_rep='')
tu_role_style
#role_tree.to_csv(f"{path_base}list_role.csv", sep = ",")


# ### Etape 2
# 
# Les modalit??s de la variable *User_role* sont regroup??es ?? l'aide d'une nouvelle variable appel??e *User_role2*. Il s'agit ici d'op??rer une premi??re r??duction des r??les. J'ai fait par exemple le choix de regrouper toutes les modalit??s contenant le terme "survivor" dans une m??ta-cat??gorie appel??e *Survivor* (avec S majuscule). De m??me toutes les modalit??s contenant le terme "cancer patient" dans la m??ta-cat??gorie *Cancer patient*, sauf celles contenant aussi le terme "survivor". 
# 

# In[24]:


users["User_role2"] = users["User_role"]
# individuals
## Advocacy
users.loc[(users['User_type'].isin(['Person'])) &
          (users['User_role']== "Advocacy"), "User_role2"] = "Advocacy"

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



## Other
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('other', case = False)) &
          (users['oncologist'] == 0) &
          (users['research'] == 0) &
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)&
          (users['health professional'] == 0)), "User_role2"] = "Other"

## Medias
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('media', case = False)) &
          (users['oncologist'] == 0) &
          (users['research'] == 0) &
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)&
          (users['health professional'] == 0)), "User_role2"] = "Media"

## NPO
users.loc[(users['User_type'].isin(['Person'])) &
          ((users['User_role'].str.contains('npo', case = False)) &
          (users['oncologist'] == 0) &
          (users['research'] == 0) &
          (users['survivor'] == 0)&
          (users['cancer patient'] == 0)&
          (users['health professional'] == 0)&
          (users['other'] == 0)), "User_role2"] = "NPO"

# collective

## Advocacy
users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role']== "Advocacy"), "User_role2"] = "Advocacy"
    
## media
users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('media', case = False)), "User_role2"] = "Media"

## NPO
users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('npo', case = False)), "User_role2"] = "NPO"
      
## Industry
users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('industry', case = False)), "User_role2"] = "Industry"

## health center
users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('health center', case = False))&
          (users['npo'] == 0) &
          (users['industry'] == 0), "User_role2"] = "Health center"

## Cancer patient
users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('cancer', case = False)), "User_role2"] = "Cancer patient"

## Survivor
users.loc[(users['User_type'].isin(['Collective'])) &
          (users['User_role'].str.contains('survivor', case = False)), "User_role2"] = "Survivor"

## Researcher
users.loc[(users['User_type'].isin(['Collective'])) &
          ((users['User_role'].str.contains('research', case = False))), "User_role2"] = "Researcher"

##Oncologists
users.loc[(users['User_type'].isin(['Collective'])) &
          ((users['User_role'].str.contains('oncologist', case = False))), "User_role2"] = "Oncologist"

           
##Health professional
users.loc[(users['User_type'].isin(['Collective'])) &
          ((users['User_role'].str.contains('professional', case = False))), "User_role2"] = "Health professional"


# In[25]:


role = users[["user_id", "User_type", "User_role2"]]
df_r=df0.merge(role,on=['user_id'], how = "inner")#how = inner by default


# In[26]:


u_role = role.groupby(["User_role2"]).agg(Accounts =('user_id', 'count'))
u_role["Account frequency"] = u_role["Accounts"]/np.sum(u_role["Accounts"])*100
t_role = df_r.groupby(["User_role2"]).agg(Tweets =('id', 'count'))
t_role["Tweet frequency"] = t_role["Tweets"]/np.sum(t_role["Tweets"])*100

tu_role = u_role.merge(t_role,on=['User_role2'], how = "inner")#how = inner by default
tu_role = tu_role.sort_values(by = ["User_role2"], ascending = True)
tu_role.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])

tu_role_style = tu_role.style.format(precision=2, na_rep='')
tu_role_style
#role_tree.to_csv(f"{path_base}list_role.csv", sep = ",")


# ### Etape 3
# 
# Partant de la variable *User_role2*, on en a cr???? une nouvelle intitul??e "User_role3" qui distingue pour chacun des r??les les comptes engag??s dans la d??fence d'une cause et ceux qui ne le sont pas. On a ainsi les modalit??s *Cance patient* et *Advocacy cancer patient*.

# In[27]:


## Advocacy


users.loc[#(users['User_type'].isin(['Person'])) & 
          (users["User_role"].str.contains("advocacy", case = False)) &
         (users['advocacy']== 1), "User_role3"] = 'Advocacy'

users.loc[#(users['User_type'].isin(['Person'])) & 
          (users["User_role"] != "Advocacy") &
         (users['advocacy']== 1), "User_role3"] = 'Advocacy' + " " + users["User_role2"]


users.loc[#(users['User_type'].isin(['Person'])) & 
         users['advocacy']== 0, "User_role3"] = users["User_role2"]


# In[28]:


role = users[["user_id", "User_type", "User_role3"]]
df_r=df0.merge(role,on=['user_id'], how = "inner")#how = inner by default


# In[29]:


u_role = role.groupby(["User_role3"]).agg(Accounts =('user_id', 'count'))
u_role["Account frequency"] = u_role["Accounts"]/np.sum(u_role["Accounts"])*100
t_role = df_r.groupby(["User_role3"]).agg(Tweets =('id', 'count'))
t_role["Tweet frequency"] = t_role["Tweets"]/np.sum(t_role["Tweets"])*100

tu_role = u_role.merge(t_role,on=['User_role3'], how = "inner")#how = inner by default
tu_role = tu_role.sort_values(by = ["User_role3"], ascending = True)
tu_role.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])

tu_role_style = tu_role.style.format(precision=2, na_rep='')
tu_role_style
#role_tree.to_csv(f"{path_base}list_role.csv", sep = ",")


# ### Etape 4
# 
# Les modalit??s constituant la variable *User_role3* sont regroup??es au sein de m??ta-cat??gories reprenant les r??les d??finis initialement. Nous proposons trois types de regroupements.
# 
# 
# Dans le premier regroupement, qui correspond ?? la variable *User_status*, la modalit?? *Advocacy* rassemble tous les comptes consid??r??s comme jouant un r??le d'advocacy, sauf si ce sont des professionnels de sant?? ou des patients. Dans le cas des professionnels de sant??, ceux qui joue un r??le d'advocacy sont plac??s avec les *Health professional*. Dans le cas des patients, on a distingu?? les patients qui font un travail de plaidoyer des autres patients. Par ailleurs, qu'il s'agisse des *Patients* ou des *Advocacy patients*, ces deux modalit??s r??unissent en leur sein les *survivors* et les *cancer patients*.
# 
# ```{note}
# Ce premier regroupement correspond ?? celui que nous avons arr??t?? ensemble
# ```
# 

# In[30]:


users["User_status"] = users["User_role3"]



users.loc[((users["User_role3"] == "Cancer patient") | 
          (users["User_role3"] == "Survivor")), "User_status" ]= "Patients"

users.loc[(users['User_role3'].str.contains('advocacy', case = False)), "User_status"] = "Advocacy"

users.loc[((users["User_role3"] == "Advocacy Cancer patient") | 
          (users["User_role3"] == "Advocacy Survivor")), "User_status" ]= "Advocacy Patients"


users.loc[(users["User_role3"] == "Advocacy Health professional"), "User_status" ]= "Health professional"


users.loc[(users["user_scree"].isin(["ros1cancer", "gyncsm", "myelomacrowd", "thebreastofus",
                                     "BestBreastNews"])), "User_status" ]= "Advocacy Patients"

users.loc[(users["user_scree"]=="Consano"), "User_status" ]= "Health center"


# In[31]:



fig = px.treemap(users, path=['User_type', 'User_status', 'User_role3', 'User_role'], color='User_status',
                title = "L'emboitement des variables 'User_type', 'User_status', 'User_role3', 'User_role'")
fig


# In[32]:


role = users[["user_id", "User_type", "User_status", "User_role"]]
df_r=df0.merge(role,on=['user_id'], how = "inner")#how = inner by default


# In[33]:


u_role = role.groupby(["User_status", "User_role"]).agg(Accounts =('user_id', 'count'))
u_role["Account frequency"] = u_role["Accounts"]/np.sum(u_role["Accounts"])*100
t_role = df_r.groupby(["User_status", "User_role"]).agg(Tweets =('id', 'count'))
t_role["Tweet frequency"] = t_role["Tweets"]/np.sum(t_role["Tweets"])*100
#u_role["Total_account"] = np.sum(u_role["Accounts"])
#t_role["Total_tweets"] = np.sum(t_role["Tweets"])

tu_role = u_role.merge(t_role,on=['User_status', 'User_role'], how = "inner")#how = inner by default
tu_role = tu_role.sort_values(by = ["User_status"], ascending = True)
tu_role.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])

tu_role_style = tu_role.style.format(precision=2, na_rep='').set_caption("Embo??tement des variables 'User_status' et 'User_role'") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)
tu_role_style
#role_tree.to_csv(f"{path_base}list_role.csv", sep = ",")


# In[34]:


u_role = role.groupby(["User_status"]).agg(Accounts =('user_id', 'count'))
u_role["Account frequency"] = u_role["Accounts"]/np.sum(u_role["Accounts"])*100
t_role = df_r.groupby(["User_status"]).agg(Tweets =('id', 'count'))
t_role["Tweet frequency"] = t_role["Tweets"]/np.sum(t_role["Tweets"])*100

tu_role = u_role.merge(t_role,on=['User_status'], how = "inner")#how = inner by default
tu_role = tu_role.sort_values(by = ["User_status"], ascending = True)
tu_role.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])

tu_role_style = tu_role.style.format(precision=2, na_rep='').set_caption("Nombre de comptes et de tweets pour chqaue modalit?? de la variable 'User_status'") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)
tu_role_style
#role_tree.to_csv(f"{path_base}list_role.csv", sep = ",")


# Dans le deuxi??me regroupement, qui correspond ?? la variable *User_status2*, la modalit?? *Advocacy* rassemble tous les comptes consid??r??s comme jouant un r??le d'advocacy, sauf si ce sont des patients. Dans le cas des patients, on conserve la distinction entre *Patients* et *Advocacy patients*.

# In[35]:


users["User_status2"] = users["User_role3"]


users.loc[((users["User_role3"] == "Cancer patient") | 
          (users["User_role3"] == "Survivor")), "User_status2" ]= "Patients"

users.loc[(users['User_role3'].str.contains('advocacy', case = False)), "User_status2"] = "Advocacy"

users.loc[((users["User_role3"] == "Advocacy Cancer patient") | 
          (users["User_role3"] == "Advocacy Survivor")), "User_status2" ]= "Advocacy Patients"

users.loc[(users["user_scree"].isin(["ros1cancer", "gyncsm", "myelomacrowd", "thebreastofus",
                                     "BestBreastNews"])), "User_status2" ]= "Advocacy Patients"

users.loc[(users["user_scree"]=="Consano"), "User_status2" ]= "Health center"


# In[36]:



fig = px.treemap(users, path=['User_type', 'User_status2', 'User_role3', 'User_role'], color='User_status2',
                title = "L'emboitement des variables 'User_type', 'User_status2', 'User_role3', 'User_role'")
fig


# In[37]:


role = users[["user_id", "User_type", "User_status2"]]
df_r=df0.merge(role,on=['user_id'], how = "inner")#how = inner by default


# In[38]:


u_role = role.groupby(["User_status2"]).agg(Accounts =('user_id', 'count'))
u_role["Account frequency"] = u_role["Accounts"]/np.sum(u_role["Accounts"])*100
t_role = df_r.groupby(["User_status2"]).agg(Tweets =('id', 'count'))
t_role["Tweet frequency"] = t_role["Tweets"]/np.sum(t_role["Tweets"])*100

tu_role = u_role.merge(t_role,on=['User_status2'], how = "inner")#how = inner by default
tu_role = tu_role.sort_values(by = ["User_status2"], ascending = True)
tu_role.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])

tu_role_style = tu_role.style.format(precision=2, na_rep='').set_caption("Nombre de comptes et tweets pour chqaue modalit?? de la variable 'User_status2'") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)
tu_role_style
#role_tree.to_csv(f"{path_base}list_role.csv", sep = ",")


# Enfin dans le troisi??me, j'ai choisi cette fois de construire une modalit?? *advocacy* qui comprend les comptes ayant uniquement un r??le d'advocacy. Dans le cas des *Advocacy oncologists*, *Advocacy researcher* ou encore *Advocacy health professional*, ils sont compt??s avec les *oncologists*, les *researchers* et les *health professionals* respectivement. On conserve la distinction entre *Patients* et *Advocacy patients*.
# 
# 

# In[39]:


users["User_status3"] = users["User_role3"]



users.loc[((users["User_role3"] == "Cancer patient") | 
          (users["User_role3"] == "Survivor")), "User_status3" ]= "Patients"

users.loc[(users['User_role3'].str.contains('advocacy', case = False)) &
          (users['Somme_role'] > 1), "User_status3"] = users["User_role2"]

users.loc[((users["User_role3"] == "Advocacy Cancer patient") | 
          (users["User_role3"] == "Advocacy Survivor")), "User_status3" ]= "Advocacy Patients"

users.loc[(users["user_scree"].isin(["ros1cancer", "gyncsm", "myelomacrowd", "thebreastofus",
                                     "BestBreastNews"])), "User_status3" ]= "Advocacy Patients"

users.loc[(users["user_scree"]=="Consano"), "User_status3" ]= "Health center"


# In[40]:



fig = px.treemap(users, path=['User_type', 'User_status3', 'User_role3', 'User_role'], color='User_status3',
                title = "L'emboitement des variables 'User_type', 'User_status3', 'User_role3', 'User_role'")
fig


# In[41]:


role = users[["user_id", "User_type", "User_status3"]]
df_r=df0.merge(role,on=['user_id'], how = "inner")#how = inner by default


# In[42]:


u_role = role.groupby(["User_status3"]).agg(Accounts =('user_id', 'count'))
u_role["Account frequency"] = u_role["Accounts"]/np.sum(u_role["Accounts"])*100
t_role = df_r.groupby(["User_status3"]).agg(Tweets =('id', 'count'))
t_role["Tweet frequency"] = t_role["Tweets"]/np.sum(t_role["Tweets"])*100

tu_role = u_role.merge(t_role,on=['User_status3'], how = "inner")#how = inner by default
tu_role = tu_role.sort_values(by = ["User_status3"], ascending = True)
tu_role.columns = pd.MultiIndex.from_product([['Comptes', 'Tweets'],
                                    ["Frequency","Percentage"]])

tu_role_style = tu_role.style.format(precision=2, na_rep='').set_caption("Nombre de comptes et tweets pour chqaue modalit?? de la variable 'User_status3'") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)
tu_role_style
#role_tree.to_csv(f"{path_base}list_role.csv", sep = ",")


# In[43]:


users.columns
users = users[['user_creat', 'user_descr', 'user_follo',
       'user_frien', 'user_id', 'user_image', 'user_likes', 'user_lists',
       'user_locat', 'user_name', 'user_numbe', 'user_scree', 'user_tweet',
       'user_url', 'user_id_id', 
       'comment_text', 'training_set', 'validation_set',
       'predicted_categories_ML', 'advocacy', 'cancer patient', 'collective',
       'corpus', 'female', 'health center', 'health professional', 'industry',
       'male', 'media', 'npo', 'oncologist', 'other', 'research', 'survivor',
               'User_type', 'Gender', 'User_status','User_role', 'User_role2', 'User_role3', 
               "User_status", "User_status2", "User_status3"]]


# In[44]:


users.to_csv(f'{path_base}recoded_user_sm_predicted.csv',sep =",")

