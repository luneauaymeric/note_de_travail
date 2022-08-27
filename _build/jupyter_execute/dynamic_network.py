#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import bamboolib
from IPython.display import display, HTML, IFrame


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


corpus = pd.read_csv(f"{path_base}corpus_tweets.csv", sep = ",")
len(corpus)


# # Visualisation d'une forme d'interaction entre les comptes: les *replying to*
# 
# Le fichier des tweets contient une variable appelée *to_username* (voir tableau ci-dessous). Elle recense toutes les fois qu'un compte a réagi à un tweet d'un autre compte en faisant "reply to". Par exemple, dans le tweet ci-dessous, on voit que le Dr. Sue Desmond-Hellmann a répondu à Pearl Freier. On peut noter également que Desmond-Hellmann mentionned la FDA à travers l'expression "@US_FDA". Dans le fichier de donnée, cette mention est enregistrée sous la variable *mentioned_names*.
# 
# ![x_reply_to_y.png](images/x_reply_to_y.png)
# 
# 

# In[ ]:


# Step: Keep rows where to_username is not missing
corpus1 = corpus.loc[(corpus['to_username'].notna()) & (corpus["user_screen_name"] == "SueDHellmann")].reset_index()


c = corpus1[['user_screen_name', "to_username", "mentioned_names", "url"]].drop_duplicates().reset_index()

c_style = c.style.format(precision=0, na_rep='').set_caption("Les variables \"to_username\" (replying to @username) et \"mentioned_names\" (@username) telles qu'elles se présentent dans le dataset") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top ; color: black ; font-size : 14pt'
 }], overwrite=False)
c_style


# Les vidéos ci-dessous illustrent la "dynamique" de cette forme d'interaction entre les 7000 et quelques comptes annotés. Réalisés à l'aide de différentes librairies disponibles sous R, les graphes ont été construits de la manière suivantes:
# 
# 1. Seuls les réponse entre comptes annotés sont conservés, ce qui donne un "sous-corpus" de 218&nbsp;263 tweets sur 1&nbsp;221&nbsp;611.
# 
# 2. Pour des raisons de temps calcul, on ne garde que les noeuds dont le degrés de centralité est supérieur ou égal à 10, ce qui siginifie qu'on a seulement les comptes qui ont publiés au moins dix réponses. Pour les mêmes raison, on a également découpé en plusieurs périodes
# 
# 3. Pour chaque noeud, on définit une date d'arrivée et une date de départ qui correspondent respectivement à la première et à la dernière réponse publiée.
# 
# 4. De la même manière, chaque réponse a un début (la date de publication du tweet) et une fin (le lendemain de la publication du tweet).
# 
# 
# Le principal intérêt de ces visualisations est d'avoir un aperçu des périodes de plus ou moins grandes intensités en terme d'interaction et de saisir dans quelle mesure des groupes de discussions se mettent en place. Pour chacune des périodes, on a une vidéo qui montre les échanges entre comptes et une autre qui rend compte des interactions entre catégories (oncologues, patients, etc.).
# 
# 

# %%html
# <video src="_static/videos/dynet_movie_2012_jan.mp4" controls width="600" ></video>
# 

# In[10]:


IFrame(src="_static/videos/dynet_2012-01-02_status.html", width = "900", height="500")


# In[11]:


IFrame(src="_static/videos/dynet_2013-12-31_status.html", width = "900", height="500")

