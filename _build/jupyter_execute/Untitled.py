#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
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


corpus = pd.read_csv(f"{path_base}corpus_tweets.csv", sep = ",")


# # Visualisation des interactions entre comptes
# 
# Les vidéos ci-dessous illustrent la "dynamique" des interactions entre les 7000 et quelques comptes annotés. Réalisés à l'aide de différentes librairies disponibles sous R, les graphes ont été construits de la manière suivantes:
# 
# - Le fichier des tweets contient une variable appelée *to_username*. Elle recense toutes mes fois qu'un compte a réagi à un tweet d'un autre compte en faisant "reply to". Par exemple, dans le tweet ci-dessous, on voit que le Dr. Sue Desmond-Hellmann a répondu à Pearl Freier. On peut noter également que Desmond-Hellmann mentionned la FDA à travers l'expression "@US_FDA". Dans le fichier de donnée, cette mention est enregistrée sous la variable *mentioned_names*.
# 
# ![x_reply_to_y.png](images/x_reply_to_y.png)
# 
# 

# In[37]:


# Step: Keep rows where to_username is not missing
corpus1 = corpus.loc[(corpus['to_username'].notna()) & (corpus["user_screen_name"] == "SueDHellmann")].reset_index()


c = corpus1[['user_screen_name', "to_username", "mentioned_names", "text", "url"]].drop_duplicates().reset_index()

c_style = c.style.format(precision=0, na_rep='').set_caption("Les variables \"to_username\" (replying to @username) et \"mentioned_names\" (@username) telles qu'elles se présentent dans le dataset") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top ; color: black ; font-size : 14pt'
 }], overwrite=False)
c_style

