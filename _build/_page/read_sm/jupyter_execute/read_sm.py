#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
from matplotlib import pyplot as plt
import numpy as np
import re
#import bamboolib


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


# Using '*' pattern 
#print('\nNamed with wildcard *:')
i=-1
dic_id={}
for x in [x for x in pd.read_csv(glob.glob(f'{path_base}sm/*.csv')[0]).columns if 'id' in x]:
    dic_id[x]=str
dic_id
for name in glob.glob(f'{path_base}/sm/*.csv'):
    #print (name)
    i+=1
    df0=pd.read_csv(name,dtype=dic_id)
    if i>0:
        df=pd.concat([df,df0])
    else:
        df=df0


# In[71]:


df0=df.drop_duplicates()


# In[72]:


df0["text"] =  df0.text.apply(lambda x : x.replace('\r', ' '))
df0["text"] =  df0.text.apply(lambda x : x.replace('\n', ' '))


# # Détection des biomarqueurs dans les tweets
# 
# 
# 
# ## Méthode initiale
# 
# Afin de compter les occurrences des biomarqueurs dans les tweets, la méthode initiale consistait à rechercher simplement les chaînes de caractères correspondant aux noms des biomarqueurs en prenant ou non en compte la casse. Pour rappel, les biomarqueurs pris en compte sont: *ROS1, ALK, EXON, EGFR, KRAS, NTRK, BRAF, MET, RET, HER2*.
# 
# Toutefois, cette première méthode conduisait à prendre en compte de nombreux faux-amis dans le cas de certains biomarqueurs. Par exemple, suivant cette méthode, "RETweet", "WALK", "VincenTRK" ou "METASTASIS" étaient compté comme une occurrence de "RET", "ALK", "NTRK" ou "MET" respectivement.
# 
# 
# 

# In[73]:


biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#
dfb = df0.copy()

for x in biom:
    if x in ['MET','RET','ALK']:
        dfb[x]=dfb.text.str.contains(x, case=True)
    else:
        dfb[x]=dfb.text.str.contains(x, case=False)
    
    


# ## Définition d'expressions régulières plus complexes
# 
# Des expressions régulières plus "complexes" ont alors été définies pour exclure les faux-amis évoqués ci-dessus et d'autres du décomptes des occurrences. Le script utilisé et affiché dans la cellule ci-dessous fonctionne de la façon suivante:
# 
# 1. Pour chaque tweet, on commence par remplacer plusieurs signes de ponctuations ("/", ".", ",", "-") par des espaces.
# 
# 
# 2. Une fois le remplacement effectué, chaque tweets est découpé afin d'obtenir la liste de toutes les chaînes de caractère précédées et suivies par un espace (dans le script ci-dessous, cette liste est nommée "tt").
# 
# 
# 3. puis on recherche la présence de chaque biormaqueur au sein de cette liste selon des expressions régulières. Par exemple, dans le cas du biomarqueur "MET", nous avons utilisé l'expression régulière suivante (avec x est égale à "MET" et le signe "|" qui signifie "ou"):
# 
# > "{x}[(+)]" **|** "[#@]{x}[^AaSsUuEeYyHh]" **|** "[#@]{x}\\$" **|** "^{x}[^AaSsUuEeYyHh]" **|** "^{x}\\$"
#     
#    
# C'est-à-dire que pour chaque terme de la liste "tt", celui-ci sera considéré comme une occurrence du biomarqueur "MET" si, et seulement si "MET" :
#     
# - est suivi du signe +" 
#     
# - **ou** est précédé par un hashtag ou un arobase et suivi par un ou plusieurs caractères sauf les lettres majuscules ou minuscules *a, s, u, e, y, h*. 
#     
# - **ou** est précédé par un hashtag ou un arobase et n'est suivi par aucun caractères
#     
# - **ou** n'est précédé par aucun caractère et n'est pas suivi par les lettres *a, s, u, e, y, h* majuscules ou minuscules". Dans ce cas, l'expression "METUProg" n'est pas prise en compte à la différence de "METmuts" 
#     
# - **ou** n'est précédé ni suivi de caractères.
#     
# Si certaines règles sont communes à l'ensemble des biomarqueurs, d'autres sont spécifiques. Par exemple, dans le cas du biomarqueur "RET", les lettres  *a, s, u, e, y, h* sont remplacée par *a, e, i, o, u, r, h, z, w* pour éviter que des mots comme "retweet" soientt considérés comme des occurrences du biomarqueur. À noter également que dans le cas des biomarqueurs "ALK", "MET" et "RET", on ne prend en compte que les formes en majuscule. Pour les autres, l'algorithme de recherche est insensible à la casse.
# 
# Enfin, ces expressions régulières sont recherchées à l'aide de la fonction "search" du module *re* du langage Python. Cette fonction retourne "vrai" si l'expression est trouvée, "faux" dans le cas contraire. Par ailleurs, elle s'arrête à la première expression trouvée. Autrement dit, la fonction indique simplement si l'expression est présente quelque soit le nombre de fois où elle apparaît le tweet. Le tableau ci-après illustre le résultat obtenu pour une dizaine de phrases avec chacune des deux méthodes (sans ou avec les expressions régulières).
# 
# 

# :::{admonition} Illustration de l'algorythme
# 
# 1. Le tweet initial : "MET copy number as a secondary driver of EGFR TKI resistance in EGFR-mutant NSCLC http://bit.ly/2HfGHjn #editorial #lcsm"</p>
#     
# 2. Le tweet nettoyé (les "/", ".", "," et "-" sont remplacés par des espaces) : "MET copy number as a secondary driver of EGFR TKI resistance in EGFR mutant NSCLC http:  bit ly 2HfGHjn #editorial #lcsm"
#     
# 3. On crée une liste de toutes les chaînes de caractère précédées et suivies par une espace : ['MET', 'copy', 'number', 'as', 'a', 'secondary', 'driver', 'of', 'EGFR', 'TKI', 'resistance', 'in', 'EGFR', 'mutant', 'NSCLC', 'http:', '', 'bit', 'ly', '2HfGHjn', '#editorial', '#lcsm']
#     
# 4. La recherche du biomarqueur se fait ensuite à partir de la liste des termes entre crochets ci-dessus. Dans cet exemple, comme on a bien le terme "MET". On constate que le tweet fait égalment référence au biomarqueur EGFR.
# 
# :::
# 
# 

# In[74]:


c = 0
c2 = 0
x = 'MET'
dict_sentence = []
index_name = []
for n, t in enumerate(df0["text"][0:100000]):
    liste = []
    if len(dict_sentence) < 21 :
        #liste.append(t)
        tt = t.replace("/", " ").replace("-", " ").replace("(", " ").replace(")", " ").replace(".", " ").replace(",", " ").split(" ")
        ttt = t.replace("/", " ").replace("-", " ").replace("(", " ").replace(")", " ").replace(".", " ").replace(",", " ")
        
        marker = [f"{x}[(+)]", f"^[#@]{x}[^AaSsUuEeYyHh]", f"[#@]{x}$", f"^{x}[^AaSsUuEeYyHh]", f"^{x}$"]
        marker2 = f"{x}"

        word = [w for w in tt if re.search('|'.join(marker), w)]
        word2 = [w for w in tt if re.search(marker2, w)]
        if len(word2) > 0:
            index_name.append(t)
            liste.append(True)
            list_word = "; ".join(word2)
            liste.append(list_word)
            if len(word) > 0:
                c = c + 1
                liste.append(True)
                list_word = "; ".join(word)
                liste.append(list_word)
                
            else:
                liste.append(False)
                liste.append(np.nan)
            dict_sentence.append(liste)
        else:
            if len(word) > 0:
                c = c + 1
                index_name.append(t)
                liste.append(False)
                liste.append(np.nan)
                liste.append(True)
                list_word = "; ".join(word)
                liste.append(list_word)
                dict_sentence.append(liste)
                
            else :
                pass
            

    else:
        pass
    
    



# In[75]:


cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'color: black; text-align:left'
}

                   
df_exemple = pd.DataFrame(dict_sentence, 
                          index=pd.Index(index_name, name='Tweets :'),
                          columns=pd.MultiIndex.from_product([['Sans régex', 'Avec régex'],
                                    ['Contient MET', 'Forme détectée']], names = ["Méthodes :", ""]))
s = df_exemple.style.format(precision=0, na_rep='Aucune', 
                formatter={('Sans régex', 'Avec régex'): lambda x: "$ {:,.1f}".format(x*-1e6)
                          })

s.set_table_styles([cell_hover, index_names, headers])
s.set_caption("Comparaison de la méthode initiale et de celle utilisant les régex sur le biomarqueur MET") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top'
 }], overwrite=False)


# In[76]:



biom=['ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]#
dict_marker = {'ROS1':[], 'ALK':[],'EXON':[], 'EGFR':[],'KRAS':[],'NTRK':[],'BRAF':[],"MET":[],'RET':[],"HER2":[]}
dict_marker2 = {'ROS1':[], 'ALK':[],'EXON':[], 'EGFR':[],'KRAS':[],'NTRK':[],'BRAF':[],"MET":[],'RET':[],"HER2":[]}
list_marker = []

list_word2 = []
for n, t in enumerate(df0['text']):
    
    list_word = []
    tt = t.replace("/", " ").replace("-", " ").split(" ")
    id_text = df0["id"].iloc[n]
    list_word.append(n)
    list_word.append(id_text)
    for x in biom:
        list_word2 = []
        if x == "MET":
            marker = [f"{x}[(+)]", f"^[#@]{x}[^AaSsUuEeYyHh]", f"[#@]{x}$", f"^{x}[^AaSsUuEeYyHh]", f"^{x}$"]
            word = [w for w in tt if re.search('|'.join(marker), w)]
            word2 = [w for w in tt if re.search(f'{x}', w) and w not in word]
            
            
        elif x == 'RET':
            marker = [f"{x}[(+)]", f"[#@]{x}[^wWaeiouAEIOUrRhHzZ]", "[#@]{x}$", f"^{x}[^wWaeiouAEIOUrRhHzZ]", f"^{x}$"]
            word = [w for w in tt if re.search('|'.join(marker), w)]
            word2 = [w for w in tt if re.search(f'{x}', w) and w not in word]
        
        elif x == 'ALK':
            marker = [f"{x}[(+)]", f"[#@]{x}",  f"^{x}", f"^{x}$"]
            word = [w for w in tt if re.search('|'.join(marker), w)]
            word2 = [w for w in tt if re.search(f'{x}', w) and w not in word]
            
        elif x == 'NTRK':
            marker = [f"{x}[(+)]", f"[#@]{x}",  f"^{x}", f"[^(vince)]{x}", f"^{x}$"]
            word = [w for w in tt if re.search('|'.join(marker), w, re.IGNORECASE)]
            word2 = [w for w in tt if re.search(f'{x}', w,  re.IGNORECASE) and w not in word]

                
        else:
            marker = [f"{x}[(-)]", f"{x}[(+)]", f"[#@/]{x}", f"^{x}", f"{x}"]
            word = [w for w in tt if re.search('|'.join(marker), w, re.IGNORECASE)]
            word2 = [w for w in tt if re.search(f'{x}', w,  re.IGNORECASE) and w not in word]
        
        if len(word) > 0:
            list_word.append(1)
            for y in word:
                list_word2.append(y)
        else:
            list_word.append(0)
        
        
        
        previous_list_word = dict_marker[x]
        current_list_word = list_word2
        upgraded_list_word = previous_list_word + current_list_word
        dict_marker[x] = [x for x in set(upgraded_list_word)]
        

        if len(word2) > 0:
            #print(n)
            #print(t)
            #print(word)
            #print(word2)
            previous_list_word2 = dict_marker2[x]
            upgraded_list_word2 = previous_list_word2 + word2
            dict_marker2[x] = [x for x in set(upgraded_list_word2)]
        else:
            pass
        
    
      
    list_marker.append(list_word)

        
            
    


# In[77]:


columns_name = ['index', 'id', 'ROS1', 'ALK','EXON', 'EGFR','KRAS','NTRK','BRAF',"MET",'RET',"HER2"]
df_marker = pd.DataFrame(np.array(list_marker), columns = columns_name)


# In[78]:


type_to_change={'id': str, 'ROS1':int,'ALK': int,'EXON':int, 'EGFR':int,'KRAS':int,'NTRK':int,
                'BRAF':int,'MET':int,'RET':int,'HER2':int}
df_marker = df_marker.drop(columns = ["index"]).astype(type_to_change)


# In[79]:



comparaison = []
for x in biom:
    liste_count = []
    df_m = df_marker.loc[df_marker[x]==1]
    dfb_b = dfb.loc[dfb[x]==True]
    liste_count.append(x)
    liste_count.append(len(df_m))
    liste_count.append(len(dfb_b))
    liste_count.append(len(df_m) - len(dfb_b))
    comparaison.append(liste_count)
    #comparaison[x] = liste_count


# Le tableau ci-dessous donne le nombre d'occurrences retrouvées pour chacun des marqueurs avec les deux méthodes.

# In[80]:


columns_name = ["Biomarqueur", "Avec expressions régulières", "Sans expression régulière", "Ecart"]
count_marker = pd.DataFrame(np.array(comparaison), columns = columns_name).sort_values("Ecart", ascending = True)
#count_marker.to_csv("comparaison_des_regles_de_matching.csv", sep = ',')
count_marker


# In[81]:


df_marker["nb_of_biomarker"] = df_marker[biom].sum(axis=1)


# In[82]:


df01 = df0[["id"]]
df_marker1 =df_marker[["id"]]


# In[83]:


df00 =  df0.merge(df_marker, how = "inner", on = ["id"])


# In[84]:


d = {}
for m in biom:
    l = [x for x in dict_marker2[m]]
    d[m] = l


# In[95]:


dfbiom1 = dfbiom1.drop(columns = ["index","ROS1","EXON","EGFR","KRAS", "BRAF"])


# In[98]:


dfbiom1.columns


# In[94]:


i = -1
for m in biom:
    i+=1
    dfbiom01= pd.DataFrame({m : d[m]}).reset_index()
    if i>0:
        dfbiom1 = dfbiom1.merge(dfbiom01, on = ["index"], how = "outer")
    else:
        dfbiom1 = dfbiom01


# In[97]:


#dfbiom1 = dfbiom1.drop(columns=["ROS1","EXON","EGFR","KRAS", "BRAF"])
dfbiom1_style = dfbiom1.style.format(precision=0, na_rep='')

dfbiom1_style.set_caption("Les formes comptées comme occurrences par la méthode 1 et exclues avec la méthodes 2") .set_table_styles([{
     'selector': 'caption',
     'props': 'caption-side: top ; color: black ; font-size : 14pt'
 }], overwrite=False)


# In[ ]:




