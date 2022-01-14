# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:54:38 2021

@author: room5
"""

import pandas as pd
import json

year = '2019'

df = pd.read_csv(r'D:\Google Drive\AMULET\Concept Drift\mesh bert datasets_for_most_frequent\mesh_'+year+'.csv')

print(df.text) 

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_2013', 'r') as f13:
    data_2013 = json.load(f13)

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_2014', 'r') as f14:
    data_2014 = json.load(f14)

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_2015', 'r') as f15:
    data_2015 = json.load(f15)

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_2016', 'r') as f16:
    data_2016 = json.load(f16)

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_2017', 'r') as f17:
    data_2017 = json.load(f17)

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_2018', 'r') as f18:
    data_2018 = json.load(f18)

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_2019', 'r') as f19:
    data_2019 = json.load(f19)


data_labels = list(data_2013.keys())+list(data_2014.keys())+list(data_2015.keys())+list(data_2016.keys())+list(data_2017.keys())+list(data_2018.keys())+list(data_2019.keys())
print('before', len(data_labels))
data_labels = list(set(data_labels))
print('after', len(data_labels))
print(data_labels)



#%%

# creating a list of dataframe columns
num_altered_texts = 0
columns = list(df.text)
texts_dict_201x = {}
for i in range(len(columns)):
  new_text = columns[i].lower().translate(str.maketrans('', '', "!,.:;?()[]{}"))
  for label in data_labels:
    #print(label)
    #print("#############################")
    #print(new_text)
    #print('############################')
    if(not label.__contains__(',')):
        processed_label = label.replace(',','').replace(' ', '#')
    elif(label.__contains__(',')):
        processed_label= label.split(', ')[len(label.split(','))-1].replace(' ','#')
        for j in range(len(label.split(','))-2,-1,-1):
            processed_label= processed_label+"#"+label.split(', ')[j].replace(' ','#')
        label = processed_label.replace('#',' ')
    #print(label)
    #print(processed_label)
    
    new_text = new_text.replace(label, processed_label)
  texts_dict_201x[i] = new_text
  if columns[i].lower() != new_text:
    num_altered_texts += 1
  #print(new_text)
print(num_altered_texts, 'out of', len(columns))
with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets_variant\texts_dict_'+year, 'w') as f: 
    json.dump(texts_dict_201x, f)