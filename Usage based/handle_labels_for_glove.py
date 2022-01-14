# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:27:41 2021

@author: room5
"""


import pandas as pd
import json


year = '2013'

df = pd.read_csv('D:\Google Drive\AMULET\Concept Drift\mesh bert datasets_for_most_frequent\mesh_'+year+'.csv')




# creating a list of dataframe columns
columns = list(df.target)
labels_dict_201x = {}
for i in range(len(columns)):
  instance_labels = columns[i].lower().translate(str.maketrans('', '', "!.:;?()[]{}")).split('#')
  # print(columns[i], columns[i].lower(), instance_labels)
  for inst_label in instance_labels:
    if inst_label not in labels_dict_201x:
      labels_dict_201x[inst_label] = set()
    labels_dict_201x[inst_label].add(i)
  # if i==100:
  #   break

for k, v in labels_dict_201x.items():
  labels_dict_201x[k] = list(v)
  
  
  



with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_'+year, 'w') as f: 
    json.dump(labels_dict_201x, f)

with open(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\glove datasets\labels_dict_'+year, 'r') as f:
  data_201x = json.load(f)
print(data_201x)
