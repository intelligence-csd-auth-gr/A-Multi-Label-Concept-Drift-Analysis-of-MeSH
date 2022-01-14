# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:38:05 2021

@author: room5
"""
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import csv

train_data = pd.read_csv(r'D:\Google Drive\AMULET\Concept Drift\train.csv')
test_data = pd.read_csv(r'D:\Google Drive\AMULET\Concept Drift\test.csv')


train_x = train_data.text.values.tolist()
test_x = test_data.text.values.tolist()
string_labels = train_data.target.values.tolist()
string_labels_test = test_data.target.values.tolist()
labels=list()
labels_test=list()
for label in string_labels:
    labels.append(label.split("#"))
for label_test in string_labels_test:
    labels_test.append(label_test.split("#"))    
del string_labels,string_labels_test
mlb = MultiLabelBinarizer()
mlb.fit(labels)


with open('train_binarized.csv', 'a', encoding='UTF8',newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(['text', 'target'])

    for i in range(0,len(labels)):
        temp_labels = labels[i]
        temp_labels = mlb.transform([temp_labels])
        temp_x_train = train_x[i]
        writer.writerow((temp_x_train,temp_labels[0]))
    
    
    
    
        
"""
for i in range(0,len(labels_test),100):
    temp_labels = labels_test[i:i+100]
    temp_labels = mlb.transform(temp_labels)
    test_data.target[i:i+100] = temp_labels.tolist()
"""