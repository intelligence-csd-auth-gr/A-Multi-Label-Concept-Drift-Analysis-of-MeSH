# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:16:49 2021

@author: room5
"""

import os
import numpy as np
import pickle

path = r'D:\Google Drive\AMULET\Concept Drift\concept drift datasets for each year'




def read_pickles(name):
	if '.pkl' in name:
		name = name[:-4]
	with open(name + ".pkl", "rb") as f:
		pickle_file = pickle.load(f)
	f.close()

	return pickle_file




ignored_labels = read_pickles(r'D:\Google Drive\AMULET\Concept Drift\ignored_labels_2013_2019.pkl')
ignored_labels = list(ignored_labels)
print(ignored_labels)
for input_name in os.listdir(path):
    if(input_name != 'desktop.ini' and input_name.__contains__('_labels')):
        with open(path + '\\'+ input_name) as f:
            file2 = open(input_name.replace('.txt','')+"_without_ignored_labels.txt","w")
            for line in f:
                label_list = list()
                for label in line[3:-3].split('","'):
                    if label.replace('"', '') not in ignored_labels:
                        #print(label.replace('"',''))
                        label_list.append(label.replace('"', ''))
                #print(str(label_list))
                file2.write(str(label_list))
                file2.write('\n')
            file2.close()
                    
            
            
                        
                    