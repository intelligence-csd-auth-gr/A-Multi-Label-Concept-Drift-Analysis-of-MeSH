# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:08:30 2021

@author: room5
"""
import os
import pickle
import random


path = r'D:\Google Drive\AMULET\Concept Drift\concept drift datasets for each year'

def read_pickles(name):
	if '.pkl' in name:
		name = name[:-4]
	with open(name + ".pkl", "rb") as f:
		pickle_file = pickle.load(f)
	f.close()

	return pickle_file



def find_label_intersection(path):
    set_list = list()
    for input_name in os.listdir(path):
        print(input_name)
        labels = set(read_pickles(path+'\\'+input_name))
        set_list.append(labels)
    
    intersection_labels = set.intersection(*set_list)
    ignored_labels = read_pickles(r'D:\Google Drive\AMULET\Concept Drift\ignored_labels_2013_2019.pkl')
    print(len(intersection_labels))
    final_labels = intersection_labels - ignored_labels
    print(len(final_labels))
    return final_labels
        

random.seed(10)
labels= find_label_intersection(r'D:\Google Drive\AMULET\Concept Drift\label list for each year')
labels = list(labels)
"""
random.shuffle(labels)
examined_labels = labels[0:300]
"""

def find_most_frequent_labels(path,year,labels,frequency_dict):
    
    for input_name in os.listdir(path):
        if(input_name != 'desktop.ini' and input_name.__contains__('_labels_without') and input_name.__contains__(year)):
            print(input_name)
            with open(path + '\\'+ input_name) as f:
                for line in f:
                    for label in line[2:-3].split("', '"):
                        if label.replace('"', '') in labels:
                            if(label.replace('"', '') not in frequency_dict.keys()):
                                frequency_dict[label.replace('"', '')] = dict()
                            if(year not in frequency_dict[label.replace('"','')].keys()):
                                frequency_dict[label.replace('"', '')][year] = 1
                            elif(year in frequency_dict[label.replace('"','')].keys()):
                                frequency_dict[label.replace('"', '')][year] +=1
    
    
    return frequency_dict

f_d = dict()
for year in ['2013','2014','2015','2016','2017','2018','2019']:
    
    f_d = find_most_frequent_labels(path, year, labels,f_d)
                            
                                 
                                



#%%
def get_intersection_of_most_frequent_labels(f_d):
    year_dict = dict()
    for key in f_d:
        for year in ['2013','2014','2015','2016','2017','2018','2019']:
            if (year not in year_dict.keys()):
                year_dict[year] = list()
            if(year in f_d[key].keys()):
                year_dict[year].append([key , f_d[key][year]])
            else:
                year_dict[year].append([key , 0])
                
        
    
    
    for key in year_dict.keys():
       year_dict[key].sort(key = lambda x:x[1],reverse = True)
                     
    
    set_list = list()
    year_dict_2=dict()
    for key in year_dict.keys():
        year_set = set()
        for label in year_dict[key][10:310]:
            #print(label)
            year_set.add(label[0])
            if(key not in year_dict_2.keys()):
                year_dict_2[key] = list()
            year_dict_2[key].append(label[0])
        set_list.append(year_set)
    examined_labels = set.intersection(*set_list)
    
    
    examined_labels_freq_per_year = dict()
    for item in examined_labels:
        for year in year_dict.keys():
            for label in year_dict[year][10:310]:
                if(label[0] == item):
                    if(label[0] not in examined_labels_freq_per_year.keys()):
                        examined_labels_freq_per_year[label[0]] = list()
                   
                    examined_labels_freq_per_year[label[0]].append([year,label[1]])
                   
    return examined_labels,examined_labels_freq_per_year,year_dict_2

examined_labels,examined_labels_freq_per_year,year_dict = get_intersection_of_most_frequent_labels(f_d)
print(len(examined_labels))
print(len(examined_labels_freq_per_year))


for key in examined_labels_freq_per_year.keys():
    print(key)
    print(examined_labels_freq_per_year[key])
    
"""
import pandas as pd
data_df = pd.DataFrame(examined_labels_freq_per_year)
data_df.to_csv('label_freqs.csv',index = False)
"""

#%%

def get_labelset_for_given_year(path,year,examined_labels):
    labels_for_year = list()
    for input_name in os.listdir(path):
        if(input_name != 'desktop.ini' and input_name.__contains__('_labels_without') and input_name.__contains__(year)):
            print(input_name)
            with open(path + '\\'+ input_name) as f:
                for line in f:
                    label_list = ''
                    for label in line[2:-3].split("', '"):
                        if label.replace('"', '') in examined_labels:
                            
                            #print(label.replace('"',''))
                            if(label_list == ''):
                                label_list = label.replace('[','').replace(']','').replace("'","")
                            else:
                                    label_list = label.replace('[','').replace(']','').replace("'","") + '#'+label_list
                       
                    
                    labels_for_year.append(label_list)
    
    
    return labels_for_year

def get_dataset_for_given_year(path,year):
    dataset = list()
    for input_name in os.listdir(path):
        if(input_name != 'desktop.ini' and input_name.__contains__('dataset') and input_name.__contains__(year)):
            print(input_name)
            with open(path + '\\'+ input_name) as f:
                for line in f:
                    #print(line[3:].split(',"journal":')[0][:-1])
                    dataset.append(line[3:].split(',"journal":')[0][:-1])
                    
    
    return dataset


dataset = get_dataset_for_given_year(path, '2019')
label_list = get_labelset_for_given_year(path, '2019',examined_labels)
print(len(dataset))
print(len(label_list))
                
           
#%%

import pandas as pd

def prepare_data_for_bert(dataset,label_list):
    dataset_dict = dict()
    for i in range(0,len(dataset)):
        if(label_list[i] != ''):
            dataset_dict[dataset[i]] = label_list[i]
     
        
    print(len(dataset_dict))
    
    return dataset_dict
    

dataset_dict = prepare_data_for_bert(dataset,label_list)
#data_df = pd.DataFrame(list(dataset_dict.items()),columns=['text','target'])
#data_df.to_csv('mesh_2013.csv',index = False)
 

    


