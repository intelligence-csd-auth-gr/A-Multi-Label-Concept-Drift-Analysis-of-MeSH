# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:08:01 2021

@author: room5
"""

import os
import pickle
import numpy as np
import pandas as pd

def create_train_test_validation(path,line_id,previous_id,year,step):
     line_dict = dict()
     for input_name in os.listdir(path):
         if input_name.__contains__(year+'_dataset'):
             print(input_name)
             with open(path + '\\'+ input_name) as f:
                 for line in f:
                     line_dict[str(line_id)+' '+input_name.split('_')[1].split('_')[0]] = line[3:-3].split(',"journal":')[0].replace('"','')
                     line_id+=1
                     
                     
                     
             
                     
         if input_name.__contains__(year+'_labels_without'):
             print(input_name)
             count=previous_id
             with open(path + '\\' + input_name) as f1:
                 for line1 in f1:
                     label_string=''
                     for thing in line1[:-1].split("', '"):
                         #print(thing.replace('[','').replace(']','').replace("'",""))
                         if(label_string == ''):
                             label_string = thing.replace('[','').replace(']','').replace("'","")
                         else:
                             label_string = thing.replace('[','').replace(']','').replace("'","") + '#'+label_string
                         #print(thing.replace('[','').replace(']','').replace("'",""))
                     
                     line_dict[str(count)+' '+input_name.split('_')[1].split('_')[0]] = label_string +"\t" +line_dict[str(count)+' '+input_name.split('_')[1].split('_')[0]]
                     count+=1
                     
                     
                     
             previous_id = count
                     
                
                 
                    
     return line_dict,line_id,previous_id
        
def removed_dupes_from_dict(line_dict):
    #print(len(line_dict))
    rev_dict=dict()
    for key,value in line_dict.items():
        rev_dict.setdefault(value,set()).add(key)
    
    
    result = filter(lambda x: len(x)>1, rev_dict.values())
    for res in list(result):
        del line_dict[next(iter(res))]
    
    return line_dict

def write_data_into_csvs(line_dict):
    os.chdir(r'D:\Google Drive\AMULET\Concept Drift')
    train_dict=dict()
    test_dict = dict()
    for key in line_dict:
        if(key.split(' ')[1] != '2014'):
            train_dict[line_dict[key].split('\t')[1]] = line_dict[key].split('\t')[0]
        elif(key.split(' ')[1] == '2014'):
            test_dict[line_dict[key].split('\t')[1]] = line_dict[key].split('\t')[0]
    
    print('train_set :'+str(len(train_dict)))
    print('test_set :'+str(len(test_dict))) 
    train_df = pd.DataFrame(list(train_dict.items()),columns=['text','target'])
    test_df = pd.DataFrame(list(test_dict.items()),columns=['text','target'])
    train_df.to_csv('train_2013_f.csv',index = False)
    test_df.to_csv('test_2014_f.csv',index = False)
            
        
            
      
        
        

path = r'D:\Google Drive\AMULET\Concept Drift\concept drift datasets for each year'
#path = '//home//myloniko//concept drift datasets'
line_id = 0
previous_id = 0



final_dict=dict()
step =0
for year in ['2013','2014']:
    print(year)
    line_dict,line_id,previous_id=create_train_test_validation(path,line_id,previous_id,year,step)
    final_dict.update(line_dict)
    print(len(final_dict))
    removed_dupes_from_dict(final_dict)
    print(len(final_dict))
    step+=len(line_dict)
    del line_dict
write_data_into_csvs(final_dict)




#%%

