# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:28:44 2021

@author: room5
"""

import os
import numpy as np
import contextlib
import time
import pickle




@contextlib.contextmanager
def timer():
    """Time the execution of a context block.

    Yields:
      None
    """
    start = time.time()
    # Send control back to the context block
    yield
    end = time.time()
    print('Elapsed: {:.2f}s'.format(end - start))







path = r'D:\Google Drive\AMULET\Concept Drift\concept drift datasets for each year'




with timer():
    for input_name in os.listdir(path):
        if(input_name != 'desktop.ini' and input_name.__contains__('labels')):
            print(input_name)
            with open(path + '\\'+ input_name) as f:
                label_list=list()
                count=0
                for line in f:
                    #print(line)
                    for label in line[3:-3].split('","'):
                            #print(label.replace('"',''))
                            label_list.append(label.replace('"',''))
                            #print(label.replace('"','').replace("'","").replace("[",'').replace("]",''))
                            #label_list.append(label.replace('"','').replace("'","").replace("[",'').replace("]",''))
                    count+=1
                    
                    if (count == 10000):
                        label_list = list(set(label_list))
                        count=0
                
                print(len(label_list))
                label_list = list(set(label_list))
                pickle.dump(label_list, open(input_name.replace('.txt', '')+'_all_labels.pkl', 'wb'))
                print(len(label_list))
                print(count)
                

#%%
                
import pickle
def read_pickles(name):
	if '.pkl' in name:
		name = name[:-4]
	with open(name + ".pkl", "rb") as f:
		pickle_file = pickle.load(f)
	f.close()

	return pickle_file

new_path = r'D:\Google Drive\AMULET\Concept Drift\label_lists_per_year'


train_labels = set()
test_labels = set()
for item in os.listdir(new_path):
    year_list = read_pickles(new_path+'\\'+item)
    year_set = set(year_list)
    if(item.__contains__('_2013')):
        train_labels = train_labels | year_set
    else:
        test_labels = test_labels | year_set
        
        
        
print(len(train_labels))
print(len(test_labels))

examined_labels = train_labels & test_labels
ignored_labels = test_labels - train_labels


print(len(examined_labels))
print(len(ignored_labels))



for la in ignored_labels:
    print(la)




pickle.dump(ignored_labels, open('ignored_labels_2013_2020.pkl', 'wb'))