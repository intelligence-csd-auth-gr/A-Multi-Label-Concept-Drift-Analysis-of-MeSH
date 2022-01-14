# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:50:06 2021

@author: room5
"""

import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from torchmetrics.functional import f1,accuracy
import numpy as np
from sklearn.metrics import f1_score

def read_pickles(name):
	if '.pickle' in name:
		name = name[:-7]
	with open(name + ".pkl", "rb") as f:
		pickle_file = pickle.load(f)
	f.close()

	return pickle_file


def handle_predictions_for_given_year(year1,year2):
    train_data = pd.read_csv(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\mesh bert datasets_for_most_frequent\mesh_'+year1+'.csv')
    test_data = pd.read_csv(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\mesh bert datasets_for_most_frequent\mesh_'+year2+'.csv')
    
    
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
    labels=mlb.transform(labels)
    labels_test = mlb.transform(labels_test)
    train_data.target = labels.tolist()
    test_data.target = labels_test.tolist()
    del labels,labels_test

    
    model_predictions = read_pickles(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\mesh bert datasets_for_most_frequent\predictions\\'+year1+'-'+year2+'\model_predictions')
    test_set_true = read_pickles(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\mesh bert datasets_for_most_frequent\predictions\\'+year1+'-'+year2+'\\test_set_true')
    
    
    print(accuracy(model_predictions,test_set_true,threshold = 0.1))
    print(f1(model_predictions,test_set_true,average='micro',num_classes=len(mlb.classes_),threshold=0.3))

    
    model_real_predictions = list()
    for pred in model_predictions:
        model_real_predictions.append(np.array((pred >= 0.3).int()))
    
    
    #y_pred_labels = mlb.inverse_transform(np.array(model_real_predictions))
    
    #print(y_pred_labels)
    
  
    predictions_per_label_dict = dict()
    
    for prediction in model_real_predictions:
        for i in range(0,len(prediction)):
            if(mlb.classes_[i] not in predictions_per_label_dict.keys()):
                predictions_per_label_dict[mlb.classes_[i]] = list()
            predictions_per_label_dict[mlb.classes_[i]].append(prediction[i])
            
    
    
 
    test_set_true_array = list()
    for true in test_set_true:
        test_set_true_array.append(np.array(true))
    
    
    real_per_label_dict = dict()
    
    for real in test_set_true_array:
        for i in range(0,len(real)):
                if(mlb.classes_[i] not in real_per_label_dict.keys()):
                    real_per_label_dict[mlb.classes_[i]] = list()
                real_per_label_dict[mlb.classes_[i]].append(real[i])
    
    
   
    return mlb.classes_, real_per_label_dict, predictions_per_label_dict

    """ 
    for i in range(0,len(mlb.classes_)):
        print(mlb.classes_[i])
        print(f1_score(real_per_label_dict[mlb.classes_[i]],predictions_per_label_dict[mlb.classes_[i]]))
   """
        


final_dict = dict()
for year in ['2013-2014','2014-2015','2015-2016','2016-2017','2017-2018','2018-2019']:
    year1 = year.split('-')[0]
    year2 = year.split('-')[1]
    print(year1)
    print(year2)
    classes,real,pred= handle_predictions_for_given_year(year1,year2)
    for i in range(0,len(classes)):
        if classes[i] not in final_dict.keys():
            final_dict[classes[i]] = list()
        final_dict[classes[i]].append({year: f1_score(real[classes[i]],pred[classes[i]])})


data_df = pd.DataFrame(final_dict)
print(data_df.head())
#%%
data_df.to_csv('f1_score_per_year_for_most_frequent_labels.csv',index = False)

#%%
################################## GET MAX DIF FOR 2013 THROUGH 2019 ############################################
diffs_dict=dict()
for key in final_dict.keys():
    diffs_list=list()
    first_pair = final_dict[key][0]['2013-2014']
    #print(first_pair)
    for pair in final_dict[key]:
        for key2 in pair:
            diffs_list.append(first_pair - pair[key2])
    diffs_dict[key] = diffs_list
    


max_diff_dict = dict()
for key in diffs_dict:
    #print(key)
    #print(np.max(diffs_dict[key]))
    max_diff_dict[key] = np.max(diffs_dict[key])


max_diff_dict= dict(sorted(max_diff_dict.items(), key=lambda item: item[1]))

for key in max_diff_dict.keys():
    print(key)
    print(max_diff_dict[key])

    
    
#%%
################################ GET MAX DIF FOR YEAR PAIRS ########################################################

    
diffs_dict=dict()
for key in final_dict.keys():
    diffs_list=list()
    previous_pair = final_dict[key][0]['2013-2014']
    #print(first_pair)
    for pair in final_dict[key]:
        for key2 in pair:
            diffs_list.append(previous_pair - pair[key2])
            previous_pair = pair[key2]
    diffs_dict[key] = diffs_list
    


max_diff_dict = dict()
for key in diffs_dict:
    #print(key)
    #print(np.max(diffs_dict[key]))
    max_diff_dict[key] = np.max(diffs_dict[key])


max_diff_dict= dict(sorted(max_diff_dict.items(), key=lambda item: item[1]))

for key in max_diff_dict.keys():
    print(key)
    print(max_diff_dict[key])


        