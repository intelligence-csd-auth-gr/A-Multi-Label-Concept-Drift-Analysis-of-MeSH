# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:48:19 2021

@author: room5
"""

import pandas as pd
import numpy as np

def get_diff_dicts(data_dict):
	diff_dict = dict()
	for i in range(0,len(data_dict['Descriptor'])):
		if(data_dict['Descriptor'][i] not in diff_dict.keys()):
			diff_dict[data_dict['Descriptor'][i]] = list()
		for year in [2014,2015,2016,2017,2018]:
			#print(year)
			if(data_dict[str(year)+' F1'][i] >= 0.1 and data_dict[str(year+1)+' F1'][i] >= 0.1):
				diff_dict[data_dict['Descriptor'][i]].append((data_dict[str(year+1)+' F1'][i]-data_dict[str(year)+' F1'][i])/data_dict[str(year)+' F1'][i])
			else:
				 diff_dict[data_dict['Descriptor'][i]].append(0)
				
	return diff_dict


def get_max_diffs(diff_dict):
	max_diffs= dict()
	for key in diff_dict.keys():
	   max_diffs[key] = [np.max(diff_dict[key]),np.argmax(diff_dict[key])]
		
		
	max_diffs= dict(sorted(max_diffs.items(), key=lambda item: item[1][0]))
	
	for key in max_diffs:
		print(key)
		print(max_diffs[key])

df1 = pd.read_csv(r'D:/Google Drive/AMULET/Concept Drift/mesh bert datasets_for_most_frequent/Passing year protocol full results.csv')

data_dict = df1.to_dict(orient='list')
#print(data_dict.keys())
#print(data_dict['Descriptor'][0])

diff_dict = get_diff_dicts(data_dict)



mean_diff_values_per_year = dict()
year_pairs=['2014-2015','2015-2016','2016-2017','2017-2018','2018-2019']
diffs_per_year = dict()
for i in range(0,5):
	sum = 0
	ignored_scores= 0
	diffs_per_year[year_pairs[i]] = list()
	for key in diff_dict.keys():
		if(diff_dict[key][i] != 0):
			sum+=diff_dict[key][i]
		
		else:
			ignored_scores+=1
		diffs_per_year[year_pairs[i]].append(diff_dict[key][i])
		   
	sum = sum/(len(diff_dict.keys())-ignored_scores)
	
	mean_diff_values_per_year[year_pairs[i]] = sum

for key in mean_diff_values_per_year.keys():
	print(key)
	print(mean_diff_values_per_year[key])
		
#%%
import matplotlib.pyplot as plt
import csv



for years in year_pairs:
	plt.boxplot(x = diffs_per_year[years])
	plt.xlabel('Descriptor')
	plt.ylabel('F1 Difference')
	plt.show()


#%%

############################### Get most drifting labels based on scatter plots ####################################
import csv

def get_max_or_min_values_for_diffs(mode,num,diff_list):
	if(mode == 'min'):
		indexes = np.argsort(diff_list)[:num]
		values = np.sort(diff_list)[:num]
	elif(mode == 'max'):
		indexes = np.argsort(diff_list)[-num:]
		values = np.sort(diff_list)[-num:]
		
	return values,indexes


maxv,maxi = get_max_or_min_values_for_diffs('max', 1, diffs_per_year['2018-2019'])
minv,mini = get_max_or_min_values_for_diffs('min', 3, diffs_per_year['2018-2019'])

print(maxv)
print(maxi)
for index in maxi:
	print(index)
	print(list(diff_dict.keys())[index])
	
print(minv)
print(mini)
for index in mini:
	print(index)
	print(list(diff_dict.keys())[index])
	

	
csv_columns = ['F1-Change','F1-Diff' ,'Descriptor']
with open('Most drifting labels_for_2018-2019.csv', 'w',newline="") as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
	writer.writeheader()
	for i in range(0,len(maxv)):
		writen_row = dict()
		writen_row['F1-Change'] = 'Increase'
		writen_row['F1-Diff'] = maxv[i]
		writen_row['Descriptor'] = list(diff_dict.keys())[maxi[i]]
		writer.writerow(writen_row)
	for i in range(0,len(minv)):
		writen_row = dict()
		writen_row['F1-Change'] = 'Decrease'
		writen_row['F1-Diff'] = minv[i]
		writen_row['Descriptor'] = list(diff_dict.keys())[mini[i]]
		writer.writerow(writen_row)
			
#%%
############################################# Find outliers based on outlier detection ###########################


from sklearn.ensemble import IsolationForest


def find_outliers_per_year(outlier_detector,contamination):
	outlier_dict= dict()
	for year_pair in diffs_per_year.keys():
		#print(year_pair)
		outlier_dict[year_pair] = list()
		diffs_list = list()
		for dif in diffs_per_year[year_pair]:
			diffs_list.append([dif])
			
		detector = outlier_detector(contamination = contamination,behaviour = 'new')
		outliers = detector.fit_predict(diffs_list)
		for i in range(0,len(outliers)):
			if(outliers[i] == -1):
				#print(diffs_per_year['2014-2015'][i])
				#print(list(diff_dict.keys())[i])
				outlier_dict[year_pair].append((diffs_per_year[year_pair][i], list(diff_dict.keys())[i]))
	return outlier_dict



outlier_dict = find_outliers_per_year(IsolationForest, 0.05)


for key in outlier_dict:
	print(key)
	print(outlier_dict[key])


#%%            
################################# Find outliers based on outlier detection for combined years ####################

from collections import Counter
from sklearn.decomposition import PCA
import numpy as np

def find_outliers_for_all_yearsr(outlier_detector,contamination):
   diffs_list = list()
   final_outliers = list()
   for i  in  range(0,len(diff_dict.keys())):
	   diffs_per_desc = list()
	   for year_pair in diffs_per_year.keys():
		   diffs_per_desc.append(diffs_per_year[year_pair][i])
	   diffs_list.append(diffs_per_desc)
   
   detector = outlier_detector(contamination = contamination,behaviour = 'new',random_state=0)
   outliers = detector.fit_predict(diffs_list)
   anom = Counter(outliers)[-1]
   norm = Counter(outliers)[1]
   contam = anom/(norm+anom)
   print(contam)
   for i in range(0,len(outliers)):
	   if(outliers[i] == -1):
		   final_outliers.append(list(diff_dict.keys())[i])
		   
				
   return final_outliers,diffs_list,contam
		
   
	   



outliers, diffs_list,contam = find_outliers_for_all_yearsr(IsolationForest, 'auto')

print(outliers)


pca = PCA(n_components = 2, random_state=0)
new_features = pca.fit_transform(diffs_list)
reconstructed_features = pca.inverse_transform(new_features)


for i in range(0,len(diffs_list)):
    reconstruction_error = np.linalg.norm(diffs_list[i] - reconstructed_features[i])
    if(reconstruction_error > contam and list(diff_dict.keys())[i] in outliers):
        print(list(diff_dict.keys())[i])
        print(diffs_list[i])


	   
	   
	   
			

