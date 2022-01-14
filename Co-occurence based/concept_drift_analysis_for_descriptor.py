# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:20:54 2021

@author: room5
"""

import os
import pickle
import random 
import pandas as pd



co_descs_per_year = dict()
full_res = pd.read_csv('D:\Google Drive\AMULET\Concept Drift\mesh bert datasets_for_most_frequent\Passing year protocol full results.csv')
for descriptor in full_res['Descriptor']:
	print(descriptor)
	co_descs_per_year[descriptor]=dict()
	for year in ['2013','2014','2015','2016','2017','2018','2019']:
		co_descs_per_year[descriptor][year] = dict()
		print(year)
		train_data = pd.read_csv('D:\Google Drive\AMULET\Concept Drift\mesh bert datasets_for_most_frequent\mesh_'+year+'.csv')
	
		for line in train_data['target']:
			if(line.__contains__(descriptor)):
				#print(line)
				#print("#################################") 
				for item in line.split('#'):
					if(item != descriptor):
							if (item not in co_descs_per_year[descriptor][year].keys()):
								co_descs_per_year[descriptor][year][item]=1
							else:
								co_descs_per_year[descriptor][year][item]+=1

#%%
import pickle



for key in co_descs_per_year.keys():
	  for year in co_descs_per_year[key]:
		  #print(co_descs_per_year[key])
		  co_descs_per_year[key][year] = {k: v for k, v in sorted(co_descs_per_year[key][year].items(), key=lambda item: item[1])}

#pickle.dump(co_descs_per_year, open('co-occurence dictionary.pkl', 'wb'))


#%%
import pickle


def read_pickles(name):
	if '.pickle' in name:
		name = name[:-4]
	with open(name + ".pkl", "rb") as f:
		pickle_file = pickle.load(f)
	f.close()
	
	return pickle_file



co_descs_per_year = read_pickles(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH/co-occurence dictionary')
diffs_per_year_pair = dict()
for key in co_descs_per_year.keys():
	print(key) 
	for year in co_descs_per_year[key]:
		print(year)
		print(list(co_descs_per_year[key][year])[-10:])
		if(str(int(year)-1) in co_descs_per_year[key]):
			print(year+' - ' + str(int(year)-1))
			if(year+' - ' + str(int(year)-1) not in diffs_per_year_pair.keys()):
					  diffs_per_year_pair[year+' - ' + str(int(year)-1)] = list()
			print(set(list(co_descs_per_year[key][year])[-10:]) - set(list(co_descs_per_year[key][str(int(year)-1)])[-10:]))
			diffs_per_year_pair[year+' - ' + str(int(year)-1)].append([key, len(set(list(co_descs_per_year[key][year])[-10:]) - set(list(co_descs_per_year[key][str(int(year)-1)])[-10:]))])
			
	   
	


#%%
from operator import itemgetter
for key in diffs_per_year_pair.keys():
	if(key == '2014 - 2013'):
		print((sorted(diffs_per_year_pair[key], key=itemgetter(1))))
		print('##################################################################################################################')
		
		
		
		
		
		
#%%
import numpy as np
import csv
counts=dict()
for key in diffs_per_year_pair.keys():
	if(key != '2014 - 2013'):
		print(key)
		counts[key] = dict()
		for diff in (sorted(diffs_per_year_pair[key], key=itemgetter(1))):
		   if(diff[1] not in counts[key].keys()):
			   counts[key][diff[1]]= 1
		   else:
				counts[key][diff[1]]+=1
				
with open('co_occ_info.csv','w',newline = '') as csv_f:
	row_wr = csv.writer(csv_f, delimiter=',')
	row_wr.writerow(['Year-Pair', 'New Co-occurrent' ,'% of Descriptors'])
	for key in counts:
		print(key)
		print(counts[key])
		for second_key in counts[key]:
			print(str(second_key) +": "+str(np.round((counts[key][second_key]/198*100),2))+"%")
			row_wr.writerow([str(key), str(second_key), str(np.round((counts[key][second_key]/198*100),2))+"%" ])
#%%

"""
path = r'D:\Google Drive\AMULET\Concept Drift\concept drift datasets for each year'
descriptor = 'Mice, Nude'

def find_most_frequent_labels(path,year,frequency_dict):
	
	for input_name in os.listdir(path):
		if(input_name != 'desktop.ini' and input_name.__contains__('_labels_without') and input_name.__contains__(year)):
			print(input_name)
			with open(path + '\\'+ input_name) as f:
				for line in f:
					if(line.__contains__(descriptor)):
						for label in line[2:-3].split("', '"):
							if(label.replace('"', '') != descriptor):
								if (label.replace('"', '') not in frequency_dict[year].keys()):
									frequency_dict[year][label.replace('"', '')]=1
								else:
									frequency_dict[year][label.replace('"', '')]+=1

						   
	
	return frequency_dict


f_d = dict()
for year in ['2013','2014','2015','2016','2017','2018','2019']:
	f_d[year] = dict()
	f_d = find_most_frequent_labels(path, year,f_d)


#%%

print('2013')
print(list(f_d['2013'])[-5:])
print('2014')
print(list(f_d['2014'])[-5:])
print('2015')
print(list(f_d['2015'])[-5:])
print('2016')
print(list(f_d['2016'])[-5:])
print('2017')
print(list(f_d['2017'])[-5:])
print('2018')
print(list(f_d['2018'])[-5:])
"""