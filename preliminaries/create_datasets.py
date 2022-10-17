# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:09:31 2021

@author: room5
"""

import ijson
import json
import os
import pickle
import pandas as pd



def read_json(input_name,number=1000):
	if (input_name == r"D:\Google Drive\AMULET\all_MeSH\allMeSH_2020.json" or input_name.__contains__('_2016') or input_name.__contains__('_2017')):
		encoding = 'windows-1252'
	else:
		encoding = 'UTF-8'

	#os.chdir(input_dir)
	dataset = []
	labels = []
	count = 0
	string = ""
	labels = list()
	ok = 'false'
	if(input_name.__contains__('_2016') or input_name.__contains__('_2017') or input_name.__contains__('_2020')):
		encoding = "windows-1252"
	else:
		encoding = "utf-8"
	year_examined = str(int(input_name.split('_')[2].replace('.json', ''))-2)
	print(year_examined)
	with open(input_name,encoding = encoding) as f:
		for line in f:
			if(line.__contains__(',"year":'+'"'+year_examined+'"')):
				#print(line)
				count+=1
			
				if(line.__contains__('"abstractText":') and line.__contains__('"meshMajor":') and (input_name.__contains__('_2017') or input_name.__contains__('_2018') or input_name.__contains__('_2019') or input_name.__contains__('_2020'))): 
					labels.append(line.split('"meshMajor":')[1].split(',"year":')[0])
					dataset.append(line.split('"meshMajor":')[1].split(',"year":')[1].split(',"abstractText":')[1].split(',"pmid":')[0])
				elif(line.__contains__('"abstractText":') and line.__contains__('"meshMajor":') and not input_name.__contains__('_2016')):
					dataset.append(line.split('"abstractText":')[1].split(',"meshMajor":')[0])
					labels.append(line.split('"abstractText":')[1].split(',"meshMajor":')[1].split(',"pmid":')[0])
				elif(line.__contains__('"abstractText":') and line.__contains__('"meshMajor":') and input_name.__contains__('_2016')):
					labels.append(line.split(',"meshMajor":')[1].split(',"year":')[0])
					dataset.append(line.split('"abstractText":')[1].split(',"pmid":')[0])
			
				
						
			
			
			if(count == number):
				break

			

	print(count)
	return dataset,labels



#%%
path= r'D:\Google Drive\AMULET\all_MeSH'
os.chdir(r'D:\Google Drive\AMULET\Concept Drift')
for input_name in os.listdir(path)[1:11]:
		print(input_name)
		dataset,labels=read_json(path+'\\'+input_name ,'all')
		"""
		file1 = open(input_name.replace('.json','').replace(input_name.replace('.json','').split('_')[1],str(int(input_name.replace('.json','').split('_')[1])-1))+"_dataset.txt","w")
		for line in dataset:
			 file1.write(str(line.encode('UTF-8')))
			 file1.write('\n')
		file1.close()
		
		file2 =  open(input_name.replace('.json','').replace(input_name.replace('.json','').split('_')[1],str(int(input_name.replace('.json','').split('_')[1])-1))+"_labels.txt","w")
		for line in labels:
			 file2.write(str(line.encode('UTF-8')))
			 file2.write('\n')
		file2.close()
		
        """
