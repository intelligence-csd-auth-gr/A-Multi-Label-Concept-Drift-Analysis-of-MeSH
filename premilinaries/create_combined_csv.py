# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:27:54 2021

@author: room5
"""

import pandas as pd
import csv


def create_csv(final_dict, frequencies):
    output_dict = dict()
    for key in final_dict.keys():
        #print(key)
        if(key not in output_dict.keys()):
            output_dict[key] = dict()
        for pair_f in frequencies[key]:
            #print(pair_f.split("', ")[0].replace('[','').replace("'",""))
            if(pair_f.split("', ")[0].replace('[','').replace("'","") not in output_dict[key].keys()):
                output_dict[key][pair_f.split("', ")[0].replace('[','').replace("'","")] = list()
            output_dict[key][pair_f.split("', ")[0].replace('[','').replace("'","")].append(pair_f.split("', ")[1].replace(']','').replace("'",""))
        year = 2014
        output_dict[key]['2013'].append('-')
        for pair in final_dict[key]:
            #print(pair.split(': ')[1].split('}')[0])
            #print("BOY")
            output_dict[key][str(year)].append(pair.split(': ')[1].split('}')[0])
            #float(first_pair) - float(pair.split(': ')[1].split('}')[0]))  
            year = year + 1
    
    csv_columns = ['Descriptor', '2013 Frequency','2013 F1' ,'2014 Frequency' ,'2014 F1', '2015 Frequency' ,'2015 F1','2016 Frequency' ,'2016 F1','2017 Frequency' ,'2017 F1','2018 Frequency' ,'2018 F1','2019 Frequency' ,'2019 F1']
    try:
        with open('Passing year protocol full results.csv', 'w',newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for key in output_dict.keys():
                writen_row = dict()
                writen_row['Descriptor'] = key
                for second_key in output_dict[key].keys():
                   writen_row[second_key+' Frequency'] = output_dict[key][second_key][0]
                   writen_row[second_key+' F1'] = output_dict[key][second_key][1]
                writer.writerow(writen_row)
                    
                    
    except IOError:
        print("I/O error")
    
    return output_dict

   

df1 = pd.read_csv(r'D:/Google Drive/AMULET/Concept Drift/mesh bert datasets_for_most_frequent/f1_score_per_year_for_most_frequent_labels.csv')
df2 = pd.read_csv(r'D:/Google Drive/AMULET/Concept Drift/mesh bert datasets_for_most_frequent/f1_score_per_year_pair_for_most_frequent_labels.csv')
frequencies = pd.read_csv(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\label_freqs.csv')


out = create_csv(df1.to_dict(orient = 'list'), frequencies.to_dict(orient = 'list'))

for key in out:
    print(key)
    print(out[key])
