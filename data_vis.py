# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:44:07 2020

@author: Abhishek
"""

from os import listdir
import matplotlib.pyplot as plt

#img_db = np.zeros((3006,200,200,3),dtype=np.float32())
#label_db = np.zeros((3006),dtype=np.str) 

#%%Visualisation
path = 'G:/Deep Learning/Age_detection/face_age/'
fld_list = listdir(path)
count = []
fin =[]
for i,fld_name in enumerate(fld_list):
    fin.append(path+fld_name)
    flg = len(listdir(fin[i]))
    count.append(flg)
fig, ax = plt.subplots()
explode = (0.1,0,0,0,0,0,0,0,0.1)
ax.pie(count,explode=explode,labels=fld_list,autopct='%1.1f%%',shadow=True, startangle=90)
ax.axis('equal')

#%%Dataset formation 
tmp = min(count)
f1 = open("img_adr.txt","w+")
f2 = open("label_adr.txt","w+")
f3 = open("val_data.txt","w+")
f4 = open("val_label.txt","w+")
for j,fld_name in enumerate(fld_list):
    fin_list = listdir(path+fld_name+'/')
    for k in range(0,tmp):
        if (k<=tmp-70):
            f1.write(path+fld_name+'/'+fin_list[k]+"\n")
            f2.write(fld_name+"\n")
        else:
            f3.write(path+fld_name+'/'+fin_list[k]+"\n")
            f4.write(fld_name+"\n")
            
f1.close()
f2.close()
f3.close()
f4.close()
