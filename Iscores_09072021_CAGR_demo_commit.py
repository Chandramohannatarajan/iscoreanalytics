#!/usr/bin/env python
# coding: utf-8

# # LOADING PYTHON MODULES

# In[1]:


import time
import math
import glob
start = time.time()
import pandas as pd
from pandas import DataFrame
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
model = LabelEncoder()
import os
from glob import glob
plt.rcParams.update({'font.size': 12})
# global option settings
pd.set_option('display.max_columns', 100) # show all column names display
pd.set_option('display.max_rows', 100) # show all rows on display
import pymongo as pym  #Interface with Python <---> MongoDB


# # DB CONNECTIVITY

# In[2]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[3]:


#creating database
db=client['dgdata']


# In[643]:


data_from_db = db.cagr_root_file.find({"NAME" : "THE TEIGNMOUTH QUAY COMPANY LTD"},{'_id':0})
data=pd.DataFrame.from_dict(data_from_db)


# SAMPLE COMPANIES

# BASF PHARMA (CALLANISH) LIMITED
# BROADSOFT LTD
# LLOYDS BANK PLC
# !OBAC LIMITED
# ABF JAPAN LIMITED
# !OBAC UK LIMITED
# ABF THE SOLDIERS' CHARITY
# THE TEIGNMOUTH QUAY COMPANY LTD

# In[644]:


#data.head()


# In[645]:


data=data.sort_values(by="YEAR",ascending=True)


# In[646]:



try:
    misc = ['misc', 'misc', 'misc', 'misc','misc','misc','misc']
    data["INDUSTRY_TYPE"]=misc
except ValueError:
    pass


# In[ ]:





# In[647]:


#data=data.replace([np.nan], 'misc')
try:
    data['INDUSTRY_TYPE'].fillna('misc', inplace=True)
except:
    pass


# In[648]:


try:
    df_all_types=data['INDUSTRY_TYPE'].unique().tolist()
except:
    pass
#df_all_types


# In[649]:


#data.count()


# In[650]:


try:
    data['SIC07'].fillna('unknown', inplace=True)
except:
    pass


# In[651]:


#data.count()


# In[652]:


data.drop_duplicates(keep=False,inplace=True)


# In[653]:


#data.count()


# # CAGR CALCULATION

# In[654]:


#data=data[data['YEAR'] != 2021]


# In[655]:


#data.count()


# In[656]:


data_cagr=data[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']]
#data_cagr.head()


# In[657]:


data_cagr=data_cagr.sort_values(by="YEAR",ascending=True)


# In[658]:


data_cagr["YEAR"] = data_cagr["YEAR"].astype(str).astype(int)


# In[659]:


data_cagr["RETAINED_PROFITS"] = data_cagr["RETAINED_PROFITS"].astype(float).astype(int)


# In[660]:


#data_cagr


# LIST OF DATAFRAMES

# In[661]:


list_dataframes= [v for k, v in data_cagr.groupby('NAME')]


# In[662]:


#list_dataframes[0]


# In[663]:


list_df=[]
#try:
for i in list_dataframes: 
    #print(i)
    #print(" ")
    r=i.values.tolist()
    #print(len(r))
    #print(" ")
    #print(r)
    #print(" ")
    try:
        while r[0][-2]<0:
            if len(r)>0:
            #if r[0][-2]<0:
                #print(r[0])
                #print(" ")
                r.pop(0)
                #print(len(r))
                #print(r)
                #print(" ")
        list_df.append(r)
    except:
        pass
    


# In[664]:


#list_df


# In[665]:


list_all=[]
list_ideal=[]
#try:
for i in list_dataframes: 
    #print(i)
    l=i.values.tolist()
    #print("list: ",l)
    list_all.append(l)
    
    #print(" ")
    r=i.values.tolist()
    #print(len(r))
    #print(r)
    #print(" ")
    try:
        while r[0][-2]<0:
            #if len(r)>0:
            #if r[0][-2]<0:
                #print(r[0])
                #print(" ")
            #print("length of r_initial: ",len(r))
            r.pop(0)
            #print("length of r_through: ",len(r))
            #print(" ")
                #print(len(r))
                #print(r)
                #print(" ")
        list_ideal.append(r)
        #print(" ")
    except:
        pass       
#print("list_all: ",list_all) 
#print(" ")
#print("list_ideal: ",list_ideal)


# In[666]:


#list_all


# In[667]:


#len(list_all)


# In[668]:


#list_ideal


# In[669]:


#len(list_df)


# In[670]:


#len(list_ideal)


# In[671]:


import math
import cmath
lst_cagr=[]
#lst_cagr_percentage=[]

try:
    for u in list_df:
        #if len(u)>1:
        try:
            for k in range(len(u)-1):
                Initial_RP=u[0][-2]
                #print("Initial_RP :",Initial_RP)
                Final_RP=u[1][-2]
                #print("Final_RP :",Final_RP)
                Initial_year=(u[0][-1])
                #print("Initial_year : ",Initial_year)
                Final_year=u[1][-1]
                #print("Final_year :", Final_year)
                reg_num=(u[0][0])
                #print("reg_num: ",reg_num)
                ind_type=(u[0][2])
                #print("ind_type ",ind_type)
                com_name=(u[0][1])
                #print("com_name: ",com_name)
                CAGR=pow((u[1][-2])/(u[0][-2]),(1/(u[1][-1]-u[0][-1])))-1
                #q = (CAGR.real, CAGR.imag)
                #A = CAGR.real
                #print("CAGR :",CAGR)
                #print(" ")
                u.pop(1)
                lst_cagr.append([reg_num,com_name,ind_type,Final_year,CAGR])
                
        except:
            pass
        #else:
            #pass
                
        #break
        
#except IndexError:
    #pass
       
except ZeroDivisionError:
    pass


# In[672]:


#lst_cagr[0]


# In[673]:


df = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])


# In[674]:


#df


# In[675]:


df['CAGR'] = pd.concat([df['CAGR'].apply(lambda x: x.real), df['CAGR'].apply(lambda x: x.imag)], 
               axis=1, 
               keys=('R','X'))


# In[676]:


#df


# In[677]:


#df.dtypes


# In[678]:


#data.dtypes


# In[679]:


data["YEAR"] = data["YEAR"].astype(str).astype(int)


# In[680]:


root_data_cagr=pd.merge(data,df, on=['REG','NAME','INDUSTRY_TYPE','YEAR'],how ="outer")


# In[681]:


#root_data_cagr.head()


# In[682]:


root_data_cagr['CAGR'] = root_data_cagr['CAGR'].fillna(0)


# In[683]:


cagr_rating=root_data_cagr[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR']]


# In[684]:


#cagr_rating.head()


# In[685]:


df_cagr_col=cagr_rating['CAGR']
df_stats=df_cagr_col.describe()
df1 =df_stats.values.tolist()


# In[686]:


#df1


# In[687]:


cagr_rating_dataframes= [v for k, v in cagr_rating.groupby('NAME')]


# In[688]:


#len(cagr_rating_dataframes)


# In[689]:


#cagr_rating_dataframes[0]


# In[690]:


list_all1=[]
list_ideal1=[]
#try:
for i in cagr_rating_dataframes: 
    #print(i)
    l=i.values.tolist()
    #print("list: ",l)
    list_all1.append(l)
    
    #print(" ")
    r=i.values.tolist()
    #print(len(r))
    #print(r)
    #print(" ")
    try:
        while r[0][-3]<0:
            #if len(r)>0:
            #if r[0][-2]<0:
                #print(r[0])
                #print(" ")
            #print("length of r_initial: ",len(r))
            r.pop(0)
            #print("length of r_through: ",len(r))
            #print(" ")
                #print(len(r))
                #print(r)
                #print(" ")
        list_ideal1.append(r)
        #print(" ")
    except:
        pass       
#print("list_all: ",list_all) 
#print(" ")
#print("list_ideal: ",list_ideal)
    


# In[691]:


#list_all1


# In[692]:


#len(list_all1)


# In[693]:


#list_all1[0][0][-2]


# In[694]:


#try:
    #list_ideal1[0][0][-2]
#except IndexError:
    #pass


# In[695]:


#try:
    #list_ideal1[0][-2]
#except:
    #pass


# In[696]:


#len(list_ideal1)


# In[697]:


#for i in cagr_rating_dataframes:
    #print(i)
conditions2_at = [
                (cagr_rating['CAGR']==0),   
                (cagr_rating['CAGR']>=df1[3])&(cagr_rating['CAGR'] <= df1[4]),
                (cagr_rating['CAGR']>df1[4])&(cagr_rating['CAGR'] <= df1[5]),
                (cagr_rating['CAGR']>df1[5])&(cagr_rating['CAGR'] <= df1[6]),
                (cagr_rating['CAGR']>df1[6])&(cagr_rating['CAGR'] <= df1[7]),   
                ]
values2_at = [0,1, 2, 3, 4]
cagr_rating['Istar_CAGR'] = np.select(conditions2_at, values2_at)


# In[698]:


#cagr_rating.head()


# In[699]:


#cagr_rating['Istar_CAGR'].sum()


# In[700]:


#cagr_rating['CAGR'][:].sum()


# In[701]:


#(cagr_rating['Istar_CAGR'][::].sum())==(cagr_rating['CAGR'][::].sum())


# In[702]:


list_all1=[]
list_ideal1=[]
#try:
for i in cagr_rating_dataframes: 
    #print(i)
    l=i.values.tolist()
    #print("list: ",l)
    list_all1.append(l)
    
    #print(" ")
    r=i.values.tolist()
    #print(len(r))
    #print(r)
    #print(" ")
    try:
        while r[0][-3]<0:
            if len(r)>0:
            #if r[0][-2]<0:
                #print(r[0])
                #print(" ")
            #print("length of r_initial: ",len(r))
                r.pop(0)
            #print("length of r_through: ",len(r))
            #print(" ")
                #print(len(r))
                #print(r)
                #print(" ")
        list_ideal1.append(r)
        #print(" ")
    except:
        pass       
#print("list_all: ",list_all) 
#print(" ")
#print("list_ideal: ",list_ideal)
    if (cagr_rating['Istar_CAGR'][::].sum())==(cagr_rating['CAGR'][::].sum()):
        try:
            conditions3_at = [
                            (cagr_rating['Istar_CAGR'][::].sum())==(cagr_rating['CAGR'][::].sum()),
                            ((cagr_rating['Istar_CAGR']==0)&(cagr_rating['YEAR']==list_all1[0][0][-2])),
                            (cagr_rating['Istar_CAGR']==0),
                            (cagr_rating['Istar_CAGR']==1),
                            (cagr_rating['Istar_CAGR']==2),
                            (cagr_rating['Istar_CAGR']==3),
                            (cagr_rating['Istar_CAGR']==4),
                            ]
            values3_at = ['Caution','Startup','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
            cagr_rating['Istar_CAGR_rating'] = np.select(conditions3_at, values3_at)
        except IndexError:
            pass
    else:
        try:
            conditions3_at = [
                            ((cagr_rating['Istar_CAGR']==0)&(cagr_rating['YEAR']==list_all1[0][0][-2])),
                            (cagr_rating['Istar_CAGR']==0)&(cagr_rating['YEAR']==list_ideal1[0][0][-2]),
                            (cagr_rating['Istar_CAGR']==0),
                            (cagr_rating['Istar_CAGR']==1),
                            (cagr_rating['Istar_CAGR']==2),
                            (cagr_rating['Istar_CAGR']==3),
                            (cagr_rating['Istar_CAGR']==4),
                            ]
            values3_at = ['Startup','Gearing_up','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
            cagr_rating['Istar_CAGR_rating'] = np.select(conditions3_at, values3_at)
        except IndexError:
            pass
    #try:
        #conditions4_at = [
                        #((cagr_rating['Istar_CAGR']==0)&(cagr_rating['YEAR']==list_ideal1[0][0][-2])),
                        #]
        #values4_at = ['Gearing_up']
        #cagr_rating['Istar_CAGR_rating'] = np.select(conditions4_at, values4_at)
         
    #except IndexError:
        #pass


# In[703]:


cagr_rating


# In[704]:


root_cagr=pd.merge(root_data_cagr,cagr_rating, on=['REG','NAME','INDUSTRY_TYPE','YEAR','RETAINED_PROFITS','CAGR'],how ="outer")


# In[705]:


root_cagr


# In[ ]:




