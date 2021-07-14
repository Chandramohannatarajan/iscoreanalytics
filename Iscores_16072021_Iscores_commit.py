#!/usr/bin/env python
# coding: utf-8

# LOADING PYTHON MODULES

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


# In[2]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[3]:


#creating database
db=client['dgdata']


# In[4]:


#data_from_db = db.i_score_root_file.find({},{'_id':0})
#data2=pd.DataFrame.from_dict(data_from_db)


# In[5]:


#data2.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/i_score_root_file.csv', index = False)


# In[6]:


data1 = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/i_score_root_file.csv",low_memory=False)


# In[7]:


data1.shape


# In[8]:


data0=data1.sort_values(by="YEAR",ascending=True)


# In[9]:


data = data0.drop_duplicates(subset=['NAME','YEAR'], keep='first')


# In[10]:


data.shape


# In[11]:


data['NAME'].tolist()


# In[12]:


#data=data.replace([np.nan], 'misc')
try:
    data['INDUSTRY_TYPE'].fillna('misc', inplace=True)
except:
    pass


# In[13]:


try:
    data['SIC07'].fillna('unknown', inplace=True)
except:
    pass


# In[14]:


conditions0_at = [
    (data['INDUSTRY_TYPE'] == 'professional, scientific and technical activities'),
    (data['INDUSTRY_TYPE'] == 'transportation and storage'),
    (data['INDUSTRY_TYPE'] == 'financial and insurance activities'),
    (data['INDUSTRY_TYPE'] == 'manufacturing'),
    (data['INDUSTRY_TYPE'] == 'other service activities'),
    (data['INDUSTRY_TYPE'] == 'real estate activities'),
    (data['INDUSTRY_TYPE'] == 'wholesale and retail trade; repair of motor vehicles and motorcycles'),
    (data['INDUSTRY_TYPE'] == 'administrative and support service activities'),
    (data['INDUSTRY_TYPE'] == 'education'),
    (data['INDUSTRY_TYPE'] == 'mining and quarrying'),
    (data['INDUSTRY_TYPE'] == 'arts, entertainment and recreation'),
    (data['INDUSTRY_TYPE'] == 'agriculture forestry and fishing'),
    (data['INDUSTRY_TYPE'] == 'information and communication'),
    (data['INDUSTRY_TYPE'] == 'construction'),
    (data['INDUSTRY_TYPE'] == 'human health and social work activities'),
    (data['INDUSTRY_TYPE'] == 'accommodation and food service activities'),
    (data['INDUSTRY_TYPE'] == 'activities of extraterritorial organisations and bodies'),
    (data['INDUSTRY_TYPE'] == 'water supply, sewerage, waste management and remediation activities'),
    (data['INDUSTRY_TYPE'] == 'public administration and defence; compulsory social security'),
    (data['INDUSTRY_TYPE'] == 'electricity, gas, steam and air conditioning supply'),
    (data['INDUSTRY_TYPE'] == 'activities of households as employers; undifferentiated goods- and services-producing activities of households for own use'),
    (data['INDUSTRY_TYPE'] == 'misc'),
    ]

values0_at = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']
data['INDUSTRY_CODE'] = np.select(conditions0_at, values0_at)


# In[15]:


#data.columns


# In[16]:


data_iscore=data[['REG','NAME','INDUSTRY_TYPE','YEAR','PRETAX_PROFIT_PERCENTAGE',
       'CURRENT_RATIO', 'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO',
       'EQUITY_RATIO', 'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO']]


# In[17]:


df_all_types=data_iscore['INDUSTRY_TYPE'].unique().tolist()
df_all_types


# In[18]:


df_all_years=data_iscore['YEAR'].unique().tolist()
df_all_years


# In[20]:


data['NAME'].tolist()


# # INDUSTRYWISE & YEARWISE

# In[22]:


df1 = pd.DataFrame(columns=['REG', 'NAME', 'INDUSTRY_TYPE','YEAR', 'PRETAX_PROFIT_PERCENTAGE',
       'CURRENT_RATIO', 'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO',
       'EQUITY_RATIO', 'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO',
       'PRETAX_PROFIT_PERCENTAGE_Iscore', 'CURRENT_RATIO_Iscore',
       'SALES_PER_NET_WORKING_CAPITAL_Iscore', 'GEARING_RATIO_Iscore',
       'EQUITY_RATIO_Iscore', 'CREDITOR_DAYS_Iscore', 'DEBTOR_DAYS_Iscore',
       'LIQUIDITY_TEST_Iscore', 'RETURN_CAPITAL_EMPLOYED_Iscore',
       'RETURN_TOTAL_ASSETS_Iscore', 'DEBT_EQUITY_Iscore',
       'RETURN_EQUITY_Iscore', 'RETURN_NET_ASSETS_Iscore',
       'TOTAL_DEBT_RATIO_Iscore', 'IScore_ALL', 'Irating_Iscoreall',
       'Irating_category'])  
for k,value_year in enumerate(df_all_years):
    for l,value_type in enumerate(df_all_types):
        print(value_year,value_type)
        data_ind1=data_iscore[(data_iscore['INDUSTRY_TYPE'] == value_type)& (data_iscore['YEAR'] == value_year)].to_dict('records')
        ind1=pd.DataFrame.from_dict(data_ind1)
        print(ind1.shape)
        ind_type1 = ind1[['REG', 'NAME', 'INDUSTRY_TYPE','YEAR','PRETAX_PROFIT_PERCENTAGE',
       'CURRENT_RATIO', 'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO',
       'EQUITY_RATIO', 'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO']]
        ind_stats1=ind_type1.describe()
        ind_stats1.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
        ind_stats_header1=ind_stats1.columns.tolist()
        #print(ind_stats_header)
        for i in ind_stats_header1:
            conditions1_at = [
                    ((ind_type1[i]) ==0),
                    ((ind_type1[i]) >=(ind_stats1[i][3]) ) & ((ind_type1[i]) <= (ind_stats1[i][4])),
                    ((ind_type1[i]) > (ind_stats1[i][4])) & ((ind_type1[i]) <=(ind_stats1[i][5])),
                    ((ind_type1[i]) > (ind_stats1[i][5])) & ((ind_type1[i]) <=(ind_stats1[i][6])),
                    ((ind_type1[i]) > (ind_stats1[i][6])) & ((ind_type1[i]) <=(ind_stats1[i][7])),
                    #((year_industry[i]) > (year_ind_stats[i][7]/2)) & ((year_industry[i]) <=(year_ind_stats[i][7])),
                    ]
            values1_at = [0,1,2,3,4]
            n = str(i)+'_'+ 'Iscore'
            ind_type1[str(n)] = np.select(conditions1_at, values1_at)
            #year_industry[str(n)] = model.fit_transform(year_industry[str(n)].astype('float'))
            ind_type1['IScore_ALL']=ind_type1.iloc[:,-14:].sum(axis=1)
            df2=ind_type1.IScore_ALL.describe()
            conditions2_at = [
                        (ind_type1['IScore_ALL']>=df2[3])&(ind_type1['IScore_ALL'] <= df2[4]),
                        (ind_type1['IScore_ALL']>df2[4])&(ind_type1['IScore_ALL'] <= df2[5]),
                        (ind_type1['IScore_ALL']>df2[5])&(ind_type1['IScore_ALL'] <= df2[6]),
                        (ind_type1['IScore_ALL']>df2[6])&(ind_type1['IScore_ALL'] <= df2[7]),   
                        ]
            values2_at = [1, 2, 3, 4]
            ind_type1['Irating_Iscoreall'] = np.select(conditions2_at, values2_at)
            #year_industry['Iservice'] = model.fit_transform(year_industry['Iservice'].astype('float'))
            df4=ind_type1.Irating_Iscoreall.describe()
            conditions3_at = [
                        (ind_type1['Irating_Iscoreall']>=df4[3])&(ind_type1['Irating_Iscoreall'] <= df4[4]),
                        (ind_type1['Irating_Iscoreall']>df4[4])&(ind_type1['Irating_Iscoreall'] <= df4[5]),
                        (ind_type1['Irating_Iscoreall']>df4[5])&(ind_type1['Irating_Iscoreall'] <= df4[6]),
                        (ind_type1['Irating_Iscoreall']>df4[6])&(ind_type1['Irating_Iscoreall'] <= df4[7]),   
                        ]
            values3_at = ['Under_Observation','Joining_League','Runner','Dynamic']
            ind_type1['Irating_category'] = np.select(conditions3_at, values3_at)
        df1 = pd.merge(df1, ind_type1, how='outer')
df2=df1[['REG', 'NAME', 'INDUSTRY_TYPE','YEAR','IScore_ALL', 'Irating_Iscoreall',
       'Irating_category']]
root_ind_iscore1=pd.merge(data_iscore,df2, on=['REG','NAME','INDUSTRY_TYPE','YEAR'],how ="outer")


# In[32]:


root_ind_iscore1[root_ind_iscore1['NAME']=="SIEMENS MOBILITY LIMITED"]


# In[24]:


#iscores=z.to_dict('records')
#stage_7_table=db['ISCORES']
#stage_7_table.insert_many(iscores)

