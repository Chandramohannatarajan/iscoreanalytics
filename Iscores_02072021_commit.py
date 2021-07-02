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


# # AC01,AC06,INDUSTRIES MASTER TO BE LOOKED UP AND Z SCORE, CAGR AND EBITDA CALCULATED AND APPENDED TO ALL DOCUMENTS FOR FURTHER PROCESSING 

# In[ ]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[ ]:


#creating database
db=client['dgdata']


# # ENTRY 1  --> To be queried directly from MongoDB

# In[ ]:


#data_from_db = db.z_score_root_file.find({"INDUSTRY_TYPE":"financial and insurance activities"},{'_id':0})
#data=pd.DataFrame.from_dict(data_from_db)


# In[9]:


data = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/root_iscore.csv",low_memory=False)


# In[10]:


data['CAGR'] = data['CAGR'].fillna(0)
data['Z'] = data['Z'].fillna(0)


# In[11]:


data.count()


# In[12]:


data=data.sort_values(by="YEAR",ascending=True)


# In[13]:


#data=data.replace([np.nan], 'misc')
data['INDUSTRY_TYPE'].fillna('misc', inplace=True)


# In[ ]:


#data['SIC07'].fillna('unknown', inplace=True)


# In[14]:


data.drop_duplicates(keep=False,inplace=True)


# In[15]:


df_all_types=data['INDUSTRY_TYPE'].unique().tolist()
df_all_types


# # HOT CODING INDUSTRIES TYPE  --> TO BE PERFORMED IN MONGODB

# In[16]:


conditions0_at = [
    (data['INDUSTRY_TYPE'] == 'manufacture of wearing apparel'),
    (data['INDUSTRY_TYPE'] == 'other manufacturing'),
    (data['INDUSTRY_TYPE'] == 'manufacture of motor vehicles, trailers and semi-trailers'),
    (data['INDUSTRY_TYPE'] == 'repair and installation of machinery and equipment'),
    (data['INDUSTRY_TYPE'] == 'manufacture of machinery and equipment (not elsewhere classified)'),
    (data['INDUSTRY_TYPE'] == 'printing and reproduction of recorded media'),
    (data['INDUSTRY_TYPE'] == 'manufacture of rubber and plastic products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of basic metals'),
    (data['INDUSTRY_TYPE'] == 'manufacture of other transport equipment'),
    (data['INDUSTRY_TYPE'] == 'manufacture of beverages'),
    (data['INDUSTRY_TYPE'] == 'manufacture of food products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of other non-metallic mineral products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of fabricated metal products, except machinery and equipment'),
    (data['INDUSTRY_TYPE'] == 'manufacture of chemicals and chemical products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of paper and paper products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of leather and related products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of electrical equipment'),
    (data['INDUSTRY_TYPE'] == 'manufacture of textiles'),
    (data['INDUSTRY_TYPE'] == 'manufacture of computer, electronic and optical products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of coke and refined petroleum products'),
    (data['INDUSTRY_TYPE'] == 'manufacture of furniture'),
    (data['INDUSTRY_TYPE'] == 'manufacture of basic pharmaceutical products and pharmaceutical preparations'),
    (data['INDUSTRY_TYPE'] =='manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials'),
    (data['INDUSTRY_TYPE'] == 'manufacture of tobacco products'),
    ]

values0_at = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
data['INDUSTRY_CODE'] = np.select(conditions0_at, values0_at)


# # ISCORE CALCULATION

# # MAIN DF FOR ISCORE CALCULATION

# In[17]:


df_all_years=data['YEAR'].unique().tolist()
#df_all_years


# In[18]:


df_all_types=data['INDUSTRY_CODE'].unique().tolist()


# In[23]:


for k,value_year in enumerate(df_all_years):
    for l,value_type in enumerate(df_all_types):
        data_year_ind=data[(data['INDUSTRY_CODE'] == value_type) & (data['YEAR'] == value_year)].to_dict('records')
        year_industry=pd.DataFrame.from_dict(data_year_ind)
        year_industry = year_industry[['REGISTRATION_NUMBER', 'COMPANY_NAME', 'INDUSTRY_TYPE','YEAR','PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
       'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO', 'EQUITY_RATIO',
       'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO', 'EBITDA', 'Z','CAGR']]
        year_ind_stats=year_industry.describe()
        year_ind_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
        year_ind_stats_header=year_ind_stats.columns.tolist()
        del year_ind_stats_header[0]
        for i in year_ind_stats_header:
            conditions1_at = [
                ((year_industry[i]) ==0),
                ((year_industry[i]) >=(year_ind_stats[i][3]) ) & ((year_industry[i]) <= (year_ind_stats[i][4])),
                ((year_industry[i]) > (year_ind_stats[i][4])) & ((year_industry[i]) <=(year_ind_stats[i][5])),
                ((year_industry[i]) > (year_ind_stats[i][5])) & ((year_industry[i]) <=(year_ind_stats[i][6])),
                ((year_industry[i]) > (year_ind_stats[i][6])) & ((year_industry[i]) <=(year_ind_stats[i][7])),
                #((year_industry[i]) > (year_ind_stats[i][7]/2)) & ((year_industry[i]) <=(year_ind_stats[i][7])),
                ]
            values1_at = [0,1,2,3,4]
            n = str(i)+'_'+ 'Iscore'
            year_industry[str(n)] = np.select(conditions1_at, values1_at)
            #year_industry[str(n)] = model.fit_transform(year_industry[str(n)].astype('float'))
        year_industry['Iscore_all']=year_industry.iloc[:,-17:].sum(axis=1)
        df=year_industry.Iscore_all.describe()
        conditions2_at = [
                    (year_industry['Iscore_all']>=df[3])&(year_industry['Iscore_all'] <= df[4]),
                    (year_industry['Iscore_all']>df[4])&(year_industry['Iscore_all'] <= df[5]),
                    (year_industry['Iscore_all']>df[5])&(year_industry['Iscore_all'] <= df[6]),
                    (year_industry['Iscore_all']>df[6])&(year_industry['Iscore_all'] <= df[7]),   
                    ]
        values2_at = [1, 2, 3, 4]
        year_industry['Irating_Iscoreall'] = np.select(conditions2_at, values2_at)
        #year_industry['Iservice'] = model.fit_transform(year_industry['Iservice'].astype('float'))
        df4=year_industry.Irating_Iscoreall.describe()
        conditions3_at = [
                    (year_industry['Irating_Iscoreall']>=df4[3])&(year_industry['Irating_Iscoreall'] <= df4[4]),
                    (year_industry['Irating_Iscoreall']>df4[4])&(year_industry['Irating_Iscoreall'] <= df4[5]),
                    (year_industry['Irating_Iscoreall']>df4[5])&(year_industry['Irating_Iscoreall'] <= df4[6]),
                    (year_industry['Irating_Iscoreall']>df4[6])&(year_industry['Irating_Iscoreall'] <= df4[7]),   
                    ]
        values3_at = ['Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
        year_industry['Irating_category'] = np.select(conditions3_at, values3_at)
        #year_industry['Irating_category'] = model.fit_transform(year_industry['Iservice'].astype('float'))
        #conditions4_at = [
                        #(year_industry['Irating_category']==1),
                        #(year_industry['Irating_category']==2),
                        #(year_industry['Irating_category']==3),
                        #(year_industry['Irating_category']==4),
                        #]
        #values4_at = ['Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
        #year_industry['Irating_category'] = np.select(conditions4_at, values4_at)
        year_industry.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore/'+'iscore_'+str(value_year)+'_'+str(value_type)+r'.csv',index = False, na_rep = 'N/A')
        #year_ind=year_industry.to_dict('records')
        #stage_7_table=db['iscore_z_cagr_'+str(value_year)+'_'+str(value_type)]
        #stage_7_table.insert_many(year_ind)


# In[24]:


import glob
os.chdir(r"C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
iscores = pd.concat([pd.read_csv(f) for f in all_filenames ])
#iscore_z_cagr=iscores_z_cagr.drop(['Iservice_category_Iscore'],axis=1)
#export to csv
iscores.to_csv( "C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore/iscores.csv", index=False)


# MERGING ISCORES FILE WITH ALL YEAR FILE

# In[25]:


x = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/root_iscore.csv",low_memory=False)


# In[26]:


y = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore/iscores.csv")


# In[27]:


z=pd.merge(x,y,how='left',on=['REGISTRATION_NUMBER', 'COMPANY_NAME', 'INDUSTRY_TYPE', 'YEAR'],suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')


# In[28]:


z.drop_duplicates(keep=False,inplace=True)


# In[29]:


z.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/all_iscores.csv",index=False)


# In[ ]:


iscores=z.to_dict('records')
stage_7_table=db['ISCORES']
stage_7_table.insert_many(iscores)


# In[30]:


z.count()


# In[31]:


df_all_years=z['YEAR'].unique().tolist()
df_all_years


# In[32]:


df_all_types=z['INDUSTRY_TYPE'].unique().tolist()
df_all_types


# In[33]:


df_all_codes=data['INDUSTRY_CODE'].unique().tolist()
df_all_codes


# In[35]:


df11 = z.loc[(z['YEAR'] == 2013) & (z['INDUSTRY_TYPE'] == 'manufacture of basic pharmaceutical products and pharmaceutical preparations')]
df11.head()


# In[36]:


list_avg=[]
list_df=[]
for i in df_all_years:
    #print(i)
    for j in df_all_types:
        #print(j)
        df11 = (z.loc[(z['YEAR'] == i) & (z['INDUSTRY_TYPE'] == str(j))])
        avg_T=((df11['TURNOVER'].sum())/(df11['TURNOVER'].count()))
        avg_EBITDA=((df11['EBITDA'].sum())/(df11['EBITDA'].count()))
        avg_iscoreall=((df11['Iscore_all'].sum())/(df11['Iscore_all'].count()))
        avg_GP=((df11['GROSS_PROFIT'].sum())/(df11['GROSS_PROFIT'].count()))

        list_avg.append((i,j,avg_T,avg_EBITDA,avg_iscoreall,avg_GP))
df_plot = DataFrame (list_avg,columns=['YEAR','industry_type','avg_T','avg_EBITDA','avg_iscoreall','avg_GP'])
list_dataframes= [v for k, v in df_plot.groupby('YEAR')]
    #list_df.append(df_plot)
    #print(df_plot)
    #break


# In[37]:


list_dataframes[7]


# In[34]:


z.groupby('INDUSTRY_TYPE')['COMPANY_NAME'].nunique().plot(kind='bar')
plt.show()


# In[38]:


# a simple line plot
list_dataframes[3].plot(kind='bar',x='industry_type',y='avg_EBITDA')


# In[39]:


list_dataframes[3].plot(kind='bar',x='industry_type',y='avg_GP')


# In[40]:


# gca stands for 'get current axis'
ax = plt.gca()

list_dataframes[6].plot(kind='bar',x='industry_type',y='avg_T',color='blue',ax=ax)
list_dataframes[6].plot(kind='bar',x='industry_type',y='avg_EBITDA', color='red', ax=ax)

plt.show()

