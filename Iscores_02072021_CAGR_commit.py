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


# # AC01,AC06,INDUSTRIES MASTER TO BE LOOKED UP FOR CAGR TO BE CALCULATED AND APPENDED TO ALL DOCUMENTS FOR FURTHER PROCESSING 

# In[43]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[44]:


#creating database
db=client['dgdata']


# # ENTRY 1  --> To be queried directly from MongoDB

# In[ ]:


data_from_db = db.cagr_root_file.find({},{'_id':0})
data=pd.DataFrame.from_dict(data_from_db)


# In[ ]:


#data.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/enhanced/dummy/cagr_root_file.csv', index = False)


# In[2]:


#data = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/enhanced/dummy/cagr_root_file.csv",low_memory=False)


# In[3]:


#data=data.replace([np.nan], 'misc')
data['INDUSTRY_TYPE'].fillna('misc', inplace=True)


# In[4]:


df_all_types=data['INDUSTRY_TYPE'].unique().tolist()
df_all_types


# In[5]:


#data.count()


# In[6]:


data['SIC07'].fillna('unknown', inplace=True)


# In[7]:


#data.count()


# In[8]:


data.drop_duplicates(keep=False,inplace=True)


# In[9]:


#data.count()


# # HOT CODING INDUSTRIES TYPE  --> TO BE PERFORMED IN MONGODB

# In[10]:


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


# # INDUSTRYWISE  --> data should be viewed here from MongoDB  -- All industrywise to be collected in one collection

# # CAGR CALCULATION

# # COLLECTION SHOULD PROVIDE REQUIRED DOCUMENTS FOR CAGR CALCULATION
# Dict to DF

# In[11]:


data=data[data['YEAR'] != 2021]


# In[12]:


#data.count()


# In[13]:


data_cagr=data[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']]
data_cagr.head()


# In[14]:


data_cagr=data_cagr.sort_values(by="YEAR",ascending=True)


# In[15]:


#data_cagr.head(10)


# LIST OF DATAFRAMES

# In[16]:


list_dataframes= [v for k, v in data_cagr.groupby('NAME')]


# In[17]:


#len(list_dataframes)


# In[18]:


list_df=[]
for i in list_dataframes:
    r=i.values.tolist()
    list_df.append(r)
#print(lst_df)    


# In[19]:


#list_df[0]


# In[20]:


lst_cagr=[]
#lst_cagr_percentage=[]

try:
    for u in list_df:
        #if len(u)>1:
        try:
            for k in range(len(u)-1):
                #print("k: ",k)
                #print(len(u))
                #print(u)
                #break
                Initial_year=(u[0][-1])
                Final_year=u[1][-1]
                #print("Final_year :", Final_year)
                #print("Initial_year : ",Initial_year)
                #break
                reg_num=(u[0][0])
                #print(reg_num)
                com_name=(u[0][1])
                #print(com_name)
                ind_type=(u[0][2])
                #print(ind_type)
                expo =(1/(Final_year-Initial_year))
                #print("expo: ",expo)
                Final_RP=u[1][-2]
                #print("Final_RP :",Final_RP)
                Initial_RP=u[0][-2]
                #print("Initial_RP :",Initial_RP)
                #if Initial_RP==0:
                    #Initial_RP=1
                #else:
                part=(Final_RP/Initial_RP)
                #print("part : ",part)
                cagr = (part)**(expo) -1
                q = (cagr.real, cagr.imag)
                A = cagr.real
                B = cagr.imag
                t = ("{:.0%}".format(A))
                #print("cagr : ",A)
                #print("cagr in %: ",t)
                #print(" ")
                #break
                u.pop(1)
                lst_cagr.append([reg_num,com_name,ind_type,Final_year,A])
                #print(lst_cagr)
                #print(len(lst_cagr))
                #lst_cagr_percentage.append([reg_num,com_name,ind_type,Final_year,t])
        except:
            pass
        #else:
            #pass
                
        #break
        
#except IndexError:
    #pass
       
except ZeroDivisionError:
    pass


# In[22]:


#lst_cagr[0]


# In[ ]:


#data_plot = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])


# In[ ]:


#data_plot.plot(x ='YEAR', y='CAGR', kind = 'line')


# In[23]:


df = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])
#df_cagr_per=DataFrame (lst_cagr_percentage,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])
#df_cagr_per.head()


# In[24]:


#df.head()


# In[ ]:


#df.head()


# In[26]:


root_data_cagr=pd.merge(data,df, on=['REG','NAME','INDUSTRY_TYPE','YEAR'],how ="outer")


# In[27]:


#root_data_cagr.head()


# In[28]:


root_data_cagr['CAGR'] = root_data_cagr['CAGR'].fillna(0)


# In[29]:


df_cagr_col=root_data_cagr['CAGR']


# In[30]:


df_stats=df_cagr_col.describe()


# In[31]:


df1 =df_stats.values.tolist()


# In[33]:


cagr_2020=root_data_cagr.loc[root_data_cagr['YEAR'] == 2020]


# In[34]:


CAGR_mark = cagr_2020.groupby(['REG','NAME','INDUSTRY_TYPE','YEAR'], as_index=False)
average_cagr = CAGR_mark.agg({'CAGR':'mean'})
top_10_companies = average_cagr.sort_values('CAGR', ascending=False).head(10)
print("TOP_10_COMPANIES")
print(" ")
print(top_10_companies)


# # CATEGORISING CAGR OF COMPANIES INTO GROUPS

# In[35]:


conditions2_at = [
                (root_data_cagr['CAGR']==0),   
                (root_data_cagr['CAGR']>=df1[3])&(root_data_cagr['CAGR'] <= df1[4]),
                (root_data_cagr['CAGR']>df1[4])&(root_data_cagr['CAGR'] <= df1[5]),
                (root_data_cagr['CAGR']>df1[5])&(root_data_cagr['CAGR'] <= df1[6]),
                (root_data_cagr['CAGR']>df1[6])&(root_data_cagr['CAGR'] <= df1[7]),   
                ]
values2_at = [0,1, 2, 3, 4]
root_data_cagr['Istar_CAGR'] = np.select(conditions2_at, values2_at)
#df['Iservice'] = model.fit_transform(df['Iservice'].astype('float'))
conditions3_at = [
                (root_data_cagr['Istar_CAGR']==0),
                (root_data_cagr['Istar_CAGR']==1),
                (root_data_cagr['Istar_CAGR']==2),
                (root_data_cagr['Istar_CAGR']==3),
                (root_data_cagr['Istar_CAGR']==4),
                ]
values3_at = ['Startup','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
root_data_cagr['Istar_CAGR_rating'] = np.select(conditions3_at, values3_at)   


# In[36]:


plt.figure(figsize=(5,3), dpi=100)
plt.style.use('ggplot')
one =root_data_cagr[(root_data_cagr.CAGR>= df1[3]) & (root_data_cagr.CAGR <= df1[4])].count()[0]
two = root_data_cagr[(root_data_cagr.CAGR  > df1[4]) & (root_data_cagr.CAGR <= df1[5])].count()[0]
three =root_data_cagr[(root_data_cagr.CAGR > df1[5]) & (root_data_cagr.CAGR <= df1[6])].count()[0]
four = root_data_cagr[(root_data_cagr.CAGR > df1[6]) & (root_data_cagr.CAGR <= df1[7])].count()[0]
weights = [1,2,3,4]
label = ['Need more analysis','Moderate','Reasonable performance','Better returns']
colors = [ '#EE82EE','#aaebcc','#FFBF00','#006400']
explode = (.1,.1,.1,.1)
plt.title('CAGR')
plt.pie(weights, labels=label, explode=explode,colors=colors, pctdistance=.8, autopct='%.2f %%')
plt.show()


# In[37]:


green =  root_data_cagr['Istar_CAGR_rating']=='Better_returns'
amber = root_data_cagr['Istar_CAGR_rating']=='Reasonable_performance'
red =  root_data_cagr['Istar_CAGR_rating']=='Moderate'
black = root_data_cagr['Istar_CAGR_rating']=='Need_more_analysis'


# In[38]:


green_report=root_data_cagr[green]
print(green_report[['NAME','YEAR','CAGR','Istar_CAGR_rating']].head(5))


# In[39]:


amber_report=root_data_cagr[amber]
print(amber_report[['NAME','YEAR','CAGR','Istar_CAGR_rating']].head(5))


# In[40]:


red_report=root_data_cagr[red]
print(red_report[['NAME','YEAR','CAGR','Istar_CAGR_rating']].head(5))


# In[41]:


black_report=root_data_cagr[black]
print(black_report[['NAME','YEAR','CAGR','Istar_CAGR_rating']].head(5))


# In[42]:


root_data_cagr.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/enhanced/dummy/root_data_cagr.csv",index=False)


# In[46]:


iscores=root_data_cagr.to_dict('records')
stage_7_table=db['CAGR_28062021']
stage_7_table.insert_many(iscores)

