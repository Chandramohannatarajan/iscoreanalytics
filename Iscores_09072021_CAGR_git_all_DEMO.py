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


# DB CONNECTIVITY

# In[2]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[3]:


#creating database
db=client['dgdata']


# In[4]:


#data_from_db = db.cagr_root_file.find({},{'_id':0})
#data0=pd.DataFrame.from_dict(data_from_db)


# FETCHING DATA

# In[5]:


data_from_db = db.cagr_root_file.find({"NAME":"BROADSOFT LTD"},{'_id':0})
data1=pd.DataFrame.from_dict(data_from_db)


# #SAMPLE COMPANIES
# 
# #BASF PHARMA (CALLANISH) LIMITED
# #BROADSOFT LTD
# #LLOYDS BANK PLC
# #!OBAC LIMITED
# #ABF JAPAN LIMITED
# #!OBAC UK LIMITED
# #ABF THE SOLDIERS' CHARITY
# #JOHN LEWIS PLC
# #HSBC BANK PLC

# In[6]:


data1


# In[7]:


#data_from_db = db.cagr_root_file.find({"INDUSTRY_TYPE":"financial and insurance activities"},{'_id':0})
#data=pd.DataFrame.from_dict(data_from_db)


# In[8]:


#data0.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/enhanced/dummy/cagr_root_file.csv', index = False)


# In[9]:


#data_root


# In[10]:


#data1 = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/enhanced/dummy/cagr_root_file.csv",low_memory=False)


# In[11]:


#is_4 =  data_1['INDUSTRY_TYPE']=='manufacturing'
#data2 = data_1[is_4]
#print(data2.shape)


# In[12]:


#data=data1.drop_duplicates(['RETAINED_PROFITS','YEAR'],keep= 'last')
#print(data.shape)


# In[13]:


data2=data1.sort_values(by="YEAR",ascending=True)


# In[14]:


data2


# In[15]:


print(data2.shape)


# In[16]:


#data3[data3['NAME']=="STAVELEY INDUSTRIES LIMITED"]


# In[17]:


#data3[data3['NAME']=="A.H. KNIGHT ASIA LTD."]


# In[18]:


data = data2.drop_duplicates(subset=['NAME','YEAR'], keep='first')


# In[19]:


data


# In[20]:


print(data.shape)


# In[21]:


#data[data['NAME']=="A.H. KNIGHT ASIA LTD."]


# In[22]:


#try:
    #misc = ['misc', 'misc', 'misc', 'misc','misc','misc','misc']
    #data["INDUSTRY_TYPE"]=misc
#except ValueError:
    #pass


# In[23]:


#data=data.replace([np.nan], 'misc')
try:
    data['INDUSTRY_TYPE'].fillna('misc', inplace=True)
except:
    pass


# In[24]:


try:
    df_all_types=data['INDUSTRY_TYPE'].unique().tolist()
except:
    pass
df_all_types


# In[25]:


#data.count()


# In[26]:


try:
    data['SIC07'].fillna('unknown', inplace=True)
except:
    pass


# In[27]:


#data.count()


# In[28]:


#data.drop_duplicates(keep=False,inplace=True)


# In[29]:


print(data.shape)


# In[30]:


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


# In[31]:


#data[data['NAME']=="STAVELEY INDUSTRIES LIMITED"]


# # CAGR CALCULATION

# In[32]:


#data=data[data['YEAR'] != 2021]


# In[33]:


#data.count()


# In[34]:


data_cagr=data[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']]
print(data_cagr.shape)


# In[35]:


data_cagr.dtypes


# In[36]:


data_cagr


# In[37]:


#data_cagr[data_cagr['NAME']=="STAVELEY INDUSTRIES LIMITED"]


# In[38]:


#data_cagr=data_cagr.sort_values(by="YEAR",ascending=True)


# In[39]:


#data_cagr["YEAR"] = data_cagr["YEAR"].astype(str).astype(int)


# In[40]:


#data_cagr.dtypes


# In[41]:


#data_cagr["RETAINED_PROFITS"] = data_cagr["RETAINED_PROFITS"].astype(float).astype(int)


# In[42]:


#data_cagr


# GROUPING COMPANIES BY NAME IN DATAFRAMES

# In[43]:


list_dataframes= [v for k, v in data_cagr.groupby('NAME')]


# In[44]:


len(list_dataframes)


# In[45]:


list_dataframes[0]


# In[46]:


#list_dataframes[1]


# APPENDING COMPANY RECORDS BASED ON CONDITION [FIRST RETAINED PROFITS RECORD SHOULD BE POSITIVE]

# In[47]:


list_df=[]

for i in list_dataframes: 
    #print(i)
    r=i.values.tolist()
    #print(r[0][1])
    try:
        while r[0][-2]<0:
            if len(r)>0:
                r.pop(0)
        #print(r)
        #print(" ")
        list_df.append(r)   
    except IndexError:
        pass
    
#for i, item in enumerate(list_df, start=0):
    #print(i,item)
    #print(" ")
    #break
#print(list_df)
    #break


# In[48]:


len(list_df)


# In[49]:


list_df


# In[50]:


#list_df[21227]


# In[51]:


#list_df[7]


# # CAGR CALCULATION

# In[52]:


import math
import cmath
lst_cagr=[]
#lst_cagr_percentage=[]

try:
    for u in list_df:
        #print(len(u))
        #if len(u)>1:
        for k in range(len(u)-1):
            #print(k)
            Initial_RP=int(u[0][-2])
            #print("Initial_RP :",Initial_RP)
            #print(" ")
            Final_RP=int(u[1][-2])
            #print("Final_RP :",Final_RP)
            #print(" ")
            Initial_year=int(u[0][-1])
            #print("Initial_year : ",Initial_year)
            #print(" ")
            Final_year=int(u[1][-1])
            #print("Final_year :", Final_year)
            #print(" ")
            reg_num=(u[0][0])
            #print("reg_num: ",reg_num)
            #print(" ")
            ind_type=(u[0][2])
            #print("ind_type ",ind_type)
            #print(" ")
            com_name=(u[0][1])
            #print("company_name: ",com_name)
            #print(" ")
            CAGR=pow(Final_RP/Initial_RP,1/(Final_year-Initial_year))-1
            #CAGR=(Final_RP/Initial_RP)**(1/(Final_year-Initial_year))-1
            #q = (CAGR.real, CAGR.imag)
            #A = CAGR.real
            print("CAGR:",CAGR)
            print(" ")
            #break
            #print(" ")
            u.pop(1)
            lst_cagr.append([reg_num,com_name,ind_type,Final_year,CAGR])
            #break

        
        #else:
            #pass
                
        #break
        
#except IndexError:
    #pass
       
except ZeroDivisionError:
    pass


# In[53]:


len(lst_cagr)


# In[54]:


lst_cagr


# In[55]:


#lst_cagr[0]


# In[56]:


df = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])


# In[57]:


len(df)


# In[58]:


df


# In[59]:


df.dtypes


# In[60]:


df['CAGR'] = pd.concat([df['CAGR'].apply(lambda x: x.real), df['CAGR'].apply(lambda x: x.imag)], 
               axis=1, 
               keys=('R','X'))


# In[61]:


df.dtypes


# In[62]:


df


# In[63]:


data.dtypes


# In[64]:


data["YEAR"] = data["YEAR"].astype(str).astype(int)


# MERGING ROOT FILE WITH CAGR RECORDS

# In[65]:


root_data_cagr=pd.merge(data,df, on=['REG','NAME','INDUSTRY_TYPE','YEAR'],how ="outer")


# In[66]:


root_data_cagr.shape


# In[67]:


root_data_cagr


# In[68]:


root_data_cagr['CAGR'] = root_data_cagr['CAGR'].fillna(0)


# In[69]:


cagr_rating=root_data_cagr[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR']]


# In[70]:


len(cagr_rating)


# In[71]:


#cagr_rating.dtypes


# In[72]:


df_cagr_col=cagr_rating['CAGR']
df_stats=df_cagr_col.describe()
df1 =df_stats.values.tolist()


# STATISTICS SURGERY

# In[73]:


df1


# GROUPING CAGR RECORDS BY COMPANY

# In[74]:


cagr_rating_dataframes= [v for k, v in cagr_rating.groupby('NAME')]


# In[75]:


len(cagr_rating_dataframes)


# In[76]:


cagr_rating_dataframes[0]


# MAPPING SCORES TO RECORDS

# In[77]:


for i in cagr_rating_dataframes:
    #print(i)
    conditions2_at = [
                    (i['CAGR']==0),   
                    (i['CAGR']>=df1[3])&(i['CAGR'] <= df1[4]),
                    (i['CAGR']>df1[4])&(i['CAGR'] <= df1[5]),
                    (i['CAGR']>df1[5])&(i['CAGR'] <= df1[6]),
                    (i['CAGR']>df1[6])&(i['CAGR'] <= df1[7]),   
                    ]
    values2_at = [0,1, 2, 3, 4]
    i['Istar_CAGR'] = np.select(conditions2_at, values2_at)
    #print(i)


# In[78]:


cagr_rating_dataframes[0]


# In[79]:


#cagr_rating_dataframes[0].dtypes


# MAPPING RATINGS TO ALL RECORDS

# In[80]:


#list_all2=[]
#list_ideal2=[]
#try:
for i in cagr_rating_dataframes: 
    list_all2=[]
    list_ideal2=[]
    #print(i)
    l=i.values.tolist()
    list_all2.append(l)
    #print("list_all2: ",list_all2)
    #print("list_all2_year: ",list_all2[0][0][-3])
    #break
    #print(" ")
    r=i.values.tolist()
    #print("length of r: ",len(r))
    #print(" ")
    #break
    
    try:
        while r[0][-4]<0:
            #print("Negative RP: ",r[0][-4])
            #print("length of r: ",len(r))
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
        list_ideal2.append(r)
        #print("ideal2_2: ",list_ideal2)
        #print("list_ideal2_year: ",list_ideal2[0][0][-3])
        #print(" ")
    except:
        pass
    #break
#print("list_all: ",list_all) 
#print(" ")
#print("list_ideal: ",list_ideal)
    if (i['Istar_CAGR'][::].sum())==(i['CAGR'][::].sum()):
        try:
            conditions3_at = [
                            (i['Istar_CAGR'][::].sum())==(i['CAGR'][::].sum()),
                            ((i['Istar_CAGR']==0)&(i['YEAR']==list_all2[0][0][-3])),
                            (i['Istar_CAGR']==0),
                            (i['Istar_CAGR']==1),
                            (i['Istar_CAGR']==2),
                            (i['Istar_CAGR']==3),
                            (i['Istar_CAGR']==4),
                            ]
            #values3_at = ['Caution','Startup','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
            values3_at = ['Under_Lens','Bud','Under_Radar','Under_Observation,','Joining_League','Runner','Dynamic']
            i['Istar_CAGR_rating'] = np.select(conditions3_at, values3_at)
        except IndexError:
            pass
    else:
        try:
            conditions3_at = [
                            ((i['Istar_CAGR']==0)&(i['YEAR']==list_all2[0][0][-3])),
                            (i['Istar_CAGR']==0)&(i['YEAR']==list_ideal2[0][0][-3]),
                            (i['Istar_CAGR']==0),
                            (i['Istar_CAGR']==1),
                            (i['Istar_CAGR']==2),
                            (i['Istar_CAGR']==3),
                            (i['Istar_CAGR']==4),
                            ]
            #values3_at = ['Startup','Gearing_up','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
            values3_at = ['Bud','Gearing_up','Under_Radar','Under_Observation','Joining_League','Runner','Dynamic']

            i['Istar_CAGR_rating'] = np.select(conditions3_at, values3_at)
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


# In[81]:


#cagr_rating_dataframes


# In[82]:


lst_df_final=[]
for i in cagr_rating_dataframes:
    t=i.values.tolist()
    lst_df_final.append(t)
lst_df_all=[]
for i in lst_df_final:
    for j in i:
        lst_df_all.append(j)


# In[83]:


lst_df_all


# In[84]:


#len(lst_df_all)


# In[85]:


df = DataFrame (lst_df_all,columns=['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR','Istar_CAGR','Istar_CAGR_rating'])


# In[86]:


df


# In[87]:


root_data_cagr.shape


# In[88]:


root_cagr=pd.merge(root_data_cagr,df, on=['REG','NAME','INDUSTRY_TYPE','YEAR','RETAINED_PROFITS','CAGR'],how ="outer")


# In[89]:


root_cagr.shape


# In[90]:


#root_cagr[root_cagr['NAME']=="LLOYDS BANK PLC"]


# In[91]:


#iscores=root_cagr.to_dict('records')
#stage_7_table=db['CAGR_09072021']
#stage_7_table.insert_many(iscores)


# In[92]:


#root_cagr.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/enhanced/dummy/root_cagr_1025484.csv', index = False)


# In[93]:


root_cagr


# In[94]:


df_all_type=cagr_rating["INDUSTRY_TYPE"].unique().tolist()
for l,value_type in enumerate(df_all_type):
    ind_type =  cagr_rating['INDUSTRY_TYPE']==str(value_type)
    data = cagr_rating[ind_type]
    k_cagr=data[['INDUSTRY_TYPE','YEAR','CAGR']]
    k_cagr.groupby(by = "YEAR").mean().plot(kind = "line")
    plt.title(str(value_type))


# In[ ]:




