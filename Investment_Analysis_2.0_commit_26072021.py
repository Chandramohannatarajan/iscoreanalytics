#!/usr/bin/env python
# coding: utf-8

# LOADING PYTHON MODULES

# In[ ]:


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


# In[ ]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[ ]:


#creating database
db=client['dgdata']


# In[ ]:


#data_from_db = db.i_score_root_file.find({},{'_id':0})
#data2=pd.DataFrame.from_dict(data_from_db)sky


# In[ ]:


#data_from_db = db.i_score_root_file.find({"INDUSTRY_TYPE":"professional, scientific and technical activities"},{'_id':0})
#data2=pd.DataFrame.from_dict(data_from_db)


# In[ ]:


#data2.columns


# In[ ]:


#data2.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/i_score_root_file1.csv', index = False)


# In[ ]:


data = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/i_score_root_file1.csv",low_memory=False)


# In[ ]:


#data_from_db = db.root_file.find({},{'_id':0})
#data2=pd.DataFrame.from_dict(data_from_db)


# In[ ]:


#data2.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/root_file.csv', index = False)


# In[ ]:


#data1 = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/root_file.csv",low_memory=False)


# In[ ]:


data.shape


# In[ ]:


data=data.sort_values(by="YEAR",ascending=True)


# In[ ]:


data = data.drop_duplicates(subset=['NAME','YEAR'], keep='first')


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


#data['NAME'].tolist()


# In[ ]:


data1 = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/enhanced/dummy/cagr_root_file.csv",low_memory=False)


# In[ ]:


data1.shape


# In[ ]:


data1=data1.sort_values(by="YEAR",ascending=True)


# In[ ]:


data1 = data1.drop_duplicates(subset=['NAME','YEAR'], keep='first')


# In[ ]:


data1.shape


# In[ ]:


data1.columns


# In[ ]:


data=pd.merge(data,data1, on=['REG', 'NAME','INC', 'SIC07', 'DIS', 'INDUSTRY_TYPE', 'LIQUIDATION',
       'ACCOUNT_FROM_DATE', 'ACCOUNT_TO_DATE', 'WEEKS', 'MONTHS', 'WEEK',
       'MONTH', 'RETAINED_PROFITS', 'YEAR'],how ="outer")


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data=data.sort_values(by="YEAR",ascending=True)


# In[ ]:


data = data.drop_duplicates(subset=['NAME','YEAR'], keep='first')


# ROOT FILE

# In[ ]:


data.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/rootfile.csv', index = False)


# In[ ]:


data=pd.read_csv('C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/16072021/rootfile.csv', low_memory=False)


# In[ ]:


data.shape


# In[ ]:


#data.dtypes


# In[ ]:


data['TURNOVER'].fillna(0)


# In[ ]:


try:
    conditions0_at = [
    (data['TURNOVER']==0.0),
    ((data['TURNOVER']>0.0) & (data['TURNOVER']<10000000.0)),
    ((data['TURNOVER']>=10000000.0) & (data['TURNOVER']<100000000.0)),
    ((data['TURNOVER']>=100000000.0) & (data['TURNOVER']<1000000000.0)),
    ((data['TURNOVER']>=1000000000.0) &(data['TURNOVER']<10000000000.0)),
    (data['TURNOVER']>=10000000000.0)
    ]

    values0_at = [0,1,2,3,4,5]
    data['TURNOVER_BAND'] = np.select(conditions0_at, values0_at)
except TypeError:
    pass


# In[ ]:


#data.dtypes


# In[ ]:


#is_0 =  data['TURNOVER_BAND']==0
#data_0 = data[is_0]
#data_0.shape


# In[ ]:


df_all_bands=data["TURNOVER_BAND"].unique().tolist()
df_all_bands


# In[ ]:


data.shape


# In[ ]:


data['TURNOVER'].describe().values.tolist()


# In[ ]:


#data = data.loc[data['TURNOVER'] > 10000000000]


# In[ ]:


#data['NAME'].values.tolist()


# In[ ]:


#data.sort_values(['REG', 'TURNOVER'], ascending=False).groupby('REG').head(10)


# In[ ]:


data[data['NAME']=="JOHN LEWIS PLC"]


# In[ ]:


#data=data.replace([np.nan], 'misc')
try:
    data['INDUSTRY_TYPE'].fillna('misc', inplace=True)
except:
    pass


# In[ ]:


try:
    data['SIC07'].fillna('unknown', inplace=True)
except:
    pass


# In[ ]:


conditions9_at = [
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

values9_at = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']
data['INDUSTRY_CODE'] = np.select(conditions9_at, values9_at)


# In[ ]:


#list_companies= [v for k, v in data.groupby('NAME')]


# In[ ]:


#len(list_companies)


# In[ ]:


#data["TOTAL_DEBT_RATIO"]=(data["TOTAL_LIAB"]/data["SHAREHOLDER_FUNDS"])


# In[ ]:


df_all_type=cagr_rating["INDUSTRY_TYPE"].unique().tolist()
df_all_band=cagr_rating["TURNOVER_BAND"].unique().tolist()
df_all_year=cagr_rating["YEAR"].unique().tolist()
for l,value_type in enumerate(df_all_type):
    for m,value_year in enumerate(df_all_year):
        for n,value_band in enumerate(df_all_band):
            ind_type = ((data['INDUSTRY_TYPE']==str(value_type)))
            cagr_indwise = cagr_rating[ind_type]


# # CAGR calculation

# In[ ]:


data.shape


# In[ ]:


data_cagr=data[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']]
#print(data_cagr.shape)


# In[ ]:


data_cagr.shape


# In[ ]:


data_cagr.head()


# In[ ]:


list_dataframes= [v for k, v in data_cagr.groupby('NAME')]


# In[ ]:


len(list_dataframes)


# In[ ]:


list_dataframes[0]


# In[ ]:


list_dataframes_df = pd.concat(list_dataframes)


# In[ ]:


list_dataframes_df.head(10)


# In[ ]:


list_dataframes_df.shape


# In[ ]:


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


# In[ ]:


len(list_df)


# In[ ]:


list_df[0]


# In[ ]:


cagr_qual = []
for sublist in list_df:
    for item in sublist:
        cagr_qual.append(item)


# In[ ]:


cagr_qual[0]


# In[ ]:


cagr_qual_df = DataFrame(cagr_qual,columns=(['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']))


# In[ ]:


cagr_qual_df.head()


# In[ ]:


cagr_qual_df.shape


# FOR CAGR

# In[ ]:


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
            CAGR=pow((u[1][-2])/(u[0][-2]),(1/(u[1][-1]-u[0][-1])))-1
            #CAGR=(Final_RP/Initial_RP)**(1/(Final_year-Initial_year))-1
            #q = (CAGR.real, CAGR.imag)
            #A = CAGR.real
            #print("CAGR :",CAGR)
            #print(" ")
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


# In[ ]:


len(lst_cagr)


# In[ ]:


df = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])


# In[ ]:


df['CAGR'] = pd.concat([df['CAGR'].apply(lambda x: x.real), df['CAGR'].apply(lambda x: x.imag)], 
               axis=1, 
               keys=('R','X'))


# In[ ]:


df.shape


# In[ ]:


list_dataframes_df.head()


# In[ ]:


cagr_qual_df.head()


# In[ ]:


df.head()


# In[ ]:


root_data_cagr=pd.merge(data,df, on=['REG','NAME','INDUSTRY_TYPE','YEAR'],how ="outer")


# In[ ]:


root_data_cagr.shape


# In[ ]:


#root_data_cagr.columns


# In[ ]:


root_data_cagr['CAGR'] = root_data_cagr['CAGR'].fillna(0)


# In[ ]:


root_data_cagr = root_data_cagr.drop_duplicates(subset=['NAME','YEAR'], keep='first')


# In[ ]:


root_data_cagr.shape


# In[ ]:


root_data_cagr.head()


# In[ ]:


cagr_rating=root_data_cagr[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','TURNOVER_BAND','YEAR','CAGR']]


# In[ ]:


cagr_rating.head()


# In[ ]:


cagr_rating_df= [v for k, v in cagr_rating.groupby('NAME')]


# In[ ]:


len(cagr_rating_df)


# In[ ]:


cagr_rating_df[0]


# In[ ]:


cagr_qual_df.head()


# In[ ]:


cagr_rating.shape


# In[ ]:


df_all_type=cagr_rating["INDUSTRY_TYPE"].unique().tolist()
df_all_type


# In[ ]:


df_all_band=cagr_rating["TURNOVER_BAND"].unique().tolist()
df_all_band


# In[ ]:


df_all_year=cagr_rating["YEAR"].unique().tolist()
df_all_year


# In[ ]:


#ind_count=[]
df8 = pd.DataFrame(columns=['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','CAGR','Istar_CAGR','Istar_CAGR_rating'])    
for l,value_type in enumerate(df_all_type):
    for m,value_year in enumerate(df_all_year):
        for n,value_band in enumerate(df_all_band):
            ind_type=cagr_rating[(cagr_rating['INDUSTRY_TYPE'] == value_type)& (cagr_rating['YEAR'] == value_year)&(cagr_rating['TURNOVER_BAND']==value_band)]
            cagr_rating_ind=ind_type[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR']]
            print("Records in df: ",value_type,value_year,value_band,len(cagr_rating_ind.values.tolist()))
            #ind_type = ((cagr_rating['INDUSTRY_TYPE']==str(value_type)))
            #cagr_indwise = cagr_rating[ind_type]
            #cagr_rating_ind=cagr_indwise[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR']]
            #print("Records in df: ",len(cagr_rating_ind.values.tolist()))
            #ind_count.append([str(value_type),cagr_rating_ind.shape])
            df2_cagr_col=cagr_rating_ind['CAGR']
            df2_stats=df2_cagr_col.describe()
            df2 =df2_stats.values.tolist()
            cagr_rating_ind_dataframes= [v for k, v in cagr_rating_ind.groupby('NAME')]
            for i in cagr_rating_ind_dataframes:
                e=i['NAME'].values.tolist()
                #print("Name of the company: ",e[0])
                #print(" ")
                y=list_dataframes_df[list_dataframes_df['NAME']==str(e[0])]
                f=y['YEAR'].values.tolist()
                try:
                    q=cagr_qual_df[cagr_qual_df['NAME']==str(e[0])]
                    #print(q)
                    #print(" ")
                    s=q['YEAR'].values.tolist()
                    #print(s)
                    #print(" ")
                    #print("Qual year: ",s[0])
                    #print(" ")
                except IndexError:
                    pass
                #print("Year of the company: ",f[0])
                #print(" ")
                conditions2_at = [
                                (i['CAGR']==0),   
                                (i['CAGR']>=df2[3])&(i['CAGR'] <= df2[4]),
                                (i['CAGR']>df2[4])&(i['CAGR'] <= df2[5]),
                                (i['CAGR']>df2[5])&(i['CAGR'] <= df2[6]),
                                (i['CAGR']>df2[6])&(i['CAGR'] <= df2[7]),   
                            ]
                values2_at = [0,1, 2, 3, 4]
                i['Istar_CAGR'] = np.select(conditions2_at, values2_at)
            for i in cagr_rating_ind_dataframes: 
                list_all3=[]
                list_ideal3=[]
                list_1=[]
                #print(i)
                l=i.values.tolist()
                list_all3.append(l)
                #print("list_all2: ",list_all2)
                #print("list_all2_year: ",list_all2[0][0][-3])
                #break
                #print(" ")
                r=i.values.tolist()
                j=len(r)
                #print("length of r: ",len(r))
                #print(" ")
                #break

                try:
                    if j==1:
                        list_1.append(r)
                        #print("list_less3: ",list_less3)
                        #break
                    else:
                        while r[0][-4]<0:
                            #print("Negative RP: ",r[0][-4])
                            #print("length of r: ",len(r))
                            try:
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
                            except AssertionError:
                                pass
                        list_ideal3.append(r)
                    #print("ideal2_2: ",list_ideal2)
                    #print("list_ideal2_year: ",list_ideal2[0][0][-3])
                    #print(" ")
                except IndexError:
                    pass
                #break
            #print("list_all: ",list_all) 
            #print(" ")
            #print("list_ideal: ",list_ideal)
                try:
                    if (i['Istar_CAGR'][::].sum())==(i['CAGR'][::].sum()):
                        try:
                            conditions5_at = [
                                            ((i['Istar_CAGR']==0)&(i['YEAR']==f[0])),
                                            (i['Istar_CAGR']==0),
                                            (i['Istar_CAGR']==1),
                                            (i['Istar_CAGR']==2),
                                            (i['Istar_CAGR']==3),
                                            (i['Istar_CAGR']==4),
                                            ]
                            #values3_at = ['Caution','Startup','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
                            values5_at = ['Bud.','Under_Radar.','Under_Observation.,','Joining_League.','Runner.','Dynamic.']
                            i['Istar_CAGR_rating'] = np.select(conditions5_at, values5_at)
                        except IndexError:
                            pass
                    #elif(len(list_ideal3)==0):
                        #try:
                            #conditions7_at = [
                                            #(i['Istar_CAGR']==0),
                                            #(i['Istar_CAGR']==1),
                                            #(i['Istar_CAGR']==2),
                                            #(i['Istar_CAGR']==3),
                                            #(i['Istar_CAGR']==4),
                                            #]
                #values3_at = ['Startup','Gearing_up','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
                            #values7_at = ['Under_Radar','Under_Observation','Joining_League','Runner','Dynamic']

                            #i['Istar_CAGR_rating'] = np.select(conditions7_at, values7_at)

                        #except IndexError:
                            #pass
                    else:
                        try:
                            conditions6_at = [
                                            ((i['Istar_CAGR']==0)&(i['YEAR']==f[0])),
                                            (i['Istar_CAGR']==0)&(i['YEAR']==s[0]),
                                            (i['Istar_CAGR']==0),
                                            (i['Istar_CAGR']==1),
                                            (i['Istar_CAGR']==2),
                                            (i['Istar_CAGR']==3),
                                            (i['Istar_CAGR']==4),
                                            ]
                            #values3_at = ['Startup','Gearing_up','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
                            values6_at = ['Bud..','Gearing_up..','Under_Radar..','Under_Observation..','Joining_League..','Runner..','Dynamic..']

                            i['Istar_CAGR_rating'] = np.select(conditions6_at, values6_at)


                        except IndexError:
                            pass
                except AssertionError:
                    pass
                            #try:

        lst_df2_final=[]
        for i in cagr_rating_ind_dataframes:
            t=i.values.tolist()
            lst_df2_final.append(t)
        lst_df2_all=[]
        for i in lst_df2_final:
            for j in i:
                lst_df2_all.append(j)
        try:
            df_ind = DataFrame (lst_df2_all,columns=['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR','Istar_CAGR','Istar_CAGR_rating'])
        except ValueError:
            pass
        df8 = pd.merge(df8, df_ind, how='outer')
        print(str(value_type)+" "+str(value_year)+" "+str(value_band)+" "+"is completed")
root_ind_cagr=pd.merge(root_data_cagr,df8, on=['REG','NAME','INDUSTRY_TYPE','YEAR','RETAINED_PROFITS','CAGR'],how ="outer")
print("Merging Completed")


# In[ ]:


#iscores=inv_analy.to_dict('records')
#stage_7_table=db['Investment_Analysis_CAGR_26072021']
#stage_7_table.insert_many(iscores)


# In[ ]:




