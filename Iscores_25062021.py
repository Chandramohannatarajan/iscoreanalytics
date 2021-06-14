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

# In[2]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[3]:


#creating database
db=client['dgdata']


# # ENTRY 1  --> To be queried directly from MongoDB

# In[4]:


data_from_db = db.root_file.find({},{'_id':0})
data=pd.DataFrame.from_dict(data_from_db)


# In[5]:


data.drop_duplicates(keep=False,inplace=True)


# # HOT CODING INDUSTRIES TYPE  --> TO BE PERFORMED IN MONGODB

# In[6]:


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


# # EBITDA AND Z SCORE ---> TO BE PERFORMED IN MONGODB AND APPEND DOCUMENTS

# In[7]:


data["TOTAL_DEBT_RATIO"]=(data["TOTAL_LIAB"]/data["SHAREHOLDER_FUNDS"])


# In[8]:


#data.head()


# In[9]:


data["EBITDA"]=((data["OPERATING_PROFITS"]+data["DEPRECIATION"]))


# In[10]:


#data.head()


# In[13]:


data["TOTAL_ASSETS"] = data["TANGIBLE_ASSETS"]+data["INTANGIBLE_ASSETS"]


# In[14]:


#data.head()


# In[15]:


#data["Z1"]=(1.2*(data["WORKING_CAPITAL"]/data["TOTAL_ASSETS"]))
#data["Z2"]=(1.4*(data["RETAINED_PROFITS"]/data["TOTAL_ASSETS"]))
#data["Z3"]=(3.3*(data["EBITDA"]/data["TOTAL_ASSETS"]))
#data["Z4"]=(.6*(data["SHAREHOLDER_FUNDS"]/data["TOTAL_LIAB"]))
#data["Z5"]=(.99*(data["TURNOVER"]/data["TOTAL_ASSETS"]))
#data["Z"]=(data["Z1"]+data["Z2"]+data["Z3"]+data["Z4"]+data["Z5"])
data["Z"]=((1.2*(data["WORKING_CAPITAL"]/data["TOTAL_ASSETS"]))+
           (1.4*(data["RETAINED_PROFITS"]/data["TOTAL_ASSETS"]))+
           (3.3*(data["EBITDA"]/data["TOTAL_ASSETS"]))+
           (.6*(data["SHAREHOLDER_FUNDS"]/data["TOTAL_LIAB"]))+
           (.99*(data["TURNOVER"]/data["TOTAL_ASSETS"])))


# In[16]:


#data.head()


# In[17]:


data.drop_duplicates(keep=False,inplace=True)


# In[21]:


data=data.replace([np.inf, -np.inf], 0)


# In[23]:


data['Z'] = data['Z'].fillna(0)


# In[24]:


data.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/data_z.csv', index = False)


# # AT THIS STAGE, THE COLLECTION SHOULD HAVE AC01,AC06 COLLATED WITH INDUSTRIES TYPE AND EBITDA,Z SCORE ADDED

# # Z SCORE GROUPING

# In[25]:


Z_mark = data.groupby(['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year'], as_index=False)
average_z = Z_mark.agg({'Z':'mean'})
top_10_companies = average_z.sort_values('Z', ascending=False).head(10)
print("TOP_10_COMPANIES")
print(" ")
print(top_10_companies)


# [1 - 2013, 2 - 2014, 3 - 2015, 4 - 2016, 5 - 2017, 6 - 2018, 7 - 2019, 8 - 2020]

# 'manufacture of wearing apparel' - 1,
#  'other manufacturing' - 2,
#  'manufacture of motor vehicles, trailers and semi-trailers' - 3,
#  'repair and installation of machinery and equipment' - 4,
#  'manufacture of machinery and equipment (not elsewhere classified)' - 5,
#  'printing and reproduction of recorded media' - 6,
#  'manufacture of rubber and plastic products' - 7,
#  'manufacture of basic metals' - 8,
#  'manufacture of other transport equipment' - 9,
#  'manufacture of beverages' - 10,
#  'manufacture of food products' - 11,
#  'manufacture of other non-metallic mineral products' - 12,
#  'manufacture of fabricated metal products, except machinery and equipment' - 13,
#  'manufacture of chemicals and chemical products' -14,
#  'manufacture of paper and paper products' - 15,
#  'manufacture of leather and related products' -16,
#  'manufacture of electrical equipment' - 17,
#  'manufacture of textiles' - 18,
#  'manufacture of computer, electronic and optical products' - 19,
#  'manufacture of coke and refined petroleum products' - 20,
#  'manufacture of furniture' - 21,
#  'manufacture of basic pharmaceutical products and pharmaceutical preparations' - 22,
#  'manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials' - 23,
#  'manufacture of tobacco products - 24'

# # INDUSTRYWISE  --> data should be viewed here from MongoDB  -- All industrywise to be collected in one collection

# # CAGR CALCULATION

# # COLLECTION SHOULD PROVIDE REQUIRED DOCUMENTS FOR CAGR CALCULATION
# Dict to DF

# In[26]:


data_cagr=data[['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','RETAINED_PROFITS','Year']]
#data_cagr.head()


# LIST OF DATAFRAMES

# In[27]:


list_dataframes= [v for k, v in data_cagr.groupby('COMPANY_NAME')]


# In[28]:


list_df=[]
for i in list_dataframes:
    r=i.values.tolist()
    list_df.append(r)
#print(lst_df)    


# In[29]:


lst_cagr=[]
lst_cagr_percentage=[]

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
                lst_cagr_percentage.append([reg_num,com_name,ind_type,Final_year,t])
        except:
            pass
        #else:
            #pass
                
        #break
        
#except IndexError:
    #pass
       
except ZeroDivisionError:
    pass


# In[30]:


df = DataFrame (lst_cagr,columns=['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year','CAGR'])
df_cagr_per=DataFrame (lst_cagr_percentage,columns=['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year','CAGR'])
#df_cagr_per.head()


# In[31]:


#df.head()


# In[32]:


data_z_cagr=pd.merge(data,df, on=['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year'],how ="outer")


# In[46]:


#data_z_cagr.head(5)


# In[37]:


data_z_cagr['CAGR'] = data_z_cagr['CAGR'].fillna(0)


# # MAIN DF FOR ISCORE CALCULATION

# In[38]:


data_z_cagr.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/data_z_cagr.csv', index = False)


# In[39]:


df_cagr_col=data_z_cagr['CAGR']


# In[40]:


df_stats=df_cagr_col.describe()


# In[41]:


df1 =df_stats.values.tolist()


# In[43]:


cagr_2020=data_z_cagr.loc[data_z_cagr['Year'] == 2020]


# In[44]:


CAGR_mark = cagr_2020.groupby(['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year'], as_index=False)
average_cagr = CAGR_mark.agg({'CAGR':'mean'})
top_10_companies = average_cagr.sort_values('CAGR', ascending=False).head(10)
print("TOP_10_COMPANIES")
print(" ")
print(top_10_companies)


# # CATEGORISING CAGR OF COMPANIES INTO GROUPS

# In[85]:


conditions2_at = [
                (data_z_cagr['CAGR']==0),   
                (data_z_cagr['CAGR']>=df1[3])&(data_z_cagr['CAGR'] <= df1[4]),
                (data_z_cagr['CAGR']>df1[4])&(data_z_cagr['CAGR'] <= df1[5]),
                (data_z_cagr['CAGR']>df1[5])&(data_z_cagr['CAGR'] <= df1[6]),
                (data_z_cagr['CAGR']>df1[6])&(data_z_cagr['CAGR'] <= df1[7]),   
                ]
values2_at = [0,1, 2, 3, 4]
data_z_cagr['Istar_CAGR'] = np.select(conditions2_at, values2_at)
#df['Iservice'] = model.fit_transform(df['Iservice'].astype('float'))
conditions3_at = [
                (data_z_cagr['Istar_CAGR']==0),
                (data_z_cagr['Istar_CAGR']==1),
                (data_z_cagr['Istar_CAGR']==2),
                (data_z_cagr['Istar_CAGR']==3),
                (data_z_cagr['Istar_CAGR']==4),
                ]
values3_at = ['Startup','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
data_z_cagr['Istar_CAGR_rating'] = np.select(conditions3_at, values3_at)   


# In[86]:


plt.figure(figsize=(5,3), dpi=100)
plt.style.use('ggplot')
one =data_z_cagr[(data_z_cagr.CAGR>= df1[3]) & (data_z_cagr.CAGR <= df1[4])].count()[0]
two = data_z_cagr[(data_z_cagr.CAGR  > df1[4]) & (data_z_cagr.CAGR <= df1[5])].count()[0]
three =data_z_cagr[(data_z_cagr.CAGR > df1[5]) & (data_z_cagr.CAGR <= df1[6])].count()[0]
four = data_z_cagr[(data_z_cagr.CAGR > df1[6]) & (data_z_cagr.CAGR <= df1[7])].count()[0]
weights = [1,2,3,4]
label = ['Need more analysis','Moderate','Reasonable performance','Better returns']
colors = [ '#EE82EE','#aaebcc','#FFBF00','#006400']
explode = (.1,.1,.1,.1)
plt.title('CAGR')
plt.pie(weights, labels=label, explode=explode,colors=colors, pctdistance=.8, autopct='%.2f %%')
plt.show()


# In[87]:


green =  data_z_cagr['Istar_CAGR_rating']=='Better_returns'
amber = data_z_cagr['Istar_CAGR_rating']=='Reasonable_performance'
red =  data_z_cagr['Istar_CAGR_rating']=='Moderate'
black = data_z_cagr['Istar_CAGR_rating']=='Need_more_analysis'


# In[88]:


green_report=data_z_cagr[green]
print(green_report[['COMPANY_NAME','Year','CAGR','Istar_CAGR_rating']].head(5))


# In[89]:


amber_report=data_z_cagr[amber]
print(amber_report[['COMPANY_NAME','Year','CAGR','Istar_CAGR_rating']].head(5))


# In[90]:


red_report=data_z_cagr[red]
print(red_report[['COMPANY_NAME','Year','CAGR','Istar_CAGR_rating']].head(5))


# In[91]:


black_report=data_z_cagr[black]
print(black_report[['COMPANY_NAME','Year','CAGR','Istar_CAGR_rating']].head(5))


# In[92]:


#data_z_cagr.head()


# # ISCORE CALCULATION

# In[93]:


df_all_years=data_z_cagr['Year'].unique().tolist()
#df_all_years


# In[94]:


df_all_types=data_z_cagr['INDUSTRY_CODE'].unique().tolist()


# In[97]:


for k,value_year in enumerate(df_all_years):
    for l,value_type in enumerate(df_all_types):
        data_year_ind=data_z_cagr[(data_z_cagr['INDUSTRY_CODE'] == value_type) & (data_z_cagr['Year'] == value_year)].to_dict('records')
        year_industry=pd.DataFrame.from_dict(data_year_ind)
        year_industry = year_industry[['REGISTRATION_NUMBER', 'COMPANY_NAME', 'INDUSTRY_TYPE','Year','PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
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
        year_industry.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/iscore/'+'iscore_'+str(value_year)+'_'+str(value_type)+r'.csv',index = False, na_rep = 'N/A')
        #year_ind=year_industry.to_dict('records')
        #stage_7_table=db['iscore_z_cagr_'+str(value_year)+'_'+str(value_type)]
        #stage_7_table.insert_many(year_ind)


# In[98]:


import glob
os.chdir(r"C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/iscore")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
iscores_z_cagr = pd.concat([pd.read_csv(f) for f in all_filenames ])
#iscore_z_cagr=iscores_z_cagr.drop(['Iservice_category_Iscore'],axis=1)
#export to csv
iscores_z_cagr.to_csv( "C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/iscore/iscores_z_cagr.csv", index=False)


# In[61]:


#ind_all=iscores_z_cagr.to_dict('records')
#stage_6_table=db['iscores_z_cagr_'+'all_ind']
#stage_6_table.insert_many(ind_all)


# MERGING ISCORES FILE WITH ALL YEAR FILE

# In[99]:


data_allscores=pd.merge(data_z_cagr,iscores_z_cagr,how='left',on=['REGISTRATION_NUMBER', 'COMPANY_NAME', 'INDUSTRY_TYPE', 'Year'],suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')


# In[100]:


data_allscores.drop_duplicates(keep=False,inplace=True)


# In[102]:


data_allscores.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/25062021/all_scores_25062021.csv",index=False)


# In[103]:


iscores=data_allscores.to_dict('records')
stage_7_table=db['ISCORES']
stage_7_table.insert_many(iscores)

