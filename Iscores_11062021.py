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


# # HOT CODING INDUSTRIES TYPE  --> TO BE PERFORMED IN MONGODB

# In[5]:


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

# In[6]:


data["TOTAL_DEBT_RATIO"]=(data["TOTAL_LIAB"]/data["SHAREHOLDER_FUNDS"])
data["EBITDA"]=((data["OPERATING_PROFITS"]+data["DEPRECIATION"]))

data["TOTAL_ASSETS"] = data["TANGIBLE_ASSETS"]+data["INTANGIBLE_ASSETS"]
data["Z1"]=(1.2*(data["WORKING_CAPITAL"]/data["TOTAL_ASSETS"]))
data["Z2"]=(1.4*(data["RETAINED_PROFITS"]/data["TOTAL_ASSETS"]))
data["Z3"]=(3.3*(data["EBITDA"]/data["TOTAL_ASSETS"]))
data["Z4"]=(.6*(data["SHAREHOLDER_FUNDS"]/data["TOTAL_LIAB"]))
data["Z5"]=(.99*(data["TURNOVER"]/data["TOTAL_ASSETS"]))
data["Z"]=(data["Z1"]+data["Z2"]+data["Z3"]+data["Z4"]+data["Z5"])


# # AT THIS STAGE, THE COLLECTION SHOULD HAVE AC01,AC06 COLLATED WITH INDUSTRIES TYPE AND EBITDA,Z SCORE ADDED

# In[7]:


data = data[['REGISTRATION_NUMBER', 'COMPANY_NAME', 'SIC_CODE', 'INDUSTRY_TYPE','INDUSTRY_CODE',
       'DISSOLVED/REMOVED/LIVE', 'LIQUIDATION', 'ACCOUNTING_FROM_DATE',
       'ACCOUNTING_TO_DATE', 'Year', 'WEEKS', 'MONTHS',
       'CONSOLIDATED-ACCOUNTS', 'ACCOUNTS_FORMAT', 'TURNOVER', 'EXPORT',
       'COST_OF_SALES', 'GROSS_PROFIT', 'WAGES_AND_SALARIES',
       'OPERATING_PROFITS', 'DEPRECIATION', 'INTEREST_PAYMENTS',
       'PRETAX_PROFITS', 'TAXATION', 'PROFIT_AFTER_TAX', 'DIVIDENDS_PAYABLE',
       'RETAINED_PROFITS', 'TANGIBLE_ASSETS', 'INTANGIBLE_ASSETS',
       'TOTAL_FIXED_ASSETS', 'TOTAL_CURRENT_ASSETS', 'TRADE_DEBTORS', 'STOCK',
       'CASH', 'OTHER_CURRENT_ASSETS', 'INCREASE_IN_CASH',
       'MISCELANEOUS_CURRENT_ASSETS', 'TOTAL_ASSETS',
       'TOTAL_CURRENT_LIABILITIES', 'TRADE_CREDITORS', 'BANK_OVERDRAFT',
       'OTHER_SHORTTERM_FIN', 'MISC_CURRENT_LIABILITIES', 'OTHER_LONGTERM_FIN',
       'TOTAL_LONGTERM_LIAB', 'BANK_OD_LTL', 'TOTAL_LIAB', 'NET_ASSETS',
       'WORKING_CAPITAL', 'PAIDUP_EQUITY', 'SHAREHOLDER_FUNDS',
       'P&L_ACCOUNT_RESERVE', 'SUNDRY_RESERVES', 'REVALUATION_RESERVE',
       'NETWORTH', 'NET_CASHFLOW_FROM_OPERATIONS',
       'NET_CASHFLOW_BEFOR_FINANCING', 'NET_CASHFLOW_FROM_FINANCING',
       'CONTINGENT_LIAB', 'CAPITAL_EMPLOYED', 'EMPLOYEES_COUNT',
       'PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
       'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO', 'EQUITY_RATIO',
       'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO', 'EBITDA','Z',
       'Z1', 'Z2', 'Z3', 'Z4', 'Z5']]


# In[ ]:


#data= data[data['TOTAL_LIAB'] != 0]


# In[ ]:


data_dict=data.to_dict('records')


# In[ ]:


#creating a collection 2
stage_2_table=db['root_file_added_scores']


# In[ ]:


#Send Dataframe from python to MongoDB
stage_2_table.insert_many(data_dict)


# WRITING AS CSV

# In[ ]:


data.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/all_industries_morescores_ver1.0.csv', index = False)


# SPLITTING YEARWISE INTO LIST

# In[8]:


df_all_years=data['Year'].unique().tolist()


# [1 - 2013, 2 - 2014, 3 - 2015, 4 - 2016, 5 - 2017, 6 - 2018, 7 - 2019, 8 - 2020]

# SPLITTING INDUSTRY_TYPE INTO LIST

# In[9]:


df_all_names=data['INDUSTRY_TYPE'].unique().tolist()
#df_all_names


# SPLITTING INDUSTRY_CODE INTO LIST

# In[10]:


df_all_types=data['INDUSTRY_CODE'].unique().tolist()


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

# In[11]:


df_all_codes=data['INDUSTRY_CODE'].unique().tolist()
#df_all_codes


# # COLLECTION SHOULD PROVIDE REQUIRED DOCUMENTS FOR CAGR CALCULATION
# Dict to DF

# In[12]:


data_cagr=data[['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','RETAINED_PROFITS','Year']]
#data_cagr.head()


# In[ ]:


#data_cagr.count()


# LIST OF DATAFRAMES

# In[13]:


list_dataframes= [v for k, v in data_cagr.groupby('COMPANY_NAME')]


# In[ ]:


#len(list_dataframes)


# In[14]:


list_df=[]
for i in list_dataframes:
    r=i.values.tolist()
    list_df.append(r)
#print(lst_df)    


# In[15]:


lst_cagr=[]
lst_cagr_percentage=[]

try:
    for u in list_df:
        for k in range(len(u)-1):
            #print("k: ",k)
            #print(len(u))
            #print(u)
            Initial_year=(u[0][-1])
            Final_year=u[1][-1]
            #print("Final_year :", Final_year)
            #print("Initial_year : ",Initial_year)
            if Final_year==Initial_year:
                pass
            else:
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
                if Initial_RP==0:
                    Initial_RP=1
                else:
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
                    u.pop(1)
                    lst_cagr.append([reg_num,com_name,ind_type,Final_year,A])
                    lst_cagr_percentage.append([reg_num,com_name,ind_type,Final_year,t])
        #break
        
except IndexError:
    pass
       
except ZeroDivisionError:
    pass


# In[ ]:


#len(lst_cagr)


# In[16]:


df = DataFrame (lst_cagr,columns=['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year','CAGR'])
df_cagr_per=DataFrame (lst_cagr_percentage,columns=['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year','CAGR'])
#df_cagr_per.head()


# In[18]:


dataa = data.merge(df, left_index=True, right_index=True,
             how='outer', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
#dataa = data.merge(df, on=[ 'REGISTRATION_NUMBER','Year'])
#dataa.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/all_scores_z_cagr.csv',index = False)


# In[ ]:


dataa_dict=dataa.to_dict('records')


# In[ ]:


stage_5_table=db['root_file_z_cagr']


# In[ ]:


stage_5_table.insert_many(dataa_dict)


# In[ ]:


#data = pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/all_industries_morescores_ver1.0.csv",engine='python')
#data.pop("Unnamed: 0")
#data_from_db = db.root_file_z_cagr.find({},{'_id':0})
#data=pd.DataFrame.from_dict(data_from_db)
for i,value in enumerate(df_all_types):
    dataa[dataa['INDUSTRY_CODE'] == value].to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/industry_code/ind_'+str(value)+r'.csv',index = False, na_rep = 'N/A') 
#for i,value in enumerate(df_all_codes):
    #data_ind=data[data['INDUSTRY_CODE'] == value].to_dict('records')
    #stage_0_table=db['root_file_added_scores_indwise_'+str(value)]
    #stage_0_table.insert_many(data_ind)


# In[ ]:


#df.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/CAGR.csv',index = False, na_rep = 'N/A')
#df_cagr_per.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/CAGR_per.csv',index = False, na_rep = 'N/A')


# In[19]:


df_cagr_final=dataa['CAGR']


# In[20]:


df_stats=df_cagr_final.describe()


# In[21]:


df1 =df_stats.values.tolist()


# In[22]:


CAGR_mark = df.groupby(['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year'], as_index=False)
average_cagr = CAGR_mark.agg({'CAGR':'mean'})
top_5_companies = average_cagr.sort_values('CAGR', ascending=False).head(5)
print("TOP_5_COMPANIES")
print(" ")
print(top_5_companies)


# # CATEGORISING CAGR OF COMPANIES INTO GROUPS

# In[23]:


conditions2_at = [
                (df['CAGR']>df1[3])&(df['CAGR'] <= df1[4]),
                (df['CAGR']>df1[4])&(df['CAGR'] <= df1[5]),
                (df['CAGR']>df1[5])&(df['CAGR'] <= df1[6]),
                (df['CAGR']>df1[6])&(df['CAGR'] <= df1[7]),   
                ]
values2_at = ['1', '2', '3', '4']
df['Iservice'] = np.select(conditions2_at, values2_at)
df['Iservice'] = model.fit_transform(df['Iservice'].astype('float'))
conditions3_at = [
                ( df['Iservice']==1),
                ( df['Iservice']==2),
                ( df['Iservice']==3),
                ( df['Iservice']==4),
                ]
values3_at = ['Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
df['Iservice_category'] = np.select(conditions3_at, values3_at)   


# In[ ]:


df_cagr.head()


# In[24]:


plt.figure(figsize=(5,3), dpi=100)
plt.style.use('ggplot')
one =df[(df.CAGR>= df1[3]) & (df.CAGR <= df1[4])].count()[0]
two = df[(df.CAGR  > df1[4]) & (df.CAGR <= df1[5])].count()[0]
three =df[(df.CAGR > df1[5]) & (df.CAGR <= df1[6])].count()[0]
four = df[(df.CAGR > df1[6]) & (df.CAGR <= df1[7])].count()[0]
weights = [one,two,three,four]
label = ['Need more analysis','Moderate','Reasonable performance','Better returns']
colors = [ '#EE82EE','#aaebcc','#FFBF00','#006400']
explode = (.1,.1,.1,.1)
plt.title('CAGR')
plt.pie(weights, labels=label, explode=explode,colors=colors, pctdistance=.8, autopct='%.2f %%')
plt.show()


# In[25]:


green =  df['Iservice_category']=='Better_returns'
amber = df['Iservice_category']=='Reasonable_performance'
red =  df['Iservice_category']=='Moderate'
black =  df['Iservice_category']=='Need_more_analysis'


# In[26]:


df2=df.Iservice.describe()


# In[28]:


#df_iserve.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
df3=df2.values.tolist()


# In[29]:


plt.figure(figsize=(5,3), dpi=100)
plt.style.use('ggplot')
one =df[(df.Iservice>= df3[3]) & (df.Iservice <= df3[4])].count()[0]
two = df[(df.Iservice  > df3[4]) & (df.Iservice <= df3[5])].count()[0]
three =df[(df.Iservice > df3[5]) & (df.Iservice <= df3[6])].count()[0]
four = df[(df.Iservice > df3[6]) & (df.Iservice <= df3[7])].count()[0]
weights = [one,two,three,four]
label = ['Need more analysis','Moderate','Reasonable performance','Better returns']
colors = [ '#EE82EE','#aaebcc','#FFBF00','#006400']
explode = (.1,.1,.1,.1)
plt.title('Company Performance')
plt.pie(weights, labels=label, explode=explode,colors=colors, pctdistance=.8, autopct='%.2f %%')
plt.show()


# In[30]:


green_report=df[green]
print(green_report[['COMPANY_NAME','Year','CAGR','Iservice_category']].head(5))


# In[31]:


amber_report=df[amber]
print(amber_report[['COMPANY_NAME','Year','CAGR','Iservice_category']].head(5))


# In[32]:


red_report=df[red]
print(red_report[['COMPANY_NAME','Year','CAGR','Iservice_category']].head(5))


# In[33]:


black_report=df[black]
print(black_report[['COMPANY_NAME','Year','CAGR','Iservice_category']].head(5))


# # FILTER COLLECTIONS INTO YEARWISE, INDUSTRYWISE AND [YEARWISE + INDUSTRYWISE] FOR PROCESSING IN FUTURE

# In[ ]:


#for i,value in enumerate(df_all_years):
    #data[data['Year'] == value].to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/year/Year_'+str(value)+r'.c   sv',index = False, na_rep = 'N/A')  


# # LOADING DATA INTO DB YEARWISE

# In[34]:


df_all_year=dataa['Year'].unique().tolist()
#df_all_year


# In[35]:


data_final = data.merge(df, left_index=True, right_index=True,
             how='outer', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')


# In[ ]:


data_final.head()


# In[ ]:


for i,value in enumerate(df_all_year):
    data_year=dataa[data_final['Year'] == value].to_dict('records')
    stage_9_table=db['root_file_z_cagr_Year_'+str(value)]
    stage_9_table.insert_many(data_year)


# # LOADING DATA INTO DB INDUSTRYCODE WISE

# In[ ]:


for i,value_year in enumerate(df_all_year):
    for l,value_type in enumerate(df_all_types):
        data_year_ind=data[(data['INDUSTRY_CODE'] == value_type) & (data['Year'] == value_year)].to_dict('records')
        stage_8_table=db['root_file_z_cagr_Year_industry_'+str(value_year)+'_'+str(value_type)]
        stage_8_table.insert_many(data_year_ind)    


# # ISCORE CALCULATION

# In[36]:


df_all_type=data_final['INDUSTRY_CODE'].unique().tolist()
for k,value_year in enumerate(df_all_year):
    for l,value_type in enumerate(df_all_type):
        data_year_ind=data_final[(data_final['INDUSTRY_CODE'] == value_type) & (data['Year'] == value_year)].to_dict('records')
        year_industry=pd.DataFrame.from_dict(data_year_ind)
        year_industry = year_industry[['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year','PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
                                       'SALES_PER_NET_WORKING_CAPITAL','GEARING_RATIO','EQUITY_RATIO','CREDITOR_DAYS', 'DEBTOR_DAYS', 
                                       'LIQUIDITY_TEST','RETURN_CAPITAL_EMPLOYED','RETURN_TOTAL_ASSETS', 'DEBT_EQUITY','RETURN_EQUITY',
                                       'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO', 'EBITDA','Z', 'CAGR','Iservice_category']]
        year_ind_stats=year_industry.describe()
        year_ind_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
        year_ind_stats_header=year_ind_stats.columns.tolist()
        del year_ind_stats_header[0]
        for i in year_ind_stats_header:
            conditions1_at = [
                ((year_industry[i]) <= 0),
                ((year_industry[i]) > 0) & ((year_industry[i]) <= (year_ind_stats[i][4])),
                ((year_industry[i]) > (year_ind_stats[i][4])) & ((year_industry[i]) <=(year_ind_stats[i][5])),
                ((year_industry[i]) > (year_ind_stats[i][5])) & ((year_industry[i]) <=(year_ind_stats[i][6])),
                ((year_industry[i]) > (year_ind_stats[i][6])) & ((year_industry[i]) <=(year_ind_stats[i][7]/2)),
                ((year_industry[i]) > (year_ind_stats[i][7]/2)) & ((year_industry[i]) <=(year_ind_stats[i][7])),
                ]
            values1_at = ['-1','1','2','3','4','5']
            n = str(i)+'_'+ 'Iscore'
            year_industry[str(n)] = np.select(conditions1_at, values1_at)
            year_industry[str(n)] = model.fit_transform(year_industry[str(n)].astype('float'))
        year_industry['Iscore_all']=year_industry.iloc[:,-17:].sum(axis=1)
        df=year_industry.Iscore_all.describe()
        conditions2_at = [
                    (year_industry['Iscore_all']>df[3])&(year_industry['Iscore_all'] <= df[4]),
                    (year_industry['Iscore_all']>df[4])&(year_industry['Iscore_all'] <= df[5]),
                    (year_industry['Iscore_all']>df[5])&(year_industry['Iscore_all'] <= df[6]),
                    (year_industry['Iscore_all']>df[6])&(year_industry['Iscore_all'] <= df[7]),   
                    ]
        values2_at = ['1', '2', '3', '4']
        year_industry['Iservice'] = np.select(conditions2_at, values2_at)
        year_industry['Iservice'] = model.fit_transform(year_industry['Iservice'].astype('float'))
        conditions3_at = [
                        (year_industry['Iservice']==1),
                        (year_industry['Iservice']==2),
                        (year_industry['Iservice']==3),
                        (year_industry['Iservice']==4),
                        ]
        values3_at = ['Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
        year_industry['Iservice_category'] = np.select(conditions3_at, values3_at)
        year_industry.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/iscore/'+'iscore_'+str(value_year)+'_'+str(value_type)+r'.csv',index = False, na_rep = 'N/A')
        #year_ind=year_industry.to_dict('records')
        #stage_7_table=db['iscore_z_cagr_'+str(value_year)+'_'+str(value_type)]
        #stage_7_table.insert_many(year_ind)


# In[37]:


import glob
os.chdir(r"C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/iscore")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
iscores_z_cagr = pd.concat([pd.read_csv(f) for f in all_filenames ])
iscore_z_cagr=iscores_z_cagr.drop(['Iservice_category_Iscore'],axis=1)
#export to csv
iscore_z_cagr.to_csv( "C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/iscore/iscores_z_cagr.csv", index=False)


# In[ ]:


iscores_z_cagr.head()


# In[39]:


ind_all=iscore_z_cagr.to_dict('records')
stage_6_table=db['iscore_z_cagr_'+'all_ind']
stage_6_table.insert_many(ind_all)


# MERGING ISCORES FILE WITH ALL YEAR FILE

# In[40]:


a = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/all_industries_morescores_ver1.0.csv")
b = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/iscore/iscores_z_cagr.csv")
b = b.dropna(axis=1)
all_scores_final = a.merge(b, left_index=True, right_index=True,
             how='outer', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
all_scores_final.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/all_scores_05062021.csv", index=False,low_memory=False)


# In[41]:


data_from_db = db.root_file.find({},{'_id':0})
a=pd.DataFrame.from_dict(data_from_db)


# In[42]:


data_from_db = db.iscore_z_cagr_all_ind.find({},{'_id':0})
b=pd.DataFrame.from_dict(data_from_db)


# In[43]:


b = b.dropna(axis=1)


# LOADING COLLATED FILE INTO DB

# In[44]:


#all_scores_05062021 = a.merge(b, on=[ 'REGISTRATION_NUMBER','Year'])
all_scores_05062021 = a.merge(b, left_index=True, right_index=True,
             how='outer', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
all_scores_05062021.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/05062021/allscores_05062021.csv", index=False)


# In[49]:


all_scores_final_dict=all_scores_05062021.to_dict('records')


# In[50]:


stage_13_table=db['all_scores_11062021']


# In[ ]:


stage_13_table.insert_many(all_scores_final_dict)

