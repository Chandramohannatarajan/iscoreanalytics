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


# ACCESS MONGODB 

# In[2]:


import pymongo
myclient = pymongo.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")
dbname="dgdata"
mydb = myclient[dbname]
#list the collections
#for coll in mydb.list_collection_names():
    #print(coll)


# SEARCH FILE IN DIRECTORY

# In[3]:


def os_any_dir_search(file):
    u = []
    for p,n,f in os.walk(os.getcwd()):
        for a in f:
            if a.endswith(file):
                    #print("A -->",a)
                    #print("P -->",p)
                    t=pd.read_csv(p+'/'+file,low_memory=False)
                    #print("T -->",t)
                    u.append(p+'/'+a)
    return t,u             


# # AC01,AC06,INDUSTRIES MASTER TO BE LOOKED UP AND Z SCORE, CAGR AND EBITDA CALCULATED AND APPENDED TO ALL DOCUMENTS FOR FURTHER PROCESSING 

# In[4]:


root_file = os_any_dir_search('all_industries_more_scores.csv')[0]


# In[ ]:


root_file_dict=root_file.to_dict('records')


# In[5]:


#making a connection to Mongo client
client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[6]:


#creating database
db=client['dgdata']


# In[ ]:


#creating a collection
#stage_1_table=db['root_file']


# In[ ]:


#Send Dataframe from python to MongoDB
#stage_1_table.insert_many(root_file_dict)


# EXPORT FROM DB TO PYTHON

# # ENTRY 1  --> To be queried directly from MongoDB

# In[ ]:


data_from_db = db.root_file.find({},{'_id':0})
data=pd.DataFrame.from_dict(data_from_db)


# # HOT CODING INDUSTRIES TYPE  --> TO BE PERFORMED IN MONGODB

# In[ ]:


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

# In[ ]:


data["TOTAL_DEBT_RATIO"]=(data["TOTAL_LIAB"]/data["SHAREHOLDER_FUNDS"])
data["EBITDA"]=((data["OPERATING_PROFITS"]+data["DEPRECIATION"]))
data["Z1"]=(1.2*(data["WORKING_CAPITAL"]/data["TOTAL_ASSETS"]))
data["Z2"]=(1.4*(data["RETAINED_PROFITS"]/data["TOTAL_ASSETS"]))
data["Z3"]=(3.3*(data["EBITDA"]/data["TOTAL_ASSETS"]))
data["Z4"]=(.6*(data["SHAREHOLDER_FUNDS"]/data["TOTAL_LIAB"]))
data["Z5"]=(.99*(data["TURNOVER"]/data["TOTAL_ASSETS"]))
data["Z"]=(data["Z1"]+data["Z2"]+data["Z3"]+data["Z4"]+data["Z5"])


# # AT THIS STAGE, THE COLLECTION SHOULD HAVE AC01,AC06 COLLATED WITH INDUSTRIES TYPE AND EBITDA,Z SCORE ADDED

# In[ ]:


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


data= data[data['TOTAL_LIAB'] != 0]


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


#data.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries_morescores_ver1.0.csv', index = False)


# SPLITTING YEARWISE INTO LIST

# In[ ]:


df_all_years=data['Year'].unique().tolist()


# [1 - 2013, 2 - 2014, 3 - 2015, 4 - 2016, 5 - 2017, 6 - 2018, 7 - 2019, 8 - 2020]

# SPLITTING INDUSTRY_TYPE INTO LIST

# In[ ]:


df_all_names=data['INDUSTRY_TYPE'].unique().tolist()
#df_all_names


# SPLITTING INDUSTRY_CODE INTO LIST

# In[ ]:


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

# In[ ]:


#data = pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/all_industries_morescores_ver1.0.csv",engine='python')
#data.pop("Unnamed: 0")
data_from_db = db.root_file_added_scores.find({},{'_id':0})
data=pd.DataFrame.from_dict(data_from_db)
#for i,value in enumerate(df_all_codes):
    #data[data['INDUSTRY_CODE'] == value].to_csv(r'E://DG/DG_DB/main/ALL/industrywise/dummy/industrywise/ind_'+str(value)+r'.csv',index = False, na_rep = 'N/A') 
for i,value in enumerate(df_all_codes):
    data_ind=data[data['INDUSTRY_CODE'] == value].to_dict('records')
    stage_0_table=db['root_file_added_scores_indwise_'+str(value)]
    stage_0_table.insert_many(data_ind)
    


# # CAGR CALCULATION

# In[ ]:


df_all_codes=data['INDUSTRY_CODE'].unique().tolist()
#df_all_codes


# # COLLECTION SHOULD PROVIDE REQUIRED DOCUMENTS FOR CAGR CALCULATION

# In[ ]:


data_from_db = db.root_file_added_scores.find({},{'_id':0})
data_cagr_root=pd.DataFrame.from_dict(data_from_db)


#data_cagr_root = pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/all_industries_morescores_ver1.0.csv",engine='python')
#data_cagr_root = pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/industrywise/ind_9.csv",engine='python')
data_cagr_root.head()


# In[ ]:


data_cagr=data_cagr_root[['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','RETAINED_PROFITS','Year']]
data_cagr.head()


# LIST OF DATAFRAMES

# In[ ]:


list_dataframes= [v for k, v in data_cagr.groupby('COMPANY_NAME')]


# In[ ]:


list_dataframes[0]


# In[ ]:


list_dataframes[-1]


# In[ ]:


lst_df=[]
for i in list_dataframes:
    r=i.values.tolist()
    lst_df.append(r)
#print(lst_df)    


# In[ ]:


import math
lst_cagr=[]
lst_cagr_percentage=[]


for i in lst_df:
    try:
        reg_num=(i[0][0])
        com_name=(i[0][1])
        g=int(i[-1][-1])
        h=int(i[0][-1])
        ind_type=(i[0][2])
        expo =(1/(g-h))
        o=(i[-1][-2])
        p=((i[0][-2]))
        k=(o/p)
        cagr = ((k ** (expo) -1))
        q = (cagr.real, cagr.imag)
        A = cagr.real
        B = cagr.imag
        t = ("{:.0%}".format(A))
        lst_cagr.append([reg_num,com_name,ind_type,A])
        lst_cagr_percentage.append([reg_num,com_name,ind_type,t])
        
    except TypeError:
        pass
    except ZeroDivisionError:
        pass


# In[ ]:


from pandas import DataFrame
df = DataFrame (lst_cagr,columns=['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','CAGR'])
#df.head()


# In[ ]:


#df.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/CAGR.csv',index = False, na_rep = 'N/A')


# In[ ]:


df_stats=df.describe()
#df_stats


# In[ ]:


df_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
df_stats
#del df_header[0]


# # VISUALS FOR CAGR

# In[ ]:


plt.figure(figsize=(5,3), dpi=100)
plt.style.use('ggplot')
one =df[(df.CAGR >= df_stats.CAGR[3])& (df.CAGR <= df_stats.CAGR[4])].count()[0]
print("Band 1 : ",one)
two = df[(df.CAGR  > df_stats.CAGR[4]) & (df.CAGR <= df_stats.CAGR[5])].count()[0]
print("Band 2 : ",two)
three =df[(df.CAGR > df_stats.CAGR[5]) & (df.CAGR <= df_stats.CAGR[6])].count()[0]
print("Band 3 : ",three)
four = df[(df.CAGR > df_stats.CAGR[6]) & (df.CAGR <= (df_stats.CAGR[7]/2))].count()[0]
print("Band 4 : ",four)
five=df[(df.CAGR > (df_stats.CAGR[7]/2)) & (df.CAGR <= df_stats.CAGR[7])].count()[0]
weights = [one,two,three,four,five]
label = ['Need more analysis','Moderate','Reasonable performance','Better returns','Bullish']
colors = [ '#EE82EE','#aaebcc','#FFBF00','#006400','#FF92FE']
explode = (.1,.1,.1,.1,.1)
plt.title('Company Performance')
plt.pie(weights, labels=label, explode=explode,colors=colors, pctdistance=.9, autopct='%.2f %%')
plt.show()


# In[ ]:


CAGR_mark = df_cagr.groupby(['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE'], as_index=False)
average_cagr = CAGR_mark.agg({'CAGR':'mean'})
top_20_companies = average_cagr.sort_values('CAGR', ascending=False).head(20)
print("TOP_20_COMPANIES")
print(" ")
PRINT(top_20_companies)


# # CATEGORISING CAGR OF COMPANIES INTO GROUPS

# In[ ]:


conditions2_at = [
                (df_cagr['CAGR']>df[3])&(df_cagr['CAGR'] <= df[4]),
                (df_cagr['CAGR']>df[4])&(df_cagr['CAGR'] <= df[5]),
                (df_cagr['CAGR']>df[5])&(df_cagr['CAGR'] <= df[6]),
                (df_cagr['CAGR']>df[6])&(df_cagr['CAGR'] <= df[7]),   
                ]
values2_at = ['1', '2', '3', '4']
df_cagr['Iservice'] = np.select(conditions2_at, values2_at)
df_cagr['Iservice'] = model.fit_transform(df_cagr['Iservice'].astype('float'))
conditions3_at = [
                ( df_cagr['Iservice']==1),
                ( df_cagr['Iservice']==2),
                ( df_cagr['Iservice']==3),
                ( df_cagr['Iservice']==4),
                ]
values3_at = ['Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
df_cagr['Iservice_category'] = np.select(conditions3_at, values3_at)   


# In[ ]:


green =  df_cagr['Iservice_category']=='Better_returns'
amber = df_cagr['Iservice_category']=='Reasonable_performance'
red =  df_cagr['Iservice_category']=='Moderate'
black =  df_cagr['Iservice_category']=='Need_more_analysis'


# In[ ]:


df=df_cagr.Iservice.describe()


# # VISUALS FOR ISERVICE BASED ON CAGR

# In[ ]:


plt.figure(figsize=(5,3), dpi=100)
plt.style.use('ggplot')
one =df_cagr[(df_cagr.Iservice>= df[3]) & (df_cagr.Iservice <= df[4])].count()[0]
print("Band 1 : ",one)
#print(type(one))
two = df_cagr[(df_cagr.Iservice  > df[4]) & (df_cagr.Iservice <= df[5])].count()[0]
print("Band 2 : ",two)
three =df_cagr[(df_cagr.Iservice > df[5]) & (df_cagr.Iservice <= df[6])].count()[0]
print("Band 3: ",three)
four = df_cagr[(df_cagr.Iservice > df[6]) & (df_cagr.Iservice <= df[7])].count()[0]
print("Band 4 : ",four)
#five = df_cagr[df_cagr.CAGR  > df[7][0]].count()[0]
weights = [one,two,three,four]
label = ['Need more analysis','Moderate','Reasonable performance','Better returns']
colors = [ '#EE82EE','#aaebcc','#FFBF00','#006400']
explode = (.1,.1,.1,.1)
plt.title('Company Performance')
plt.pie(weights, labels=label, explode=explode,colors=colors, pctdistance=.8, autopct='%.2f %%')
plt.show()


# In[ ]:


print("GREEN_COMPANIES")
print(" ")
green_report=df_cagr[green]
print(green_report.COMPANY_NAME.head(10))


# In[ ]:


print("AMBER_COMPANIES")
print(" ")
amber_report=df_cagr[amber]
print(amber_report.COMPANY_NAME.head(10))


# In[ ]:


print("RED_COMPANIES")
print(" ")
red_report=df_cagr[red]
print(red_report.COMPANY_NAME.tail(10))


# In[ ]:


print("BLACK_COMPANIES")
print(" ")
black_report=df_cagr[black]
print(black_report.COMPANY_NAME.tail(10))


# # FILTER COLLECTIONS INTO YEARWISE, INDUSTRYWISE AND [YEARWISE + INDUSTRYWISE] FOR PROCESSING IN FUTURE

# In[ ]:


#for i,value in enumerate(df_all_years):
    #data[data['Year'] == value].to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/year/Year_'+str(value)+r'.c   sv',index = False, na_rep = 'N/A')  


# # LOADING DATA INTO DB YEARWISE

# In[ ]:


for i,value in enumerate(df_all_years):
    data_year=data[data['Year'] == value].to_dict('records')
    stage_9_table=db['root_file_added_scores_Year_'+str(value)]
    stage_9_table.insert_many(data_year)


# # LOADING DATA INTO DB INDUSTRYCODE WISE

# In[ ]:


for i,value_year in enumerate(df_all_years):
    for l,value_type in enumerate(df_all_types):
        data_year_ind=data[(data['INDUSTRY_CODE'] == value_type) & (data['Year'] == value_year)].to_dict('records')
        stage_8_table=db['root_file_added_scores_Year_industry_'+str(value_year)+'_'+str(value_type)]
        stage_8_table.insert_many(data_year_ind)    


# # ISCORE CALCULATION

# In[ ]:


for k,value_year in enumerate(df_all_years):
    for l,value_type in enumerate(df_all_types):
        data_year_ind=data[(data['INDUSTRY_CODE'] == value_type) & (data['Year'] == value_year)].to_dict('records')
        year_industry=pd.DataFrame.from_dict(data_year_ind)
        year_industry = year_industry[['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','Year','PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
       'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO', 'EQUITY_RATIO',
       'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO', 'EBITDA', 'Z']]
        year_ind_stats=year_industry.describe()
        data_year_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
        year_ind_stats_header=data_year_stats.columns.tolist()
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
        year_industry['Iscore_all']=year_industry.iloc[:,-16:-1].sum(axis=1)
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
        #year_industry.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/iscores/dummy/'+'iscore_'+str(value_year)+'_'+str(value_type)+r'.csv',index = False, na_rep = 'N/A')
        year_ind=year_industry.to_dict('records')
        stage_7_table=db['iscore_'+str(value_year)+'_'+str(value_type)]
        stage_7_table.insert_many(year_ind)

#extension = 'csv'
#all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
#iscore_all_industry_year = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
#iscore_all_industry_year.to_csv( "C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/iscore_all_industry_year.csv", index=False)
#ind_all=iscore_all_industry_year.to_dict('records')
#stage_6_table=db['iscore_'+'all_ind']
#stage_6_table.insert_many(ind_all)


# MERGING ISCORES FILE WITH ALL YEAR FILE

# In[ ]:


a = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/all_industries_more_scores.csv")
b = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/iscore_all_industry_year.csv")
b = b.dropna(axis=1)
all_scores_final = a.merge(b, on=[ 'COMPANY_NAME', 'Year'])
all_scores_final.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/all_scores_final.csv", index=False)


# In[ ]:


data_from_db = db.root_file.find({},{'_id':0})
a=pd.DataFrame.from_dict(data_from_db)


# In[ ]:


data_from_db = db.iscore_all_industry_year.find({},{'_id':0})
b=pd.DataFrame.from_dict(data_from_db)


# In[ ]:


b = b.dropna(axis=1)


# LOADING COLLATED FILE INTO DB

# In[ ]:


all_scores_final = a.merge(b, on=[ 'COMPANY_NAME', 'Year'])
all_scores_final.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/all_industries/all_scores_final.csv", index=False)


# In[ ]:


all_scores_final_dict=all_scores_final.to_dict('records')


# In[ ]:


stage_7_table=db['all_scores_final']


# In[ ]:


stage_7_table.insert_many(all_scores_final_dict)

