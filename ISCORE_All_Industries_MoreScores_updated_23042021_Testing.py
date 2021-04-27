#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start = time.time()

import pandas as pd
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


# In[2]:


data= pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/all_industries_more_scores.csv",engine='python')


# In[3]:


data["TOTAL_DEBT_RATIO"]=(data["TOTAL_LIAB"]/data["SHAREHOLDER_FUNDS"])
data["EBITDA"]=((data["OPERATING_PROFITS"]+data["DEPRECIATION"]))
data["Z1"]=(1.2*(data["WORKING_CAPITAL"]/data["TOTAL_ASSETS"]))
data["Z2"]=(1.4*(data["RETAINED_PROFITS"]/data["TOTAL_ASSETS"]))
data["Z3"]=(3.3*(data["EBITDA"]/data["TOTAL_ASSETS"]))
data["Z4"]=(.6*(data["SHAREHOLDER_FUNDS"]/data["TOTAL_LIAB"]))
data["Z5"]=(.99*(data["TURNOVER"]/data["TOTAL_ASSETS"]))
data["Z"]=(data["Z1"]+data["Z2"]+data["Z3"]+data["Z4"]+data["Z5"])


# In[4]:


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


# In[5]:


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


# In[6]:


data= data[data['TOTAL_LIAB'] != 0]


# In[7]:


data.to_csv(r'E://DG/DG_DB/main/ALL/industrywise/dummy/all_industries_morescores_ver1.0.csv', index = False)


# In[8]:


df_all_years=data['Year'].unique().tolist()


# [1 - 2013, 2 - 2014, 3 - 2015, 4 - 2016, 5 - 2017, 6 - 2018, 7 - 2019, 8 - 2020]

# In[9]:


df_all_names=data['INDUSTRY_TYPE'].unique().tolist()


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

# SPLITTING MAIN CSV INTO YEARWISE 

# In[11]:


data = pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/all_industries_morescores_ver1.0.csv",engine='python')
#data.pop("Unnamed: 0")
for i,value in enumerate(df_all_years):
    data[data['Year'] == value].to_csv(r'E://DG/DG_DB/main/ALL/industrywise/dummy/year/Year_'+str(value)+r'.csv',index = False, na_rep = 'N/A')     


# SPLITTING YEARWISE CSV INTO YEAR/INDUSTRY

# In[12]:


PATH = "E://DG/DG_DB/main/ALL/industrywise/dummy/year"
EXT = "*.csv"
all_csv_files = []
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        all_csv_files.append(file)
#print(all_csv_files)

for j in all_csv_files:
    x=j.split('/')
    y=x[8].split('\\')
    z=y[1]
    w=z.split('.')
    v=w[0]
    yearwise = pd.read_csv(str(j),low_memory = False, dtype='unicode')
    #print(yearwise)
    for l,value in enumerate(df_all_types):
        yearwise[yearwise['INDUSTRY_CODE'] == value].to_csv(r'E://DG/DG_DB/main/ALL/industrywise/dummy/industry/'+str(v)+'_'+str(value)+r'.csv',index = False, na_rep = 'N/A')  


# SPLITTING YEAR/INDUSTRY CSV INTO STATISTICAL DATA

# In[13]:


PATH = "E://DG/DG_DB/main/ALL/industrywise/dummy/industry"
EXT = "*.csv"
all_csv_files_industry = []
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        all_csv_files_industry.append(file)
        
for j in all_csv_files_industry:
    a=j.split('/')
    b=a[8].split('\\')
    c=b[1]
    d=c.split('.')
    e=d[0]
    year_industry = pd.read_csv(str(j),low_memory = False)
    year_industry = year_industry[['REGISTRATION_NUMBER','COMPANY_NAME','INDUSTRY_TYPE','PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
       'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO', 'EQUITY_RATIO',
       'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO', 'EBITDA', 'Z']]
    year_industry_stat=year_industry.describe()
    year_industry_stat.to_csv(r'E://DG/DG_DB/main/ALL/industrywise/dummy/stats/'+str(e)+r'.csv',index = True, na_rep = 'N/A')


# In[14]:


PATH = "E://DG/DG_DB/main/ALL/industrywise/dummy/stats"
EXT = "*.csv"
all_csv_files_stats = []
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        all_csv_files_stats.append(file)


# In[15]:


for j in all_csv_files_stats:
    f=j.split('/')
    g=f[8].split('\\')
    h=g[1]
    k=h.split('.')
    l=k[0]
    m=k[1]
    year_ind_stats = pd.read_csv(str(j))
    year_ind_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
    year_ind_stats_header=year_ind_stats.columns.tolist()
    del year_ind_stats_header[0]
    break


# In[16]:


for j in all_csv_files_industry:
    f=j.split('/')
    g=f[8].split('\\')
    h=g[1]
    k=h.split('.')
    l=k[0]
    m=k[1]
    year_industry = pd.read_csv(str(j),low_memory = False)
    year_industry = year_industry[['REGISTRATION_NUMBER','COMPANY_NAME','Year','INDUSTRY_TYPE','PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
       'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO', 'EQUITY_RATIO',
       'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
       'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
       'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO', 'EBITDA', 'Z']]
    year_indus_stats=year_industry.describe()
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
    
    year_industry.to_csv(r'E://DG/DG_DB/main/ALL/industrywise/dummy/iscore/'+str(l)+'_'+str(m)+r'.csv',index = False, na_rep = 'N/A')


# In[17]:


os.chdir("E://DG/DG_DB/main/ALL/industrywise/dummy/iscore")
extension = 'csv'
import glob
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
iscore_all_industry_year = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
iscore_all_industry_year.to_csv( "iscore_all_industry_year.csv", index=False)


# In[18]:


iscore_all_industry_years= pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/iscore/iscore_all_industry_year.csv",engine='python')


# In[19]:


a = pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/all_industries_more_scores.csv")
b = pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/iscore/iscore_all_industry_year.csv")
b = b.dropna(axis=1)
all_scores_final = a.merge(b, on=[ 'COMPANY_NAME', 'Year'])
all_scores_final.to_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/all_scores_final.csv", index=False)


# # TOP SCORERS COUNT

# In[20]:


df_2020_22= pd.read_csv("E://DG/DG_DB/main/ALL/industrywise/dummy/iscore/Year_2020_22_csv.csv",engine='python')


# # SAMPLE - YEAR 2020 FOR PHARMA

# # PIE CHART BASED ON Iscore

# In[21]:


plt.figure(figsize=(5,3), dpi=100)
plt.style.use('ggplot')
one =df_2020_22.loc[(df_2020_22.Iscore_all >= df[3]) & (df_2020_22.Iscore_all <= df[4])].count()[0]
two = df_2020_22[(df_2020_22.Iscore_all  > df[4]) & (df_2020_22.Iscore_all <= df[5])].count()[0]
three =df_2020_22[(df_2020_22.Iscore_all  > df[5]) & (df_2020_22.Iscore_all  <= df[6])].count()[0]
four = df_2020_22[(df_2020_22.Iscore_all  > df[6]) & (df_2020_22.Iscore_all <= df[7])].count()[0]
#five = Iscore2019_df[Iscore2019_df.Iindex  >= 250].count()[0]
weights = [one,two,three,four]
label = ['Need more analysis','Moderate','Reasonable performance','Better returns']
colors = [ '#EE82EE','#aaebcc','#FFBF00','#006400']
explode = (.1,.1,.1,.1)
plt.title('Company Performance')
plt.pie(weights, labels=label, explode=explode,colors=colors, pctdistance=.8,autopct='%.2f %%')
plt.show()


# # WORTH INVESTING

# In[22]:


I_score = df_2020_22.groupby(['REGISTRATION_NUMBER','COMPANY_NAME','Iservice_category'], as_index=False)
average_Iscore = I_score.agg({'Iscore_all':'mean'})
top_10_companies = average_Iscore.sort_values('Iscore_all', ascending=False).head(10)
print("TOP_10_COMPANIES")
print(" ")
top_10_companies


# In[23]:


Iperformance=top_10_companies[['COMPANY_NAME','Iservice_category']]
#Iperformance


# In[24]:


top_10_companies.head(10)


# In[25]:


Iperformance.head(10)


# In[26]:


green =  df_2020_22['Iservice_category']=='Better_returns'
amber = df_2020_22['Iservice_category']=='Reasonable_performance'
red =  df_2020_22['Iservice_category']=='Moderate'
black =  df_2020_22['Iservice_category']=='Need_more_analysis'


# In[27]:


print("GREEN_COMPANIES")
print(" ")
green_report=df_2020_22[green]
print(green_report.COMPANY_NAME.head(10))


# In[28]:


print("AMBER_COMPANIES")
print(" ")
amber_report=df_2020_22[amber]
print(amber_report.COMPANY_NAME.head(10))


# In[29]:


print("RED_COMPANIES")
print(" ")
red_report=df_2020_22[red]
print(red_report.COMPANY_NAME.tail(10))


# In[30]:


print("RED_COMPANIES")
print(" ")
black_report=df_2020_22[black]
print(black_report.COMPANY_NAME.tail(10))


# In[31]:


end = time.time()
print("Time taken to run the code is {} seconds".format(end - start))

