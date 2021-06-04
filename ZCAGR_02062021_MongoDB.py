#!/usr/bin/env python
# coding: utf-8

# LOADING PYTHON MODULES

# In[1]:


import time
start = time.time()
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
#get_ipython().run_line_magic('matplotlib', 'inline')
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
from pandas import DataFrame
from bson.raw_bson import RawBSONDocument
from bson.codec_options import CodecOptions
from pymongo import MongoClient
from datetime import datetime

global absDirName
absDirName = os.path.dirname(os.path.abspath(__file__))

codec_options = CodecOptions(unicode_decode_error_handler='ignore')


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

def mongoconnection():
    with open(os.path.join(absDirName,"constants.json"),"r")  as constants:
        global constantsData
        constantsData = json.load(constants)
        print(constantsData)
    #   constantsData = json.load(constants)

    #dgSafeConstants = constantsData["dgSafe"]
    global uri 
    #uri = "mongodb://" + dgSafeConstants['username'] + ":" + dgSafeConstants['password'] + "@" + dgSafeConstants["server"] +":" + dgSafeConstants['port'] + "/dgsafe?ssl=false&authSource=dgsafe"
    uri = "mongodb://" + constantsData['username'] + ":" + constantsData['password'] + "@" + "testapi.datagardener.com" +":" + constantsData['port'] + "/dgsafe?ssl=falses&authSource=dgsafe"
    #data_gardener_liveConstants = constantsData["data_gardener_live"]
    print (uri)
    #uri_dg_live = "mongodb://" + data_gardener_liveConstants['username'] + ":" + data_gardener_liveConstants['password'] + "@" + data_gardener_liveConstants["server"] +":" + data_gardener_liveConstants['port'] + "/data-gardener_Live?ssl=false&authSource=data-gardener_Live"

    client = MongoClient(uri)
    #client1 =  MongoClient(uri)
    global dgsafe
    dgsafe = client["dgsafe"]

    # global dg_live
    # dg_live = client1["data-gardener_Live"]            


# # AC01,AC06,INDUSTRIES MASTER TO BE LOOKED UP AND Z SCORE, CAGR AND EBITDA CALCULATED AND APPENDED TO ALL DOCUMENTS FOR FURTHER PROCESSING 

if __name__ == "__main__":
    mongoconnection()
    

    #root_file_path = os_any_dir_search('all_industries_more_scores.csv')[1]
    start = datetime.now()
    currentyear = datetime.today().year
    mainCollection = dgsafe.get_collection("cagr_root_file", codec_options=codec_options)
    uniqueIndustryList = list(mainCollection.distinct("INDUSTRY_TYPE"))
    for industry in uniqueIndustryList:
        print("industry " ,industry)

        data_from_db = list(mainCollection.find({"INDUSTRY_TYPE": industry},{"_id":0}))
        # queryData = list(data_from_db)
        if len(data_from_db) == 0:
                continue

        data_cagr_root = pd.DataFrame.from_dict(data_from_db)
        print("pass")
        data_cagr = data_cagr_root[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']]
        data_cagr.head()
        list_dataframes = []
        for k, v in data_cagr.groupby('NAME'):
            if v.shape[0] >= 4:
                list_dataframes.append(v)
        #list_dataframes= [v for k, v in data_cagr.groupby('COMPANY_NAME')]
        print("dataframe list ",len(list_dataframes))
        if len(list_dataframes) == 0:
            continue
        

        lst_df=[]
        for i in list_dataframes:
            r=i.values.tolist()
            lst_df.append(r)
        #print(lst_df) 

        lst_cagr = []
        lst_cagr_percentage = []
        try:
            for u in lst_df:
        #if len(u)>0:
        #print("number of records in df : ",len(lst_df))
        #print(u)
        #print(" ")
                for k in u:
                #print("number of records per company: ",len(u))
                    year=(k[-1])
                    Final_year=u[-1][-1]
                    Initial_year=k[-1]
                    if (Final_year!=Initial_year):
                        reg_num=(k[0])
                        #print(reg_num)
                        com_name=(k[1])
                        #print(com_name)
                        ind_type=(k[2])
                        #print(ind_type)
                        #print("Final_year :", Final_year)
                        #print("Initial_year : ",Initial_year)
                        expo =(1/(Final_year-Initial_year))
                        #print("expo: ",expo)
                        Final_RP=u[-1][-2]
                        #print("Final_RP :",Final_RP)
                        Initial_RP=k[-2]
                        if Initial_RP==0:
                            Initial_RP==1
                        else:
                            #print("Initial_RP :",Initial_RP)
                            part=(Final_RP/Initial_RP)
                            #print("part : ",part)
                            cagr = ((part ** (expo)) -1)
                            q = (cagr.real, cagr.imag)
                            A = cagr.real
                            B = cagr.imag
                            t = ("{:.0%}".format(A))
                            #print("cagr: ",A)
                            #print("cagr in %: ",t)
                            #print(" ")
                        lst_cagr.append([reg_num,com_name,ind_type,year+1,A])
                        lst_cagr_percentage.append([reg_num,com_name,ind_type,year+1,t])
                    else:
                        pass
        #break     
#except TypeError:
    #pass
        except ZeroDivisionError:
            pass
        
        df = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','CAGR'])

        df_stats = df.describe()

        df_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)

        one = df[(df.CAGR >= df_stats.CAGR[3]) & (df.CAGR <= df_stats.CAGR[4])].count()[0]
        print("Band 1 : ",one)
        two = df[(df.CAGR  > df_stats.CAGR[4]) & (df.CAGR <= df_stats.CAGR[5])].count()[0]
        print("Band 2 : ",two)
        three =df[(df.CAGR > df_stats.CAGR[5]) & (df.CAGR <= df_stats.CAGR[6])].count()[0]
        print("Band 3 : ",three)
        four = df[(df.CAGR > df_stats.CAGR[6]) & (df.CAGR <= (df_stats.CAGR[7]/2))].count()[0]
        print("Band 4 : ",four)
        five = df[(df.CAGR > (df_stats.CAGR[7]/2)) & (df.CAGR <= df_stats.CAGR[7])].count()[0]

        CAGR_mark = df.groupby(['REG','NAME','INDUSTRY_TYPE'], as_index=False)
        average_cagr = CAGR_mark.agg({'CAGR':'mean'})
        top_20_companies = average_cagr.sort_values('CAGR', ascending=False).head(5)
        print("TOP_20_COMPANIES")
        print(" ")
        print(top_20_companies)
        break

        # all_scores_final_dict = all_scores_final.to_dict('records')

        # stage_7_table = dgsafe['all_scores_final']

        # stage_7_table.insert_many(all_scores_final_dict)








