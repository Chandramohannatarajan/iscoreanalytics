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
            if v.shape[0] >= 3:
                list_dataframes.append(v)
        #list_dataframes= [v for k, v in data_cagr.groupby('COMPANY_NAME')]
        print("dataframe list ",len(list_dataframes))
        if len(list_dataframes) == 0:
            continue
        

        list_df=[]
        for i in list_dataframes:
            r=i.values.tolist()
            list_df.append(r)
        #print(lst_df) 

        lst_cagr = []
        lst_cagr_percentage = []
        try:
            for u in list_df:
                for k in range(len(u)-1):
                    #print("k: ",k)
                    #print(len(u))
                    #print(u)
                    Initial_year=(u[0][-1])
                    Final_year=u[1][-1]
                    print("Final_year :", Final_year)
                    print("Initial_year : ",Initial_year)
                    if Final_year==Initial_year:
                        pass
                    else:
                        reg_num=(u[0][0])
                        #print(reg_num)
                        com_name=(u[0][1])
                        #print(com_name)
                        ind_type=(u[0][2])
                        #print(ind_type)
                        expo =(1/(int(Final_year)-int(Initial_year)))
                        #print("expo: ",expo)
                        Final_RP=u[1][-2]
                        #print("Final_RP :",Final_RP)
                        Initial_RP=u[0][-2]
                        #print("Initial_RP :",Initial_RP)
                        if Initial_RP==0:
                            Initial_RP=1
                        else:
                            part=(Final_RP/Initial_RP)
                            print("part : ",part)
                            cagr = (part)**(expo) -1
                            q = (cagr.real, cagr.imag)
                            A = cagr.real
                            B = cagr.imag
                            t = ("{:.0%}".format(A))
                            #print("cagr : ",A)
                            #print("cagr in %: ",t)
                            #print(" ")
                            u.pop(1)
                            lst_cagr.append([reg_num,com_name,ind_type,int(Final_year),A])
                            lst_cagr_percentage.append([reg_num,com_name,ind_type,int(Final_year),t])
        #break
        except IndexError:
            pass
       
        except ZeroDivisionError:
            pass
        
        df = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])
        df_cagr_per=DataFrame (lst_cagr_percentage,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])
        df_cagr_per.head()

        #dataa = data_cagr_root.merge(df, on=[ 'REG','NAME','INDUSTRY_TYPE','YEAR'])
        

        df_cagr_final= df['CAGR']
        df_stats = df_cagr_final.describe()

        df1 =df_stats.values.tolist()
        #df_cagr_final=dataa['CAGR']

        #df_stats=df_cagr_final.describe()
        #df_stats

        #df.stats=df_stats.values.tolist

        # df_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
        # df_stats

        CAGR_mark = df.groupby(['REG','NAME','INDUSTRY_TYPE','YEAR'], as_index=False)
        average_cagr = CAGR_mark.agg({'CAGR':'mean'})
        top_20_companies = average_cagr.sort_values('CAGR', ascending=False).head(20)
        print("TOP_20_COMPANIES")
        print(" ")
        print(top_20_companies)

        industry_top_com_dict = top_20_companies.to_dict('records')

        stage_5_table = dgsafe['top_ind_cagr_com']

        print("START INSERT DATA INTO COLLECTION" )
        
        stage_5_table.insert_many(industry_top_com_dict)

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

        one =df[(df.CAGR>= df1[3]) & (df.CAGR <= df1[4])].count()[0]
        two = df[(df.CAGR  > df1[4]) & (df.CAGR <= df1[5])].count()[0]
        three =df[(df.CAGR > df1[5]) & (df.CAGR <= df1[6])].count()[0]
        four = df[(df.CAGR > df1[6]) & (df.CAGR <= df1[7])].count()[0]
        weights = [one,two,three,four]
        label = ['Need more analysis','Moderate','Reasonable performance','Better returns']

        green =  df['Iservice_category']=='Better_returns'
        amber = df['Iservice_category']=='Reasonable_performance'
        red =  df['Iservice_category']=='Moderate'
        black =  df['Iservice_category']=='Need_more_analysis'

        df2=df.Iservice.describe()

        df3=df2.values.tolist()

        one =df[(df.Iservice>= df3[3]) & (df.Iservice <= df3[4])].count()[0]
        two = df[(df.Iservice  > df3[4]) & (df.Iservice <= df3[5])].count()[0]
        three =df[(df.Iservice > df3[5]) & (df.Iservice <= df3[6])].count()[0]
        four = df[(df.Iservice > df3[6]) & (df.Iservice <= df3[7])].count()[0]
        weights = [one,two,three,four]
        label = ['Need more analysis','Moderate','Reasonable performance','Better returns']

        green_report=df[green]

        print(green_report[['REG','NAME','YEAR','INDUSTRY_TYPE','CAGR','Iservice_category']].head(5))

        dataa_dict = green_report.to_dict('records')

        stage_5_table = dgsafe['final_cagr_scores']

        print("START INSERT DATA INTO COLLECTION" )
        
        stage_5_table.insert_many(dataa_dict)
        
        #quit()              
        # df_stats = df.describe()

        # df_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)

        # one = df[(df.CAGR >= df_stats.CAGR[3]) & (df.CAGR <= df_stats.CAGR[4])].count()[0]
        # print("Band 1 : ",one)
        # two = df[(df.CAGR  > df_stats.CAGR[4]) & (df.CAGR <= df_stats.CAGR[5])].count()[0]
        # print("Band 2 : ",two)
        # three =df[(df.CAGR > df_stats.CAGR[5]) & (df.CAGR <= df_stats.CAGR[6])].count()[0]
        # print("Band 3 : ",three)
        # four = df[(df.CAGR > df_stats.CAGR[6]) & (df.CAGR <= (df_stats.CAGR[7]/2))].count()[0]
        # print("Band 4 : ",four)
        # five = df[(df.CAGR > (df_stats.CAGR[7]/2)) & (df.CAGR <= df_stats.CAGR[7])].count()[0]
        # CAGR_mark = df.groupby(['REG','NAME','INDUSTRY_TYPE'], as_index=False)
        # average_cagr = CAGR_mark.agg({'CAGR':'mean'})
        # top_20_companies = average_cagr.sort_values('CAGR', ascending=False).head(5)
        # print("TOP_20_COMPANIES")
        # print(" ")
        # print(top_20_companies)
        # break

        # all_scores_final_dict = all_scores_final.to_dict('records')

        # stage_7_table = dgsafe['all_scores_final']

        # stage_7_table.insert_many(all_scores_final_dict)








