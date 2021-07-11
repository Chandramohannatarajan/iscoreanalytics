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
import pandas as pd

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

        #05507387 00002065 04967001 financial and insurance activities
        #data_from_db = list(mainCollection.find({"INDUSTRY_TYPE": "financial and insurance activities"},{"_id":0}))
        data_from_db = list(mainCollection.find({"REG": {"$in" :  ["00002404"]}},{"_id":0}))
        #data_from_db = list(mainCollection.find({"INDUSTRY_TYPE": industry},{"_id":0}))
        # queryData = list(data_from_db)

        if len(data_from_db) == 0:
                continue

        data_cagr_root = pd.DataFrame.from_dict(data_from_db)
        print("pass")

        data1 = data_cagr_root[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']]

        data2 = data1.sort_values(by="YEAR",ascending=True)

        data = data2.drop_duplicates(subset=['NAME','YEAR'], keep='first')

        try:
            data['INDUSTRY_TYPE'].fillna('misc', inplace=True)
        except:
            pass

        try:
            df_all_types = data['INDUSTRY_TYPE'].unique().tolist()
        except:
            pass

        try:
            data['SIC07'].fillna('unknown', inplace=True)
        except:
            pass

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

        data_cagr = data[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR']]

        #data_cagr["YEAR"] = data_cagr["YEAR"].astype(str).astype(int)
        #data = data[data['YEAR'] != 2021]
        #data_cagr["RETAINED_PROFITS"] = data_cagr["RETAINED_PROFITS"].astype(float).astype(int)
        data.head()

        # list_dataframes = []
        # for k, v in data.groupby('NAME'):
        #     if v.shape[0] >= 3:
        #         list_dataframes.append(v)
        
        list_dataframes = [v for k, v in data.groupby('NAME')]
        #print(list_dataframes)
        print("dataframe list ",len(list_dataframes))
        if len(list_dataframes) == 0:
            continue
        
        
        list_df=[]
        for i in list_dataframes: 
            #print(i)
            r=i.values.tolist()
            #print(r[0][1])
            try:
                print("Error Value ",r[0][-2])
                while int(r[0][-2])<0:
                    if len(r)>0:
                        r.pop(0)
                #print(r)
                #print(" ")
                list_df.append(r)   
            except IndexError:
                pass

        print("len ",len(list_df))
        print(list_df)

        lst_cagr = []
        lst_cagr_percentage = []
        try:
            for u in list_df:
                #if len(u)>1:
                try:
                    for k in range(len(u)-1):
                        Initial_RP = int(u[0][-2])
                        print("Initial_RP :",Initial_RP)
                        Final_RP = int(u[1][-2])
                        print("Final_RP :",Final_RP)
                        Initial_year = int(u[0][-1])
                        print("Initial_year : ",Initial_year)
                        Final_year = int(u[1][-1])
                        print("Final_year :", Final_year)
                        reg_num = (u[0][0])
                        ind_type = (u[0][2])
                        com_name = (u[0][1])
                        # print("****** 1 ****",(u[1][-2]))
                        # print("****** 2 ****",(u[0][-2]))
                        # print("****** 3 ****",(u[1][-1]-u[0][-1]))
                        # if (u[1][-2]) < 0  and (u[0][-2]) < 0:
                        #     print("print both negative")
                        #     CAGR = 0
                        # else:
                        #     #print("both not Negative Condition")
                        #     CAGR = pow((u[1][-2])/(u[0][-2]),(1/(u[1][-1]-u[0][-1])))-1

                        CAGR=pow((u[1][-2])/(u[0][-2]),(1/(u[1][-1]-u[0][-1])))-1
                        #CAGR = CAGR.real
                        #q = (CAGR.real, CAGR.imag)
                        #CAGR = CAGR.real
                        print("CAGR :",CAGR)
                        #print(" ")
                        u.pop(1)
                        lst_cagr.append([reg_num,com_name,ind_type,Final_year,CAGR])
                
                except:
                    pass
        except ZeroDivisionError:
            pass
        
        df = DataFrame (lst_cagr,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])
        
        #df.head()

        df['CAGR'] = pd.concat([df['CAGR'].apply(lambda x: x.real), df['CAGR'].apply(lambda x: x.imag)], axis=1, keys=('R','X'))
        
        #df_cagr_per=DataFrame (lst_cagr_percentage,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])
        #df_cagr_per.head()
        
        #dataa = data_cagr_root.merge(df, on=[ 'REG','NAME','INDUSTRY_TYPE','YEAR'])
        #df.head()

        #df['CAGR'] = df['CAGR'].fillna(0)
        root_data_cagr=pd.merge(data,df, on=['REG','NAME','INDUSTRY_TYPE','YEAR'],how ="outer")

        root_data_cagr.shape

        root_data_cagr['CAGR'] = root_data_cagr['CAGR'].fillna(0)

        cagr_rating = root_data_cagr[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR']]

        df_cagr_col=cagr_rating['CAGR']
        df_stats=df_cagr_col.describe()
        df1 =df_stats.values.tolist()
        
        # cagr_2020 = df.loc[df['YEAR'] == 2020]
        
        # cagr_2020.dtypes
        # #df_cagr_final=dataa['CAGR']

        # #df_stats=df_cagr_final.describe()
        # #df_stats

        # #df.stats=df_stats.values.tolist

        # # df_stats.rename(columns = {"Unnamed: 0" : "Stats"}, inplace = True)
        # # df_stats

        # CAGR_mark = cagr_2020.groupby(['REG','NAME','INDUSTRY_TYPE','YEAR'], as_index=False)

        # average_cagr = CAGR_mark.agg({'CAGR':'mean'})
        # top_20_companies = average_cagr.sort_values('CAGR', ascending=False).head(20)
        # print("TOP_20_COMPANIES")
        # print(" ")
        # print(top_20_companies)

        # industry_top_com_dict = top_20_companies.to_dict('records')

        # #stage_5_table = dgsafe['final_top_cagr']

        # print("START INSERT DATA INTO COLLECTION")
        
        # #stage_5_table.insert_many(industry_top_com_dict)
        cagr_rating_dataframes = [v for k, v in cagr_rating.groupby('NAME')]
        #list_dataframes_cagr = [v for k, v in df.groupby('NAME')]

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
        
        lst_df_final=[]
        for i in cagr_rating_dataframes:
            t=i.values.tolist()
            lst_df_final.append(t)
        lst_df_all=[]
        for i in lst_df_final:
            for j in i:
                lst_df_all.append(j)


        df = DataFrame(lst_df_all,columns=['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR','Istar_CAGR','Istar_CAGR_rating'])

        root_cagr = pd.merge(root_data_cagr,df, on=['REG','NAME','INDUSTRY_TYPE','YEAR','RETAINED_PROFITS','CAGR'],how ="outer")

        root_cagr.shape
        print(root_cagr)
        iscores = root_cagr.to_dict('records')

        #stage_5_table = dgsafe['final_cagr_scores_test_fin']

        print("START INSERT DATA INTO COLLECTION" )
        
        
        #stage_5_table.insert_many(iscores)
        
        quit()              
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







