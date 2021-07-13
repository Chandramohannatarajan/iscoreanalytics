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

from bson.raw_bson import RawBSONDocument
from bson.codec_options import CodecOptions
from pymongo import MongoClient
from datetime import datetime

global absDirName
absDirName = os.path.dirname(os.path.abspath(__file__))

codec_options = CodecOptions(unicode_decode_error_handler='ignore')




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


#making a connection to Mongo client
#client=pym.MongoClient("mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata")


# In[ ]:


#creating database
#db=client['dgdata']


# # ENTRY 1  --> To be queried directly from MongoDB

# In[ ]:


#data_from_db = db.z_score_root_file.find({"INDUSTRY_TYPE":"financial and insurance activities"},{'_id':0})
#data=pd.DataFrame.from_dict(data_from_db)


# In[9]:


#data = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/root_iscore.csv",low_memory=False)


# In[10]:


#data['CAGR'] = data['CAGR'].fillna(0)
#data['Z'] = data['Z'].fillna(0)



if __name__ == "__main__":
    mongoconnection()

    start = datetime.now()
    currentyear = datetime.today().year
    mainCollection = dgsafe.get_collection("i_score_root_file", codec_options=codec_options)
    uniqueIndustryList = list(mainCollection.distinct("INDUSTRY_TYPE"))
    for industry in uniqueIndustryList:
        print("industry " ,industry)

        for year in range (currentyear- 7 , currentyear+1):

            print("year " ,year)

            data_from_db = list(mainCollection.find({"YEAR" : str(year),"INDUSTRY_TYPE": industry},{"_id":0}))
            #data_from_db = list(mainCollection.find({"INDUSTRY_TYPE": industry},{"_id":0}))
            print("pass")
            if len(data_from_db) == 0:
                continue
            
            data = pd.DataFrame.from_dict(data_from_db)
            data.count()

            data = data.sort_values(by="YEAR",ascending=True)


            #data=data.replace([np.nan], 'misc')
            data['INDUSTRY_TYPE'].fillna('misc', inplace=True)

            
            data.drop_duplicates(keep=False,inplace=True)

            #data["YEAR"] = data["YEAR"].astype(str).astype(int)
            data["PRETAX_PROFIT_PERCENTAGE"] = data["PRETAX_PROFIT_PERCENTAGE"].astype(str).astype(float)
            data["CURRENT_RATIO"] = data["CURRENT_RATIO"].astype(str).astype(float)
            data["SALES_PER_NET_WORKING_CAPITAL"] = data["SALES_PER_NET_WORKING_CAPITAL"].astype(str).astype(float)
            data["GEARING_RATIO"] = data["GEARING_RATIO"].astype(str).astype(float)
            data["EQUITY_RATIO"] = data["EQUITY_RATIO"].astype(str).astype(float)
            data["CREDITOR_DAYS"] = data["CREDITOR_DAYS"].astype(str).astype(float)
            data["DEBTOR_DAYS"] = data["DEBTOR_DAYS"].astype(str).astype(float)
            data["LIQUIDITY_TEST"] = data["LIQUIDITY_TEST"].astype(str).astype(float)
            data["RETURN_CAPITAL_EMPLOYED"] = data["RETURN_CAPITAL_EMPLOYED"].astype(str).astype(float)
            data["RETURN_TOTAL_ASSETS"] = data["RETURN_TOTAL_ASSETS"].astype(str).astype(float)
            data["DEBT_EQUITY"] = data["DEBT_EQUITY"].astype(str).astype(float)
            data["RETURN_EQUITY"] = data["RETURN_EQUITY"].astype(str).astype(float)
            data["RETURN_NET_ASSETS"] = data["RETURN_NET_ASSETS"].astype(str).astype(float)
            data["TOTAL_DEBT_RATIO"] = data["TOTAL_DEBT_RATIO"].astype(str).astype(float)

            df_all_types = data['INDUSTRY_TYPE'].unique().tolist()

            print("ind types ")
            print(data)

            conditions0_at = [
                (data['INDUSTRY_TYPE'] == "accommodation and food service activities"),
                (data['INDUSTRY_TYPE'] == "activities of extraterritorial organisations and bodies"),
                (data['INDUSTRY_TYPE'] == "activities of households as employers; undifferentiated goods- and services-producing activities of households for own use"),
                (data['INDUSTRY_TYPE'] == "administrative and support service activities"),
                (data['INDUSTRY_TYPE'] == "agriculture forestry and fishing"),
                (data['INDUSTRY_TYPE'] == "arts, entertainment and recreation"),
                (data['INDUSTRY_TYPE'] == "construction"),
                (data['INDUSTRY_TYPE'] == "education"),
                (data['INDUSTRY_TYPE'] == "electricity, gas, steam and air conditioning supply"),
                (data['INDUSTRY_TYPE'] == "financial and insurance activities"),
                (data['INDUSTRY_TYPE'] == "human health and social work activities"),
                (data['INDUSTRY_TYPE'] == "information and communication"),
                (data['INDUSTRY_TYPE'] == "manufacturing"),
                (data['INDUSTRY_TYPE'] == "mining and quarrying"),
                (data['INDUSTRY_TYPE'] == "other service activities"),
                (data['INDUSTRY_TYPE'] == "professional, scientific and technical activities"),
                (data['INDUSTRY_TYPE'] == "public administration and defence; compulsory social security"),
                (data['INDUSTRY_TYPE'] == "real estate activities"),
                (data['INDUSTRY_TYPE'] == "transportation and storage"),
                (data['INDUSTRY_TYPE'] == "water supply, sewerage, waste management and remediation activities"),
                (data['INDUSTRY_TYPE'] == "wholesale and retail trade; repair of motor vehicles and motorcycles"),
                ]

            values0_at = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']
            data['INDUSTRY_CODE'] = np.select(conditions0_at, values0_at)


            # # ISCORE CALCULATION

            # # MAIN DF FOR ISCORE CALCULATION

            df_all_years = data['YEAR'].unique().tolist()
            #df_all_years

            #print("years ",df_all_years)
            df_all_types = data['INDUSTRY_CODE'].unique().tolist()

            print("df_all_types")
            print(df_all_types)
            for k,value_year in enumerate(df_all_years):
                print("k value ",k)
                print("year val ",value_year)
                #quit()
                for l,value_type in enumerate(df_all_types):
                    print("value type ",value_type)
                    data_year_ind = data[(data['INDUSTRY_CODE'] == value_type) & (data['YEAR'] == value_year)].to_dict('records')
                    year_industry = pd.DataFrame.from_dict(data_year_ind)
                    year_industry = year_industry[['REG', 'NAME', 'INDUSTRY_TYPE','YEAR','PRETAX_PROFIT_PERCENTAGE', 'CURRENT_RATIO',
                    'SALES_PER_NET_WORKING_CAPITAL', 'GEARING_RATIO', 'EQUITY_RATIO',
                    'CREDITOR_DAYS', 'DEBTOR_DAYS', 'LIQUIDITY_TEST',
                    'RETURN_CAPITAL_EMPLOYED', 'RETURN_TOTAL_ASSETS', 'DEBT_EQUITY',
                    'RETURN_EQUITY', 'RETURN_NET_ASSETS', 'TOTAL_DEBT_RATIO']]
                    year_ind_stats=year_industry.describe()
                    print("stats ",year_ind_stats)
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
                    print(year_industry)
                    
                        ##year_industry.to_csv(r'C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore/'+'iscore_'+str(value_year)+'_'+str(value_type)+r'.csv',index = False, na_rep = 'N/A')

                    year_ind = year_industry.to_dict('records')
                    stage_7_table = dgsafe['final_i_scores']
                    stage_7_table.insert_many(year_ind)
                    #quit()

                # import glob
                # os.chdir(r"C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore")
                # extension = 'csv'
                # all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
                # #combine all files in the list
                # iscores = pd.concat([pd.read_csv(f) for f in all_filenames ])
                # #iscore_z_cagr=iscores_z_cagr.drop(['Iservice_category_Iscore'],axis=1)
                # #export to csv
                # iscores.to_csv( "C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore/iscores.csv", index=False)


                # # MERGING ISCORES FILE WITH ALL YEAR FILE

                # x = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/root_iscore.csv",low_memory=False)

                # y = pd.read_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/iscore/iscores.csv")

                # z = pd.merge(x,y,how='left',on=['REGISTRATION_NUMBER', 'COMPANY_NAME', 'INDUSTRY_TYPE', 'YEAR'],suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')


                # z.drop_duplicates(keep=False,inplace=True)

                # z.to_csv("C://Users/44740/Machine learning/DG/Credit_Score_Analysis-master/11062021/all_iscores.csv",index=False)

                # iscores=z.to_dict('records')
                # stage_7_table=db['ISCORES']
                # stage_7_table.insert_many(iscores)

                # z.count()

                # df_all_years=z['YEAR'].unique().tolist()
                # df_all_years


                # df_all_types=z['INDUSTRY_TYPE'].unique().tolist()
                # df_all_types

                # df_all_codes=data['INDUSTRY_CODE'].unique().tolist()
                # df_all_codes

                # df11 = z.loc[(z['YEAR'] == 2013) & (z['INDUSTRY_TYPE'] == 'manufacture of basic pharmaceutical products and pharmaceutical preparations')]
                # df11.head()

                # list_avg=[]
                # list_df=[]
                # for i in df_all_years:
                #     #print(i)
                #     for j in df_all_types:
                #         #print(j)
                #         df11 = (z.loc[(z['YEAR'] == i) & (z['INDUSTRY_TYPE'] == str(j))])
                #         avg_T=((df11['TURNOVER'].sum())/(df11['TURNOVER'].count()))
                #         avg_EBITDA=((df11['EBITDA'].sum())/(df11['EBITDA'].count()))
                #         avg_iscoreall=((df11['Iscore_all'].sum())/(df11['Iscore_all'].count()))
                #         avg_GP=((df11['GROSS_PROFIT'].sum())/(df11['GROSS_PROFIT'].count()))

                #         list_avg.append((i,j,avg_T,avg_EBITDA,avg_iscoreall,avg_GP))
                # df_plot = DataFrame (list_avg,columns=['YEAR','industry_type','avg_T','avg_EBITDA','avg_iscoreall','avg_GP'])
                # list_dataframes= [v for k, v in df_plot.groupby('YEAR')]
                #     #list_df.append(df_plot)
                #     #print(df_plot)
                #     #break


                # list_dataframes[7]

                # z.groupby('INDUSTRY_TYPE')['COMPANY_NAME'].nunique().plot(kind='bar')
                # plt.show()


                # # a simple line plot
                # list_dataframes[3].plot(kind='bar',x='industry_type',y='avg_EBITDA')


                # list_dataframes[3].plot(kind='bar',x='industry_type',y='avg_GP')


                # # gca stands for 'get current axis'
                # ax = plt.gca()

                # list_dataframes[6].plot(kind='bar',x='industry_type',y='avg_T',color='blue',ax=ax)
                # list_dataframes[6].plot(kind='bar',x='industry_type',y='avg_EBITDA', color='red', ax=ax)

                # plt.show()
    quit()   

