import pymongo as pym  # Interface with Python <---> MongoDB
import pymongo
from pymongo import UpdateOne
from datetime import datetime
from pymongo import MongoClient
from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
from glob import glob
import os
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import figure, winter
from matplotlib import pyplot, use
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
from pandas import DataFrame
import pandas as pd
import glob
import math
import time

start = time.time()
start = time.time()
#get_ipython().run_line_magic('matplotlib', 'inline')
# import seaborn as sns
model = LabelEncoder()
plt.rcParams.update({'font.size': 12})
# global option settings
pd.set_option('display.max_columns', 100)  # show all column names display
pd.set_option('display.max_rows', 100)  # show all rows on display

global absDirName
absDirName = os.path.dirname(os.path.abspath(__file__))

codec_options = CodecOptions(unicode_decode_error_handler='ignore')

# ACCESS MONGODB

# In[2]:

myclient = pymongo.MongoClient(
    "mongodb://dgdataUser:Dg-Data-TD-2021@testapi.datagardener.com:52498/dgdata"
)
dbname = "dgdata"
mydb = myclient[dbname]
# list the collections
# for coll in mydb.list_collection_names():
# print(coll)

# SEARCH FILE IN DIRECTORY

# In[3]:


def os_any_dir_search(file):
    u = []
    for p, n, f in os.walk(os.getcwd()):
        for a in f:
            if a.endswith(file):
                #print("A -->",a)
                #print("P -->",p)
                t = pd.read_csv(p + '/' + file, low_memory=False)
                #print("T -->",t)
                u.append(p + '/' + a)
    return t, u


def mongoconnection():
    with open(os.path.join(absDirName, "constants.json"), "r") as constants:
        global constantsData
        constantsData = json.load(constants)
        print(constantsData)
    #   constantsData = json.load(constants)

    #dgSafeConstants = constantsData["dgSafe"]
    global uri
    #uri = "mongodb://" + dgSafeConstants['username'] + ":" + dgSafeConstants['password'] + "@" + dgSafeConstants["server"] +":" + dgSafeConstants['port'] + "/dgsafe?ssl=false&authSource=dgsafe"
    uri = "mongodb://" + constantsData['username'] + ":" + constantsData['password'] + "@" + \
        "testapi.datagardener.com" + ":" + \
        constantsData['port'] + "/dgsafe?ssl=falses&authSource=dgsafe"
    #data_gardener_liveConstants = constantsData["data_gardener_live"]
    print(uri)
    #uri_dg_live = "mongodb://" + data_gardener_liveConstants['username'] + ":" + data_gardener_liveConstants['password'] + "@" + data_gardener_liveConstants["server"] +":" + data_gardener_liveConstants['port'] + "/data-gardener_Live?ssl=false&authSource=data-gardener_Live"

    client = MongoClient(uri)
    #client1 =  MongoClient(uri)
    global dgsafe
    dgsafe = client["dgsafe"]

    # global dg_live
    # dg_live = client1["data-gardener_Live"]


# # AC01,AC06,INDUSTRIES MASTER TO BE LOOKED UP AND Z SCORE, CAGR AND EBITDA CALCULATED AND APPENDED TO ALL DOCUMENTS FOR FURTHER PROCESSING

def comparindDataFrames(df2,df1):
    return pd.merge(df2, df1, how='outer',
    indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])

if __name__ == "__main__":
    mongoconnection()

    #root_file_path = os_any_dir_search('all_industries_more_scores.csv')[1]
    start = datetime.now()
    currentyear = datetime.today().year

    df_reg = pd.read_csv("Distinct_REG_AC01.txt", header=None, usecols=[0], delimiter="\t", dtype="str")
    
    df_reg = pd.read_csv("/opt/dailyDistinctFiles/Distinct_REG_AC01.txt", header=None, usecols=[0], delimiter="\t", dtype="str")
    

    print(len(list(df_reg[0])))

    df_reg.drop_duplicates(keep="first", inplace=True)

    # print(df_reg.head())
    reg_array = list(df_reg[0])

    # reg_array = ["00216214"]
    batchSize = 100
    numBatches = int(len(reg_array)) + 1
    print(len(reg_array))

    mainCollection = dgsafe.get_collection("cagr_root_file",
                                           codec_options=codec_options)
    """ 
        CAGR CALCULATION START
    """
    for i_batch in range(0, 1):
        print("CAGR CALCULATION PROCESSING FOR BATCH .....", i_batch)
        start = i_batch * batchSize
        end = (i_batch + 1) * batchSize
        batch = reg_array[start:end]
        print(batch)
        ci01 = dgsafe.get_collection("ci01_main_company_information",
                                     codec_options=codec_options)
        query = [
            {
            "$match": {
                "REG": {
                    "$in": batch
                }
            }
        }, {
            "$lookup": {
                "localField": "REG",
                "foreignField": "REG",
                "from": "ac01_statutory_company_accounts",
                "as": "account_details"
            }
        }, {
            "$lookup": {
                "from": "industriesMaster",
                "localField": "SIC07",
                "foreignField": "sic_code",
                "as": "industry"
            }
        }, {
            "$project": {
                "REG": 1,
                "NAME": 1,
                "INC": 1,
                "DIS": 1,
                "ALPHA": 1,
                "SIC07": 1,
                "industry": {
                    "$arrayElemAt": ["$industry", 0]
                },
                "account_details": 1
            }
        }, {
            "$unwind": "$account_details"
        }, {
            "$match": {
                "$and": [{
                    "account_details.AC022": {
                        "$exists": True,
                        "$nin": [0]
                    }
                }, {
                    "account_details.AC049": {
                        "$exists": True,
                        "$nin": [0]
                    }
                }, {
                    "account_details.AC033": {
                        "$exists": True,
                        "$nin": [0]
                    }
                }, {
                    "account_details.AC042": {
                        "$exists": True,
                        "$nin": [0]
                    }
                }]
            }
        }, {
            "$project": {
                "_id": 0,
                "REG": 1,
                "NAME": 1,
                "INC": 1,
                "SIC07": 1,
                "INDUSTRY_TYPE": "$industry.industry_name",
                "DIS": 1,
                "LIQUIDATION": "",
                "ACCOUNT_FROM_DATE": "$account_details.AC001",
                "ACCOUNT_TO_DATE": "$account_details.AC002",
                "WEEKS": "$account_details.AC003",
                "MONTHS": "$account_details.AC004",
                "DATE": {
                    "$dateFromString": {
                        "dateString": "$account_details.AC002",
                        "format": "%d/%m/%Y"
                    }
                },
                "YEAR": {
                    "$dateToString": {
                        "format": "%dd/%MM/%YYYY",
                        "date": "$DATE"
                    }
                },
                "WEEK": "$account_details.AC003",
                "MONTH": "$account_details.AC004",
                "CONSOLIDATED_ACCOUNTS": "$account_details.AC006",
                "ACCOUNTS_FORMAT": "$account_details.AC007",
                "TURNOVER": {
                    "$ifNull": ["$account_details.AC008", 0]
                },
                "EXPORT": {
                    "$ifNull": ["$account_details.AC009", 0]
                },
                "COST_OF_SALES": "$account_details.AC010",
                "GROSS_PROFIT": "$account_details.AC011",
                "WAGES_AND_SALARIES": "$account_details.AC012",
                "OPERATING_PROFITS": {
                    "$ifNull": ["$account_details.AC014", 0]
                },
                "DEPRECIATION": {
                    "$ifNull": ["$account_details.AC015", 0]
                },
                "INTEREST_PAYMENTS": "$account_details.AC017",
                "PRETAX_PROFITS": "$account_details.AC018",
                "TAXATION": "$account_details.AC019",
                "PROFIT_AFTER_TAX": "$account_details.AC020",
                "DIVIDENDS_PAYABLE": "$account_details.AC021",
                "RETAINED_PROFITS": {
                    "$ifNull": ["$account_details.AC022", 0]
                },
                "TANGIBLE_ASSETS": "$account_details.AC023",
                "INTANGIBLE_ASSETS": "$account_details.AC024",
                "TOTAL_FIXED_ASSETS": "$account_details.AC025",
                "TOTAL_CURRENT_ASSETS": "$account_details.AC026",
                "TRADE_DEBTORS": "$account_details.AC027",
                "STOCK": "$account_details.AC028",
                "CASH": "$account_details.AC029",
                "OTHER_CURRENT_ASSETS": "$account_details.AC030",
                "INCREASE_IN_CASH": {
                    "$ifNull": ["$account_details.AC031", 0]
                },
                "MISCELANEOUS_CURRENT_ASSETS": "$account_details.AC032",
                "TOTAL_ASSETS": "$account_details.AC033",
                "TOTAL_CURRENT_LIABILITIES": "$account_details.AC034",
                "TRADE_CREDITORS": "$account_details.AC035",
                "BANK_OVERDRAFT": "$account_details.AC036",
                "OTHER_SHORTTERM_FIN": "$account_details.AC037",
                "MISC_CURRENT_LIABILITIES": "$account_details.AC038",
                "OTHER_LONGTERM_FIN": "$account_details.AC039",
                "TOTAL_LONGTERM_LIAB": {
                    "$ifNull": ["$account_details.AC040", 0]
                },
                "BANK_OD_LTL": "$account_details.AC041",
                "TOTAL_LIAB": "$account_details.AC042",
                "NET_ASSETS": "$account_details.AC043",
                "WORKING_CAPITAL": {
                    "$ifNull": ["$account_details.AC044", 0]
                },
                "PAIDUP_EQUITY": "$account_details.AC045",
                "P_L_ACCOUNT_RESERVE": "$account_details.AC046",
                "SUNDRY_RESERVES": "$account_details.AC047",
                "REVALUATION_RESERVE": "$account_details.AC048",
                "SHAREHOLDER_FUNDS": "$account_details.AC049",
                "NETWORTH": "$account_details.AC050",
                "NET_CASHFLOW_FROM_OPERATIONS": {
                    "$ifNull": ["$account_details.AC051", 0]
                },
                "NET_CASHFLOW_BEFOR_FINANCING": "$account_details.AC052",
                "NET_CASHFLOW_FROM_FINANCING": {
                    "$ifNull": ["$account_details.AC053", 0]
                },
                "CONTINGENT_LIAB": "$account_details.AC054",
                "CAPITAL_EMPLOYED": "$account_details.AC055",
                "EMPLOYEES_COUNT": "$account_details.AC056"
            }
        }, {
            "$project": {
                "_id": 0,
                "REG": 1,
                "NAME": 1,
                "INC": 1,
                "SIC07": 1,
                "INDUSTRY_TYPE": 1,
                "DIS": 1,
                "LIQUIDATION": 1,
                "ACCOUNT_FROM_DATE": 1,
                "ACCOUNT_TO_DATE": 1,
                "WEEKS": 1,
                "MONTHS": 1,
                "YEAR": {
                    "$dateToString": {
                        "format": "%Y",
                        "date": "$DATE"
                    }
                },
                "WEEK": 1,
                "MONTH": 1,
                "CONSOLIDATED_ACCOUNTS": 1,
                "ACCOUNTS_FORMAT": 1,
                "TURNOVER": 1,
                "EXPORT": 1,
                "COST_OF_SALES": 1,
                "GROSS_PROFIT": 1,
                "WAGES_AND_SALARIES": 1,
                "OPERATING_PROFITS": 1,
                "DEPRECIATION": 1,
                "INTEREST_PAYMENTS": 1,
                "PRETAX_PROFITS": 1,
                "TAXATION": 1,
                "PROFIT_AFTER_TAX": 1,
                "DIVIDENDS_PAYABLE": 1,
                "RETAINED_PROFITS": 1,
                "TANGIBLE_ASSETS": 1,
                "INTANGIBLE_ASSETS": 1,
                "TOTAL_FIXED_ASSETS": 1,
                "TOTAL_CURRENT_ASSETS": 1,
                "TRADE_DEBTORS": 1,
                "STOCK": 1,
                "CASH": 1,
                "OTHER_CURRENT_ASSETS": 1,
                "INCREASE_IN_CASH": 1,
                "MISCELANEOUS_CURRENT_ASSETS": 1,
                "TOTAL_ASSETS": 1,
                "TOTAL_CURRENT_LIABILITIES": 1,
                "TRADE_CREDITORS": 1,
                "BANK_OVERDRAFT": 1,
                "OTHER_SHORTTERM_FIN": 1,
                "MISC_CURRENT_LIABILITIES": 1,
                "OTHER_LONGTERM_FIN": 1,
                "TOTAL_LONGTERM_LIAB": 1,
                "BANK_OD_LTL": 1,
                "TOTAL_LIAB": 1,
                "NET_ASSETS": 1,
                "WORKING_CAPITAL": 1,
                "PAIDUP_EQUITY": 1,
                "P_L_ACCOUNT_RESERVE": 1,
                "SUNDRY_RESERVES": 1,
                "REVALUATION_RESERVE": 1,
                "SHAREHOLDER_FUNDS": 1,
                "NETWORTH": 1,
                "NET_CASHFLOW_FROM_OPERATIONS": 1,
                "NET_CASHFLOW_BEFOR_FINANCING": 1,
                "NET_CASHFLOW_FROM_FINANCING": 1,
                "CONTINGENT_LIAB": 1,
                "CAPITAL_EMPLOYED": 1,
                "EMPLOYEES_COUNT": 1
            }
        }]

        data = list(ci01.aggregate(query, allowDiskUse=True))

        finalArray = []
        # print(data)
        new_batch_array = []
        for tempObj in data:
            new_batch_array.append(tempObj["REG"])
            finalArray.append(
                UpdateOne(
                    {
                        "REG": tempObj["REG"],
                        "ACCOUNT_FROM_DATE": tempObj["ACCOUNT_FROM_DATE"],
                        "ACCOUNT_TO_DATE": tempObj["ACCOUNT_TO_DATE"]
                    }, {"$set": tempObj},
                    upsert=True))
            # break
        if len(new_batch_array) == 0:
            continue

        if len(finalArray) > 0:
            result = mainCollection.bulk_write(finalArray)

        # with open("test.json","w") as testFile:
        #     json.dump(data, testFile, indent=2)
        # quit()

        uniqueIndustryList = list(
            mainCollection.distinct("INDUSTRY_TYPE",
                                    {"REG": {
                                        "$in": new_batch_array
                                    }}))
        print("len uniqueIndustryList ", len(uniqueIndustryList))

        # uniqueIndustryList = ["construction"]
        """ 
        """
        # for industry in uniqueIndustryList:
        # for reg in batch:

        data_from_db = list(mainCollection.find({"REG": { "$in" : batch }}, {"_id": 0}))

        if len(data_from_db) == 0:
            continue

        data_cagr_root = pd.DataFrame.from_dict(data_from_db)
        print("pass")

        data1 = data_cagr_root[[
            'REG', 'NAME', 'INDUSTRY_TYPE', 'RETAINED_PROFITS', 'YEAR'
        ]]

        data2 = data1.sort_values(by="YEAR", ascending=True)

        data = data2.drop_duplicates(subset=['NAME', 'YEAR'], keep='first')

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

        data["YEAR"] = data["YEAR"].astype(str).astype(int)



        data_cagr = data[[
            'REG', 'NAME', 'INDUSTRY_TYPE', 'RETAINED_PROFITS', 'YEAR'
        ]]

        data_cagr["YEAR"] = data_cagr["YEAR"].astype(str).astype(int)
        #data = data[data['YEAR'] != 2021]
        #data_cagr["RETAINED_PROFITS"] = data_cagr["RETAINED_PROFITS"].astype(float).astype(int)
        # data.head()

        # list_dataframes = []
        # for k, v in data.groupby('NAME'):
        #     if v.shape[0] >= 3:
        #         list_dataframes.append(v)

        list_dataframes = [v for k, v in data_cagr.groupby('NAME')]
        # print(list_dataframes)
        # print("dataframe list ",list_dataframes)
        if len(list_dataframes) == 0:
            continue

        # quit()
        list_df = []
        for i in list_dataframes:
            # print(i)
            r = i.values.tolist()
            # print(r[0][1])
            try:
                #print("Error Value ",r[0][-2])
                while r[0][-2] < 0:
                    if len(r) > 0:
                        r.pop(0)
                # print(r)
                #print(" ")
                list_df.append(r)
            except IndexError:
                pass

        print("len ", len(list_df))
        # print(list_df)

        lst_cagr = []
        lst_cagr_percentage = []
        try:
            for u in list_df:
                # if len(u)>1:
                try:
                    for k in range(len(u) - 1):
                        Initial_RP = int(u[0][-2])
                        #print("Initial_RP :",Initial_RP)
                        Final_RP = int(u[1][-2])
                        #print("Final_RP :",Final_RP)
                        Initial_year = u[0][-1]
                        #print("Initial_year : ",Initial_year)
                        Final_year = u[1][-1]
                        #print("Final_year :", Final_year)
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

                        CAGR = pow(
                            (u[1][-2]) / (u[0][-2]),
                            (1 / (int(u[1][-1]) - int(u[0][-1])))) - 1
                        #CAGR = CAGR.real
                        #q = (CAGR.real, CAGR.imag)
                        #CAGR = CAGR.real
                        #print("CAGR :",CAGR)
                        #print(" ")
                        u.pop(1)
                        lst_cagr.append([
                            reg_num, com_name, ind_type, Final_year, CAGR
                        ])

                except:
                    pass
        except ZeroDivisionError:
            pass

        df = DataFrame(
            lst_cagr,
            columns=['REG', 'NAME', 'INDUSTRY_TYPE', 'YEAR', 'CAGR'])

        df.head()

        df['CAGR'] = pd.concat([
            df['CAGR'].apply(lambda x: x.real),
            df['CAGR'].apply(lambda x: x.imag)
        ],axis=1, keys=('R', 'X'))

        #df_cagr_per=DataFrame (lst_cagr_percentage,columns=['REG','NAME','INDUSTRY_TYPE','YEAR','CAGR'])
        # df_cagr_per.head()

        #dataa = data_cagr_root.merge(df, on=[ 'REG','NAME','INDUSTRY_TYPE','YEAR'])
        # df.head()

        #df['CAGR'] = df['CAGR'].fillna(0)
        root_data_cagr = pd.merge(
            data,
            df,
            on=['REG', 'NAME', 'INDUSTRY_TYPE', 'YEAR'],
            how="outer")

        # root_data_cagr.shape

        root_data_cagr['CAGR'] = root_data_cagr['CAGR'].fillna(0)

        print("CAGR DF")

        print(root_data_cagr)


        cagr_rating = root_data_cagr[[
            'REG', 'NAME', 'INDUSTRY_TYPE', 'RETAINED_PROFITS', 'YEAR',
            'CAGR'
        ]]

        df_cagr_col = cagr_rating['CAGR']
        df_stats = df_cagr_col.describe()
        df1 = df_stats.values.tolist()

        print(cagr_rating.to_json(orient="records"))

        cagr_rating_json = cagr_rating.to_dict('records')

        # with open("CAGR.json", "w") as outFile:
        #     json.dump(cagr_rating_json, outFile, indent=2)

        stage_5_table = dgsafe['final_cagr_scores']

        print("START INSERT DATA INTO COLLECTION")

        finalArray = []

        for tempObj in cagr_rating_json:
            finalArray.append(
                UpdateOne({
                    "REG": tempObj["REG"],
                    "YEAR": tempObj["YEAR"]
                }, {"$set": tempObj},
                            upsert=True))
        # break

        # stage_5_table.insert_many(cagr_rating_json)
        if len(finalArray) > 0:
            result = stage_5_table.bulk_write(finalArray)

        print("CAGR CALCULATION COMPLETED REG")

    """ 
        CAGR CALCULATION END
    """

    """ 
        ISTAR CALCULATION START
    """

    print("ISTAR CALCULATION STARTS ")


    ind_count=[]

    df = pd.DataFrame(columns=['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR','Istar_CAGR','Istar_CAGR_rating'])


    delta_reg = []

    stage_5_table = dgsafe['final_cagr_scores']

    df_all_typez = list(
            stage_5_table.distinct("INDUSTRY_TYPE"))

    for l,value_type in enumerate(df_all_typez):
        print("Fetcting Data for ... ", value_type)

        value_type = "construction"

        data_industry = list( stage_5_table.find( { "INDUSTRY_TYPE" : value_type } , { "_id" : 0}) )

        print("Data Fetcted for ... ", value_type)

        # ind_type =  cagr_rating['INDUSTRY_TYPE']==str(value_type)

        # cagr_indwise = cagr_rating[ind_type]

        cagr_indwise = pd.DataFrame(data_industry)

        df_old = cagr_indwise

        df_old["Istar_CAGR"] = df_old["Istar_CAGR"].fillna(0)


        print(cagr_indwise.shape)

        # print(cagr_indwise.head())

        # quit()

        cagr_rating_ind=cagr_indwise[['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR']]
        ind_count.append([str(value_type),cagr_rating_ind.shape])
        df2_cagr_col=cagr_rating_ind['CAGR']
        df2_stats=df2_cagr_col.describe()
        df2 =df2_stats.values.tolist()
        cagr_rating_ind_dataframes= [v for k, v in cagr_rating_ind.groupby('NAME')]
        for i in cagr_rating_ind_dataframes:
            #print(i)
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
            #print(i)
            l=i.values.tolist()
            list_all3.append(l)
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
                list_ideal3.append(r)
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
                    conditions5_at = [
                                    (i['Istar_CAGR'][::].sum())==(i['CAGR'][::].sum()),
                                    ((i['Istar_CAGR']==0)&(i['YEAR']==list_all3[0][0][-3])),
                                    (i['Istar_CAGR']==0),
                                    (i['Istar_CAGR']==1),
                                    (i['Istar_CAGR']==2),
                                    (i['Istar_CAGR']==3),
                                    (i['Istar_CAGR']==4),
                                    ]
                    #values3_at = ['Caution','Startup','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
                    values5_at = ['Under_Lens','Bud','Under_Radar','Under_Observation,','Joining_League','Runner','Dynamic']
                    i['Istar_CAGR_rating'] = np.select(conditions5_at, values5_at)
                except IndexError:
                    pass
            else:
                try:
                    conditions6_at = [
                                    ((i['Istar_CAGR']==0)&(i['YEAR']==list_all3[0][0][-3])),
                                    (i['Istar_CAGR']==0)&(i['YEAR']==list_ideal3[0][0][-3]),
                                    (i['Istar_CAGR']==0),
                                    (i['Istar_CAGR']==1),
                                    (i['Istar_CAGR']==2),
                                    (i['Istar_CAGR']==3),
                                    (i['Istar_CAGR']==4),
                                    ]
                    #values3_at = ['Startup','Gearing_up','Negative_growth','Need_more_analysis','Moderate','Reasonable_performance','Better_returns']
                    values6_at = ['Bud','Gearing_up','Under_Radar','Under_Observation','Joining_League','Runner','Dynamic']

                    i['Istar_CAGR_rating'] = np.select(conditions6_at, values6_at)
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
        lst_df2_final=[]
        for i in cagr_rating_ind_dataframes:
            t=i.values.tolist()
            lst_df2_final.append(t)
        lst_df2_all=[]
        for i in lst_df2_final:
            for j in i:
                lst_df2_all.append(j)
        df_ind = DataFrame (lst_df2_all,columns=['REG','NAME','INDUSTRY_TYPE','RETAINED_PROFITS','YEAR','CAGR','Istar_CAGR','Istar_CAGR_rating'])

        df = pd.merge(df, df_ind, how='outer')

        print( "df_ind DF ...... " )

        df_delta = comparindDataFrames( df_ind , df_old)

        delta_reg +=  list(df_delta["REG"])

        df_delta = df_delta.to_dict('records')

        stage_5_table = dgsafe['final_cagr_scores']

        print("START INSERT DATA INTO COLLECTION")

        finalArray = []

        for tempObj in df_delta:
            finalArray.append(
                UpdateOne({
                    "REG": tempObj["REG"],
                    "YEAR": tempObj["YEAR"]
                }, {"$set": tempObj},
                            upsert=True))

        if len(finalArray) > 0:
            result = stage_5_table.bulk_write(finalArray)

        break


    df = pd.DataFrame(delta_reg, columns=["REG"])
    print(df.head())

#    df.to_csv("/dailyDistinctFiles/Distinct_AC01_CAGR.csv", index=False, header=Nones)

    df.to_csv("Distinct_AC01_CAGR.csv", index=False, header=Nones)


    print("complete ind")

    """ 
        ISTAR CALCULATION START
    """
print("complete")
