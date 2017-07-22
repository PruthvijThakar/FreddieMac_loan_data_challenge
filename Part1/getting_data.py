
# coding: utf-8

# In[14]:

import sys
import requests
import os, shutil
import glob
from lxml import html
import os
import sys
import logging 
import shutil 
import zipfile
import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import logging
import requests
import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve

 
# year = 'Q12005'
# username = 'thakar.p@husky.neu.edu'
# password = 'ZKBfc;5P'

def initialize_part2():
    
    if not os.path.exists('historical_files'):
        os.makedirs('historical_files', mode=0o777, exist_ok=False)
        print("Created Directory called historical_files")
    else:
        shutil.rmtree(os.path.join('historical_files')) 
        os.makedirs('historical_files', mode=0o777, exist_ok=False)
        print("Created Directory called historical_files")




    
def initialize_part1():
    
    if not os.path.exists('Sample_zipfiles'):
        os.makedirs('Sample_zipfiles', mode=0o777, exist_ok=False)
        print("Created Directory called Sample_zipfiles")
    else:
        shutil.rmtree(os.path.join('Sample_zipfiles')) 
        os.makedirs('Sample_zipfiles', mode=0o777, exist_ok=False)
        print("Created Directory called Sample_zipfiles")
        
def get_samples_data(username,password):
#   username='sumedhsaraf.s@gmail.com'
#   password='Cmm\Bz^n'

    credentials = {
        'username': username, 
        'password': password
    }
    rs = requests.session()
    link = "https://freddiemac.embs.com/FLoan/secure/auth.php"
    result = rs.post(
        link, 
        data = credentials, 
        headers = dict(referer=link)
    )
    

    link1 = 'https://freddiemac.embs.com/FLoan/Data/download.php'
    tandc={
        "accept":"Yes",
        "action":"acceptTandC",
        "acceptSubmit":"Continue"
        }
    result = rs.post(
        link1, 
        tandc,
        headers = dict(referer = link1)
    )
    
    if "Please log in" not in result.text:
        
        current_page = BeautifulSoup(result.text,"html.parser")

        print("beginning to Download sample files")

        for a in current_page.find_all('a', href=True):
            if "sample" in a['href']:
                url_pattern = 'https://freddiemac.embs.com/FLoan/Data/'+ a['href']
                url_sorted = int(url_pattern[61:65])
                print(url_pattern[54:65])
                if url_sorted > 2004: 
                    req = rs.get(url_pattern,stream=True)
                    with open(os.path.join('Sample_zipfiles',url_pattern[61:65]+'.zip'), 'wb') as f:
                        for block in req.iter_content(1024):
                            f.write(block)
        print("Unzipping sample files")                   
        try:
            for data in os.listdir('Sample_zipfiles'):
                if data.endswith(".zip"): 
                    file_name = 'Sample_zipfiles' +"/"+ data
                    print(file_name)
                    zipfile.ZipFile(file_name).extractall('Sample_zipfiles')
                    os.remove(file_name)
        except Exception as e:
            print((str(e)))
    else:
        print("Invalid Credentials")
        
        
def Merge_Files():
    print("Beginning to merge sample orinial files")
    column_name_orig = ['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units',
                          'occpy_sts','cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty',
                          'prod_type','st', 'prop_type','zipcode','id_loan','loan_purpose', 
                          'orig_loan_term','cnt_borr','seller_name','servicer_name', 'flag_sc']

    files = glob.glob('Sample_zipfiles' + '/sample_orig*.txt')
    i = 0
  
    for f in files:
        if (i == 0):
            df = pd.read_csv(os.path.join('Sample_zipfiles','sample_orig_2005.txt'), sep = '|',names = column_name_orig, low_memory = False)
            df.to_csv('agg_sample_files.csv',index = False)
        else:
            df = pd.read_csv(f, sep = '|',names = column_name_orig, low_memory = False)
            with open('agg_sample_files.csv','a') as agg:
                df.to_csv(agg,index = False,header = False)
        i = i + 1 
    print("All sample orinial files merged to agg_sample_files.csv")    
    print("Beginning to merge sample svcg files")
    column_name_orig_svc = ['id_loan','svcg_cycle','current_upb','delq_sts','loan_age','mths_remng',
                               'repch_flag','flag_mod', 'cd_zero_bal',
                               'dt_zero_bal','current_int_rt','non_int_brng_upb','dt_lst_pi','mi_recoveries',
                               'net_sale_proceeds','non_mi_recoveries','expenses', 'legal_costs',
                               'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost']

    files_svcg = glob.glob('Sample_zipfiles' + '/sample_svcg*.txt')
    j = 0
    for f in files_svcg:
        if (j == 0):
            df = pd.read_csv(os.path.join('Sample_zipfiles','sample_svcg_2005.txt'), sep = '|',names = column_name_orig_svc, low_memory = False)
            df.to_csv('agg_sample_files_svcg.csv',index = False)
        else:
            df = pd.read_csv(f, sep = '|',names = column_name_orig, low_memory = False)
            with open('agg_sample_files_svcg.csv','a') as agg:
                df.to_csv(agg,index = False,header = False)
        j = j + 1    

    print("All sample svgc files merged to agg_sample_files_svcg.csv")         
        

        
def get_next_quarter_name(q):
    quarter = {}
    quarter['Q1'] = 'Q2'
    quarter['Q2'] = 'Q3'
    quarter['Q3'] = 'Q4'
    quarter['Q4'] = 'Q1'

    if (((q)[0:2]) == (quarter['Q3'] )):
        year_num = int((q)[2:6]) + 1
        next_year = quarter[(q)[0:2]]  + str(year_num)
    else:
        year_num = int((q)[2:6]) 
        next_year =  quarter[(q)[0:2]] +str(year_num)
        
    print("Next Quarter: ", next_year)
    return next_year         
        
def Download_specific_File(year,username,password):    
#     year = 'Q22006'
#     username='sumedhsaraf.s@gmail.com'
#     password='Cmm\Bz^n'

    credentials = {
        'username': username, 
        'password': password
    }
    rs = requests.session()
    link = "https://freddiemac.embs.com/FLoan/secure/auth.php"
    result = rs.post(
        link, 
        data = credentials, 
        headers = dict(referer=link)
    )

    link1 = 'https://freddiemac.embs.com/FLoan/Data/download.php'
    tandc={
        "accept":"Yes",
        "action":"acceptTandC",
        "acceptSubmit":"Continue"
        }
    result = rs.post(
        link1, 
        tandc,
        headers = dict(referer = link1)
    )

    if "Please log in" not in result.text:
        current_page = BeautifulSoup(result.text,"html.parser")

        if not os.path.exists('historical_files'):
            os.makedirs('historical_files', mode=0o777, exist_ok=False)

        for a in current_page.find_all('a', href=True):
            if "historical" in a['href']:
                url_pattern = 'https://freddiemac.embs.com/FLoan/Data/'+ a['href']
                url_sorted = (url_pattern[71:77])
                #print(type(url_sorted))
                if url_sorted == str(year):
                    print("Dowloading required files")
                    req = rs.get(url_pattern,stream=True)
                    with open(os.path.join('historical_files',url_pattern[71:77]+'.zip'), 'wb') as f:
                        for block in req.iter_content(1024):
                            f.write(block)
               

        try:
            for data in os.listdir('historical_files'):
                if data.endswith(".zip"): 
                        file_name = 'historical_files' +"/"+ data
                        zipfile.ZipFile(file_name).extractall('historical_files')
                        print(file_name)
                        os.remove(file_name)
        except Exception as e:
            print(error(str(e)))
    else:
        print("Invalid Credentials")

def Download_train_and_test_data(year,username,password):
    Download_specific_File(year,username,password) 
    Download_specific_File(get_next_quarter_name(year),username,password) 
    
def initialize_classification(year):
    train_file = 'historical_data1_time_'+year+'.txt'
    test_file = 'historical_data1_time_'+get_next_quarter_name(str(year))+'.txt'
    
    data_chunks = pd.read_csv(os.path.join('historical_files',train_file),low_memory=False,sep="|", nrows=250000)
    data_chunks1 = pd.read_csv(os.path.join('historical_files',test_file),low_memory=False,sep="|", nrows=250000)
    print(data_chunks.shape())
    print(data_chunks.shape1())
    
def compute_Matrix(data_chunks,data_chunks1):
    y_train = pd.DataFrame(data_chunks['new_delinq'])
    y_test = pd.DataFrame(data_chunks1['new_delinq'])
    x_train_raw = data_chunks.drop('new_delinq',axis = 1)
    x_train = preprocessing.minmax_scale(x_train_raw)
    y_train_raw = data_chunks1.drop('new_delinq',axis = 1)
    x_test = preprocessing.minmax_scale(y_train_raw)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_train_estimated = logreg.predict(x_train)
    y_test_estimated = logreg.predict(x_test)
    conf_mat = metrics.confusion_matrix(y_test_estimated,y_test)
    print(conf_mat)
    No_of_actual_delq = conf_mat[1][0] + conf_mat[1][1]
    No_of_pred_delq = conf_mat[0][1] + conf_mat[1][1]
    No_of_records = conf_mat[0][1] + conf_mat[1][1] + conf_mat[1][0] + conf_mat[0][0]
    No_of_delq_properly_classified = conf_mat[1][1]
    No_of_nonDelq_improperly_classified_as_delq = conf_mat[0][1]
    output = {'No_of_actual_delq':No_of_actual_delq,
           'No_of_pred_delq':No_of_pred_delq,
           'No_of_records':No_of_records,
           'No_of_delq_properly_classified':No_of_delq_properly_classified,
           'No_of_nonDelq_improperly_classified_as_delq':No_of_nonDelq_improperly_classified_as_delq}
    print(output) 


def clean_data(data_chunks):
    data_chunks.columns = ['id_loan','svcg_cycle','current_upb','delq_sts','loan_age','mths_remng',
                           'repch_flag','flag_mod', 'cd_zero_bal',
                           'dt_zero_bal','current_int_rt','non_int_brng_upb','dt_lst_pi','mi_recoveries',
                           'net_sale_proceeds','non_mi_recoveries','expenses', 'legal_costs',
                           'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost']

    for i, row in data_chunks.iterrows():
            data_chunks.set_value(i,'svcg_cycle',((int(str(row['svcg_cycle'])[:4]) - 1990)*12*30 + (int(str(row['svcg_cycle'])[4:6])*30)))

    data_chunks.delq_sts.replace('R', 1 , inplace = True)
    data_chunks.delq_sts.replace('XX', 1 , inplace = True)
    data_chunks.delq_sts = data_chunks.delq_sts.astype('float64')     


    data_chunks.repch_flag.replace(np.nan, 2 , inplace = True)
    data_chunks.repch_flag.replace('N', 0 , inplace = True)
    data_chunks.repch_flag.replace('Y', 1 , inplace = True)
    data_chunks.repch_flag = data_chunks.repch_flag.astype('float64')

    data_chunks.flag_mod.replace(np.nan,0,inplace = True)
    data_chunks.flag_mod.replace('Y',1,inplace = True)
    data_chunks.flag_mod = data_chunks.flag_mod.astype('float64')

    data_chunks.cd_zero_bal.replace(np.nan,0,inplace = True)
    data_chunks.cd_zero_bal = data_chunks.cd_zero_bal.astype('float64')

    data_chunks.dt_zero_bal.replace(np.nan,0,inplace = True)
    data_chunks.dt_zero_bal = data_chunks.dt_zero_bal.astype('int64')

    for i, row in data_chunks.iterrows():
        if (row['dt_zero_bal'] > 0):
            data_chunks.set_value(i,'dt_zero_bal',((int(str(row['dt_zero_bal'])[:4]) - 1990)*12*30 + (int(str(row['dt_zero_bal'])[4:6])*30)))

    data_chunks.dt_lst_pi.replace(np.nan,0,inplace = True)
    data_chunks.dt_lst_pi = data_chunks.dt_lst_pi.astype('int64')

    for i, row in data_chunks.iterrows():
        if (row['dt_lst_pi'] > 0):
            data_chunks.set_value(i,'dt_lst_pi',((int(str(row['dt_lst_pi'])[:4]) - 1990)*12*30 + (int(str(row['dt_lst_pi'])[4:6])*30)))

    data_chunks.mi_recoveries.replace(np.nan,0,inplace = True)	

    data_chunks.non_mi_recoveries.replace(np.nan, 0,inplace = True)

    data_chunks.net_sale_proceeds.replace(np.nan, 0,inplace = True)
    data_chunks.net_sale_proceeds.replace('C', 1, inplace = True)
    data_chunks.net_sale_proceeds.replace('U', 0, inplace = True)
    data_chunks.net_sale_proceeds = data_chunks.net_sale_proceeds.astype('float64')

    data_chunks.expenses.replace(np.nan, 0,inplace = True)
    data_chunks.legal_costs.replace(np.nan, 0,inplace = True)
    data_chunks.maint_pres_costs.replace(np.nan, 0,inplace = True)
    data_chunks.taxes_ins_costs.replace(np.nan, 0,inplace = True)
    data_chunks.misc_costs.replace(np.nan, 0,inplace = True)
    data_chunks.actual_loss.replace(np.nan, 0,inplace = True)
    data_chunks.modcost.replace(np.nan, 0,inplace = True)
    data_chunks['new_delinq'] = (data_chunks.delq_sts > 0.0).astype('float64')
    
    df_selected = data_chunks[['svcg_cycle','current_upb','loan_age','mths_remng','repch_flag',
                               'flag_mod','cd_zero_bal','dt_zero_bal','current_int_rt','non_int_brng_upb','new_delinq']]



    return df_selected    

no_of_aruments = len(sys.argv)
print("No of arguments passed",no_of_aruments - 1)
username=''
password=''
part=''
if no_of_aruments > 3:
    username=sys.argv[1]
    password=sys.argv[2]
    part=sys.argv[3]
    if str(part) == 'part1':
        print("Part1 Started")
        initialize_part1()
        get_samples_data(username,password)
        Merge_Files()
        
    elif str(part) == 'part2':
        
        print("Part2 Started")
        year = sys.argv[4]
        print("Initializing")
        initialize_part2()
        print("Downloading Part")
        Download_train_and_test_data(str(year),username,password)
        train_file = 'historical_data1_time_'+year+'.txt'
        test_file = 'historical_data1_time_'+get_next_quarter_name(str(year))+'.txt'
        print("Reading csv")
        data_chunks = pd.read_csv(os.path.join('historical_files',train_file),low_memory=False,sep="|", nrows=250000)
        data_chunks1 = pd.read_csv(os.path.join('historical_files',test_file),low_memory=False,sep="|", nrows=250000)
        data_chunks = clean_data(data_chunks)
        data_chunks1 = clean_data(data_chunks1)
        compute_Matrix(data_chunks,data_chunks1)
    else:
        print("Invalid Input") 

        
elif no_of_aruments == 2:
    print('enter defualt login')
    username='thakar.p@husky.neu.edu'
    password='ZKBfc;5P'
    initialize_part1()
    get_samples_data(username,password) 
    Merge_Files()
    print("Part2 Started")
    year = 'Q12005'
    print("Initializing")
    initialize_part2()
    print("Downloading Part")
    Download_train_and_test_data(str(year),username,password)
    train_file = 'historical_data1_time_'+year+'.txt'
    test_file = 'historical_data1_time_'+get_next_quarter_name(str(year))+'.txt'
    print("Reading csv")
    data_chunks = pd.read_csv(os.path.join('historical_files',train_file),low_memory=False,sep="|", nrows=250000)
    data_chunks1 = pd.read_csv(os.path.join('historical_files',test_file),low_memory=False,sep="|", nrows=250000)
    data_chunks = clean_data(data_chunks)
    data_chunks1 = clean_data(data_chunks1)
    compute_Matrix(data_chunks,data_chunks1)
    
else:
    print("Invalid Input")

