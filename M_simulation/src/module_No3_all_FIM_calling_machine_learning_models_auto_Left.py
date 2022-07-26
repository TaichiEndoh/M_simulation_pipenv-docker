#python FIM_calling_machine_learning_models.py
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import codecs
import os
import glob
import shap
import pickle
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import ttk


def erasing_large_numbers(marge_data_no1,c_1,c_2,c_3,c_4,c_5,num):

    taking_out=marge_data_no1[[c_1,c_2,c_3,c_4,c_5]]
    marge_data_intp=marge_data_no1.astype('int')
    marge_data_int=marge_data_intp.drop([c_1,c_2,c_3,c_4,c_5], axis=1)
    marge_data_no2=marge_data_int.where(marge_data_int<num)
    marge_data_no3=marge_data_no2.dropna()
    marge_data_no4=pd.concat([marge_data_no3,taking_out], axis=1)

    return marge_data_no4


def Model_Call_Posterior_Prediction():

    c_code = "Shift-JIS"

    target_All=['F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
        'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19']

    path = './learned_model/model/'
    files = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)): 
            files.append(filename)
    print("Check the data in the folder containing the trained machine learning model",files)
    with open("./learned_model/model/"+"Left"+"model_save_time_pass.pkl", 'rb') as f:
        Out_Result_pass_name_model_L = pickle.load(f)
    model_save_time_pass =Out_Result_pass_name_model_L
    Out_Result_pass_name = model_save_time_pass
    Out_Result_tk = model_save_time_pass
    num = 'Left'
    disease = 'Disease_Type'


    def Rename_the_data():
        path = './input/Test_data_input/*.csv'
        i = 1
        flist = glob.glob(path)
        print(flist)
        for file in flist:
            os.rename(file, "./input/Test_data_input/test_data_inp" + str(i) + '.csv')
            i+=1
        list = glob.glob(path)
        print(list)


    image_file_path = './input/Test_data_input/input_data.csv'
    def open_rehabilitation_data(image_file_path_v):
                image_file_path = './input/Test_data_input/input_data.csv'
                image_file_path2 = './input/columns/columns.csv'
                with codecs.open(image_file_path2, "r", c_code, "ignore") as file:
                        dfpp_c = pd.read_table(file, delimiter=",")
                test_columns = dfpp_c
                df = pd.read_csv(image_file_path,encoding=c_code,header=None)
                columns_name =test_columns.columns
                columns_name
                df.columns = columns_name
                dfpp_m_rehabilitation_test = df
                df_only=dfpp_m_rehabilitation_test
                dfpp_m_rehabilitation_p=df_only[df_only.Disease==disease]
                dfpp_m_rehabilitation=dfpp_m_rehabilitation_p
                all_data_p1=dfpp_m_rehabilitation
                all_data_after_get_dummies_gender=pd.get_dummies(all_data_p1['gender'])
                all_data_after_get_dummies_gender.reset_index()
                all_data_non_gender=all_data_p1.drop(['gender'], axis=1)
                all_data_o = pd.concat([all_data_after_get_dummies_gender, all_data_non_gender], axis=1)
                all_data_out=all_data_o
                all_data_out_rename=all_data_out.rename(columns={'W':"woman",'M':"man"})
                marge_data_p1=all_data_out_rename.drop(["day"], axis=1)
                marge_data_p=marge_data_p1.drop(['Disease'], axis=1)
                feature_value_p=marge_data_p.drop(['F1','F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
            'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'], axis=1)

                df=feature_value_p
                def if_left_right(num):
                    if num == 'Right':
                        df2 = df.drop(["part"],axis=1)
                        return df2
                    elif num == 'Left':
                        df_or=df
                        Paralysis_of_the_left__right_side_of_the_brain_infarction=df_or[df_or['part'] == "R"]
                        df2 = Paralysis_of_the_left__right_side_of_the_brain_infarction.drop(["part"],axis=1)
                        return df2
                    else:
                        print("Please select right or left")
                        
                feature_value_p = if_left_right(num)
                feature_value = feature_value_p.drop_duplicates(subset='ID1')
                return feature_value
              
    rehabilitation1p=open_rehabilitation_data(image_file_path)
    rehabilitation1=rehabilitation1p
    
    X_test=rehabilitation1
    def Create_Variables_df(columns_name_df):
        y_testp=rehabilitation1[columns_name_df]
        y_test=y_testp.reset_index(drop=True)
        return y_test
    y_test=Create_Variables_df('Number')
    record_ID=Create_Variables_df('ID1')
    Order_ID=Create_Variables_df('ID2')

    def Machine_Learning_Predictive_Batch(Title):
        filename = Out_Result_pass_name+Title+'model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(X_test, num_iteration=loaded_model.best_iteration)
        print(result)
        y_pred = np.argmax(result, axis=1)
        rezurt_1=pd.DataFrame({Title: y_pred})
        rezurt_1['Number']=y_test
        rezurt_1['ID1']=record_ID
        rezurt_1['ID2']=Order_ID
        importance = pd.DataFrame(loaded_model.feature_importance(importance_type='gain'), columns=['importance'])
        return rezurt_1

    rezurt_F1=Machine_Learning_Predictive_Batch("F1")
    rezurt_F2=Machine_Learning_Predictive_Batch(target_All[0])
    rezurt_F3=Machine_Learning_Predictive_Batch(target_All[1])
    rezurt_F4=Machine_Learning_Predictive_Batch(target_All[2])
    rezurt_F5=Machine_Learning_Predictive_Batch(target_All[3])
    rezurt_F6=Machine_Learning_Predictive_Batch(target_All[4])
    rezurt_F7=Machine_Learning_Predictive_Batch(target_All[5])
    rezurt_F8=Machine_Learning_Predictive_Batch(target_All[6])
    rezurt_F9=Machine_Learning_Predictive_Batch(target_All[7])
    rezurt_F10=Machine_Learning_Predictive_Batch(target_All[8])
    rezurt_F11=Machine_Learning_Predictive_Batch(target_All[9])
    rezurt_F12=Machine_Learning_Predictive_Batch(target_All[10])
    rezurt_F13=Machine_Learning_Predictive_Batch(target_All[11])
    rezurt_F14=Machine_Learning_Predictive_Batch(target_All[12])
    rezurt_F15=Machine_Learning_Predictive_Batch(target_All[13])
    rezurt_F16=Machine_Learning_Predictive_Batch(target_All[14])
    rezurt_F17=Machine_Learning_Predictive_Batch(target_All[15])
    rezurt_F18=Machine_Learning_Predictive_Batch(target_All[16])
    rezurt_F19=Machine_Learning_Predictive_Batch(target_All[17])
    #Combining_the_results
    rezurt_F1_2=pd.merge(rezurt_F1, rezurt_F2, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F3_4=pd.merge(rezurt_F3, rezurt_F4, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F1_4=pd.merge(rezurt_F1_2, rezurt_F3_4, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F5_6=pd.merge(rezurt_F5, rezurt_F6, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F7_8=pd.merge(rezurt_F7, rezurt_F8, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F5_8=pd.merge(rezurt_F5_6, rezurt_F7_8, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F1_8=pd.merge(rezurt_F1_4, rezurt_F5_8, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F9_10=pd.merge(rezurt_F9, rezurt_F10, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F11_12=pd.merge(rezurt_F11, rezurt_F12, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F9_12=pd.merge(rezurt_F9_10, rezurt_F11_12, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F1_12=pd.merge(rezurt_F1_8, rezurt_F9_12, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F13_14=pd.merge(rezurt_F13, rezurt_F14, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F15_16=pd.merge(rezurt_F15, rezurt_F16, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F13_16=pd.merge(rezurt_F13_14, rezurt_F15_16, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F1_16=pd.merge(rezurt_F1_12, rezurt_F13_16, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F17_18=pd.merge(rezurt_F17, rezurt_F18, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F1_18=pd.merge(rezurt_F1_16, rezurt_F17_18, how="inner", on = ['Number','ID1','ID2'])
    rezurt_F1_19p=pd.merge(rezurt_F1_18,rezurt_F19, how="inner", on = ['Number','ID1','ID2'])
    #Get today's date and time.
    import datetime
    dt_now = datetime.datetime.now()
    s=str(dt_now)
    dt_now2=s.replace(':', '_').replace(' ', '_')
    #Erase the letters behind it.
    Comment=dt_now2[:16]
    print(Comment)
    outname=str(Comment)
    print(rezurt_F1_19p)
    rezurt_F1_19=rezurt_F1_19p.reindex(columns=['ID1','ID2','Number','F1',"F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19"])
    print(rezurt_F1_19)
    rezurt_F1_19.to_csv(r""+"./learned_model"+'/output_log_data/'+"left_rezurt_1_19_"+outname+'.csv', encoding = 'shift-jis')


    def now_time():
	    import datetime
	    dt_now = datetime.datetime.now()
	    s=str(dt_now)
	    dt_now2=s.replace(':', '_').replace(' ', '_')
	    #Erase the letters behind it.
	    Comment=dt_now2[:16]
	    print(Comment)
	    outname=str(Comment)
	    return outname


    now_time_name = now_time()
    image_file_path = './input/Test_data_input/input_data.csv'
    with codecs.open(image_file_path, "r", c_code, "ignore") as file:
                        dfpp = pd.read_table(file, delimiter=",")
    dfp_input = dfpp
    dfp_input.to_csv(r""+"./learned_model"+'/input_log_data/'+"use_left_inp_"+now_time_name+'.csv', encoding = 'shift-jis')
    return rezurt_F1_19

#left_rezurt_F1_19 = Model_Call_Posterior_Prediction()
