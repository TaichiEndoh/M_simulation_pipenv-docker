import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecs
import os
import glob
import pickle

def input_data_change():
    c_code = "Shift-JIS"
    image_file_path_1 = './input/Test_data_input/input_data.csv'
    df = pd.read_csv(image_file_path_1,encoding=c_code,header=None)
    predata = df

    columns_pass = './input/columns/columns.csv'
    data_working = './input/columns/data_working.csv'
    with codecs.open(columns_pass, "r", c_code, "ignore") as file:
            dfpp_c = pd.read_table(file, delimiter=",")
    test_columns = dfpp_c

    df_data_working = pd.read_csv(data_working,encoding=c_code,header=None)
    df_v = pd.concat([predata, df_data_working], axis=0)


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


    df_v.to_csv(r""+"./input"+'/Test_data_input/'+"input_data.csv", encoding = 'shift-jis', index = False, header=False)


    with open("./learned_model"+'/input_log_data/'+"use_data_working_inp_"+now_time_name+"_pickled.pkl", 'wb') as f:
        pickle.dump(df_v, f)



def input_data_del():
    c_code = "Shift-JIS"
    image_file_path_1 = './input/Test_data_input/input_data.csv'
    df = pd.read_csv(image_file_path_1,encoding=c_code,header=None)
    predata = df

    columns_pass = './input/columns/columns.csv'
    data_working = './input/columns/data_working.csv'
    with codecs.open(columns_pass, "r", c_code, "ignore") as file:
            dfpp_c = pd.read_table(file, delimiter=",")
    test_columns = dfpp_c

    df_data_working = pd.read_csv(data_working,encoding=c_code,header=None)
    df_v = pd.concat([predata, df_data_working], axis=0)

    rezurt_df1_2=df_v[df_v['ID1'] != 1]
    rezurt_df1=rezurt_df1_2[rezurt_df1_2['ID1'] != 2]
    return rezurt_df1