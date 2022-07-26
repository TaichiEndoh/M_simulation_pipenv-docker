#!python Preparing_data_for_training_FIM2.py の内容
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecs
import os
import glob
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import ttk
import pickle


def Rename_the_data():
    path = './input/Training_data_input/*.csv'
    i = 1
    flist = glob.glob(path)
    print(flist)
    for file in flist:
        os.rename(file, "./input/Training_data_input/train_data_inp" + str(i) + '.csv')
        i+=1
    list = glob.glob(path)
    print(list)


#Used inside function ⑤ that reads data
def imp_data_f():
    image_file_path = './input/Training_data_input/train_data_inp1.csv'
    def open_rehabilitation_data(image_file_path_v):
        with codecs.open(image_file_path_v, "r", "Shift-JIS", "ignore") as file:
                dfpp = pd.read_table(file, delimiter=",")
        dfpp_m_rehabilitation = dfpp
        return dfpp_m_rehabilitation
    rehabilitation1=open_rehabilitation_data(image_file_path)
    return rehabilitation1


#①Use Tkinter 
#GUI Function for disease-specific rehabilitation
def tkinter_disease_selection():
    root = Tk()
    root.title('AI Learning Diseases')
    def close_window():
        root.destroy()
    # Frame
    frame1 = ttk.Frame(root, padding=10)
    # Style - Theme
    ttk.Style().theme_use('classic')
    # Label Frame
    label_frame = ttk.Labelframe(
        frame1,
        text='Disease',
        padding=(10),
        style='My.TLabelframe')
    # Radiobutton 1
    v1 = StringVar()
    rb1 = ttk.Radiobutton(
        label_frame,
        text='Disease_Type',
        value='Disease_Type',
        variable=v1)
    button1 = ttk.Button(
        frame1,
        text='OK',
        padding=(20, 5),
        command=lambda :[print("v1=%s" % v1.get()) ,close_window()])
    # Layout
    frame1.grid()
    label_frame.grid(row=0, column=0)
    rb1.grid(row=0, column=0) # LabelFrame
    #rb2.grid(row=0, column=1) # LabelFrame
    button1.grid(row=1, pady=5)
    # Start App
    root.mainloop()
    print("v1.get",v1.get())
    Result=v1.get()
    return Result



def tkinter_left_right():
    root = Tk()
    root.title('Machine Learning Model Selection')
    def close_window():
        root.destroy()
    # Frame
    frame1 = ttk.Frame(root, padding=10)
    # Style - Theme
    ttk.Style().theme_use('classic')
    # Label Frame
    label_frame = ttk.Labelframe(
        frame1,
        text='Options',
        padding=(10),
        style='My.TLabelframe')
    # Radiobutton 1
    v1 = StringVar()
    rb1 = ttk.Radiobutton(
        label_frame,
        text='lightGBM',
        value='lightGBM',
        variable=v1)
    # Radiobutton 2
    rb2 = ttk.Radiobutton(
        label_frame,
        text='XGBoost',
        value='XGBoost',
        variable=v1)
    # Radiobutton 3
    rb3 = ttk.Radiobutton(
        label_frame,
        text='Random_forest',
        value='Rf',
        variable=v1)
    # Radiobutton 2
    rb4 = ttk.Radiobutton(
        label_frame,
        text='SVC',
        value='SVC',
        variable=v1)
    # Button
    button1 = ttk.Button(
        frame1,
        text='OK',
        padding=(20, 5),
        command=lambda :[print("v1=%s" % v1.get()) ,close_window()])
    # Layout
    frame1.grid()
    label_frame.grid(row=0, column=0)
    rb1.grid(row=0, column=0) # LabelFrame
    rb2.grid(row=0, column=1) # LabelFrame
    rb3.grid(row=0, column=0) # LabelFrame
    rb4.grid(row=0, column=1) # LabelFrame
    button1.grid(row=1, pady=5)
    # Start App
    root.mainloop()
    Result2=v1.get()
    return Result2


#③
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
#now_time_name = now_time()


#⑤
def read_data_dummies_gender(disease_f):
    Read_data=imp_data_f()
    all_data_p1=Read_data
    all_data_after_get_dummies_gender=pd.get_dummies(all_data_p1['gender'])
    all_data_after_get_dummies_gender.reset_index()
    all_data_non_gender=all_data_p1.drop(["gender"], axis=1)
    all_data_o = pd.concat([all_data_after_get_dummies_gender, all_data_non_gender], axis=1)
    all_data_out_p=all_data_o.fillna(1)
    df_only=all_data_out_p
    df_filters=df_only[df_only.Disease==disease_f]
    print("df_filters",df_filters)
    df=df_filters.drop(['Disease'], axis=1)
    return df

#⑥
def if_left_right(num,df_Result):
    if num == 'Right':
        df_or=df_Result
        Paralyzed_side_right__left_cerebral_infarction=df_or[df_or["part"] == "L"]
        df2 = Paralyzed_side_right__left_cerebral_infarction.drop(["part"],axis=1)
        return df2

    elif num == 'Left':
        df_or=df_Result
        Paralysis_of_the_left__right_side_of_the_brain_infarction=df_or[df_or['part'] == "R"]
        df2 = Paralysis_of_the_left__right_side_of_the_brain_infarction.drop(['part'],axis=1)
        return df2
    else:
        print("Please select right or left")
#all_data_out = if_left_right(Result)


def erasing_large_numbers(marge_data_no1,c_1,c_2,c_3,c_4,c_5,num):

    taking_out=marge_data_no1[[c_1,c_2,c_3,c_4,c_5]]
    marge_data_intp=marge_data_no1.astype('int')
    marge_data_int=marge_data_intp.drop([c_1,c_2,c_3,c_4,c_5], axis=1)
    marge_data_no2=marge_data_int.where(marge_data_int<num)
    marge_data_no3=marge_data_no2.dropna()
    marge_data_no4=pd.concat([marge_data_no3,taking_out], axis=1)

    return marge_data_no4



def advance_preparation_implementation():
    #①
    Rename_the_data()
    Result_R = 'Right'
    Result_L = 'Left'
    #③
    now_time_name = now_time()
    #④
    Out_Result_pass_name_R = "./learned_model/learning_results/"+now_time_name+Result_R+'output'
    Out_Result_pass_name_L = "./learned_model/learning_results/"+now_time_name+Result_L+'output'
    #④
    Out_Result_pass_name_model_R = "./learned_model/model/"+now_time_name+Result_R+'output'
    Out_Result_pass_name_model_L = "./learned_model/model/"+now_time_name+Result_L+'output'
    #⑤
    df_six=read_data_dummies_gender('Disease_Type')
    #⑥
    all_data_out_R = if_left_right(Result_R,df_six)
    all_data_out_L = if_left_right(Result_L,df_six)
    #⑦
    df_r_R=all_data_out_R
    df_r_L=all_data_out_L

    def meke_target(df_a_or,first_num,Result):
        a=df_a_or[first_num]
        feature_value=df_a_or.drop(['F1','F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
            'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'], axis=1)
        feature_value2=pd.concat([a, feature_value], axis=1)
        merge_data1=feature_value2
        merge_data2=merge_data1.rename(columns={first_num:"target",'W':"woman",'M':"man"})
        merge_data2
        df_t_F1=merge_data2
        merge_data=df_t_F1
        merge_data.columns
        marge_data_out=merge_data
        marge_data_p=marge_data_out.drop(["day"], axis=1)
        print("marge_data_p",marge_data_p)
        marge_data_no1=marge_data_p
        df_bd=marge_data_no1
        with open("./learned_model/learned_data/"+first_num+Result+".pkl", 'wb') as f:
            pickle.dump(df_bd, f)
        return df_bd

    target_All=['F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
        'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19']
    print("target_All",target_All)
    df_F1=meke_target(df_r_R,"F1","right")
    df_F1=meke_target(df_r_L,"F1","Left")
    print("df_F1",df_F1)
    df_F2=meke_target(df_r_R,target_All[0],"right")
    df_F2=meke_target(df_r_L,target_All[0],"Left")
    print("df_F2",df_F2)
    df_F3=meke_target(df_r_R,target_All[1],"right")
    df_F3=meke_target(df_r_L,target_All[1],"Left")
    print("df_F3",df_F3)
    df_F4=meke_target(df_r_R,target_All[2],"right")
    df_F4=meke_target(df_r_L,target_All[2],"Left")
    print("df_F4",df_F4)
    df_F5=meke_target(df_r_R,target_All[3],"right")
    df_F5=meke_target(df_r_L,target_All[3],"Left")
    print("df_F5",df_F5)
    df_F6=meke_target(df_r_R,target_All[4],"right")
    df_F6=meke_target(df_r_L,target_All[4],"Left")
    print("df_F6",df_F6)
    df_F7=meke_target(df_r_R,target_All[5],"right")
    df_F7=meke_target(df_r_L,target_All[5],"Left")
    print("df_F7",df_F7)
    df_F8=meke_target(df_r_R,target_All[6],"right")
    df_F8=meke_target(df_r_L,target_All[6],"Left")
    print("df_F8",df_F8)
    df_F9=meke_target(df_r_R,target_All[7],"right")
    df_F9=meke_target(df_r_L,target_All[7],"Left")
    print("df_F9",df_F9)
    df_F10=meke_target(df_r_R,target_All[8],"right")
    df_F10=meke_target(df_r_L,target_All[8],"Left")
    print("df_F10",df_F10)
    df_F11=meke_target(df_r_R,target_All[9],"right")
    df_F11=meke_target(df_r_L,target_All[9],"Left")
    print("df_F11",df_F11)
    df_F12=meke_target(df_r_R,target_All[10],"right")
    df_F12=meke_target(df_r_L,target_All[10],"Left")
    print("df_F12",df_F12)
    df_F13=meke_target(df_r_R,target_All[11],"right")
    df_F13=meke_target(df_r_L,target_All[11],"Left")
    print("df_F13",df_F13)
    df_F14=meke_target(df_r_R,target_All[12],"right")
    df_F14=meke_target(df_r_L,target_All[12],"Left")
    print("df_F14",df_F14)
    df_F15=meke_target(df_r_R,target_All[13],"right")
    df_F15=meke_target(df_r_L,target_All[13],"Left")
    print("df_F15",df_F15)
    df_F16=meke_target(df_r_R,target_All[14],"right")
    df_F16=meke_target(df_r_L,target_All[14],"Left")
    print("df_F16",df_F16)
    df_F17=meke_target(df_r_R,target_All[15],"right")
    df_F17=meke_target(df_r_L,target_All[15],"Left")
    print("df_F17",df_F17)
    df_F18=meke_target(df_r_R,target_All[16],"right")
    df_F18=meke_target(df_r_L,target_All[16],"Left")
    print("df_F18",df_F18)
    df_F19=meke_target(df_r_R,target_All[17],"right")
    df_F19=meke_target(df_r_L,target_All[17],"Left")

    with open("./learned_model/model/"+Result_R+"learned_model_new_pickled.pkl", 'wb') as f:
        pickle.dump(Out_Result_pass_name_R, f)

    with open("./learned_model/model/"+Result_L+"learned_model_new_pickled.pkl", 'wb') as f:
        pickle.dump(Out_Result_pass_name_L, f)

    with open("./learned_model/model/"+Result_R+"model_save_time_pass.pkl", 'wb') as f:
        pickle.dump(Out_Result_pass_name_model_R, f)
        
    with open("./learned_model/model/pass_name_log/"+now_time_name+Result_R+"model_save_time_pass.pkl", 'wb') as f:
        pickle.dump(Out_Result_pass_name_model_R, f)

    with open("./learned_model/model/"+Result_L+"model_save_time_pass.pkl", 'wb') as f:
        pickle.dump(Out_Result_pass_name_model_L, f)

    with open("./learned_model/model/pass_name_log/"+now_time_name+Result_L+"model_save_time_pass.pkl", 'wb') as f:
        pickle.dump(Out_Result_pass_name_model_L, f)

advance_preparation_implementation()