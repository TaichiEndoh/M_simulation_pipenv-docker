
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import codecs
import os
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def EDA_all():

    def open_pass_pickle():
            with open('./learned_model/model/Leftlearned_model_new_pickled.pkl', 'rb') as f:
                Out_Result_pass_name_p = pickle.load(f)
            Out_Result = Out_Result_pass_name_p
            Out_Result_pass_name = Out_Result
            print("Out_Result_pass_name",Out_Result_pass_name)

            with open("./learned_model/model/"+"Left"+"model_save_time_pass.pkl", 'rb') as f:
                Out_Result_pass_name_model_L = pickle.load(f)
            model_save_time_pass =Out_Result_pass_name_model_L

            return Out_Result_pass_name,model_save_time_pass

    Out_Result_pass_name,model_save_time_pass = open_pass_pickle()
    def open_pickle(Title):
        with open("./learned_model/learned_data/"+Title+"Left.pkl", 'rb') as f:merge_data2 = pickle.load(f)
        X = merge_data2.drop("target",axis=1).values
        y_pre=merge_data2
        df_pre=y_pre
        df_pre.loc[df_pre['target'] > 9, 'target'] = 7
        y = df_pre["target"].values
        y = merge_data2["target"].values
        return merge_data2

    target_All=['F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
        'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19']
    F1pre=open_pickle("F1")
    F2pre=open_pickle(target_All[0])
    F3pre=open_pickle(target_All[1])
    F4pre=open_pickle(target_All[2])
    F5pre=open_pickle(target_All[3])
    F6pre=open_pickle(target_All[4])
    F7pre=open_pickle(target_All[5])
    F8pre=open_pickle(target_All[6])
    F9pre=open_pickle(target_All[7])
    F10pre=open_pickle(target_All[8])
    F11pre=open_pickle(target_All[9])
    F12pre=open_pickle(target_All[10])
    F13pre=open_pickle(target_All[11])
    F14pre=open_pickle(target_All[12])
    F15pre=open_pickle(target_All[13])
    F16pre=open_pickle(target_All[14])
    F17pre=open_pickle(target_All[15])
    F18pre=open_pickle(target_All[16])
    F19pre=open_pickle(target_All[17])


    def feature_selection_fim(merge_data2):
        X = merge_data2.drop(["target","ID1","ID2","O","Number"],axis=1).values
        y = merge_data2["target"].values
        column_list=merge_data2.drop(["target"], axis=1)
        column_l =list(column_list.columns)
        print('merge_data2:',merge_data2)
        print('X:',X.shape)
        select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=10)
        select.fit(X, y)
        print('Feature ranking by RFF:', select.ranking_)
        X_train_selected = X[:, select.support_]
        print('no1',select.ranking_[0])
        select.ranking_2=list(dict.fromkeys(select.ranking_))
        print('select.ranking_2',select.ranking_2)
        merge_data3=merge_data2[["target","ID1","ID2","O","Number",column_l[select.ranking_2[0]+5],column_l[select.ranking_2[1]+5],column_l[select.ranking_2[2]+5],column_l[select.ranking_2[3]+5],column_l[select.ranking_2[4]+5],column_l[select.ranking_2[5]+5],column_l[select.ranking_2[6]+5],column_l[select.ranking_2[7]+5],column_l[select.ranking_2[8]+5],column_l[select.ranking_2[9]+5]]]
        return merge_data3

    F1pre_selection=feature_selection_fim(F1pre)
    F2pre_selection=feature_selection_fim(F2pre)
    F3pre_selection=feature_selection_fim(F3pre)
    F4pre_selection=feature_selection_fim(F4pre)
    F5pre_selection=feature_selection_fim(F5pre)
    F6pre_selection=feature_selection_fim(F6pre)
    F7pre_selection=feature_selection_fim(F7pre)
    F8pre_selection=feature_selection_fim(F8pre)
    F9pre_selection=feature_selection_fim(F9pre)
    F10pre_selection=feature_selection_fim(F10pre)
    F11pre_selection=feature_selection_fim(F11pre)
    F12pre_selection=feature_selection_fim(F12pre)
    F13pre_selection=feature_selection_fim(F13pre)
    F14pre_selection=feature_selection_fim(F14pre)
    F15pre_selection=feature_selection_fim(F15pre)
    F16pre_selection=feature_selection_fim(F16pre)
    F17pre_selection=feature_selection_fim(F17pre)
    F18pre_selection=feature_selection_fim(F18pre)
    F19pre_selection=feature_selection_fim(F19pre)

    def EDA_Analysis(indexNames,name_T):
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = "MS Gothic"
        data_dir_out="./learned_model/EDA_data"
        if not os.path.exists(data_dir_out):
            os.makedirs(data_dir_out)
            print("make_new")

        data=indexNames

        class_group = data.groupby("target")
        pd.options.display.max_columns = None
        class_group.describe()
        data.hist(figsize=(20,10))
        plt.tight_layout()
        class_group = data.groupby("target")
        class_group["target"].hist(alpha=0.7)
        plt.savefig(data_dir_out+"/"+name_T+"class_group.png")

        plt.figure(figsize=(20,10))
        for n, name in enumerate(data.columns.drop("target")):
            plt.subplot(4,4,n+1)
            class_group[name].hist(alpha=0.7)
            plt.title(name,fontsize=13,x=0, y=0)
            plt.legend([1,2,3,4,5,6,7,8])
        plt.savefig(data_dir_out+"/"+name_T+"evaluation_hist"+"all"+".png")
        plt.figure()

        class_group = data.groupby("target")
        class_group["target"].hist(alpha=0.7)
        plt.legend([1,2,3,4,5,6,7,8])
        plt.savefig(data_dir_out+"/"+name_T+"evaluation_hist.png")
        plt.figure()

        df_corr = df_corr = data.corr()
        print(df_corr)
        sns.heatmap(df_corr)
        data2=data.dropna()
        plt.savefig(data_dir_out+"/"+name_T+"heatmap.png")
        plt.figure()
        
    EDA_Analysis(F1pre_selection,"F1")
    EDA_Analysis(F2pre_selection,target_All[0])
    EDA_Analysis(F3pre_selection,target_All[1])
    EDA_Analysis(F4pre_selection,target_All[2])
    EDA_Analysis(F5pre_selection,target_All[3])
    EDA_Analysis(F6pre_selection,target_All[4])
    EDA_Analysis(F7pre_selection,target_All[5])
    EDA_Analysis(F8pre_selection,target_All[6])
    EDA_Analysis(F9pre_selection,target_All[7])
    EDA_Analysis(F10pre_selection,target_All[8])
    EDA_Analysis(F11pre_selection,target_All[9])
    EDA_Analysis(F12pre_selection,target_All[10])
    EDA_Analysis(F13pre_selection,target_All[11])
    EDA_Analysis(F14pre_selection,target_All[12])
    EDA_Analysis(F15pre_selection,target_All[13])
    EDA_Analysis(F16pre_selection,target_All[14])
    EDA_Analysis(F17pre_selection,target_All[15])
    EDA_Analysis(F18pre_selection,target_All[16])
    EDA_Analysis(F19pre_selection,target_All[17])

EDA_all()