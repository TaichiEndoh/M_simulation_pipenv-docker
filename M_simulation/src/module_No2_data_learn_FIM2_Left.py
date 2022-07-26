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


def Learning_and_Model_Saving():
    
    c_code = "Shift-JIS"

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


    def Machine_Learning_Model_Creation(Title):
        with open("./learned_model/learned_data/"+Title+"Left.pkl", 'rb') as f:merge_data2 = pickle.load(f)
        X = merge_data2.drop("target",axis=1).values
        y_pre=merge_data2
        df_pre=y_pre
        df_pre.loc[df_pre['target'] > 9, 'target'] = 7
        y = df_pre["target"].values
        y = merge_data2["target"].values
        columns_name = merge_data2.drop("target",axis=1).columns
        def Test_data_and_training_data_split(df,X,Y):
                    N_train = int(len(df) * 0.8)
                    N_test = len(df) - N_train
                    X_train, X_test, y_train, y_test = \
                        train_test_split(X, Y, test_size=N_test,shuffle=False,random_state=42)
                    return X_train, X_test, y_train, y_test
        # Execute a function that separates data for training and data for testing.
        X_train, X_test, y_train, y_test = Test_data_and_training_data_split(merge_data2,X,y)
        X_train = pd.DataFrame(X_train, columns=columns_name)
        X_test = pd.DataFrame(X_test, columns=columns_name)
        X_test_df = pd.DataFrame(X_test)
        y_test_df = pd.DataFrame(y_test)
        test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
        test_df=test_dfp.rename(columns={0:"target"})
        y_trainp = pd.DataFrame(y_train)
        X_trainp = pd.DataFrame(X_train)
        train=pd.concat([y_trainp, X_trainp], axis=1)
        merge_data_p=train.rename(columns={0:"target"})
        X = merge_data_p.drop("target",axis=1).values
        y = merge_data_p["target"].values
        columns_name = merge_data_p.drop("target",axis=1).columns
        def Test_data_and_training_data_split(df,X,Y):
                    N_train = int(len(df) * 0.80)
                    N_test = len(df) - N_train
                    X_train, X_test, y_train, y_test = \
                        train_test_split(X, Y, test_size=N_test,random_state=42)
                    return X_train, X_test, y_train, y_test
        # Execute a function that separates the data for training from the data for validation.
        X_train,X_val, y_train,y_val = Test_data_and_training_data_split(merge_data_p,X,y)
        X_train = pd.DataFrame(X_train, columns=columns_name)
        X_val = pd.DataFrame(X_val, columns=columns_name)
        #training verification Combine test data vertically
        y_trainp = pd.DataFrame(y_train)
        X_trainp = pd.DataFrame(X_train)
        train=pd.concat([y_trainp, X_trainp], axis=1)
        y_valp = pd.DataFrame(y_val)
        X_valp = pd.DataFrame(X_val)
        val=pd.concat([y_valp, X_valp], axis=1)
        train_vol=pd.concat([train, val])
        order_of_things=train_vol.rename(columns={0:"target"})
        X_test_df = pd.DataFrame(X_test)
        y_test_df = pd.DataFrame(y_test)
        test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
        test_df=test_dfp.rename(columns={0:"target"})
        marge_data_out=pd.concat([order_of_things, test_df])


        print("train shape", X_train.shape)
        print("test shape", X_test.shape)
        print("Xtest", X_test)
        print("validation shape", X_val.shape)
        print("y_train shape", y_train.shape)
        print("y_test shape", y_test.shape)
        print("y_validation shape", y_val.shape)
        print("y_test describe",y_test_df.describe())
        print("ytest", y_test)
        print("not_ y_test describe",(~y_test_df.duplicated()).sum())
        print("y_test_df.duplicated().sum()",y_test_df.duplicated().sum())

        import lightgbm as lgb
        train = lgb.Dataset(X_train, label=y_train)
        valid = lgb.Dataset(X_val, label=y_val)
        params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': 8,
        }
        model = lgb.train(params,
                        train,
                        valid_sets=valid,
                        num_boost_round=10000,
                        early_stopping_rounds=100)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        print(model)
        
        
        filename = model_save_time_pass+Title+'model.sav'
        pickle.dump(model, open(filename, 'wb'))


        y_pred = np.argmax(y_pred, axis=1)


        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import accuracy_score


        colnames = [Title+"予測値1",Title+"予測値2",Title+"予測値3",Title+"予測値4",Title+"予測値5",Title+"予測値6",Title+"予測値7"]
        index=["実際1", "実際2", "実際3","実際4", "実際5", "実際6", "実際7"]
        result_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred))


        class_accuracy = [(result_matrix[i][i]/result_matrix[i].sum())*1 for i in range(len(result_matrix))]
        result_matrix["class_accuracy"] = class_accuracy
        print("class_accuracy",class_accuracy)
        print("result_matrix",result_matrix)


        data_dir_o=Out_Result_pass_name+"/"
        if not os.path.exists(data_dir_o):
                os.mkdir(data_dir_o)
        print(data_dir_o+"Folder created")


        print(accuracy_score(y_test, y_pred))
        accuracy_all=accuracy_score(y_test, y_pred)
        result_matrix[Title+"accuracy_all"]=accuracy_all


        kappa = cohen_kappa_score(y_test,y_pred)
        print("kappa score:",kappa)
        result_matrix[Title+"kappa score"]=kappa
        result_matrix2=result_matrix.rename(columns={0:Title+"予測値1",1:Title+"予測値2",2:Title+"予測値3",3:Title+"予測値4",4:Title+"予測値5",5:Title+"予測値6",6:Title+"予測値7"}, index={0:"実際1",1:"実際2", 2:"実際3",3:"実際4", 4:"実際5",5: "実際6",6: "実際7"})
        result_matrix2.to_csv(r""+data_dir_o+Title+"result_matrix.csv", encoding = 'shift-jis')


        importance = pd.DataFrame(model.feature_importance(importance_type='gain'), columns=[Title+'importance'])
        column_list=merge_data2.drop(["target"], axis=1)
        importance[Title+"columns"] =list(column_list.columns)
        i_df=importance.sort_values(by=Title+'importance',ascending=False)
        i_df.to_csv(r""+data_dir_o+Title+"importance.csv", encoding = 'shift-jis')


    Out_Result_pass_name,model_save_time_pass = open_pass_pickle()


    target_All=['F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
        'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19']
    Machine_Learning_Model_Creation("F1")
    Machine_Learning_Model_Creation(target_All[0])
    Machine_Learning_Model_Creation(target_All[1])
    Machine_Learning_Model_Creation(target_All[2])
    Machine_Learning_Model_Creation(target_All[3])
    Machine_Learning_Model_Creation(target_All[4])
    Machine_Learning_Model_Creation(target_All[5])
    Machine_Learning_Model_Creation(target_All[6])
    Machine_Learning_Model_Creation(target_All[7])
    Machine_Learning_Model_Creation(target_All[8])
    Machine_Learning_Model_Creation(target_All[9])
    Machine_Learning_Model_Creation(target_All[10])
    Machine_Learning_Model_Creation(target_All[11])
    Machine_Learning_Model_Creation(target_All[12])
    Machine_Learning_Model_Creation(target_All[13])
    Machine_Learning_Model_Creation(target_All[14])
    Machine_Learning_Model_Creation(target_All[15])
    Machine_Learning_Model_Creation(target_All[16])
    Machine_Learning_Model_Creation(target_All[17])


    def read_result_csv(Title):
        data_dir_o=Out_Result_pass_name+"/"
        image_file_path = data_dir_o+Title+"result_matrix.csv"
        with codecs.open(image_file_path, "r", c_code, "ignore") as file:
                        dfpp = pd.read_table(file, delimiter=",")
        result_csv = dfpp
        return result_csv


    rezurt_F1=read_result_csv("F1")
    rezurt_F2=read_result_csv(target_All[0])
    rezurt_F3=read_result_csv(target_All[1])
    rezurt_F4=read_result_csv(target_All[2])
    rezurt_F5=read_result_csv(target_All[3])
    rezurt_F6=read_result_csv(target_All[4])
    rezurt_F7=read_result_csv(target_All[5])
    rezurt_F8=read_result_csv(target_All[6])
    rezurt_F9=read_result_csv(target_All[7])
    rezurt_F10=read_result_csv(target_All[8])
    rezurt_F11=read_result_csv(target_All[9])
    rezurt_F12=read_result_csv(target_All[10])
    rezurt_F13=read_result_csv(target_All[11])
    rezurt_F14=read_result_csv(target_All[12])
    rezurt_F15=read_result_csv(target_All[13])
    rezurt_F16=read_result_csv(target_All[14])
    rezurt_F17=read_result_csv(target_All[15])
    rezurt_F18=read_result_csv(target_All[16])
    rezurt_F19=read_result_csv(target_All[17])

    #Combining_the_results
    rezurt_F1_2=pd.concat([rezurt_F1, rezurt_F2], axis=1)
    rezurt_F3_4=pd.concat([rezurt_F3, rezurt_F4], axis=1)
    rezurt_F1_4=pd.concat([rezurt_F1_2, rezurt_F3_4], axis=1)
    rezurt_F5_6=pd.concat([rezurt_F5, rezurt_F6], axis=1)
    rezurt_F7_8=pd.concat([rezurt_F7, rezurt_F8], axis=1)
    rezurt_F5_8=pd.concat([rezurt_F5_6, rezurt_F7_8], axis=1)
    rezurt_F1_8=pd.concat([rezurt_F1_4, rezurt_F5_8], axis=1)
    rezurt_F9_10=pd.concat([rezurt_F9, rezurt_F10], axis=1)
    rezurt_F11_12=pd.concat([rezurt_F11, rezurt_F12], axis=1)
    rezurt_F9_12=pd.concat([rezurt_F9_10, rezurt_F11_12], axis=1)
    rezurt_F1_12=pd.concat([rezurt_F1_8, rezurt_F9_12], axis=1)
    rezurt_F13_14=pd.concat([rezurt_F13, rezurt_F14], axis=1)
    rezurt_F15_16=pd.concat([rezurt_F15, rezurt_F16], axis=1)
    rezurt_F13_16=pd.concat([rezurt_F13_14, rezurt_F15_16], axis=1)
    rezurt_F1_16=pd.concat([rezurt_F1_12, rezurt_F13_16], axis=1)
    rezurt_F17_18=pd.concat([rezurt_F17, rezurt_F18], axis=1)
    rezurt_F1_18=pd.concat([rezurt_F1_16, rezurt_F17_18], axis=1)
    rezurt_F1_19=pd.concat([rezurt_F1_18,rezurt_F19], axis=1)

    #Save in CSV format
    data_dir_o=Out_Result_pass_name+"/"
    rezurt_F1_19.to_csv(r""+data_dir_o+"result_all_data.csv", encoding = 'shift-jis')


    def read_importance_csv(Title):
        data_dir_o=Out_Result_pass_name+"/"
        image_file_path = data_dir_o+Title+"importance.csv"
        with codecs.open(image_file_path, "r", c_code, "ignore") as file:
                        dfpp = pd.read_table(file, delimiter=",")
        result_csv = dfpp
        return result_csv

    rezurt_F1=read_importance_csv("F1")
    rezurt_F2=read_importance_csv(target_All[0])
    rezurt_F3=read_importance_csv(target_All[1])
    rezurt_F4=read_importance_csv(target_All[2])
    rezurt_F5=read_importance_csv(target_All[3])
    rezurt_F6=read_importance_csv(target_All[4])
    rezurt_F7=read_importance_csv(target_All[5])
    rezurt_F8=read_importance_csv(target_All[6])
    rezurt_F9=read_importance_csv(target_All[7])
    rezurt_F10=read_importance_csv(target_All[8])
    rezurt_F11=read_importance_csv(target_All[9])
    rezurt_F12=read_importance_csv(target_All[10])
    rezurt_F13=read_importance_csv(target_All[11])
    rezurt_F14=read_importance_csv(target_All[12])
    rezurt_F15=read_importance_csv(target_All[13])
    rezurt_F16=read_importance_csv(target_All[14])
    rezurt_F17=read_importance_csv(target_All[15])
    rezurt_F18=read_importance_csv(target_All[16])
    rezurt_F19=read_importance_csv(target_All[17])

    #Combining_the_results
    rezurt_F1_2=pd.concat([rezurt_F1, rezurt_F2], axis=1)
    rezurt_F3_4=pd.concat([rezurt_F3, rezurt_F4], axis=1)
    rezurt_F1_4=pd.concat([rezurt_F1_2, rezurt_F3_4], axis=1)
    rezurt_F5_6=pd.concat([rezurt_F5, rezurt_F6], axis=1)
    rezurt_F7_8=pd.concat([rezurt_F7, rezurt_F8], axis=1)
    rezurt_F5_8=pd.concat([rezurt_F5_6, rezurt_F7_8], axis=1)
    rezurt_F1_8=pd.concat([rezurt_F1_4, rezurt_F5_8], axis=1)
    rezurt_F9_10=pd.concat([rezurt_F9, rezurt_F10], axis=1)
    rezurt_F11_12=pd.concat([rezurt_F11, rezurt_F12], axis=1)
    rezurt_F9_12=pd.concat([rezurt_F9_10, rezurt_F11_12], axis=1)
    rezurt_F1_12=pd.concat([rezurt_F1_8, rezurt_F9_12], axis=1)
    rezurt_F13_14=pd.concat([rezurt_F13, rezurt_F14], axis=1)
    rezurt_F15_16=pd.concat([rezurt_F15, rezurt_F16], axis=1)
    rezurt_F13_16=pd.concat([rezurt_F13_14, rezurt_F15_16], axis=1)
    rezurt_F1_16=pd.concat([rezurt_F1_12, rezurt_F13_16], axis=1)
    rezurt_F17_18=pd.concat([rezurt_F17, rezurt_F18], axis=1)
    rezurt_F1_18=pd.concat([rezurt_F1_16, rezurt_F17_18], axis=1)
    rezurt_F1_19=pd.concat([rezurt_F1_18,rezurt_F19], axis=1)

    #Save in CSV format
    data_dir_o=Out_Result_pass_name+"/"
    rezurt_F1_19.to_csv(r""+data_dir_o+"importance_all_data.csv", encoding = 'shift-jis')


Learning_and_Model_Saving()