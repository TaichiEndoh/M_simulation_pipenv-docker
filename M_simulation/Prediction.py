import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import schedule
from time import sleep
from src import input_make_data_working
from src import chatwork_m_api
from src import module_No3_all_FIM_calling_machine_learning_models_auto_Left
from src import module_No3_all_FIM_calling_machine_learning_models_auto_Right

def main():
    inp_file_path = './input/Test_data_input/input_data.csv'
    path= './input/Test_data_input'
    files = os.listdir(path)
    files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
    print(files_file)
    #Check if there are files in the folder
    print(os.path.exists(inp_file_path))

    if os.path.exists(inp_file_path):
        while True:
            input_make_data_working.input_data_change()
            left_rezurt_F1_19 = module_No3_all_FIM_calling_machine_learning_models_auto_Left.Model_Call_Posterior_Prediction()
            right_rezurt_F1_19 = module_No3_all_FIM_calling_machine_learning_models_auto_Right.Model_Call_Posterior_Prediction()
            rezurt_F1_19_all_p=pd.concat([left_rezurt_F1_19, right_rezurt_F1_19])
            rezurt_df=rezurt_F1_19_all_p
            rezurt_df1=rezurt_df[rezurt_df['Number'] != 1]
            rezurt_df2=rezurt_df1[rezurt_df1['Number'] != 2]
            rezurt_F1_19_all=rezurt_df2
            rezurt_F1_19_all.to_csv(r""+"./output"+"/Result.csv", encoding = 'shift-jis', index = False)
            #If you want to add API comments, do the following
            #chatwork_m_api.api_chat('We have the data and the periodic run was successfully completed.') 

            def delete_test_imput_data():
             #read imput data
                test_pass_all = './input/Test_data_input/*.csv'
                for file in glob.glob(test_pass_all):
                    os.remove(file)

            delete_test_imput_data()
            #chatwork_m_api.api_chat('Data deleted after periodic execution.')
            break

    else:
            print("No data were available")
            #chatwork_m_api.api_chat('No data were available')

if __name__ == "__main__":
        main()
        
        #If you run the task function every day, it will be under
        #schedule.every().day.at("11:32").do(main)
        #while True:
        #    schedule.run_pending()
        #    sleep(1)