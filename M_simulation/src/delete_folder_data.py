import os
import glob

def delete_test_imput_data():
 #read imput data
 test_pass_all = './input/Test_data_input/*.csv'
 for file in glob.glob(test_pass_all):
    os.remove(file)
     
delete_test_imput_data()
