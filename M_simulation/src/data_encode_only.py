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
from module_python import Label_Encoder_delete

def imp_data_f():
    image_file_path = './input/Training_data_input/train_data_inp1.csv'
    def open_rehabilitation_data(image_file_path_v):
        with codecs.open(image_file_path_v, "r", "Shift-JIS", "ignore") as file:
                dfpp = pd.read_table(file, delimiter=",")
        dfpp_m_rehabilitation = dfpp
        return dfpp_m_rehabilitation
    rehabilitation1=open_rehabilitation_data(image_file_path)
    return rehabilitation1

def read_data_dummies_gender(disease_f):
    Read_data=imp_data_f()
    all_data_p1=Read_data
    df=all_data_p1
    return df

