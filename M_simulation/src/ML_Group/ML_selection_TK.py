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
def tkinter_left_right():
    root = Tk()
    root.title('Machine Learning Model Selection')
    def close_window():
        root.destroy()
    # Frame
    frame1 = ttk.Frame(root, padding=20)
    # Style - Theme
    ttk.Style().theme_use('classic')
    # Label Frame
    label_frame = ttk.Labelframe(
        frame1,
        text='selection',
        padding=(18),
        style='My.TLabelframe')
    # Radiobutton 1
    v1 = StringVar()
    rb1 = ttk.Radiobutton(
        label_frame,
        text='lightGBM',
        value='lightGBM',
        variable=v1)
    # Radiobutton 2
    v2 = StringVar()
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
    rb3.grid(row=0, column=2) # LabelFrame
    rb4.grid(row=0, column=3) # LabelFrame
    button1.grid(row=1, pady=5)
    # Start App
    root.mainloop()
    Result2=v1.get()
    return Result2