import os
import glob


inp_file_path = '../input/Test_data_input/input_data.csv'
path= '../input/Test_data_input'

files = os.listdir(path)
files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
print(files_file)

name = files_file[0]

if name == "input_data.csv":
        	print("de-taarimasu")
else:
        	print("de-taarimasenn")
        	print("de-taarimasenn riyuu",os.path.isfile(inp_file_path))
