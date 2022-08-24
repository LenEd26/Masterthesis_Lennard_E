import csv
import os
import json
import glob
from re import X
from turtle import right
import pandas as pd   
import pathlib 
from pathlib import Path


### previously set parameters depending on folder structure
dataset = "data_sim"

if dataset == "data_kiel":
    prefixes = ('pp')
    method_name = "treadmill"

elif dataset == "data_sim":
    prefixes  = ("1P", "2P", "3P", "regular")
    method_name = "N/A"
    subjects = ("S1", "S2")


### Adds a column to a dataframe and saves it in different name 
# files -> list of file directories
# name -> list of subject names
# method -> method of gait conduction (e.g. treadmill)

def add_column_LF(files_LF, name, method):
    for i,j,z  in zip(files_LF, name, method):
        df = pd.read_csv(i)
        df["subject_id"] = j
        df["method"] = z
        df.to_csv(aggregated_data_path + "left_foot_core_params_" + j + ".csv", index = False)


def add_column_RF(files_RF, name, method):
    for i,j,z  in zip(files_RF, name, method):
        df = pd.read_csv(i)
        df["subject_id"] = j
        df["method"] = z
        df.to_csv(aggregated_data_path + "right_foot_core_params_" + j + ".csv", index = False)


def concat_df_LR(side):
    # define left/right side as string
    files = glob.glob(aggregated_data_path + side +"_foot_core_params_*" + "*.csv")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df.to_csv(aggregated_data_path + "all_"+side + "_parameters.csv", index = False)


def add_column_LR(processed_files,side):
    '''Takes a list of file paths of LF or RF and adds subject + method column for LF or RF'''
    for subject in processed_files :
        # only method name
        method_cut = []
        names_cut = [] 
        for j in names:
            if j != processed_base_path +"/pipeline_figures":
                names_cut = os.path.basename(os.path.normpath(j))
                #print(names_cut +"_names cut")
                count = 0
                while count < len(prefixes):
                    for i in method:
                        method_cut= os.path.basename(os.path.normpath(i))
                        #print(method_cut + "_method_cut")
        
                        if method_cut not in prefixes:
                            print(method_cut + "_not in folder structure")
                        elif method_cut in prefixes:
                            df = pd.read_csv(subject)
                            df["method"] = method_cut
                            df["subject"] = names_cut
                            df.to_csv(aggregated_data_path + side +"_foot_core_params_" + method_cut + names_cut + ".csv", index = False)
                    count = count + 1

    
    ### List of Process files ###
#### Processed datapath for each dataset
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
processed_base_path = os.path.join(paths[dataset], "processed")
aggregated_data_path = os.path.join(paths[dataset], "aggregated_data/")

# aggregated data ordner erstellen
ordner = pathlib.Path(aggregated_data_path)
if not ordner.exists():
    ordner.mkdir()

   
## list of all files in folder
all_processed_files = glob.glob(processed_base_path + "/**/*.csv", recursive = True)

processed_files_left = glob.glob(processed_base_path + "/**/left_*.csv", recursive = True)
# print("left")
# print(processed_files_left)

processed_files_right = glob.glob(processed_base_path + "/**/right_*.csv", recursive = True) 
# print("right")
# print(processed_files_right)

## names of the Subject Files
names = glob.glob(processed_base_path + "/*")
# print("names")
# print(names)

## filter method
method = glob.glob(processed_base_path + "/*/*")
# print("method")
# print(method)

############################## MAIN
add_column_LR(processed_files_right, "right")
add_column_LR(processed_files_left, "left")
concat_df_LR("left")
concat_df_LR("right")
    ## only subject name
     
    # for i in names:
        
    #         names_cut = os.path.basename(os.path.normpath(i))
    #         print(names_cut +"_names cut")


    # if method_cut != "":
    #     add_column_LF(processed_files_left, names_cut, method_cut)

## only subject name
# names_cut = []
# for i in names:
#     if i != processed_base_path +"/pipeline_figures":
#         names_cut.append(os.path.basename(os.path.normpath(i)))
# print(names_cut)

# ## only method name
# method_cut = []

# if dataset == "data_kiel":
#     for i in method:
#         method_cut.append(os.path.basename(os.path.normpath(i)))
#         print(method_cut)
#         for word in method_cut[:]:
#             if word.startswith(prefixes) or word != method_name:
#                 method_cut.remove(word)
    
# else:
#     for i in method:
#         method_cut.append(os.path.basename(os.path.normpath(i)))
#         print(method_cut)
#         for word in method_cut[:]:
#             if word not in prefixes: # if word.startswith(prefixes):
#                 method_cut.remove(word)

# print(method_cut)
############ main

# if dataset == "data_kiel":
# ##adding subject number + method used to every df for each subject
#     add_column_LF(processed_files_left, names_cut, method_cut)
#     add_column_RF(processed_files_right, names_cut, method_cut)

# ## creating two bi df for each side 
#     concat_df_LR("left")
#     concat_df_LR("right")


# ## creating two big df for each side 
#     concat_df_LR("left")
#     concat_df_LR("right")