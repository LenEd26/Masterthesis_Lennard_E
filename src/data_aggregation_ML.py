import os
import json
import glob
from re import M, X
from turtle import right
import pandas as pd   
import pathlib 
from src.features.symmetry_index import construct_windowed_df, create_DF_SI, merge_df
from src.features.symmetry_index import calculate_SI_new
from src.features.symmetry_index import merge_df
from src.features.symmetry_index import create_DF_SI


### previously set parameters depending on folder structure
dataset = "data_sim"
win_size = 10
win_slide = 2

exclude_columns_left = ["stride_length_CV_left","clearance_CV_left","stride_time_CV_left","swing_time_CV_left","stance_time_CV_left","stance_ratio_CV_left","cadence_CV_left","speed_CV_left"]
exclude_columns_right = ["stride_length_CV_right","clearance_CV_right","stride_time_CV_right","swing_time_CV_right","stance_time_CV_right","stance_ratio_CV_right","cadence_CV_right","speed_CV_right"]

### List of Process files ###
#### Processed datapath for each dataset
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
processed_base_path = os.path.join(paths[dataset], "processed")
aggregated_data_path = os.path.join(paths[dataset], "aggregated_data/")
final_data_path = os.path.join(paths[dataset], "final_data/")

# aggregated data ordner erstellen
ordner = pathlib.Path(aggregated_data_path)
if not ordner.exists():
    ordner.mkdir()

ordner = pathlib.Path(final_data_path)
if not ordner.exists():
    ordner.mkdir()

## names of the Subject Files
subjects = [entry for entry in os.listdir(processed_base_path) if os.path.isdir(os.path.join(processed_base_path, entry)) and entry != "pipeline_figures"]
## filter method
methods = set()
for subject in subjects:
    subject_path = os.path.join(processed_base_path, subject)
    for entry in os.listdir(subject_path):
         if os.path.isdir(os.path.join(subject_path, entry)):
            methods.add(entry)

### Adds a column to a dataframe and saves it in different name 
# files -> list of file directories
# name -> list of subject names
# method -> method of gait conduction (e.g. treadmill)


# def concat_df_LR(side):
#     # define left/right side as string
#     files = glob.glob(aggregated_data_path + side +"_foot_core_params_*" + "*.csv")
#     df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
#     df["side"] = side
#     df.to_csv(aggregated_data_path + "all_"+side + "_parameters.csv", index = False)
#     return df

def concat_df_LR():
            files_left = sorted(glob.glob(aggregated_data_path + "left_foot_core_params_*" + "*.csv"))
            print("files_left", files_left)
            files_right = sorted(glob.glob(aggregated_data_path + "right_foot_core_params_*" + "*.csv"))
            print("files_right", files_right)
            for path_left, path_right in zip(files_left, files_right):
                # print("path_left", path_left)
                # print("path_right", path_right)
                df_left = pd.read_csv(path_left)
                # print("df_left:", df_left.describe)
                # print("left columns__", df_left. columns)
                df_right = pd.read_csv(path_right)
                # print("df_right:", df_right.describe)
                # print("right columns__", df_right. columns)
                SI_df = calculate_SI_new(df_left, df_right)
                # print("list SI", len(SI_df))
                SI_df = create_DF_SI(SI_df)
                #print("df_SI:", SI_df.describe)
                df_left = df_left.drop(columns = exclude_columns_left)
                df_right = df_right.drop(columns = exclude_columns_right)
                params_df = merge_df(df_left, df_right, SI_df)
                #df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
                s = df_left["subject"].iat[0] 
                m = df_left["severity"].iat[0] 
                params_df.to_csv(final_data_path + "all_" + s +"_"+ m + "_parameters.csv", index = False)
            return params_df


def add_column_LR(side, subjects, methods):
    '''Takes a list of file paths of LF or RF and adds subject + method column for LF or RF'''
    for subject in subjects:
        for method in methods:
            sub_method_dir = os.path.join(os.path.join(processed_base_path, subject), method)
            if os.path.exists(sub_method_dir):
                files = glob.glob(sub_method_dir + f"/{side}_*.csv", recursive = False)
                if len(files) == 0:
                    print(f"Missing file for {subject} {method}")
                elif len(files) > 1:
                    print("Too many files for case")
                else:
                    #create SI and av df
                    df = pd.read_csv(files[0])
                    #print("dataframe before CW", df)
                    #print("Colnames dataframe before CW", df.columns)
                    win_df = construct_windowed_df(df, win_size, win_slide, side)
                    win_df = win_df.add_suffix("_" + side)
                    win_df["severity"] = method
                    win_df["subject"] = subject
                    #print("Colnames win_dataframe ", win_df.columns)
                    win_df.to_csv(aggregated_data_path + side +"_foot_core_params_" + method +"_" + subject + ".csv", index = False)


############################## MAIN ###########################
add_column_LR( "right", subjects, methods)
add_column_LR( "left", subjects, methods)

### window the df and add symmetry index column?

final_df_ = concat_df_LR()

