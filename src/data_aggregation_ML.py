import os
import json
import glob
from re import X
from turtle import right
import pandas as pd   
import pathlib 


### previously set parameters depending on folder structure
dataset = "data_kiel"

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


def concat_df_LR(side):
    # define left/right side as string
    files = glob.glob(aggregated_data_path + side +"_foot_core_params_*" + "*.csv")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df.to_csv(aggregated_data_path + "all_"+side + "_parameters.csv", index = False)


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
                    df = pd.read_csv(files[0])
                    df["method"] = method
                    df["subject"] = subject
                    df.to_csv(aggregated_data_path + side +"_foot_core_params_" + method +"_" + subject + ".csv", index = False)


############################## MAIN ###########################
add_column_LR( "right", subjects, methods)
add_column_LR( "left", subjects, methods)
concat_df_LR("left")
concat_df_LR("right")

