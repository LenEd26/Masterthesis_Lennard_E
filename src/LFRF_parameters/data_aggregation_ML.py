import csv
import os
import json
import glob
import pandas as pd    
#from pathlib import Path


### previously set parameters depending on folder structure
dataset = "data_kiel"
prefixes = ('pp')
method_name = "treadmill"

#### Processed datapath for each dataset
with open(os.path.join(os.path.dirname("Masterthesis_Lennard_E/src/"), '..', 'path.json')) as f:
    paths = json.loads(f.read())
processed_base_path = os.path.join(paths[dataset], "processed")
aggregated_data_path = os.path.join(paths[dataset], "aggregated_data/")

   
## list of all files in folder
all_processed_files = glob.glob(processed_base_path + "/**/*.csv", recursive = True)

processed_files_left = glob.glob(processed_base_path + "/**/left_*.csv", recursive = True)
#print("left")
#print(processed_files_left)

processed_files_right = glob.glob(processed_base_path + "/**/right_*.csv", recursive = True) 
#print("right")
#print(processed_files_right)


## names of the Subject Files
names = glob.glob(processed_base_path + "/*")
#print(names)

## only subject name
names_cut = []
for i in names:
    if i != processed_base_path +"/pipeline_figures":
        names_cut.append(os.path.basename(os.path.normpath(i)))
#print(names_cut)

## filter method
method = glob.glob(processed_base_path + "/*/*")
#print(method)

# for i,j in zip(names_cut, method_cut):
    #path_csv = glob.glob(processed_base_path +"/"+ names_cut[0] +"/"+ method_cut[0] +"/*.csv")
#     if "left" is in path_csv[0]:


## only method name
method_cut = []
for i in method:
    method_cut.append(os.path.basename(os.path.normpath(i)))
    for word in method_cut[:]:
        if word.startswith(prefixes) or word != method_name:
             method_cut.remove(word)

#print(method_cut)


### Adds a column to a dataframe and saves it in different name 
# files -> list of file directories
# name -> list of subject names
# save -> specified into left/right foot
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


###########

# append df_left/right in one big DF_left/right 
def concat_df_LR(site):
    files = glob.glob(aggregated_data_path + site +"_foot_core_params_*" + "*.csv")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df.to_csv(aggregated_data_path + "all_"+site + "_parameters.csv", index = False)


############ main

add_column_LF(processed_files_left, names_cut, method_cut)
add_column_RF(processed_files_right, names_cut, method_cut)
concat_df_LR("left")
concat_df_LR("right")