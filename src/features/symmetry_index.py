from curses.ascii import SI
import os
import json
from pydoc import describe
import sys
import pandas as pd
import numpy as np
from scipy.stats import variation
import pathlib 

## previously set parameters depending on folder structure
dataset = "data_sim"

### List of Process files ###
#### Processed datapath for each dataset
with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
processed_base_path = os.path.join(paths[dataset], "processed")
AV_SI_data_path = os.path.join(paths[dataset], "AV_SI/")

# AV_SI data ordner erstellen
ordner = pathlib.Path(AV_SI_data_path)
if not ordner.exists():
    ordner.mkdir()

def aggregate_parameters_from_df(df, select_strides=False, onesided=False):
    '''creates a row of each average and variation for each colummn of the df'''

    print(".", end="")
    df.reset_index(inplace=True)
    # load data and filter out all outliers
    df = df[df.is_outlier != 1]  # filter outliers
    if "turning_interval" in df.columns:
        df = df[df.turning_interval != 1]  # filter turning strides
    if "interrupted" in df.columns:
        df = df[df.interrupted != 1]  # filter interrupted strides
    df_param = df.filter(items=[
        'stride_length',
        'clearance',
        'stride_time',
        'swing_time',
        'stance_time',
        'stance_ratio'
    ])

    # calculate cadence and speed for single foot
    df_param['cadence'] = 120 / df_param['stride_time']  # cadence in (step / min)
    df_param['speed'] = df_param['stride_length'] / df_param['stride_time']

    avg_list = df_param.mean().tolist()
    CV_list = variation(df_param, axis=0).tolist()
    aggregate_list = avg_list + CV_list
    aggregate_params = pd.DataFrame(
        columns=[
            'stride_length_avg',
            'clearance_avg',
            'stride_time_avg',
            'swing_time_avg',
            'stance_time_avg',
            'stance_ratio_avg',
            'cadence_avg',
            'speed_avg',
            'stride_length_CV',
            'clearance_CV',
            'stride_time_CV',
            'swing_time_CV',
            'stance_time_CV',
            'stance_ratio_CV',
            'cadence_CV',
            'speed_CV',
        ])
    aggregate_params.loc[0] = aggregate_list
    #print("aggregated params", aggregate_params)

    all_aggregate_params = aggregate_params

    return all_aggregate_params

def calculate_SI_new(win_df_left, win_df_right):
    """
    calculate symmetry index from average and variation df for left and right foot
    """
    avg_list_left = win_df_left.iloc[:, :8]
    avg_list_right = win_df_right.iloc[:, :8]
    diff_avg = []
    sum_avg = []
    # for name in SI_List:
    for left_col, right_col in zip(avg_list_left, avg_list_right):
            for left_tuple, right_tuple in zip(avg_list_left[left_col], avg_list_right[right_col]):
                value = abs(left_tuple - right_tuple)
                diff_avg.append(value)
            for t in zip(avg_list_left[left_col], avg_list_right[right_col]):
                sum_avg.append(sum(t))
            print("diff_avg_", diff_avg)
            print(len(diff_avg))
            print("sum_avg_", sum_avg)
            print(len(sum_avg))
        
    SI_List = [x / (0.5 * y) for x, y in zip(diff_avg, sum_avg)]

    return SI_List

def construct_windows(df, window_sz=10, window_slide=2):
    """

    Parameters
    ----------
    df: the dataframe
    window_sz: the window size

    Returns list of windowed dataframes
    -------

    """
    windowed_dfs = []
    start_idx = 0
    end_idx = window_sz
    while end_idx <= len(df) - 1:
        windowed_dfs.append(df[start_idx:end_idx])
        start_idx += window_slide
        end_idx += window_slide

    return windowed_dfs

def construct_windowed_df (df, window_sz, window_slide, side):
    ''' creates a windowed df from the gait parameters 
        and calculates the average and variation for each column 
        -> for Symmetry Index use '''
    dat = []
    #for df in df_list:
    windows = construct_windows(df, window_sz=window_sz, window_slide=window_slide)
    dat.extend(windows)
    # aggregate parameters and save it to csv for other methods such as SVM
    #if aggregate_windows:
    agg_dat = [aggregate_parameters_from_df(df) for df in dat]
    all_windows_df = pd.concat(agg_dat)
    all_windows_df.reset_index(drop=True, inplace=True)
    all_windows_df.dropna(inplace=True)  # in case SI parameters are NaN
    #all_windows_df.drop(all_windows_df[all_windows_df.clearances_min_CV > 50000].index, inplace=True)
    
    return all_windows_df

def create_DF_SI(list_SI):
    '''creates the df for further use of the Symmetry Index'''
    d = len(list_SI)/8
    split_list_SI = np.array_split(list_SI, d)
    df_SI = pd.DataFrame (split_list_SI, columns = [
                'stride_length_SI',
                'clearance_SI',
                'stride_time_SI',
                'swing_time_SI',
                'stance_time_SI',
                'stance_ratio_SI',
                'cadence_SI',
                'speed_SI',
            ])

    print(df_SI.describe())
    print(df_SI)
    return df_SI

def merge_df(df1, df2, SI_df):#, score, subject):
    #merges two df with different length, chooses shorter df first and saves it in folder
    print("length df1_:", len(df1))
    print("length df2_:", len(df2))
    print("length SI_df_:", len(SI_df))
    if len(df1) < len(df2):
        mergedf = pd.concat([df1,df2], join='inner', axis = 1)
    else:
        mergedf = pd.concat([df2,df1], join='inner', axis = 1)
    merge_final = pd.concat([mergedf,SI_df], axis = 1)

    # merge_final["score"] = score
    # merge_final["subject"] = subject
    print('Merged data frame:', merge_final)
    print(merge_final.describe())
    print("length merge_:", len(merge_final))
    #merge_final.to_csv(AV_SI_data_path + "AV_SI" + "_parameters.csv", index = True, header=True)
    return merge_final
#all_windows_df.to_csv(AV_SI_data_path + "AV_SI"+ side + "_parameters.csv", index = False)

# ### Main
# df_left = pd.read_csv('/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_sim/aggregated_data/left_foot_core_params_1P_S1.csv')
# df_right = pd.read_csv('/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_sim/aggregated_data/right_foot_core_params_1P_S1.csv')
# # TODO -> get all single LF/RF  from each patient -> see if you can pass Subject + run on without calculating it and keep it in
# win_size = 10
# win_slide = 2
# # df_left = df_left.drop(['ic_time', 'fo_time'], axis=1)
# # df_right = df_right.drop(['ic_time', 'fo_time'], axis=1)
# print(df_left.describe())
# print(df_right.describe())

# win_df_left = construct_windowed_df(df_left, win_size, win_slide, side="left")
# win_df_left = win_df_left.add_suffix('_left')
# win_df_right = construct_windowed_df(df_right, win_size, win_slide, side="right")
# win_df_right = win_df_right.add_suffix('_right')
# print("df_left, ",  win_df_left)
# print("df_right, ",  win_df_right)

# #print(win_df_left.describe())

# Sym_list_1PS1 = calculate_SI_new(win_df_left, win_df_right)
# #print(len(Sym_list_1PS1))
# #print(Sym_list_1PS1)

# df_SI = create_DF_SI(Sym_list_1PS1)
# merge_df(win_df_left, win_df_right, df_SI, score = "1P", subject = "S1")