from curses.ascii import SI
import os
import json
from pydoc import describe
import sys
import pandas as pd
import numpy as np
from scipy.stats import variation

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
    calculate symmetry index from averge and variation df for left and right foot
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
            # print("diff_avg_", diff_avg)
            # print(len(diff_avg))
            # print("sum_avg_", sum_avg)
            # print(len(sum_avg))
        
    SI_List = [x / (0.5 * y) for x, y in zip(diff_avg, sum_avg)]

    return SI_List

def calculate_SI(avg_list_left, avg_list_right):
    """
    calculate symmetry index
    """
    diff_avg = [abs(j - i) for i, j in zip(avg_list_left, avg_list_right)]
    sum_avg = [sum(x) for x in zip(avg_list_left, avg_list_right)]
    SI_list = [x / (0.5 * y) for x, y in zip(diff_avg, sum_avg)]
    return SI_list

def get_SI_series(SI_list) -> pd.Series:
    return pd.Series(
        data=SI_list,
        index=[
            'stride_length_SI',
            'clearance_SI',
            'stride_time_SI',
            'swing_time_SI',
            'stance_time_SI',
            'stance_ratio_SI',
            'cadence_SI',
            'speed_SI',
        ]
    )

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

def construct_windowed_df (df, window_sz, window_slide):
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

def SI_LFRF(left, right):
    """ 
    for seperate windowed Dataframes of LF and RF parameters to calculate a SI 
    creates a df with the SI of the Subject
    """
    for row1, row2 in zip(left[:], right[:]):
        SI = calculate_SI(row1, row2)
    return SI

def create_DF_SI(list_SI):
    '''creates the df for further use of the Symmetry Index'''
    split_list_SI = np.array_split(list_SI,13)
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


### Main
df_left = pd.read_csv('/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_sim/aggregated_data/left_foot_core_params_1P_S1.csv')
df_right = pd.read_csv('/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_sim/aggregated_data/right_foot_core_params_1P_S1.csv')
win_size = 10
win_slide = 2
# df_left = df_left.drop(['ic_time', 'fo_time'], axis=1)
# df_right = df_right.drop(['ic_time', 'fo_time'], axis=1)
# print(df_left)
# print(df_left.describe())

win_df_left = construct_windowed_df(df_left, win_size, win_slide)
win_df_right = construct_windowed_df(df_right, win_size, win_slide)
print("df_left, ",  win_df_left)
print("df_right, ",  win_df_right)
#print(win_df_left.describe())

Sym_list_1PS1 = calculate_SI_new(win_df_left, win_df_right)
#print(len(Sym_list_1PS1))
#print(Sym_list_1PS1)

create_DF_SI(Sym_list_1PS1)