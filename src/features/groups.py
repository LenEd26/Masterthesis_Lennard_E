import json
import os
import pandas as pd
import warnings

from data.imu import IMU
import visualization.plot as plot

"""
This script is cutting the six minute processed and the signal data into 3 groups of 2 minutes each and creates plots 
for the cut signal data 
"""

"""
test gait param differences for time groups
=> cut raw (from interim) data and gait params: drop first 6 sec, then 2min intervals and drop everything at the end
=> cut processed first, then use minimum timestamp also for raw data
=> plot each group and save in grouped subfolder for the raw data
"""


def cut_processed_data(dataset, processed_base_path, sub_list, conditions, tests, start_time, end_time):
    for sub in sub_list:
        for cond in conditions:
            for test in tests:
                save_path = os.path.join(
                    processed_base_path,
                    'fatigue_dual_task',
                    'OG_' + cond + '_' + test,
                    sub,
                    'groups'
                )

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for foot in ["left", "right"]:
                    path = os.path.join(processed_base_path,
                                        'fatigue_dual_task',
                                        'OG_' + cond + '_' + test,
                                        sub,
                                        foot + '_foot_core_params_py_n.csv')
                    dat = pd.read_csv(path)
                    if dat.timestamps[0] > start_time:
                        warnings.warn(
                            f"First stride's timestamp of {sub}, {cond}, {test}, {foot} is greater than specified start_time")
                        print(f"Skipping {sub}, {cond}, {test}, {foot} due to too late first timestamp.")
                        continue

                    dat_02 = dat[(dat.timestamps >= start_time) & (dat.timestamps < end_time)]
                    dat_24 = dat[(dat.timestamps >= end_time) & (dat.timestamps < end_time + 120)]
                    dat_46 = dat[(dat.timestamps >= end_time + 120) & (dat.timestamps < end_time + 240)]

                    for idx, df in enumerate([dat_02, dat_24, dat_46]):
                        df.to_csv(os.path.join(save_path, f"group_{idx}_{foot}_foot_core_params_py_n.csv"), index=False)


def cut_raw_data(dataset, base_path, sub_list, conditions, tests, start_time, end_time):
    for sub in sub_list:
        for cond in conditions:
            for test in tests:
                path_folder = os.path.join(
                    base_path,
                    'fatigue_dual_task',
                    'OG_' + cond + '_' + test,
                    sub
                )
                save_path = os.path.join(path_folder, "groups")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for sensor in ["LF", "RF", "RL", "LL", "RW", "LW", "SA", "ST", "HE"]:  #
                    path_file = os.path.join(path_folder, sensor + ".csv")
                    # cut
                    imu = IMU(path_file)
                    imu02 = imu.crop_by_time(start_time, end_time, inplace=False)
                    imu24 = imu.crop_by_time(end_time, end_time + 120, inplace=False)
                    imu46 = imu.crop_by_time(end_time + 120, end_time + 240, inplace=False)

                    # save data and plots
                    for idx, imu in enumerate([imu02, imu24, imu46]):
                        imu.data.to_csv(os.path.join(save_path, sensor + f"group_{idx}.csv"), index=False)
                        plot.plot_accel_gyro(imu, path=os.path.join(save_path, sensor + f"group_{idx}.png"))


if __name__ == "__main__":
    # params
    dataset = 'fatigue_dual_task'
    with open('../../path.json') as f:
        paths = json.load(f)
    interim_base_path = paths['interim_data']
    processed_base_path = paths['processed_data']

    sub_list = [
        "sub_01",
        "sub_02",
        "sub_03",
        "sub_05",
        "sub_06",
        "sub_07",
        "sub_08",
        "sub_09",
        "sub_10",
        "sub_11",
        "sub_12",
        "sub_13",
        "sub_14",
        "sub_15",
        "sub_17",
        "sub_18"
    ]
    conditions = [
        "st",
        "dt"
    ]
    tests = [
        "control",
        "fatigue"
    ]
    start_time = 6.0
    end_time = start_time + 120

    cut_processed_data(dataset, processed_base_path, sub_list, conditions, tests, start_time, end_time)
    cut_raw_data(dataset, interim_base_path, sub_list, conditions, tests, start_time, end_time)
