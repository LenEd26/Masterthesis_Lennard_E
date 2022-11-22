import json
import os
from pickle import FALSE
import pandas as pd
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from src.LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr
from src.LFRF_parameters.preprocessing.get_imu_gyro_thresholds import AccPlot, GyroPlot
from data_reader.DataLoader import DataLoader

#### PARAMS START ####
dataset = "data_charite"
load_raw = True  # load (and plot) raw IMU data into interim data
get_stance_threshold = True  # determine stance threshold
get_initial_contact = False # determine IMU initial contact

if dataset == "data_charite":
    sub_list = [
        #"imu0001",
        #"imu0002"
        #"imu0003",  #IMU0003 DataLoader Xsense time is used
        "imu0006"
        ]

    runs = [
        "visit1",
        #"visit2"
    ]


if dataset == "data_kiel":
    # kiel dataset
    sub_list = [
        # "pp001",
        "pp077",
        #"pp111",
        "pp122",
        #"pp152",
        # "pp105",
        # "pp112",
        # "pp114",
        # "pp139"
    ]
    runs = [
        #"gait1", 
        #"gait2",
        # "walk_slow",
        #"walk_preferred",
        # "walk_fast",
        "treadmill"
    ]

elif dataset == "data_kiel_val":
    sub_list = [
        "pp001",
        "pp002",
        "pp003",
        "pp004",
        "pp005",
        "pp006",
        "pp007",
        "pp008",
        "pp009",
        "pp010"
    ]
    runs = [
        "gait1",
        # "gait2",
        # "walk_slow",
        # "walk_preferred",
        # "walk_fast"
    ]

elif dataset == "data_imu_validation":
    # Xsens IMU validation dataset
    sub_list = [
        # "physilog",
        # "xsens1",
        "xsens2"
    ]
    runs = [
        "slow",
        "normal",
        "fast"
    ]

elif dataset == "data_sim":
    # Simulated dataset
    sub_list = [
        "S1",
        "S2"
         ]

    runs = [
        "1P", 
        "2P",
        "3P",
        "regular"   
    ]

elif dataset == "data_sim_cbf":
    sub_list = ["S1"]
    
    runs = [
        #"leicht", 
        "leicht2",
        # "leicht3",
        # "normal",
        #"stark"   
    ]

elif dataset == "data_stanford":
    sub_list = ["S1"   
                 ]
    
    runs = [
        #"dummy_ts",
        "NLA"   
    ]


with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
raw_base_path = os.path.join(paths[dataset], "raw")
interim_base_path = os.path.join(paths[dataset], "interim")
#### PARAMS END ####

#### plot and load raw data ####
if load_raw:
    for i in range(len(sub_list)):
        for j in range(len(runs)):
            from_interim = False
            data_path = os.path.join(sub_list[i], runs[j], "imu")  # folder containing the raw IMU data
            read_folder_path = os.path.join(raw_base_path, data_path)
            save_folder_path = os.path.join(interim_base_path, data_path)

            # select IMU locations to load
            IMU_loc_list = ['LF', 'RF']
            for loc in IMU_loc_list:
                if from_interim:  # load interim data
                    df_loc = pd.read_csv(os.path.join(read_folder_path, loc + ".csv"))
                else:  # load raw data (& save file to the interim folder)
                    data_loader = DataLoader(read_folder_path, loc, sub_list, runs)
                    #df_loc = data_loader.load_kiel_data()
                    #df_loc = data_loader.load_xsens_data()
                    #df_loc = data_loader.load_xsens_data_time() 
                    df_loc = data_loader.load_sim_data()
                    #df_loc = data_loader.load_stanford_data()
                    # df_loc = data_loader.load_GaitUp_data()
                    # df_loc = data_loader.cut_data(500, 10500)  # (if necessary: segment data)
                    data_loader.save_data(save_folder_path)  # save re-formatted data into /interim folder

                # df_loc = df_loc.dropna()
                if df_loc is not None:  # if the IMU data is loaded, plot the signals
                    plot_acc_gyr(df_loc, ['AccX', 'AccY', 'AccZ'], 'raw_Acc_' + loc, save_folder_path)
                    plot_acc_gyr(df_loc, ['GyrX', 'GyrY', 'GyrZ'], 'raw_Gyr_' + loc, save_folder_path)

            #plt.show()


#### get gyro stance threshold ####
if get_stance_threshold:
    overwrite = False  # if False: append to existing file 

    # if no file, create one. Otherwise append to the existing file
    if not os.path.isfile(os.path.join(interim_base_path, 'stance_magnitude_thresholds.csv')) or overwrite:
        pd.DataFrame(
            columns=[
                "subject",
                "run",
                "stance_magnitude_threshold_left",
                "stance_magnitude_threshold_right",
                "stance_count_threshold_left",
                "stance_count_threshold_right",
            ],
        ).to_csv(
            os.path.join(interim_base_path, "stance_magnitude_thresholds.csv"), index=False
        )

    for subject_id, subject in enumerate(sub_list):
        subject_directory = os.path.join(interim_base_path, subject)
        
        runs = [
            x
            for x in os.listdir(subject_directory)
            if os.path.isdir(os.path.join(subject_directory, x))
        ]
        for run_id, run in enumerate(runs):
            print(
                "subject",
                subject_id + 1,
                "/",
                len(sub_list),
                "/",
                "run",
                run_id + 1,
                "/",
                len(runs)
            )
            file_directory = os.path.join(subject_directory, run, "imu")
            print("file directory is" + file_directory)
            print("interimbp_" + interim_base_path)
            print("subject" + subject)
            print(run)

            # run interactive gyro stance phase detection
            gp = GyroPlot(file_directory, interim_base_path, subject, run)
            gp.gyro_threshold_slider()


#### get IMU initial contact ####
overwrite = True  # if False: append to existing file 
if get_initial_contact:
    # if no file, create one. Otherwise append to the existing file
    if not os.path.isfile(os.path.join(interim_base_path, 'imu_initial_contact.csv')) or overwrite:
        pd.DataFrame(
            columns=[
                "subject",
                "run",
                "imu_initial_contact_left",
                "imu_initial_contact_right"
            ],
        ).to_csv(
            os.path.join(interim_base_path, "imu_initial_contact.csv"), index=False
        )

    for subject_id, subject in enumerate(sub_list):
        subject_directory = os.path.join(interim_base_path, subject)
        runs = [
            x
            for x in os.listdir(subject_directory)
            if os.path.isdir(os.path.join(subject_directory, x))
        ]
        for run_id, run in enumerate(runs):
            print(
                "subject",
                subject_id + 1,
                "/",
                len(sub_list),
                "/",
                "run",
                run_id + 1,
                "/",
                len(runs)
            )
            file_directory = os.path.join(subject_directory, run, "imu")

            # run interactive imu initial contact detection
            ap = AccPlot(file_directory, interim_base_path, subject, run)
            ap.acc_ic_plot()

    