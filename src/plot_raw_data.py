### Data Kiel
#Folder path: data_kiel/raw/*any/all patients*/treadmill/imu/LF and RF
# patients with treadmill data? -> 
import json
from data_reader.DataLoader import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr

#### PARAMS START ####
dataset = "data_sim_cbf"
load_raw = True   # load (and plot) raw IMU data into interim data

if dataset == "data_sim_cbf":
    # Simulated dataset
    sub_list = [
        "S1"
         ]

    runs = [
        "leicht", 
        "leicht2",
        "leicht3",
        "normal",
        "stark"   
    ]

if dataset == "data_sim":
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

elif dataset == "data_kiel":
    # kiel dataset

    sub_list = [
        #"pp076",
        #"pp077",
        #"pp107"
        "pp111",
        #"pp122",
        #"pp127"
        #"pp136"
        #"pp152"
    ]
    runs = [
        #"gait1", 
        #"gait2",
        # "walk_slow",
        "walk_preferred",
        # "walk_fast",
        # "treadmill"
    ]


#TODO Use path.json to wokr with root path
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
raw_base_path = os.path.join(paths[dataset], "raw")
interim_base_path = os.path.join(paths[dataset], "interim")
# raw_base_path = os.path.join("Masterthesis_Lennard_E", dataset, "raw")
# interim_base_path = os.path.join("Masterthesis_Lennard_E", dataset, "interim")
# print(raw_base_path)


if dataset == "data_RNN":
    # dataset RNN

    sub_list = [
        #"S1",
        "S2",
        #"S3",
        #"S4",
        #"S5",
    ]
    runs = [
        # "gait1", 
        # "gait2",
        # "walk_slow",
        "walk_preferred",
        # "walk_fast",
        #"treadmill"
    ]

elif dataset == "data_stanford":
    sub_list = ["S1", "S2"]
    
    runs = [
        "NLA"  
    ]

#### plot and load raw data ####
if load_raw:
    from_interim = False
    data_path = os.path.join(sub_list[0], runs[0])#, "imu")  # folder containing the raw IMU data
    read_folder_path = os.path.join(raw_base_path, data_path)
    save_folder_path = os.path.join(interim_base_path, data_path)

    # select IMU locations to load
    if dataset == "data_sim_cbf":
        IMU_loc_list = ['LF', 'RF', "LW", "RW", "ST" ]

    else:
        IMU_loc_list = ['LF', 'RF']
    for loc in IMU_loc_list:
        if from_interim:  # load interim data
            df_loc = pd.read_csv(os.path.join(read_folder_path, loc + ".csv"))
        else:  # load raw data (& save file to the interim folder)
            data_loader = DataLoader(read_folder_path, loc, sub_list, runs)
            df_loc = data_loader.load_csv_data()
            
            if dataset == "data_RNN":
                df_loc.columns = ['AccX', 'AccY', 'AccZ','GyrX', 'GyrY', 'GyrZ']
            # df_loc = data_loader.load_xsens_data()
            # df_loc = data_loader.load_GaitUp_data()
            # df_loc = data_loader.cut_data(500, 10500)  # (if necessary: segment data)
            data_loader.save_data(save_folder_path)  # save re-formatted data into /interim folder





            columns_acc = ['AccX', 'AccY', 'AccZ']
            columns_gyr = ['GyrX', 'GyrY', 'GyrZ']
            save_fig_path = "/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/src/visualization"
            title = ("raw_acc_xyz_" + sub_list[0] + "_" + loc)
            #columns = df_loc.iloc[:, 2:4]
            #acc = [['AccX', 'AccY', 'AccZ']]
            df_vis = df_loc#.iloc[2100:2300]

            plot_acc_gyr(df_vis, columns_acc, title ,save_fig_path)


            fig= plt.figure()
            ax = plt.axes(projection = '3d')
            ax.plot3D(df_loc['AccX'], df_loc['AccY'], df_loc['AccZ'])
            ax.set_title('Acc 3D')
            plt.show()
            title3D = "raw_acc_3D %r" % sub_list[0]
            plt.savefig(os.path.join(save_fig_path, str(title3D + '.png')), bbox_inches='tight')
            #plt.close(fig)