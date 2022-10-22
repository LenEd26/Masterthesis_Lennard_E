from sklearn import datasets
#from data_reader.DataLoader import DataLoader
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from src.LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr
#from src.main_LFRF_preprocessing import IMU_loc_list
#plt.matplotlib.use("WebAgg")



df_S1_6B = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/no_use/6B.csv", delimiter="\t")
df_S1_6D = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/no_use/6D.csv", delimiter="\t")
df_S1_79 = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/no_use/79.csv", delimiter="\t")
df_S1_E6 = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/no_use/E6.csv", delimiter="\t")
df_S1_EF = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/no_use/EF.csv", delimiter="\t")
df_S1_F3 = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/no_use/F3.csv", delimiter="\t")

####### add dummy timestamp column
df_S1_E6["SampleTimeFine"] = (range(0,len(df_S1_E6)))
df_S1_E6["SampleTimeFine"] = 0.01*df_S1_E6["SampleTimeFine"]
#print(df_S1_E6["SampleTimeFine"])
df_S1_E6.to_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/NLA/LF.csv")
df_S1_EF["SampleTimeFine"] = (range(0,len(df_S1_EF)))
df_S1_EF["SampleTimeFine"] = 0.01*df_S1_EF["SampleTimeFine"]
#print(df_S1_EF["SampleTimeFine"])
df_S1_EF.to_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/NLA/RF.csv")


def raw_stanford_plt(df, name):

    print(df.head())
    df = df[['Acc_X', 'Acc_Y', 'Acc_Z','Gyr_X', 'Gyr_Y', 'Gyr_Z' ]]
    print(df.describe())
    # sub_list = ["S1"]
        
    # runs = [
    #         "6B", 
    #         "6D",
    #         "79",
    #         "E6",
    #         "EF",
    #         "F3" ]
   


    columns_acc = ['Acc_X', 'Acc_Y', 'Acc_Z']
    columns_gyr = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
    save_fig_path = "/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/src/visualization"
    title = ("raw_acc_xyz"+"_"+ name)
    #plot_acc_gyr(df, columns_acc, title, save_fig_path)
    #columns = df_loc.iloc[:, 2:4]
    #acc = [['AccX', 'AccY', 'AccZ']]
    df_vis = df.iloc[10000:12000]

    plot_acc_gyr(df_vis, columns_acc, title ,save_fig_path)


    fig= plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot3D(df['Acc_X'], df['Acc_Y'], df['Acc_Z'])
    ax.set_title('Acc 3D')
    plt.show()
    plt.close()
    # title3D = "raw_acc_3D %r" % sub_list[0]
    # plt.savefig(os.path.join(save_fig_path, str(title3D + '.png')), bbox_inches='tight')
    # plt.close(fig)
    
    # fig= plt.figure()
    # ax = plt.axes(projection = '3d')
    # ax.plot3D(df['Gyr_X'], df['Gyr_Y'], df['Gyr_Z'])
    # ax.set_title('Gyr 3D')
    # plt.show()
    # plt.close()
    # title3D = "raw_gyr_3D %r" #% sub_list[0]
    #plt.savefig(os.path.join(save_fig_path, str(title3D + '.png')), bbox_inches='tight')
    #plt.close(fig)
###############





def plot_sim_cbf(dataset, sub_list, runs):

    for sub in sub_list:
        for run in runs:
            with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
                paths = json.loads(f.read())
            raw_base_path = os.path.join(paths[dataset], "raw")
            data_path = os.path.join(sub, run)#, "imu")  # folder containing the raw IMU data

            read_folder_path = os.path.join(raw_base_path, data_path)
            print(read_folder_path)
            if dataset == "data_sim_cbf":
                IMU_loc_list = ['LF', 'RF', "LW", "RW", "ST" ]

            else:
                IMU_loc_list = ['LF', 'RF']
            
            for loc in IMU_loc_list:
                file_path = os.path.join(read_folder_path, loc + ".csv")
                print(file_path)
                df_loc = pd.read_csv(file_path, skiprows= 7, on_bad_lines='skip') ###### ?? format wird nicht gelsen?
                print(df_loc.describe())

                columns_acc = ['Acc_X', 'Acc_Y', 'Acc_Z']
                columns_gyr = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
                save_fig_path = "/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/src/visualization"
                title = ("raw_acc_xyz" + "_" + sub + "_" + loc +"_"+ run)
                #columns = df_loc.iloc[:, 2:4]
                #acc = [['AccX', 'AccY', 'AccZ']]
                df_vis = df_loc.iloc[2000:4000]

                plot_acc_gyr(df_vis, columns_acc, title ,save_fig_path)

                fig= plt.figure()
                ax = plt.axes(projection = '3d')
                ax.plot3D(df_loc['Acc_X'], df_loc['Acc_Y'], df_loc['Acc_Z'])
                ax.set_title('Acc 3D')
                plt.show()
                title3D = ("raw_acc_3D" + sub)
                plt.savefig(os.path.join(save_fig_path, str(title3D + '.png')), bbox_inches='tight')
                #plt.close(fig)


################ MAIN 

sub_list = [
     "S1"]

runs = [
    "leicht", 
    "leicht2",
    "leicht3",
    "normal",
    "stark"]

#plot_sim_cbf(dataset = "data_sim_cbf", sub_list = sub_list, runs = runs)

raw_stanford_plt(df_S1_6B, "6B")
raw_stanford_plt(df_S1_6D, "6D")
raw_stanford_plt(df_S1_79, "79")
raw_stanford_plt(df_S1_E6, "E6")
raw_stanford_plt(df_S1_EF, "EF")
raw_stanford_plt(df_S1_F3, "F3")

# df = df_S1_EF[['Acc_X', 'Acc_Y', 'Acc_Z','Gyr_X', 'Gyr_Y', 'Gyr_Z', 'UTC_Valid','UTC_Second']]
# print(df.describe())

 

