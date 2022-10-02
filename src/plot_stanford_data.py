from data_reader.DataLoader import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr
plt.matplotlib.use("WebAgg")

df_S1_6B = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/no_use/6B.csv", delimiter="\t")
df_S1_6D = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/no_use/6D.csv", delimiter="\t")
df_S1_79 = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/no_use/79.csv", delimiter="\t")
df_S1_E6 = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/no_use/E6.csv", delimiter="\t")
df_S1_EF = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/no_use/EF.csv", delimiter="\t")
df_S1_F3 = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/no_use/F3.csv", delimiter="\t")


def raw_stanford_plt(df):

    print(df.describe())
    df = df[['Acc_X', 'Acc_Y', 'Acc_Z','Gyr_X', 'Gyr_Y', 'Gyr_Z', ]]
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
    title = ("raw_acc_xyz_F3")
    plot_acc_gyr(df, columns_acc, title, save_fig_path)
    #columns = df_loc.iloc[:, 2:4]
    #acc = [['AccX', 'AccY', 'AccZ']]
    df_vis = df#.iloc[2100:2300]

    #plot_acc_gyr(df_vis, columns_acc, title ,save_fig_path)


    fig= plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot3D(df['Acc_X'], df['Acc_Y'], df['Acc_Z'])
    ax.set_title('Acc 3D')
    plt.show()
    plt.close()
    title3D = "raw_acc_3D %r" #% sub_list[0]
    #plt.savefig(os.path.join(save_fig_path, str(title3D + '.png')), bbox_inches='tight')
    #plt.close(fig)
    
    fig= plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot3D(df['Gyr_X'], df['Gyr_Y'], df['Gyr_Z'])
    ax.set_title('Gyr 3D')
    plt.show()
    plt.close()
    title3D = "raw_gyr_3D %r" #% sub_list[0]
    #plt.savefig(os.path.join(save_fig_path, str(title3D + '.png')), bbox_inches='tight')
    #plt.close(fig)


raw_stanford_plt(df_S1_6B)
raw_stanford_plt(df_S1_6D)
raw_stanford_plt(df_S1_79)
raw_stanford_plt(df_S1_E6)
raw_stanford_plt(df_S1_EF)
raw_stanford_plt(df_S1_F3)

df = df_S1_EF[['Acc_X', 'Acc_Y', 'Acc_Z','Gyr_X', 'Gyr_Y', 'Gyr_Z', 'UTC_Valid','UTC_Second']]
print(df.describe())