from data_reader.DataLoader import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr
plt.matplotlib.use("WebAgg")

df_loc = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_stanford/raw/S1/6B.csv")
df_loc = df_loc[['Acc_X', 'Acc_Y', 'Acc_Z']]
print(df_loc)
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
title = ("raw_acc_xyz_")
#columns = df_loc.iloc[:, 2:4]
#acc = [['AccX', 'AccY', 'AccZ']]
df_vis = df_loc#.iloc[2100:2300]

#plot_acc_gyr(df_vis, columns_acc, title ,save_fig_path)


fig= plt.figure()
ax = plt.axes(projection = '3d')
ax.plot3D(df_loc['Acc_X'], df_loc['Acc_Y'], df_loc['Acc_Z'])
ax.set_title('Acc 3D')
plt.show()
title3D = "raw_acc_3D %r" #% sub_list[0]
plt.savefig(os.path.join(save_fig_path, str(title3D + '.png')), bbox_inches='tight')
#plt.close(fig)