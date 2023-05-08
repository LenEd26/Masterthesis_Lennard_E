import pandas as pd
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
plt.matplotlib.use("WebAgg")
from math import pi
#from visualize import exclude_outlier


dataset = "data_kiel"

# stroke
subjects_1 = [        
        "pp077",
        "pp122",
        ]

#stroke healthy
subjects_2 = [        
        "pp101",
        "pp109",        
        "pp123",
        "pp145",
        "pp149",
        "pp112"
        ] 

# healthy
subjects_3 = [
        "pp105", 
        "pp028",
        "pp010", 
        "pp011",
        "pp099",
        "pp079",
        "pp105",
        "pp106",
        "pp137",
        "pp139",
        "pp158",
        "pp165"
]
#all_sub = list(zip(subjects_1, subjects_2, subjects_3))

#print("all subjects", all_sub)
# Define filepath

with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
final_datapath= os.path.join(paths[dataset], "final_data")

print("path", final_datapath)

csv_file = glob.glob(os.path.join(final_datapath, "*.csv"))

df_list = []
for path in csv_file:
    df_s = pd.read_csv(path)
    df_list.append(df_s)
data_full = pd.concat(df_list, axis = 0, ignore_index= True)    
print("appending data:", data_full.describe())
data_descr = data_full.describe()
data_mean = data_full.mean(axis=0)
data_mean = pd.DataFrame(data=data_mean)
data_mean = data_mean.transpose()
data_descr.to_csv(final_datapath + "_summary.csv", sep=',', header=True)
data_mean.to_csv(final_datapath + "_mean.csv", sep=',', header=True, index=False)
print(data_full.columns)
print(data_full.isnull().values.any())
data_full = data_full.dropna()

data_columns = list(data_full.columns.values)


data_1_list = []
for sub1 in subjects_1:
    data_sub1 = data_full[data_full["subject"] == sub1]
    data_1_list.append(data_sub1)
data_1 = pd.concat(data_1_list)

data_2_list = []
for sub2 in subjects_2:
    data_sub2 = data_full[data_full["subject"] == sub2]
    data_2_list.append(data_sub2)
data_2 = pd.concat(data_2_list)

data_3_list = []
for sub3 in subjects_3:
    data_sub3 = data_full[data_full["subject"] == sub3]
    data_3_list.append(data_sub3)
data_3 = pd.concat(data_3_list)

### keep subject as identifier

features = ['stride_length_avg_left', 'clearance_avg_left', 'stride_time_avg_left',
       'swing_time_avg_left', 'stance_time_avg_left', 'stance_ratio_avg_left',
       'cadence_avg_left', 'speed_avg_left', 'stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left', 'subject', 'stride_length_avg_right',
       'clearance_avg_right', 'stride_time_avg_right', 'swing_time_avg_right',
       'stance_time_avg_right', 'stance_ratio_avg_right', 'cadence_avg_right',
       'speed_avg_right', 'stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right',
       'stride_length_SI', 'clearance_SI', 'stride_time_SI', 'swing_time_SI',
       'stance_time_SI', 'stance_ratio_SI', 'cadence_SI', 'speed_SI']

categories = ['stride_length_avg_left', 'clearance_avg_left', 'stride_time_avg_left',
       'swing_time_avg_left', 'stance_time_avg_left', 'stance_ratio_avg_left',
       'cadence_avg_left', 'speed_avg_left', 'stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left', 'stride_length_avg_right',
       'clearance_avg_right', 'stride_time_avg_right', 'swing_time_avg_right',
       'stance_time_avg_right', 'stance_ratio_avg_right', 'cadence_avg_right',
       'speed_avg_right', 'stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right',
       'stride_length_SI', 'clearance_SI', 'stride_time_SI', 'swing_time_SI',
       'stance_time_SI', 'stance_ratio_SI', 'cadence_SI', 'speed_SI']

featureset1 = data_1[features]
print("1",featureset1.describe())
featureset2 = data_2[features]
print("2",featureset2.describe())
featureset3 = data_3[features]
print("3",featureset3.describe())

fset1 = featureset1.drop("subject", axis = 1)
fset2 = featureset2.drop("subject", axis = 1)
fset3 = featureset3.drop("subject", axis = 1)
########## generate plots 
healthy_mean = fset3.mean()
healthy_mean = healthy_mean.to_frame()
healthy_mean = healthy_mean.transpose()
print("healthy mean", healthy_mean)
fset_1_norm = fset1.div(healthy_mean.iloc[0], axis = 'columns').reset_index()
fset_2_norm = fset2.div(healthy_mean.iloc[0], axis = 'columns').reset_index()
fset_3_norm = fset3.div(fset3).reset_index()
print("fset1_NORM", fset_1_norm.describe())

fset_1_norm.drop("index", inplace = True, axis = 1)
fset_2_norm.drop("index", inplace = True, axis = 1)
fset_3_norm.drop("index", inplace = True, axis = 1)
fset_1_norm.dropna(inplace=True)
fset_2_norm.dropna(inplace=True)
fset_3_norm.dropna(inplace=True)
print("fset3_NORM", fset_3_norm.describe())
### overall mean in each category as series -> [0] for value selection
fset1_mean = fset_1_norm.mean()
fset2_mean =fset_2_norm.mean()
fset3_mean =fset_3_norm.mean()
fs3m = fset3_mean.transpose()
### create df for each subject and then the mean
mean_1_list = []
for sub1 in subjects_1:
    feature_sub1 = featureset1[featureset1["subject"] == sub1]
    print("feature sub 1", feature_sub1)
    feature_sub1 = feature_sub1.drop("subject", axis = 1)
    feature_sub1 = feature_sub1.mean()
    mean_1_list.append(feature_sub1)
mean_data_1 = pd.concat(mean_1_list, axis=1)
mean_data_1 = mean_data_1.transpose()
print("Mean Data1", mean_data_1.describe())


mean_2_list = []
for sub2 in subjects_2:
    feature_sub2 = featureset2[featureset2["subject"] == sub2]
    print("feature sub 2", feature_sub2)
    feature_sub2 = feature_sub2.drop("subject", axis = 1)
    print("feature sub2_-sub", feature_sub2)
    feature_sub2 = feature_sub2.mean()
    mean_2_list.append(feature_sub2)
mean_data_2 = pd.concat(mean_2_list, axis=1)
mean_data_2 = mean_data_2.transpose()
print("Mean Data2", mean_data_2.describe())

### normalize group data 
norm_mean_data_1 = mean_data_1.div(healthy_mean.iloc[0], axis = 'columns').reset_index()
norm_mean_data_2 = mean_data_2.div(healthy_mean.iloc[0], axis = 'columns').reset_index()
##drop index
norm_mean_data_1.drop("index", inplace = True, axis = 1)
norm_mean_data_2.drop("index", inplace = True, axis = 1)

print("Norm mean 1", norm_mean_data_1)
print("Norm mea 2", norm_mean_data_2)

# remove outliers?

# ------- PART 1: Create background
# number of variable
def Radar_plot_Kiel(N_of_categories, df1, df2, df3, label_df1, label_df2):
    N = len(N_of_categories)
    print(N)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
    ax.set_xticklabels(categories, fontsize = 17)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    max_val_1 = df1.max() 
    max_val_2 = df2.max()
    max_val = np.concatenate((max_val_1, max_val_2), axis=None).max()
    print("max of the 2 dfs = ", max_val)
    
    
    plt.yticks(np.arange(0, max_val, step=0.5), color="grey", size=7)
    plt.ylim(0,max_val)

    # ------- PART 2: Add plots
    
    # Plot each individual = each line of the data
    # Ind1
    for row in df1.index:
        values=df1.iloc[row].values.flatten().tolist()
        #print("VALUES1", values)
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label= label_df1, color = "b")
        #ax.fill(angles, values, 'b', alpha=0.1)
    
    # Ind2
    for row in df2.index:
        values2=df2.iloc[row].values.flatten().tolist()
        #print("VALUES1", values)
        values2 += values2[:1]
        ax.plot(angles, values2, linewidth=1, linestyle='solid', label= label_df2, color = "g")
        #ax.fill(angles, values2, 'g', alpha=0.1)

    values=df3.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label= "healthy", color = "r")
    #ax.fill(angles, values, 'r', alpha=0.1)
    # for row in df2.index:
    #     values=df2.iloc[row].values.flatten().tolist()
    #     print("VALUES2", values)
    #     values += values[:1]
    #     ax.plot(angles, values, linewidth=1, linestyle='solid', label="healty stroke")
    #     ax.fill(angles, values, 'r', alpha=0.1) 

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Add title
    BLUE = "#2a475e"
    fig.suptitle(
        "Radarplot of Kiel Data stroke, healthy stroke and healthy",
        x = 0.1,
        y = 1,
        ha="left",
        fontsize=24,
        fontname="DejaVu Sans",
        color=BLUE,
        weight="bold",    
    )
    # Show the graph
    plt.show()
    plt.close()
#plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ dataset +"_" + "Radar_plot_three_classes"".png")

###Main
Radar_plot_Kiel(categories, norm_mean_data_1,norm_mean_data_2, fs3m, label_df1="stroke", label_df2 = "healthy stroke")
#Radar_plot_Kiel(categories, norm_mean_data_2, fs3m, label_df1= "not severe stroke")