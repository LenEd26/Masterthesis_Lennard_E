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
        #"pp101",
        #"pp109",
        #"pp112",
        "pp122",
        #"pp123",
       # "pp145",
        #"pp149",
        ]

#stroke healthy
subjects_2 = ["pp028"] 

# healthy
subjects_3 = [
        "pp010", 
        "pp011",
        "pp099",
        "pp079",
        "pp105",
        #"pp028",

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

featureset1 = data_1[data_1.columns.intersection(features)]
print("1",featureset1.describe())
featureset2 = data_2[features]
print("2",featureset2.describe())
featureset3 = data_3[features]
print("3",featureset3.describe())


visit_1 = featureset[featureset["severity"] == "visit1"]
visit_1 = visit_1.reset_index()
visit_1.drop("severity",inplace = True, axis = 1)
visit_1.drop("index", inplace = True, axis = 1)
print("visit_1", visit_1.describe())

visit_2 = featureset[featureset["severity"] == "visit2"]
visit_2 = visit_2.reset_index()
visit_2.drop("severity",inplace = True, axis = 1)
visit_2.drop("index", inplace = True, axis = 1)
print("visit_2", visit_2.describe())


visit_1_norm = visit_1.div(visit_1).reset_index()
visit_1_norm.dropna(inplace=True)
print("VISIT1 NORM", visit_1_norm.describe())

visit1_mean = visit_1.mean()
print("Mean V1", visit1_mean)
visit_2_norm = visit_2.div(visit1_mean).reset_index()
visit_2_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
visit_2_norm.dropna(inplace=True)
visit_1_norm.drop("index", inplace = True, axis = 1)
visit_2_norm.drop("index", inplace = True, axis = 1)
print("VISIT2 NORM", visit_2_norm)

### remove outliers ?
#exclude_outlier(visit_2_norm, categories)
#exclude_outlier(visit_1_norm, categories)


v_2_norm_mean = visit_2_norm.mean()
v_1_norm_mean = visit_1_norm.mean()
print("Mean V2", v_2_norm_mean)
print("VISIT 2 LOC",visit_2_norm.transpose())
# print("VISIT 1 LOC",visit_1_norm.transpose())
#visit_1_norm = visit_1_norm.transpose()
#visit_2_norm = visit_2_norm.transpose()

######### normalize data
# trans = MinMaxScaler()
# visit_1_scaled = DataFrame(trans.fit_transform(visit_1), columns = visit_1.columns)
# visit_2_scaled = DataFrame(trans.fit_transform(visit_2), columns = visit_2.columns)
# print(visit_1_scaled.describe())

features_v1v2 = featureset.transpose()

# ------- PART 1: Create background
 
# number of variable
#categories=list(df)[1:]
N = len(categories)
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
 
# Draw ylabels
ax.set_rlabel_position(0)
max_val = v_2_norm_mean.max()
print(max_val)
plt.yticks(np.arange(0, max_val, step=0.5), color="grey", size=7)
plt.ylim(0,max_val)

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# Ind1
values=v_1_norm_mean.values.flatten().tolist()
print("VALUES1", values)
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Visit 1")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=v_2_norm_mean.values.flatten().tolist()
print("VALUES2", values)
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Visit 2")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Add title
BLUE = "#2a475e"
fig.suptitle(
    "Radarplot of both visits for subject " + subject,
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
plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ dataset +"_" + subject +"_" + "Radar_plot_v1_mean"".png")
plt.close()