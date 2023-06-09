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
from visualize import exclude_outlier


dataset = "data_charite"
subject = "imu0013"
feature_group = "CV" #"avg" or "SI" or "CV" or "CV_SI"  -> only makes sense to check with SI not multiplied by 100
windowed = False
# Define filepath

with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
final_datapath= os.path.join(paths[dataset], "final_data")
final_datapath_full_data = os.path.join(paths[dataset], "one_window_final_data")

print("final data path 1_", final_datapath)
print("final data path 2_", final_datapath_full_data)

if windowed == True:
    csv_file = glob.glob(os.path.join(final_datapath, "*.csv"))

if windowed == False:
    csv_file = glob.glob(os.path.join(final_datapath_full_data, "*.csv"))
df_list = []

for path in csv_file:
    df_s = pd.read_csv(path)
    df_list.append(df_s)
data_full = pd.concat(df_list, axis = 0, ignore_index= True)    
print("appending data:", data_full.describe())
print(data_full.columns)
print(data_full.isnull().values.any())
data_full = data_full.dropna()

#load reference data
reference_df = pd.read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_kiel/final_data_mean.csv")
#reference_df = pd.DataFrame(data = reference_df)
#reference_df= reference_df.transpose()

data_columns = list(data_full.columns.values)

data_subject = data_full[data_full["subject"] == subject]


features_charite_avg = data_subject[['stride_length_avg_left', 'clearance_avg_left', 'stride_time_avg_left',
       'swing_time_avg_left', 'stance_time_avg_left', 'stance_ratio_avg_left',
       'cadence_avg_left', 'speed_avg_left', 'severity', 'stride_length_avg_right',
       'clearance_avg_right', 'stride_time_avg_right', 'swing_time_avg_right',
       'stance_time_avg_right', 'stance_ratio_avg_right', 'cadence_avg_right',
       'speed_avg_right']]

features_charite_CV = data_subject[['stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left', 'severity', 'stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right']]

reference_df_CV = reference_df[['stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left', 'stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right']]

features_charite_SI = data_subject[[
       'severity', 'stride_length_SI', 'clearance_SI', 'stride_time_SI', 'swing_time_SI',
       'stance_time_SI', 'stance_ratio_SI', 'cadence_SI', 'speed_SI']]

features_charite_CV_SI = data_subject[['stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left', 'severity', 'stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right',
       'stride_length_SI', 'clearance_SI', 'stride_time_SI', 'swing_time_SI',
       'stance_time_SI', 'stance_ratio_SI', 'cadence_SI', 'speed_SI']]

print("features descr.", features_charite_avg.describe())
print("features descr.", features_charite_SI.describe())


categories_avg = ['stride_length_avg_left', 'clearance_avg_left', 'stride_time_avg_left',
       'swing_time_avg_left', 'stance_time_avg_left', 'stance_ratio_avg_left',
       'cadence_avg_left', 'speed_avg_left',  'stride_length_avg_right',
       'clearance_avg_right', 'stride_time_avg_right', 'swing_time_avg_right',
       'stance_time_avg_right', 'stance_ratio_avg_right', 'cadence_avg_right',
       'speed_avg_right']

categories_CV = ['stride length left',
       'clearance left', 'stride time left', 'swing time left',
       'stance time left', 'stance ratio left', 'cadence left',
       'speed left','stride length right', 'clearance right',
       'stride time right', 'swing time right', 'stance time right',
       'stance ratio right', 'cadence right', 'speed right']

categories_SI = [
       'stride length', 'clearance', 'stride time', 'swing time',
       'stance time', 'stance ratio', 'cadence', 'speed']

categories_CV_SI = ['stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left','stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right', 
       'stride_length_SI', 'clearance_SI', 'stride_time_SI', 'swing_time_SI',
       'stance_time_SI', 'stance_ratio_SI', 'cadence_SI', 'speed_SI']

#create labels for plot description
label_SI = [
       'stride length', 'clearance', 'stride time', 'swing time',
       'stance time', 'stance ratio', 'cadence', 'speed'] 

label_CV = ['stride length left',
       'clearance left', 'stride time left', 'swing time left',
       'stance time left', 'stance ratio left', 'cadence left',
       'speed left','stride length right', 'clearance_right',
       'stride time right', 'swing time right', 'stance time right',
       'stance ratio right', 'cadence right', 'speed right']

if feature_group == "avg":
    features_charite = features_charite_avg
    categories = categories_avg

elif feature_group == "CV":
    features_charite = features_charite_CV
    categories = categories_CV

elif feature_group == "SI":
    features_charite = features_charite_SI
    categories = categories_SI

elif feature_group == "CV_SI":
    features_charite = features_charite_CV_SI
    categories = categories_CV_SI

visit_1 = features_charite[features_charite["severity"] == "visit1"]
visit_1 = visit_1.reset_index()
visit_1.drop("severity",inplace = True, axis = 1)
visit_1.drop("index", inplace = True, axis = 1)
print("visit_1", visit_1.describe())

visit_2 = features_charite[features_charite["severity"] == "visit2"]
visit_2 = visit_2.reset_index()
visit_2.drop("severity",inplace = True, axis = 1)
visit_2.drop("index", inplace = True, axis = 1)
print("visit_2", visit_2.describe())


#for check of radarplots
diff_in_values = visit_1 - visit_2
print("diff in values", diff_in_values.transpose())

if feature_group == "avg":
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

else:
    visit_1_norm = visit_1
    visit_2_norm = visit_2

### remove outliers
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

features_v1v2 = features_charite.transpose()

print("visit2!", visit_2)
print("visit2! transposed", visit_2.transpose())


visit_2_trans = visit_2_norm.transpose()
visit_1_trans = visit_1_norm.transpose()
reference_trans = reference_df_CV.transpose()

if windowed == False:
    combined_trans = np.concatenate((visit_2_trans, visit_1_trans, reference_trans), axis=0)
# ------- PART 1: Create background

# number of variable
#categories=list(df)[1:]
N = len(categories)
print(N)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
if feature_group == "SI":
    plt.xticks(angles[:-1], label_SI)
    ax.set_xticklabels(label_SI, fontsize = 8)  
    visit_zero = [0,0,0,0,0,0,0,0] 
elif feature_group == "CV":
    plt.xticks(angles[:-1], label_CV)
    ax.set_xticklabels(label_CV, fontsize = 8)
else:    
    plt.xticks(angles[:-1], categories)
    ax.set_xticklabels(categories, fontsize = 8)

# Draw ylabels
ax.set_rlabel_position(0)
if windowed == True:
    max_val = visit_2_norm.max()
    min_val = visit_2_norm.min()
    print("max_val", max_val)
    step = 0.01
    y_ticks = [round(num, 2) for num in np.arange(min_val, max_val, step=step)]
    plt.yticks(y_ticks, color="grey", size=7)
    plt.ylim(min_val.min(),max_val.max()) 
elif windowed == False:
    max_val = combined_trans.max()
    min_val = combined_trans.min()
    print("max_val", max_val)
    step = 0.02
    y_ticks = [round(num, 2) for num in np.arange(min_val, max_val, step=step)]
    plt.yticks(y_ticks, color="grey", size=7)   
    plt.ylim(min_val.min(),max_val.max())

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# Ind1
if windowed == True:
    values1 = v_1_norm_mean.values.flatten().tolist()
elif windowed == False:
    values1 = visit_1_trans.values.flatten().tolist()
print("VALUES1", values1)
values1 += values1[:1]
ax.plot(angles, values1, linewidth=1, linestyle='solid', label="Visit 1")
ax.fill(angles, values1, 'b', alpha=0.1)
 
# Ind2
if windowed == True:
    values2 = v_2_norm_mean.values.flatten().tolist()
elif windowed == False:
    values2 = visit_2_trans.values.flatten().tolist()
print("VALUES2", values2)
values2 += values2[:1]
ax.plot(angles, values2, linewidth=1, linestyle='solid', label="Visit 2")
ax.fill(angles, values2, 'r', alpha=0.1)

#CV reference data
if feature_group == "CV":
    reference_data = reference_df_CV.values.flatten().tolist()
    reference_data += reference_data[:1]
    ax.plot(angles, reference_data, linewidth=1, linestyle='dashed', label="Reference")
    ax.fill(angles, reference_data , 'g', alpha=0)

#Ind Zero
if feature_group == "SI":
    values_zero = visit_zero
    values_zero += values_zero[:1]
    ax.plot(angles, values_zero, linewidth=1, linestyle='dashed', label="Zero Reference")
    ax.fill(angles, values_zero, 'g', alpha=0)
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.2, 0.1))

# Add title
BLUE = "#2a475e"
fig.suptitle(
    "Radarplot of the " + feature_group + " for both visits - subject " + subject,
    x = 0.23,
    y = 0.95,
    ha="left",
    fontsize=10,
    fontname="DejaVu Sans",
    color=BLUE,
    weight="bold",    
)
# Show the graph
plt.show()
#plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ dataset +"_" + subject +"_" + "Radar_plot_v1_mean"".png")
plt.close()