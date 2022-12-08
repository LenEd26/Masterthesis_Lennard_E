import matplotlib.pyplot as plt
from joypy import joyplot
import os
import pandas as pd
import json
import glob
import seaborn as sns
plt.matplotlib.use("WebAgg")

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

featureset3 = featureset3.loc[~featureset3.index.duplicated(), :]
fset3 = fset3.loc[~fset3.index.duplicated(), :]
print("duplicates", featureset3.index.duplicated())
print("columns duplicate",featureset3.columns.duplicated())


##### distributions plots 
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
fig.suptitle("Data distribution for Stride length and speed in the healthy control group")
sns.histplot(data= featureset3, x = 'stride_length_avg_left',hue = "subject", alpha  = 0.5, kde = True, ax=axes[0], binwidth=0.02,legend = False)
sns.histplot(data= featureset3, x = 'speed_avg_left',hue = "subject", alpha  = 0.5, kde = True, ax=axes[1], binwidth=0.02,legend = True)
plt.show()
plt.close()

sns.displot(
  data=featureset3,
  x="stride_length_avg_left",
  hue="subject",
  kind="kde",
  height=6,
  aspect=1.4,
  log_scale=10
)
plt.show()
plt.close()

 
sns.displot(data= featureset3, x = 'stride_length_avg_left',hue = "subject", kind="kde",legend = False)
sns.displot(data= featureset3, x = 'stride_length_avg_right',hue = "subject", kind="kde", legend = False)
sns.displot(data= featureset3, x = 'speed_avg_left',hue = "subject", kind="kde",legend = False)
sns.displot(data= featureset3, x = 'speed_avg_right',hue = "subject", kind="kde", legend = False)

plt.show()
plt.close()

plt.show()
plt.close()
# sns.kdeplot(fset3['stride_length_avg_left'])
# sns.rugplot(fset3['stride_length_avg_left'])
# plt.show()
# plt.close()

fig, axs = plt.subplots(ncols=5, nrows=8, figsize = (20,20))
fig.tight_layout(pad=2.0)
for column, ax in zip(categories, axs.ravel()):
    sns.kdeplot(fset3[column] ,ax=ax)
    sns.rugplot(fset3[column], ax=ax)
    # chart formatting
    ax.set_title(column.lower())
    #ax.get_legend().remove()
    ax.set_xlabel("")
plt.show()
plt.close()

fig, axs = plt.subplots(ncols=5, nrows=8, figsize = (20,20))
fig.tight_layout(pad=2.0)
for column, ax in zip(categories, axs.ravel()):
    sns.histplot(data= fset3, x = column, kde = False, ax=ax)
    sns.histplot(data= fset1, x = column, kde = False, ax=ax, color = "r")
    sns.histplot(data= fset2, x = column, kde = False, ax=ax, color = "g")
    # chart formatting
    ax.set_title(column.lower())
    #ax.get_legend().remove()
    ax.set_xlabel("")
plt.show()
plt.close()
######### plot
 
# for column in categories:
#     plt.figure()

#     joyplot(
#         data=featureset1[[column, "subject"]], 
#         by='subject',
#         figsize=(12, 8)
#     )
#     plt.title('Ridgeline Plot of Max Temperatures in Sydney', fontsize=20)
    
# plt.show()