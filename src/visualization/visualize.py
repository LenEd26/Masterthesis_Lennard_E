import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
matplotlib.use("WebAgg")

df_sim_right = pd.read_csv("data_sim/aggregated_data/all_right_parameters.csv")

print(df_sim_right.shape)
print(df_sim_right.info()) #last two columns are dtype object
print("old_describe")
print(df_sim_right.describe())
# df_sim_right['method'] = df_sim_right['method'].astype('category')
# df_sim_right['subject'] = df_sim_right['subject'].astype('category')
print('old shape_' , df_sim_right.shape)
# Upper bound
#upper = np.where(df_sim_right['method'] >= 2)
# Lower bound

#''' Removing the Outliers '''
#df_sim_right.drop(upper[0], inplace = True)

df_sim_right = df_sim_right[(df_sim_right["stance_time"]) < 2]
df_sim_right = df_sim_right[(df_sim_right["stride_length"]) > 1.2]

print('new shape_' , df_sim_right.shape)

print(df_sim_right.describe())
print(df_sim_right.method.unique())
print(df_sim_right.method.value_counts())

print("S1")
print(df_sim_right[df_sim_right["subject"] == 'S1'].describe())
print("S2")
print(df_sim_right[df_sim_right["subject"] == 'S2'].describe())

############# Plots
#heat_map = sb.heatmap(df_sim_right.iloc[:,:-2])
#plt.show()

sns.boxplot( x="stance_time", y='subject',data=df_sim_right) 
sns.displot(df_sim_right, x="stance_time", bins = 20)
plt.show()
# sb.pairplot(df_sim_right, hue='method', height=2)
# plt.show()
print(np.where(df_sim_right['stance_time']>2))

##### remove outlier


fig, ax = plt.subplots(figsize = (18,10))
ax.scatter(df_sim_right['method'], df_sim_right['stance_time'])

# x-axis label
ax.set_xlabel('methods used/scores')

# y-axis label
ax.set_ylabel('stance time')
plt.show()