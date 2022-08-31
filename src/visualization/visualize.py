from cmath import nan
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

#''' Removing the Outliers '''
# is_outlier column
print("outlier = ", sum(df_sim_right["is_outlier"] == True))
df_is_outlier = df_sim_right[df_sim_right["is_outlier"] == False]
print(df_is_outlier)
sns.boxplot( x= 'subject', y='stride_length',data=df_is_outlier)
plt.show()

#IQR Method
def outliers(df, feature):
    Q1= df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit

for column in df_sim_right.iloc[:,:-4]:
    #print("column",column)
    for cell in df_sim_right[column]:
        upper, lower = outliers(df_sim_right.iloc[:,:-4], column)
        #print(column, "_Upper whisker: ", upper)
        #print(column, "_Lower Whisker: ", lower)
        #print("cell",cell)
        if cell > upper: 
            df_sim_right.loc[df_sim_right[column] == cell, column] = np.nan
        if cell < lower:
            df_sim_right.loc[df_sim_right[column] == cell, column] = np.nan
        #df_sim_right = df_sim_right.loc[df_sim_right[column] cell > upper or cell < lower].replace((cell > upper or cell < lower),np.nan)
        # df_sim_right.loc[(df_sim_right[i]) < upper] = np.nan
        # df_sim_right.loc[(df_sim_right[i]) > lower] = np.nan
        #print("old_describe", df_sim_right.describe())

# check for selected outlier set to NA
print(df_sim_right.describe())
print(df_sim_right.isnull().sum())

# drop selected outlier = NA
df_sim_right = df_sim_right.dropna(axis = 0)
print(df_sim_right.isnull().sum())

print('new shape_' , df_sim_right.shape)

print(df_sim_right.describe())
print(df_sim_right.method.unique())
print(df_sim_right.method.value_counts())

# check Different Subjects
print("S1")
print(df_sim_right[df_sim_right["subject"] == 'S1'].describe())
print("S2")
print(df_sim_right[df_sim_right["subject"] == 'S2'].describe())


############# Plots
#heat_map = sb.heatmap(df_sim_right.iloc[:,:-2])
#plt.show()
plt.close()
sns.boxplot( x= 'subject', y='stride_length',data=df_sim_right)
plt.show()
plt.close()
sns.boxplot( x= "subject", y='clearance',data=df_sim_right)
plt.show()
plt.close()
sns.boxplot( x= "subject", y='stride_time',data=df_sim_right)
plt.show()
# sns.displot(df_sim_right, x="stance_time", bins = 20)
# plt.show()
# sb.pairplot(df_sim_right, hue='method', height=2)
# plt.show()

# fig, ax = plt.subplots(figsize = (18,10))
# ax.scatter(df_sim_right['method'], df_sim_right['stance_time'])

# # x-axis label
# ax.set_xlabel('methods used/scores')

# # y-axis label
# ax.set_ylabel('stance time')
# plt.show()