from cmath import nan
from enum import auto
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
matplotlib.use("WebAgg")

df_sim_right = pd.read_csv("data_sim/aggregated_data/all_right_parameters.csv")
df_sim_left = pd.read_csv("data_sim/aggregated_data/all_left_parameters.csv")

def df_check_outlier_column(df):
    print(df.shape)
    print(df.info()) #last two columns are dtype object
    print("old_describe")
    print(df.describe())
    # df_sim_right['method'] = df_sim_right['method'].astype('category')
    # df_sim_right['subject'] = df_sim_right['subject'].astype('category')
    print('old shape_' , df.shape)

    #''' Removing the Outliers '''
    print("outlier = ", sum(df["is_outlier"] == True))
    df_is_outlier = df[df["is_outlier"] == False]
    print(df_is_outlier)
    sns.boxplot( x= 'subject', y='stride_length',data=df_is_outlier)
    plt.show()

#IQR Method
def iqr_outliers(df, feature):
    Q1= df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit

def exclude_outlier(df):
    for column in df.iloc[:,:-4]:
        #print("column",column)
        for cell in df[column]:
            upper, lower = iqr_outliers(df.iloc[:,:-4], column)
            #print(column, "_Upper whisker: ", upper)
            #print(column, "_Lower Whisker: ", lower)
            #print("cell",cell)
            if cell > upper: 
                df.loc[df[column] == cell, column] = np.nan
            if cell < lower:
                df.loc[df[column] == cell, column] = np.nan


    # check for selected outlier set to NA
    print("describe: ", df.describe())
    print("sum of NAs: ", df.isnull().sum())

    # drop selected outlier = NA
    df = df.dropna(axis = 0)
    print(df.isnull().sum())

    print('new shape_' , df.shape)

    print(df.describe())
    print(df.method.unique())
    print(df.method.value_counts())
    
    return df


def plot_check_df(df):
    df_corr = df.iloc[:,:-6]
    print("df_corr_", df_corr.columns.tolist())
    sns.heatmap(df_corr.corr(), xticklabels = 1, yticklabels = 1)
    plt.show()
    plt.close()
    sns.boxplot(x='variable', y='value', data = pd.melt(df_corr.loc[:,["stride_index", 'timestamp','fo_time', 'ic_time']]))
    plt.show()
    plt.close()
    sns.boxplot(x='variable', y='value', data = pd.melt(df_corr.loc[:,['stride_length', 'clearance', 'stride_time',
    'swing_time', 'stance_time', 'stance_ratio']]))
    plt.show()
    plt.close()
    sns.boxplot(x= 'subject', y='stride_length',data=df)
    plt.show()
    plt.close()
    sns.boxplot(x= "subject", y='clearance',data=df)
    plt.show()
    plt.close()
    sns.boxplot( x= "subject", y='stride_time',data=df)
    plt.show()
    sns.displot(df, x="stance_time", bins = 20)
    plt.show()
    sns.pairplot(df_corr, hue='method', height=2)
    plt.show()

# fig, ax = plt.subplots(figsize = (18,10))
# ax.scatter(df_sim_right['method'], df_sim_right['stance_time'])

# # x-axis label
# ax.set_xlabel('methods used/scores')

# # y-axis label
# ax.set_ylabel('stance time')
# plt.show()

########## MAIN #########
df_sim_right = exclude_outlier(df_sim_right)
df_sim_left = exclude_outlier(df_sim_left)
#print(df_sim_right.describe())
plot_check_df(df_sim_right)