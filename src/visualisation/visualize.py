from cmath import nan
from enum import auto
from turtle import title
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
matplotlib.use("WebAgg")
import os
import json
import glob 


dataset = "data_sim_cbf"
# Define filepath

with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
final_datapath= os.path.join(paths[dataset], "final_data")
# Load Excel file using Pandas
#file = pd.ExcelFile(final_datapath)
csv_file = glob.glob(os.path.join(final_datapath, "*.csv"))



def df_check_outlier_column(df):
    print(df.shape)
    print(df.info()) #last two columns are dtype object
    print("old_describe")
    print(df.describe())
    #df['method'] = df['method'].astype('category')
    #df['subject'] = df['subject'].astype('category')
    print('old shape_' , df.shape)

    #''' Removing the Outliers '''
    print("outlier = ", sum(df["is_outlier"] == True))
    df  = df[df["is_outlier"] == False]
    print(df)
    sns.boxplot( x= 'subject', y='stride_length',data=df)
    plt.show()
    return df

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
    print("subject unique before drop", df.subject.unique())
    print("sum of NAs: ", df.isnull().sum())

    # drop selected outlier = NA
    df = df.dropna(axis = 0)
    print(df.isnull().sum())

    print('new shape_' , df.shape)

    print(df.describe())
    print(df.method.unique())
    print(df.subject.unique())
    print(df.method.value_counts())
    print(df.subject.value_counts())
    
    return df


def plot_check_df(df, subject, color_dict, column):

    # plt.figure(figsize = (15,8))
    # sns.heatmap(df.corr(), xticklabels = 1, yticklabels = 1)
    # plt.show()
    # plt.close()
    sns.boxplot(x= 'subject', y= column, data=df).set(title= "stride length of the different stroke categories " + subject)
    plt.show()
    plt.close()
    sns.boxplot(x= 'severity', y= column, data=df, palette= color_dict, order = color_dict).set(title= "stride length of the different stroke categories " + subject)
    plt.show()
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + column +".png")  
    plt.close()
    sns.pairplot(df)
    plt.show()
    plt.close()


# def subject_boxplots(df, name, subject, color_dict):
#     sns.boxplot(x= df.loc[df["subject"] == subject, 'severity'], y='stride_length', data=df, palette= color_dict, order = color_dict).set(title= "stride length of the different stroke categories " + name)
#     plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + name +"_stride_length_.png") 
#     plt.close() 
    
#     sns.boxplot(x= df.loc[df["subject"] == subject, 'severity'], y='clearance',data=df, palette= color_dict, order = color_dict).set(title= "clearance of the different stroke categories " + name)
#     plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + name +"_clearance_.png") 
#     plt.close()
    
#     sns.boxplot( x= df.loc[df["subject"] == subject, 'severity'], y='stride_time',data=df, palette= color_dict, order = color_dict).set(title= "stride time of the different stroke categories " + name)
#     plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + name +"_stride_time_.png") 
#     plt.close()

def subject_boxplots(df, name, subject):
    sns.boxplot(x= df.loc[df["subject"] == subject, 'severity'], y='stride_length_avg', data=df).set(title= "stride length of the different stroke categories " + name)
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + name +"_stride_length_.png") 
    plt.close() 
    
    sns.boxplot(x= df.loc[df["subject"] == subject, 'severity'], y='clearance',data=df).set(title= "clearance of the different stroke categories " + name)
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + name +"_clearance_.png") 
    plt.close()
    
    sns.boxplot( x= df.loc[df["subject"] == subject, 'severity'], y='stride_time_avg',data=df).set(title= "stride time of the different stroke categories " + name)
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + name +"_stride_time_.png") 
    plt.close()
def plot_check_kiel(df, name, color_dict_kiel):
    df_corr = df.iloc[:,:-6]
    print("df_corr_", df_corr.columns.tolist())
    plt.figure(figsize = (15,8))
    sns.heatmap(df_corr.corr(), xticklabels = 1, yticklabels = 1)
    plt.show()
    plt.close()
    sns.boxplot(x='variable', y='value', data = pd.melt(df_corr.loc[:,["stride_index", 'timestamp','fo_time', 'ic_time']]))
    plt.close()
    sns.boxplot(x='variable', y='value', data = pd.melt(df_corr.loc[:,['stride_length', 'clearance', 'stride_time',
    'swing_time', 'stance_time', 'stance_ratio']]))
    plt.close()
    sns.boxplot(x= 'subject', y='stride_length',data = df, palette= color_dict_kiel).set(title = "stride length of the different stroke categories " + name)
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ name +"_stride_length_.png")  
    plt.close()
    sns.boxplot(x= "subject", y='clearance',data = df, palette= color_dict_kiel).set(title = "clearance of the different stroke categories " + name)
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ name +"_clearance_.png") 
    plt.close()
    sns.boxplot( x= "subject", y='stride_time',data = df, palette= color_dict_kiel).set(title = "stride time of the different stroke categories " + name)
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ name +"_stride_time_.png") 
    plt.close()

def plot_check_stanford(df, subject, color_dict, column):

    # plt.figure(figsize = (15,8))
    # sns.heatmap(df.corr(), xticklabels = 1, yticklabels = 1)
    # plt.show()
    # plt.close()
    sns.boxplot(x= 'subject', y= column, data=df).set(title= "stride length of the different stroke categories " + subject)
    plt.show()
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + column +".png")  
    plt.close()
    #plt.close()
    # sns.pairplot(df)
    # plt.show()
def plot_check_sim_cbf(df, subject, color_dict, column):

    # plt.figure(figsize = (15,8))
    # sns.heatmap(df.corr(), xticklabels = 1, yticklabels = 1)
    # plt.show()
    # plt.close()
    sns.boxplot(x= 'severity', y= column, data=df, palette= color_dict, order = color_dict).set(title= "stride length of the different stroke categories " + subject)
    #plt.show()
    plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ subject + "_cbf_" + column +".png")  
    plt.close()
    sns.pairplot(df)
    plt.show()
    plt.close()

########## MAIN #########
# Data Sim --> CHECK WITH GIVEN OUTLIER AGAIN!
# df_sim_right = exclude_outlier(df_sim_right)
# df_sim_left = exclude_outlier(df_sim_left)
# print(df_sim_right.describe())
# side ="_left" #"_right"
# feature = "stride_length_avg"
subject = "S1"
df_list = []
#color_dict_sim = {"NLA":"C0"}
#color_dict = {"regular":"C0", "1P":"C1", "2P":"C2", "3P":"C3"}
color_dict = {"leicht":"C0", "leicht3":"C1", "normal":"C2", "stark":"C3"}

print(csv_file)
for path in csv_file:
    df_s = pd.read_csv(path)
    df_list.append(df_s)
data = pd.concat(df_list, axis = 0, ignore_index= True)

data_columns = list(data.columns.values)
data_columns.remove("subject")
data_columns.remove("severity")
print(data_columns)   
print("appending data:", data.describe())

for i in data_columns:
    column = i
    plot_check_sim_cbf(data, subject, color_dict, column)
#color_dict_sim = {"regular":"C0", "1P":"C1", "2P":"C2", "3P":"C3"}


#plot_check_df(data, subject, color_dict_sim, side, feature, filter_for_sub=False)
#plot_check_df(df_sim_left, "sim_left", color_dict_sim)
#subject_boxplots(data, side, subject)
#subject_boxplots(df_sim_left, "_sim_left", "S2", color_dict_sim)

## Data Kiel
# data_kiel_right = df_check_outlier_column(data_kiel_right)
# # df_kiel_right = exclude_outlier(data_kiel_right)
# # plot_check_kiel(df_kiel_right, "kiel_right")
# color_dict_kiel = {subject: "r" if (subject == "pp077") or (subject == "pp122") else "b" for subject in data_kiel_right.subject.unique()}
# plot_check_kiel(data_kiel_right, "kiel_right",color_dict_kiel)
