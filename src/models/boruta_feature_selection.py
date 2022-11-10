from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import glob
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step2 Load dataset
dataset = "data_sim_cbf"

with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
final_datapath= os.path.join(paths[dataset], "final_data")
# Load CSV file using Pandas
csv_file = glob.glob(os.path.join(final_datapath, "*.csv"))

### concatenate all csv files into one DF
df_list = []
for path in csv_file:
    df_s = pd.read_csv(path)
    df_list.append(df_s)
data = pd.concat(df_list, axis = 0, ignore_index= True)    
print("appending data:", data.describe())
print(data.columns)
print(data.isnull().values.any())
data = data.dropna()


## Split Data
data_X = data.loc[:, ~data.columns.isin(["severity", "subject"])] ####### define X and Y from the overal Dataset!
print("old Data-X:", data_X.describe())
data_y = data.loc[:, data.columns.isin(["severity"])]#, "subject"])] 

## Data Normalization 
''' MinMaxScaler/Normalization rescales all values to a range between 0 and 1 -> should be ued if the distribution
    is not normal/gaussian'''
# perform a robust scaler transform of the dataset
trans = MinMaxScaler()
data_X_new = pd.DataFrame(trans.fit_transform(data_X), columns = data_X.columns)
print(data_X_new.describe())

X = data_X_new
#print(X)
y = data_y
print("Y__", y)

# Step3 Split data in training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 1)
''' For further evaluation: https://scikit-learn.org/stable/modules/cross_validation.html '''


###initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 5
)
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 100 # number of trials to perform
)
X_np = X#.to_numpy()
y_np = y#.to_numpy()
### fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X_np), np.array(np.ravel(y_np)))
### print results
green_area = X.columns[boruta.support_].to_list()
blue_area = X.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)