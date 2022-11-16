import glob
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from statistics import mean, stdev
plt.matplotlib.use("WebAgg")


# Step2 Load dataset
dataset = "data_charite"

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

## Data Standardization 
# '''rescales the distribution of values so that the mean is 0 and std is 1 -> use for normally distributed data'''
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_scaled = scaler.transform(X_train)
# print("Scaled mean should be 0:", X_scaled.mean(axis = 0))
# print("Scaled std should be 1:", X_scaled.std(axis=0))

## Cross validation


###### Logistic Regression -> Multinomial logistic regression
'''https://machinelearningmastery.com/multinomial-logistic-regression-with-python/'''

# define the model
#model_log = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty = "l2", C=1.0))
model_log = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty = "l2", C=1.0)
# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
skf = StratifiedKFold(n_splits=3)
# evaluate the model and collect the scores

n_scores = cross_val_score(model_log, X, np.ravel(y), scoring='accuracy', cv=skf, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
# fit the model

model_log.fit(X, np.ravel(y))
print("TEST")
# score the model
print(f'model score on training data: {model_log.score(X_train, np.ravel(y_train))}')
print(f'model score on testing data: {model_log.score(X_test, y_test)}')

# get importance
importances = pd.DataFrame(data={'Attribute': X_train.columns,
    'Importance': abs(model_log.coef_[0])})
importances = importances.sort_values(by='Importance', ascending=True)
# summarize feature importance
plt.figure(figsize=(20,15))
#plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.barh(importances['Attribute'], importances['Importance'])#, align='center')
plt.title('LogReg Feature importances obtained from coefficients' + "_" + dataset, size=20)
plt.yticks(range(len(importances['Attribute'])), importances['Attribute'])
#plt.xticks(rotation='vertical')
plt.show()

### Stratified k fold model ###
skf = StratifiedKFold(n_splits=5)
log_mod = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty = "l2", C=1.0)
lst_accu_stratified = []

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    #x_train_fold, x_test_fold = X[train_index], X[test_index]
    #y_train_fold, y_test_fold = y[train_index], y[test_index]
    X_train_fold = X.iloc[train_index]
    X_test_fold = X.iloc[test_index]
    y_train_fold = y.iloc[train_index]
    y_test_fold = y.iloc[test_index]
    log_mod.fit(X_train_fold, np.ravel(y_train_fold))
    log_mod_pred = log_mod.predict(X_test_fold)
    lst_accu_stratified.append(log_mod.score(X_test_fold, y_test_fold))

print('List of possible accuracy for log Model:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accu_stratified)*100, '%')
print('\nMinimum Accuracy:',
      min(lst_accu_stratified)*100, '%')
print('\nOverall Accuracy:',
      mean(lst_accu_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_accu_stratified))