# Step1 Import Packages:
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Step2 Load dataset
data_sim_LF = read_csv("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/data_sim/aggregated_data/all_left_parameters.csv")
data_sim_LF_X = data_sim_LF.iloc[:,-2:]
data_sim_LF_y = data_sim_LF.iloc[:,:-2]
X = data_sim_LF_X
print(X)
y = data_sim_LF_y
print(y)
# Step3 Split data in training and testing subsets
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)

# Step4 Classifier Training using SVM
#linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)

# Step5 CHeck classifier accuracy on test data and see results