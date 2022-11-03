# Step1 Import Packages:
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm, datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import svm
plt.matplotlib.use("WebAgg")
import glob
import os
import json
from sklearn.linear_model import LogisticRegression


# Step2 Load dataset
dataset = "data_sim_cbf"

with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
final_datapath= os.path.join(paths[dataset], "final_data")
# Load CSV file using Pandas
csv_file = glob.glob(os.path.join(final_datapath, "*.csv"))

#SVM
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


data_X = data.loc[:, ~data.columns.isin(["severity", "subject"])] ####### define X and Y from the overal Dataset!
data_y = data.loc[:, data.columns.isin(["severity"])]#, "subject"])] 

X = data_X
#print(X)
y = data_y
print("Y__", y)

# Step3 Split data in training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 1)
''' For further evaluation: https://scikit-learn.org/stable/modules/cross_validation.html '''

def f_importances(coef, names):
    imp = abs(coef)
    print(imp)
    imp,names = zip(*sorted(zip(imp,names)))
    plt.figure(figsize=(20,15))
    plt.title("SVC feature importance obtained from coefficients"+ "_" + dataset, size =20)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
    plt.close()
# Step 4 Classifiert Training using SVM
# poly, rbf, linear
feature_names = data_X.columns.tolist()  #data_X.columns.values.tolist()
print("Feature_Names",feature_names)

##### linear SVM clasifier
svclassifier_lin = SVC(kernel='linear', C=1, random_state= 42)
# fit the model
svclassifier_lin.fit(X_train, np.ravel(y_train, order="C"))
# importance plot
f_importances(svclassifier_lin.coef_[0], feature_names)
#efficiency of the models
lin_pred = svclassifier_lin.predict(X_test)
print(confusion_matrix(y_test,lin_pred))
print(classification_report(y_test,lin_pred))
### score the model 
lin_cross_scores = cross_val_score(svclassifier_lin, X, y, cv=5)
lin_accuracy = accuracy_score(y_test, lin_pred)
lin_f1 = f1_score(y_test, lin_pred, average='weighted')
print(lin_cross_scores, "%0.2f accuracy with a standard deviation of %0.2f" % (lin_cross_scores.mean(), lin_cross_scores.std()))
print('Accuracy (Linear Kernel): ', "%.2f" % (lin_accuracy*100))
print('F1 (Linear Kernel): ', "%.2f" % (lin_f1*100))

##### ploynomial SVM clasifier
svclassifier_poly = SVC(kernel='poly', C=1)
# fit the model 
svclassifier_poly.fit(X_train, np.ravel(y_train, order="C"))
# importance plot
'''rbf kernel implicitly transforms your features into a higher dimensional space of ' transformed features' and tries to separate 
 linearly in that new space. As the transformation is implicit you do not know the link with your original features. '''
# efficiency of the model
poly_pred = svclassifier_poly.predict(X_test)
### score model 
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))


###### Logistic Regression
# define the model
model_log = LogisticRegression()
# fit the model
model_log.fit(X, y)
# score the model
print(f'model score on training data: {model_log.score(X_train, y_train)}')
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





# # Step4 Classifier Training using SVM
# # select kernel
# linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
# rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
# poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
# sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)

# # create the mesh in which the results will be plotted
# h = .01
# #create the mesh
# x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
# y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# # create the title that will be shown on the plot
# titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']

# # loop to plot all kernel functions
# for i, clf in enumerate((linear, rbf, poly, sig)):
#     #defines how many plots: 2 rows, 2columns=> leading to 4 plots
#     plt.subplot(2, 2, i + 1) #i+1 is the index
#     #space between plots
#     plt.subplots_adjust(wspace=0.4, hspace=0.4) 
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn,     edgecolors='grey')
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(titles[i])
#     plt.show()

# # make predictions from the data 
# linear_pred = linear.predict(X_test)
# poly_pred = poly.predict(X_test)
# rbf_pred = rbf.predict(X_test)
# sig_pred = sig.predict(X_test)

# # Step5 CHeck classifier accuracy on test data and see results
# # retrieve the accuracy and print it for all 4 kernel functions
# accuracy_lin = linear.score(X_test, y_test)
# accuracy_poly = poly.score(X_test, y_test)
# accuracy_rbf = rbf.score(X_test, y_test)
# accuracy_sig = sig.score(X_test, y_test)
# print("Accuracy Linear Kernel:", accuracy_lin)
# print("Accuracy Polynomial Kernel:", accuracy_poly)
# print("Accuracy Radial Basis Kernel:", accuracy_rbf)
# print("Accuracy Sigmoid Kernel:", accuracy_sig)
