# Step1 Import Packages:
from pandas import read_csv
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Step2 Load dataset


# Step3 Split data in training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)

# Step4 Classifier Training using SVM


# Step5 CHeck classifier accuracy on test data and see results