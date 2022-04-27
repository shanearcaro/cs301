from random import random
from sklearn.datasets import make_classification, make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
from numpy import mean
from numpy import absolute
from numpy import sqrt
from numpy import std

# ===============================
#       Q U E S T I O N # 1
# ===============================
data = pd.read_csv('/home/shane/Documents/School/Junior/CS301/Python/data/winequality-red.csv') 

# Set X and Y data
X = data.iloc[:, 1:5]
Y = data.iloc[:, 11]

# Set train and test variable
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
dt = DecisionTreeClassifier(criterion='gini', random_state=None, max_depth=7, min_samples_leaf=5)
dt.fit(x_train, y_train)
y_prediction = dt.predict(x_test)
plt.show()

# ===============================
#       Q U E S T I O N # 2
# ===============================
print("10 cross-validation score", np.mean(cross_val_score(dt, X, Y, cv=10)))
print("Precision: ", precision_score(y_test, y_prediction, average='macro'))
print("Recall: ", recall_score(y_test, y_prediction, average='macro'))
print("Accuracy: ", accuracy_score(y_test, y_prediction))
print("F1: ", f1_score(y_test, y_prediction, average='macro'))

# ===============================
#       Q U E S T I O N # 3
# ===============================
X, Y = make_regression(n_samples=1599, n_features=12)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
regression = LinearRegression().fit(x_train, y_train)
y_prediction_linear = regression.predict(x_test)

print("Linear Regression Coefficients:", regression.coef_)

# ===============================
#       Q U E S T I O N # 4
# ===============================
print("Mean Squared Error:", mean_squared_error(y_test, y_prediction_linear))