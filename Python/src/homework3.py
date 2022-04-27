from random import random
from sklearn.datasets import make_classification, make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
from numpy import mean
from numpy import absolute
from numpy import sqrt
from numpy import std

# Use iloc command to select all attributes
# Functions precesion_score, recall_score use with y_test, y_predict

data = pd.read_csv('Python/data/winequality-red.csv').iloc[0: 12]
X, Y = make_classification(n_samples=1599, n_classes=2, random_state=1)
x_train , x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Create the decision tree
regression_dec = DecisionTreeRegressor(random_state=0, max_depth=5)
reg = regression_dec.fit(x_train, y_train)
y_prediction_tree = reg.predict(x_test)
tree.plot_tree(regression_dec, fontsize=10)
plt.show()

X, Y = make_regression(n_samples=1599,n_features=12)
model = LinearRegression()
##cross_val_score() function will be used to perform the evaluation for 10 fold
cross_val = KFold(n_splits=10, random_state=1, shuffle=True)

##you can plug in different scoring attributes like accuracy
scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cross_val, n_jobs=-1)

# create dataset
X, Y = make_classification(n_samples=1599, n_classes=2, random_state=1)
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
precision = cross_val_score(model, X, Y, scoring='precision', cv=cv, n_jobs=-1)
recall = cross_val_score(model, X, Y, scoring='recall', cv=cv, n_jobs=-1)
accuracy = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
f1 = cross_val_score(model, X, Y, scoring='f1', cv=cv, n_jobs=-1)
# report performance
print("Precision: ", mean(precision))
print("Recall: ", mean(recall))
print("Accuracy:", mean(accuracy))
print("F1: ", mean(f1))