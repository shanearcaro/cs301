from decimal import Decimal
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, max_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("/home/shane/Documents/School/Junior/CS301/Python/data/train.csv")
data_test = pd.read_csv('/home/shane/Documents/School/Junior/CS301/Python/data/test-1.csv')

# TASK 1

# Pick the correct columns for data
X = data.iloc[:, 2:12]
Y = data.iloc[:, 1]

# Alter data to work with 1 and 0 instead of M and B
Y = Y.map({'M': 1, 'B': 0})

# Set the training data
X_test = data_test.iloc[:, 2:12]
Y_test = data_test.iloc[:, 1]

# Alter training data to work with 1 and 0 instead of M and B
Y_test = Y_test.map({'M':1, 'B':0})

# Create model and prediction
model= LogisticRegression().fit(X,Y)
Y_pred = model.predict(X_test)

print("Recall:", recall_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))

# TASK 2
regression_decision = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0).fit(X, Y)
y_prediction_tree = regression_decision.predict(X_test)

print("F1:", f1_score(Y_test, y_prediction_tree))
print("Accuracy:", accuracy_score(Y_test, y_prediction_tree))

# TASK 3
X_mean_min = X.mean() + X.min()
Y_mean_min = Y.mean() + Y.min()

X_test_mean_min = X_test.mean() + X_test.min()

model = LogisticRegression().fit(X_mean_min, Y_mean_min)
Y_pred = model.predict(X_test_mean_min)
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))



