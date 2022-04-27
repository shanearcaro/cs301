from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd


X,Y=make_regression(n_samples=1000,n_features=10)
model=LinearRegression()
##cross_val_score() function will be used to perform the evaluation for 10 fold
cv = KFold(n_splits=10, random_state=1, shuffle=True)
##you can plug in different scoring attributes like accuracy
scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
print(mean(absolute(scores)))
