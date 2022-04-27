import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_socre, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test-1.csv")
pd.set_option('display.float_format', lambda x: '%.5f' % x)

X, Y = make_regression(n_samples=569, n_features=10)
model = LogisticRegression.fit(X, Y)
plt.show()