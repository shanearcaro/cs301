# Author: Shane Arcaro and Nicholas DeLello
# Date: 4/26/2022
# Assignment: Final Project

from random import seed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import SGDClassifier, LinearRegression, LogisticRegression

# Read the data
data = pd.read_csv('../data/fetal_health-1.csv')

# Three classes: normal, suspect, pathological

# ===============================
#           T A S K # 1          
# ===============================

# Get X and Y values from the data setcd cd
X = data.iloc[:, : -1]
Y = data.iloc[:,   -1]

# Find and print classification values
values, count = np.unique(Y, return_counts=True)
print(values, count)

# Function used to show percentage on pi chart
def func(pct):
    return "{:1.1f}%".format(pct)

# Data is very unbalanced: Normal: 1655, Suspect: 295, Pathological: 176
# labels = ['Normal', 'Suspect', 'Pathological']
# plt.pie(count, labels=labels, autopct=lambda pct: func(pct))
# plt.title('Data Balance Distribution')
# plt.show()

# How to fix imbalanced data? One way is to oversample data
data_majority1 = data[Y == 1]
data_minority2 = data[Y == 2]
data_minority3 = data[Y == 3]

# Number of data points in new sample size
resample_size = 750

data_sample_majority1 = resample(data_majority1, replace=True, n_samples=resample_size)
data_sample_minority2 = resample(data_minority2, replace=True, n_samples=resample_size)
data_sample_minority3 = resample(data_minority3, replace=True, n_samples=resample_size)

data_sample = pd.concat([data_sample_majority1, data_sample_minority2, data_sample_minority3])
x_distribution = data_sample.iloc[:, : -1]
y_distribution = data_sample.iloc[:,   -1]
x_distribution_train, x_distribution_test, y_distribution_train, y_distribution_test = train_test_split(x_distribution, y_distribution)

# Find and print classification values
values, count = np.unique(y_distribution, return_counts=True)
print(values, count)

# Data is very unbalanced: Normal: 1655, Suspect: 295, Pathological: 176
# labels = ['Normal', 'Suspect', 'Pathological']
# plt.pie(count, labels=labels, autopct=lambda pct: func(pct))
# plt.title('Data Balance Distribution')
# plt.show()

# ===============================
#           T A S K # 2          
# ===============================

# Get feature importance using Random Forest Regression
# model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=10000, warm_start=True).fit(x_distribution_train, y_distribution_train)
# importance = pd.DataFrame(data={
#     'Attribute': x_distribution_train.columns,
#     'Importance': model.coef_[0]
# })
# importance = importance.sort_values(by='Importance', ascending=False)

# plt.bar(x=importance['Attribute'], height=importance['Importance'], color='#087E8B')
# plt.title('Feature importance', size=21)
# plt.xticks(rotation='vertical')
# plt.show()

# print(importance)

# Significance values change every time because of oversampling and undersampling
# Don't know how to keep these the same yet

# I'm not exactly sure how to do this statistical significance part Nick. I'm going to leave this up to you and keep working

# ===============================
#           T A S K # 3          
# ===============================

# This is a multiclass problem that can be treated as regression
# Just need 2 different models to classify the features into the three fetal health states
# Using Statify classifier and RandomForest classifier

# Create stratify classifier
x_stratify_train, x_stratify_test, y_stratify_train, y_stratify_test = train_test_split(x_distribution, y_distribution, test_size=0.3, stratify=y_distribution)
model_stratify = SGDClassifier().fit(x_stratify_train, y_stratify_train)
y_prediction_stratify = model_stratify.predict(x_stratify_test)

# Create Random Forest classifier
model_stratify_randomforest = RandomForestClassifier().fit(x_stratify_train, y_stratify_train)
y_predicition_stratify_randomforest = model_stratify_randomforest.predict(x_stratify_test)

# ===============================
#           T A S K # 4          
# ===============================

confusionmatrix = confusion_matrix(y_stratify_test, y_prediction_stratify)
plt.figure(figsize=(15, 8))
sns.heatmap(confusionmatrix, annot=True)
plt.show()

confusionmatrix_forest = confusion_matrix(y_stratify_test, y_predicition_stratify_randomforest)
plt.figure(figsize=(15, 8))
sns.heatmap(confusionmatrix_forest, annot=True)
plt.show()

# ===============================
#           T A S K # 5          
# ===============================

# Stratify scores
# print("Area Under ROC:      ", roc_auc_score(y_stratify_test, y_prediction_stratify))
print("F1 Score:            ", f1_score(y_stratify_test, y_prediction_stratify))
print("Area Under Precision:", precision_recall_curve(y_stratify_test, y_prediction_stratify))