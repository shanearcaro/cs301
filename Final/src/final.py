# Author: Shane Arcaro and Nicholas DeLello
# Date: 4/26/2022
# Assignment: Final Project

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the data
data = pd.read_csv('../data/fetal_health-1.csv')

# Three classes: normal, suspect, pathological

# Find correlation data between the data
correlation = data.corr()
print(correlation)

# Create and plot correlation map
# plt.figure(figsize=(15, 8))
# sns.heatmap(correlation, annot=True)
# plt.show()

# Get X and Y values from the data set
X = data.iloc[:, : -1]
Y = data.iloc[:,   -1]

# Find and print classification values
values, count = np.unique(Y, return_counts=True)
for i in range(0, len(count)):
    print(i, ":", count[i])

# Data is very unbalanced: Normal: 1655, Suspect: 295, Pathological: 176
labels = ['Normal', 'Suspect', 'Pathological']
plt.pie(count, labels=labels)
plt.title('Data Balance Distribution')
plt.show()

# How to fix data imbalance? Stratification of data
resample_data = resample(X, n_samples=2000, replace=True, stratify=X, random_state=1)
np.unique(resample_data, return_counts=True)




