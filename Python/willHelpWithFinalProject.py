from random import Random
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
data = pd.read_csv('data/winequality-red.csv')

# Find the correlation between the data
correlation = data.corr()
print(correlation)

# Create and plot correlation map
plt.figure(figsize=(15,8))
sns.heatmap(correlation, annot=True)
# plt.show()

# Get X and Y values from the data set
X = data.iloc[:, :-1]
Y = data.iloc[:,  -1]

values, count = np.unique((Y), return_counts=True)
print(values, count)

# Get training data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Get predictions using a model
model = SGDClassifier().fit(x_train, y_train)
y_prediction = model.predict(x_test)

# Get the accuracy score
print("Accuracy Score:", accuracy_score(y_test, y_prediction))

# Create stratify classifier
x_stratify_train, x_stratify_test, y_stratify_train, y_straitfy_test = train_test_split(X, Y, stratify=data['quality'], test_size=0.3, random_state=True)
model_stratify = SGDClassifier().fit(x_stratify_train, y_stratify_train)
y_prediction_stratify = model_stratify.predict(x_stratify_test)

# Get the accuracy score of stratify
print('Accuracy Stratify:', accuracy_score(y_straitfy_test, y_prediction_stratify))

# Random Forest
model_stratify_randomforest = RandomForestClassifier().fit(x_stratify_train, y_stratify_train)
y_prediction_stratify_randomforest = model_stratify_randomforest.predict(x_stratify_test)

# Get the accuracy score of stratify random forest
print("Accuracy Random Forest:", accuracy_score(y_straitfy_test, y_prediction_stratify_randomforest))

confusionmatrix = confusion_matrix(y_straitfy_test, y_prediction_stratify_randomforest)
plt.figure(figsize=(15, 8))
sns.heatmap(confusionmatrix, annot=True)
# plt.show()

# Features minority or majority (excluding feature 7)
data_minority3 = data[Y==3]
data_minority4 = data[Y==4]
data_majority5 = data[Y==5]
data_7 = data[Y==7]
data_majority6 = data[Y==6]
data_minority8 = data[Y==8]

data_sample_minority3 = resample(data_minority3, replace=True, n_samples=250)
data_sample_minority4 = resample(data_minority4, replace=True, n_samples=250)
data_sample_minority8 = resample(data_minority8, replace=True, n_samples=250)
data_sample_majority5 = resample(data_majority5, replace=True, n_samples=250)
data_sample_majority6 = resample(data_majority6, replace=True, n_samples=250)

data_sample = pd.concat([data_sample_minority3, data_sample_minority4, data_sample_majority5, data_sample_majority6, data_7, data_sample_minority8])
x_distribution = data_sample.iloc[:, : -1]
y_distribution = data.sample.iloc[:,   -1]
x_distribution_train, x_distribution_test, y_distribution_train, y_distribution_test = train_test_split(x_distribution, y_distribution)

model_distributed = RandomForestClassifier().fit(x_distribution_train, y_distribution_train)
y_distributed_predict = model_distributed.predict(x_distribution_test)