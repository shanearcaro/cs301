import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

# read the dataset
df = pd.read_csv('winequality-red.csv')
print('dataset length', len(df))
print(df.tail(10))
# to extract the correlation between features
correlation = df.corr()
print('correlation', correlation)
# resize the figure
plt.figure(figsize=(15, 8))
# use heatmap to visualise the correlation
sns.heatmap(correlation, annot=True)
plt.show()
# extract the features to X and label to Y
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
# to understand the number of labels in each class we use unique()
values, count = np.unique((Y), return_counts=True)
print(values, count)
# use train_test_split from sklearn
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# SDG classifier is good for binary classification but here we have multi class classification
model = SGDClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
# we use accuracy_score and pass the true label and predicted label
print('Accuracy of model', accuracy_score(y_test, y_pred))
# 1) use stratify from train_test split to acknowledge the imbalance in data
x_stratify_train, x_stratify_test, y_stratify_train, y_stratify_test = train_test_split(X, Y, stratify=df['quality'],
                                                                                        test_size=0.3, random_state=42)
model_stratify = SGDClassifier().fit(x_stratify_train, y_stratify_train)
y_predict_stratify = model_stratify.predict(x_stratify_test)
print('Accuracy of startified k-fol model', accuracy_score(y_stratify_test, y_predict_stratify))
# random forest
# accuracy with SDG classifier is pretty low hence we switch to random forest classifier
model_stratify_randomforest = RandomForestClassifier().fit(x_stratify_train, y_stratify_train)
y_predict_stratify_randomforest = model_stratify_randomforest.predict(x_stratify_test)
print('Accuracy of startified k-fol model with random forest',
      accuracy_score(y_stratify_test, y_predict_stratify_randomforest))
# to use confusion_matrix() pass in the parameters true label followed by predicted label
confusionmatrix = confusion_matrix(y_stratify_test, y_predict_stratify_randomforest)
plt.figure(figsize=(15, 8))
sns.heatmap(confusionmatrix, annot=True)
# plt.show()
# 2 method of stratification is perform oversampling in minority class and under sampling in majority class
# we start off by extracting the columns corresponding to each label
df_minority3 = df[Y == 3]
df_minority4 = df[Y == 4]
df_majority5 = df[Y == 5]
df_majority6 = df[Y == 6]
df_minority8 = df[Y == 8]
df_7 = df[Y == 7]
# sinc label 7 has 199 feature we neglect that and add more data points to class 3,4,8 and reduce no of samples from 5 and 6
# to perform oversampling /add more data tuples we keep replace as TRue and to undersample we keep replace=False
df_sample_minority3 = resample(df_minority3, replace=True, n_samples=250)
df_sample_minority4 = resample(df_minority4, replace=True, n_samples=250)
df_sample_minority8 = resample(df_minority8, replace=True, n_samples=250)
df_sample_majority5 = resample(df_majority5, replace=False, n_samples=250)
df_sample_majority6 = resample(df_majority6, replace=False, n_samples=250)
# once we have resampled we concatinate it back to create a new dataset with equal distribution of most classes
df_sample = pd.concat([df_sample_minority3, df_sample_minority4,
                       df_sample_majority5, df_sample_majority6, df_7,
                       df_sample_minority8])
# we extract the features and label from df_Sample
X_distribution = df_sample.iloc[:, :-1]
y_distribution = df_sample.iloc[:, -1]
# perform a 70-30 split
x_distribution_train, x_distirbution_test, y_distribution_train, y_distribution_test \
    = train_test_split(X_distribution, y_distribution)
model_distributed = RandomForestClassifier().fit(x_distribution_train, y_distribution_train)
y_distributed_predcit = model_distributed.predict(x_distirbution_test)
print("Accuracy of sampled model", accuracy_score(y_distribution_test, y_distributed_predcit))
