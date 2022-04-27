import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics
# Read from data files
data = pd.read_csv("train.csv")
data_test=pd.read_csv("test-1.csv")
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#################question 1#############
X=data.iloc[:,2:12]
#print(X.head)
Y=data.iloc[:,1]
#print(Y)
Y=Y.map({'M':1,'B':0})
X_test=data_test.iloc[:,2:12]
#print(X_test)
Y_test=data_test.iloc[:,1]
Y_test=Y_test.map({'M':1,'B':0})
#print(Y_test)
#=    SGDClassifier().fit(X,Y)

model= LogisticRegression().fit(X,Y)
#model = LogisticRegression(solver='lbfgs', max_iter=10).fit(X,Y)
y_pred= model.predict(X_test)
print("Recall of Task",recall_score(Y_test,y_pred))
#####Recall 83.33 %

print("Precision of Task",precision_score(Y_test,y_pred))
print("Accuracy of Task",accuracy_score(Y_test,y_pred))
######Precision score 59.677%

