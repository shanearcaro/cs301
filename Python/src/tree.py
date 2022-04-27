import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
data=pd.read_csv('Python/data/advertising.csv')
#print(data.tail(10))
X=data['TV'].values.reshape(-1,1)
Y=data['sales'].values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(X,Y)
#########Linear regression
regression=LinearRegression().fit(x_train,y_train)
y_predict_linear=regression.predict(x_test)
print(y_predict_linear)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_predict_linear)
plt.xlabel("Amount spent in ad for TV")
plt.ylabel("Sales obtained")
plt.show()
#########Desicion tree regression
###if using decision tree classifier we can specify entropy and gini index
regression_dec=DecisionTreeRegressor(random_state=0,max_depth=3)
reg=regression_dec.fit(x_train,y_train)
y_predict_tree=reg.predict(x_test)
####adding fontsize we can customise the fonts within tree
tree.plot_tree(regression_dec,fontsize=10)
plt.show()