import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

data = pd.read_csv('Advertising(1).csv')
# print(data.tail(10))

X = data['TV'].values.reshape(-1, 1)
Y = data['sales'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y)
regression = LinearRegression.fit(x_train, y_train)
y_prediction_linear = regression.predict(x_test)

print(y_prediction_linear)
plt.scatter(x_test, y_test)
plt.plot(x_test, y_test)

plt.xlabel("Amount spent in ad for TV")
plt.ylabel("Sales obtained")
plt.show()

regression_dec = DecisionTreeRegressor(random_state=0, max_depth=5)
reg = regression_dec.fit(x_train, y_train)
y_prediction_tree = reg.predict(x_test)

tree.plot_tree(regression_dec)
plt.show()

