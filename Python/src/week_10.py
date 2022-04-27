from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import statsmodels.api as sm
import matplotlib.pyplot as plt

X, Y = make_classification(n_samples=1000, n_classes=2, random_state=1)

# Create normal plot
check_data = X[:, 1]
sm.qqplot(check_data, line='45')
# plt.show()

# Create model from training data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
model = GaussianNB().fit(x_train, y_train)

# Predict data
y_prediction = model.predict(x_test)

# Check how close the prediction is
print(confusion_matrix(y_test, y_prediction))

# Get accuracy score
print(accuracy_score(y_test, y_prediction))

# Plot ROC curve (need false positive rate and true positive rate)
fpr, tpr, _ = roc_curve(y_test, y_prediction)
plt.plot(fpr, tpr, label='ROC Curve')

# Set the limitations of the axis
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

# Set the labels
plt.xlabel("False Positive Rate of 1 - specificity")
plt.ylabel("True Positive Rate or sensitivity")

# On the graph, the blue curve is the ROC curve

plt.show()

# Get AOC value (area under curve)
print(roc_auc_score(y_test, y_prediction))
