import pandas as pd
import matplotlib as plt
import statsmodels.api as sm
from scipy.stats import norm, chi2_continquency, chi2
import numpy as np

# Read in data from excel sheet
data = pd.read_csv('./data.csv')

data.boxplot(by='diagnosis', column='radius_mean', grid=False)
data.plot.scatter(x='radius_mean', y='area_mean', s=10, c='red')

data.hist(column='area_mean', bins=30)
sm.qqplot(data.area_mean, norm, fit=True, line='45')

plt.show()


table = [[45, 55], [55, 45]]

stats, p, dof, expected = chi2_continquency(table)
print(chi2_continquency(table))

prob = 0.95
critical = chi2(prob, dof)

if (abs(stats) >= critical):
    print("Smoking status is related to gender")
else:
    print("Smoking status is not related to geneder")

# Replace diagnosis data with 0 for M and 1 for B
data.diagnosis.replace({"M":0, "B":1}, inplace=True)
print(data)