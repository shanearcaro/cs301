import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('./data.csv')
print(data.head())

radius_mean = data['radius_mean']
mean_rd = np.mean(radius_mean)
std_rd  = np.std(radius_mean)
# print(mean_rd, std_rd)

# qq Plot for normality check
# prb_plt=stats.probplot((data['texture_mean']), plot=plt)
# plt.show()

# Uniform distribution for sampling distribution problems
uniform = np.random.uniform(0, 20, 50)

mean_uniform = [np.mean(np.random.choice(uniform)) for i in range (0, 100)]
plt.hist(mean_uniform)
plt.show()



