import numpy as np
import matplotlib.pyplot as plt

sample_data = np.random.uniform(low=0.0, high=10.0, size=100)
sample_size = 10

sample_mean = [np.mean(np.random.choice(sample_data)) for i in range(0, 100)]
plt.hist(sample_mean, bins=sample_size)
plt.show()

sample_size = 30
sample_mean = [np.mean(np.random.choice(sample_data)) for i in range(0, 100)]
plt.hist(sample_mean, bins=sample_size)
plt.show()

mean = np.mean(sample_mean)
std  = np.std(sample_mean)

print(mean, std)