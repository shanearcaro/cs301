import pandas as pd
import numpy as np
from sklearn import model_selection

data = pd.read_csv("./data.csv")
print(data.shape)

row = data.iloc[1]