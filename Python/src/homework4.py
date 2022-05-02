import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the data
data = pd.read_excel('./Python/data/homework4.xlsx', index_col=0)

te = TransactionEncoder()
fitted = te.fit(data)
te_ary = fitted.transform(data, sparse=True)
fittedData = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

# I cannot get column names to work properly for the life of me

freq_items = apriori(fittedData, min_support=0.03, use_colnames=True, verbose=1)
print("frequent Items:")
print(freq_items)

rules = association_rules(freq_items, metric="confidence", min_threshold=0.3)
print("Rules: ")
print(rules.head())


# Top 10 frequent items
print("Most Frequent Items:")
print(freq_items.sort_values(by='support', ascending=False).head(10))
