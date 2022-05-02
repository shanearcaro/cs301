import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt


df = pd.read_csv('./Python/data/retail_dataset.csv', sep=',')

## Print top 5 rows
print(df.head(5))


#Each row of the dataset represents items that were purchased together on the same day at the same store.
# The dataset is a sparse dataset as relatively high percentage of data is NA or NaN or equivalent.
#These NaNs make it hard to read the table.
# Let’s find out how many unique items are actually there in the table.

items = (df['0'].unique())
print("Unique items:", items)

#There are only 9 items in total that make up the entire dataset.


#############Data Preprocessing

#To make use of the apriori module given by mlxtend library, we need to convert the dataset according to
# it’s liking. apriori module requires a dataframe that has either 0 and 1 or True and False as data.
# The data we have is all string (name of items), we need to One Hot Encode the data.


itemset = set(items)
encoded_vals = []
for index, row in df.iterrows():
    rowset = set(row)
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)

print(encoded_vals[0])
ohe_df = pd.DataFrame(encoded_vals)




####################### Applying Apriori

freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
print("frequent Items:")
print(freq_items.head(7))


##########################  Mining Association Rules

rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
print("Rules: ")
print(rules.head())

