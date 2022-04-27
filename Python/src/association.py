from turtle import clear
import pandas as pd
from mlxtend.frequent_patters import apri

# Read in the data
data = pd.read_csv("data/retail_dataset.csv")
print(data.tail(5))

# Get unique values from the data
itemset = set(data['0'].unique())
print(itemset)

# Transform data into numbers
encoded_values = []

for index, row in data.iterrows():
    current_row = set(row)
    common = itemset.intersection(current_row)
    uncommon = list(itemset - current_row)

    # Change data values
    label = {}
    for item in common:
        label[item] = 1
    for item in uncommon:
        label[item] = 0

    encoded_values.append(label)

onehot_encoding = pd.DataFrame(encoded_values)
items = aprioori(onehot_encoding, min_support=0.2, use_colnames=True)
rules = association_rules(items, metric='confidence', min_threshold=0.60)