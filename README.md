# Association-Rules
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
data=pd.read_csv(r"C:\Users\haree\OneDrive\Desktop\Online retail.csv",names=['items'])
data
data.info()
data.isnull().sum()
data.duplicated().sum()
data.drop_duplicates(inplace=True)
data.duplicated().sum()
data.describe()
data['items'].unique()
# Assuming the column name is 'items'
transactions = data['items'].apply(lambda x: x.split(',')).tolist()

transactions
data
# Use TransactionEncoder to transform the transactions into a binary matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_binary = pd.DataFrame(te_ary, columns=te.columns_)

# Display the transformed dataframe
print(df_binary.head())
# Apply the Apriori algorithm to find frequent itemsets with a minimum support of 0.02
frequent_itemsets = apriori(df_binary, min_support=0.02, use_colnames=True)

# Display the frequent itemsets
print(frequent_itemsets)
# Generate association rules with a minimum confidence of 0.3
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# Display the association rules
print(rules)
