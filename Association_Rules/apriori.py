import pandas as pd

dataset = pd.read_csv('data/Market_Basket_Optimisation.csv', header=None)

# rule1
from efficient_apriori import apriori
transcations = []
for i in range(dataset.shape[0]):
    # 记录一笔transaction
    temp = set()
    for j in range(dataset.shape[1]):
        if str(dataset.values[i, j]) != 'nan':
            temp.add(dataset.values[i, j])
    transcations.append(temp)

itemsets, rules = apriori(transcations, min_support=0.05, min_confidence=0.3)
print('频繁项集：', itemsets)
print('关联规则：', rules)
'''

# rule2
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np

one_hot_encoder = TransactionEncoder()
data = np.array(dataset.values.tolist())
data = [[i for i in row if i != 'nan'] for row in data]
data_oh = one_hot_encoder.fit(data).transform(data)
df = pd.DataFrame(data_oh, columns=one_hot_encoder.columns_)
fre_itemsets = apriori(df, min_support=0.05, use_colnames=True)
rules = association_rules(fre_itemsets, metric='confidence', min_threshold=0.3)
print('频繁项集：', fre_itemsets)
print('关联规则：', rules)
'''