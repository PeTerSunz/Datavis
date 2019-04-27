__author__ = 'Nattachai Chaiwiriya'

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
x = boston.data
y = boston.target
columns = boston.feature_names
#create the dataframe
boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
print(boston_df.head())
print("_________________________________")
print(boston_df.shape)
print("_________________________________")
sns.boxplot(x=boston_df['DIS'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_df['INDUS'], boston_df['TAX'])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')

plt.show()

Q1 = boston_df.quantile(0.25)
Q3 = boston_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df_out = boston_df[~((boston_df < (Q1 - 1.5 * IQR)) |(boston_df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_out.shape

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(boston_df))
print(z)

threshold = 3
print(np.where(z > 3))

