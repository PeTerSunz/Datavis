__author__ = 'Nattachai Chaiwiriya'

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="darkgrid")

# load data from https://github.com/mwaskom/seaborn-data


tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker", data=tips);
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);
plt.show()


'''
sns.catplot(x="day", y="total_bill", data=tips);
sns.catplot(x="day", y="total_bill", kind="swarm", data=tips);
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);
sns.catplot(x="size", y="total_bill", kind="swarm",
data=tips.query("size != 3"));
plt.show()
'''

'''
sns.catplot(data=tips, orient="h", kind="box");
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
sns.catplot(data=tips, orient="h", kind="box");
plt.show()
'''


'''
sns.relplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker", col="time", aspect=.6,
kind="swarm", data=tips);
plt.show()
'''