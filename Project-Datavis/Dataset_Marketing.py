# import pandas
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set(style="darkgrid")
# data = pandas.read_csv("D:\\Data_vis\\Project-Datavis\\dataset\\marketing.csv")
# print(data)
# print("_________________________________")
# groupby_gender = data.groupby('Sex')
# print(groupby_gender.mean())
#
#

import matplotlib.pyplot as plt
import pandas as pd

# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
# planets = sns.load_dataset('planets')

import pandas as pd
tips = pd.read_csv('seaborn-data-master/tips.csv')

# # Create the boxplot
# ax = sns.boxplot(x="total_bill", data=tips)
# ax.set(xlim=(0, 100))
# plt.show()