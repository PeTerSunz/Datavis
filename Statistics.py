_author_ = 'Nattachai Chaiwiriya'
import pandas
data = pandas.read_csv('brain_size.csv', sep=';', na_values=".")
print(data)
print("_________________________________")
print(data.shape)  #จะออกเป็น Row กับ column
print(data['Gender'])
print("_________________________________")
print(data[data['Gender'] == 'Female']['VIQ'].mean())
print(data[data['Gender'] == 'Female']['VIQ'].std())


# จัดกลุ่มตามเพสและปริ้นค่าเฉลี่ย
groupby_gender = data.groupby('Gender')
print(groupby_gender.mean())

for gender, value in groupby_gender['VIQ']:
          print((gender, value.mean()))


from pandas.tools import plotting
# import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
# plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
# plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])
# plt.show()

print("_________________________________")
data['Height'].fillna(method='pad', inplace=True)
data['Weight'].fillna(method='pad', inplace=True)

x = data['Height'].values
y = data['Weight'].values

import numpy as np
print (np.corrcoef(x, y))

# import matplotlib.pyplot as plt
# plt.xlabel("Height")
# plt.ylabel("Weight")
# plt.scatter(x, y)
# plt.show()

# from statsmodels.formula.api import ols
# model = ols("y ~ x", data).fit() # y ถูกทำนายโดย X  ก็คือ น้ำหนักถูกทำนายโดยส่วนสูง
# print(model.summary())


print("_________________________________")
import numpy as np
import scipy.stats
confidence = 0.95
a = 1.0 * data['Height'].values
n = len(a)
m, se = np.mean(a), scipy.stats.sem(a)
h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
print (m, m-h, m+h)