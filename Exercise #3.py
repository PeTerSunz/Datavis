__author__ = 'Nattachai Chaiwiriya'
import pandas
data = pandas.read_csv('tips.csv', sep=',', na_values=".")
print(data)
print("_________________________________")
# print(data['total_bill'].describe(include='all'))

#ดูข้อมูลทั้งหมดที่มาคำนวณแล้ว
print(data['total_bill'].describe())
# print(data['total_bill'].mean())

print("_________________________________")
# จำนวนคนเฉลี่ยต่อโต๊ะของลูกค้าที่เป็นเพศชาย (sex = male)
print(data[data['sex'] == 'Male']['size'].mean())

print("_________________________________")
# ค่าเฉลี่ยของ tip แบ่งตามกลุ่มช่วงเวลา (time)
groupby = data.groupby('time')
print(groupby.mean())

print("_________________________________")
# ค่า Correlation ของ size และ tip
data['size'].fillna(method='pad', inplace=True)
data['tip'].fillna(method='pad', inplace=True)

x = data['size'].values
y = data['tip'].values

import numpy as np
print (np.corrcoef(x, y))

# แสดงเป็นกราฟ
# import matplotlib.pyplot as plt
# plt.xlabel("size")
# plt.ylabel("tip")
# plt.scatter(x, y)
# plt.show()

print("_________________________________")
# หาค่าเฉลี่ยแบบช่วงของค่า tip ทีระดับ confidence = 0.95
import numpy as np
import scipy.stats
confidence = 0.95
a = 1.0 * data['tip'].values
n = len(a)
m, se = np.mean(a), scipy.stats.sem(a)
h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
print (m, m-h, m+h)


