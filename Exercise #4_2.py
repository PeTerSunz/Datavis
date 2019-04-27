__author__ = 'Nattachai Chaiwiriya'
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# Detect & remove duplicate data
# การเช็คข้อมูลที่ซ้ำกัน และ การลบข้อมูลที่ซ้ำกันออก

titanic = sns.load_dataset("titanic")
print(titanic.head)
print("_________________________________1")
print(titanic.duplicated())
print("_________________________________2")
print(titanic.shape)
print("_________________________________3")
cd_titanic = titanic.drop_duplicates()
print(cd_titanic.shape)

# Detect & replace missing data
# ตรวจจับและแทนที่ข้อมูลที่หายไป

for i in range(len(cd_titanic.index)) :
    print("Nan in row ", i , " : " , cd_titanic.iloc[i].isnull().sum())

print(cd_titanic.isna().sum())
cd_titanic1 = cd_titanic.fillna(method='ffill')
cd_titanic2 = cd_titanic.sort_index().fillna(method='bfill')
cd_titanic3= cd_titanic.fillna(cd_titanic.mean())
print (cd_titanic1)
print (cd_titanic2)
print (cd_titanic3)
print("_________________________________4")
print (titanic.isna().sum())