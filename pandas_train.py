import pandas as pd

df = pd.read_csv("C:\\Users\\PeTerSunz\\Desktop\\Datavis\\Students.csv")

# print(df)
# print (df.head()) # Show top 5 rows of dataframe
# print("_________________________________")
# print (len (df)) # Number of rows
# print("_________________________________")
# print (df.shape) # Show Number of rows & attributes
# print("_________________________________")
# print (df.tail())
# print("_________________________________")
# print (df.describe( ) ) # Show basic statistics of each attribute
# print("_________________________________")
# print (df['Gender'].describe(include= 'all') )
# print("_________________________________")
# print (df['Gender'].value_counts())
#     print (df(df['Male'].mean()))

# print (df.mean())

# print (df[['Age']].head())
# print (df[['Age']].tail())
# print (df.Age.max())
# print (df.Height.max())
# print (df.Weight.max())


# print (df[['Age','Gender','Weight','Height']].head())
# print("_________________________________")
# print(df[df.columns[0:3]].head())
# print("_________________________________")
# print(df.select_dtypes(include = 'number'))
# print("_________________________________")
# print(df.iloc[1:10,0:4])
# print("_________________________________")
# print(df[df['Age']>=23])
# print(len (df[df['Height']>160]))
print("_________________________________")
# print(df.sort_values('Age',ascending=False))
# print(df.sort_values(by='Height',ascending=False))
# print(df.sort_values(by='Height',ascending=False))
print(df[df.Gender.isin(['Male'])].Height.mean())
print(df[df.Gender.isin(['Female'])].Weight.sum())
# print (type(df[['Age']]))


# df[['total_bill', 'tip']] # Show only values from total_bill and tip columns
# df[df.columns [1:3]] # Show only values from columns from 1 to 2
# df.select_dtypes (include= 'number') # Show only numeric columns
# df.select_dtypes (exclude= 'number') # Show non-numeric columns
