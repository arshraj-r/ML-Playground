from sklearn.datasets import fetch_california_housing
import pandas as pd
housing = fetch_california_housing()
# print(housing.data.shape, housing.target.shape)
# print(housing.feature_names[0:6])
print
df=pd.DataFrame(housing.data,columns=[housing.feature_names])

# df.to_csv("classification/clifornia_housing.csv",index=False)
df["Price_Target"]=housing.target
df.to_csv("regression/california_housing.csv",index=False)
print(df.head(5))
print(housing.DESCR)