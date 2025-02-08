from ucimlrepo import fetch_ucirepo 
  

import os
print("Current working directory is:",os.getcwd())

# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

X.to_csv(r"classification/iris_X.csv",index=False)
y.to_csv(r"classification/iris_y.csv",index=False)