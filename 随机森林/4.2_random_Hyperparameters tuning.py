import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
# Load the airquality dataset
#airquality = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv', index_col=0)
airquality = pd.read_csv('airquality.csv')
# Check summary statistics
summary = airquality.describe()
print(summary)

# Remove rows with NaN values
airquality = airquality.dropna()

# 数据拆分
X= airquality.drop(columns=['Ozone'])
y= airquality['Ozone']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# 定义超参数网格
param_grid ={'n_estimators':[100,500,1000],
             "max_features":[1,2,3,4],
             'min_samples_split':[2,5,10],
             'min_samples_leaf':[1,2,4]}
# 创建随机森林回归器
rf_regressor =RandomForestRegressor(random_state=1)
#使用网格搜索寻找最佳超参数
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train,y_train)
# 打印最佳参数
best_params = grid_search.best_params_
print("Best Parameters:",best_params)







