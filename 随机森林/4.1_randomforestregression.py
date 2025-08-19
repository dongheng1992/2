import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the airquality dataset
#airquality = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv', index_col=0)
airquality = pd.read_csv('airquality.csv')
# 保存 airquality
#airquality.to_csv("airquality.csv",index=False)
# Check summary statistics
summary = airquality.describe()
print(summary)

# Remove rows with NaN values
airquality = airquality.dropna()

# 数据拆分
idx = np.random.choice(range(len(airquality)), size=int(len(airquality) * 0.7), replace=False)
train = airquality.iloc[idx]
test = airquality.iloc[np.delete(range(len(airquality)), idx)]

# 创建随机森林回归器
rf_regressor = RandomForestRegressor(n_estimators=500, max_features=2, random_state=1)

# 训练随机森林模型
rf_regressor.fit(train.drop(columns=['Ozone']), train['Ozone'])

# 预测测试集
predictions = rf_regressor.predict(test.drop(columns=['Ozone']))

# 计算R2分数
r2 = r2_score(test['Ozone'], predictions)
print("R-squared:", r2)

# 绘制预测值和真实值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(test['Ozone'], predictions, alpha=0.7)
plt.plot([test['Ozone'].min(), test['Ozone'].max()], [test['Ozone'].min(), test['Ozone'].max()], linestyle='--', color='gray')  # Diagonal line
plt.xlabel('Actual Ozone')
plt.ylabel('Predicted Ozone')
plt.title('Actual vs Predicted Ozone')
plt.show()

