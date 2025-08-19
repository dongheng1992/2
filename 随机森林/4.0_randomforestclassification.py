import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# 设置随机数种子
np.random.seed(1)

# 加载鸢尾花数据集
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target_names[iris.target]
X = iris.data
y = iris.target

# 查看摘要统计信息
summary = data.describe()

print(summary)

# 数据拆分
idx = np.random.choice(range(len(iris.target)), size=int(len(iris.target) * 0.7), replace=False)
train = X[idx]
train_labels = y[idx]
test = np.delete(X, idx, axis=0)
test_labels = np.delete(y, idx, axis=0)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=1000, max_features=3, random_state=1) # 树的数量，

# 训练随机森林模型
rf_classifier.fit(train, train_labels)

# 预测测试集
predictions = rf_classifier.predict(test)

# 计算混淆矩阵
confusion = confusion_matrix(test_labels, predictions)

# 打印混淆矩阵
print("Confusion Matrix:\n", confusion)

# 计算分类报告
class_report = classification_report(test_labels, predictions, target_names=iris.target_names)

# 打印分类报告
print("Classification Report:\n", class_report)

# 计算整体准确度
overall_accuracy = accuracy_score(test_labels, predictions)

# 打印整体准确度
print("Overall Accuracy:", overall_accuracy)


