# 在终端写pip install sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 加载Iris数据集
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target_names[iris.target]
X = iris.data
y = iris.target

# 查看摘要统计信息
summary = data.describe()

print(summary)

# 创建决策树分类器
clf = DecisionTreeClassifier()

# 训练模型
clf.fit(X, y)

# 绘制决策树的树状图
plt.figure(figsize=(12, 8), dpi=300)
# 绘制决策树，设置字号为12
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names.tolist(), fontsize=8)
# 保存图像为高分辨率的PNG文件
#plt.savefig('decision_tree_high_res.png', bbox_inches='tight')
# 显示图像
plt.show()

# 对数据集进行预测(没有进行切割处理)
X_pred = iris.data
y_true = iris.target

# 进行预测
y_pred = clf.predict(X_pred)

# 导入pandas库，将预测结果转换为DataFrame
y_pred_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_true})

# 计算混淆矩阵
confusion = confusion_matrix(y_true, y_pred)

# 打印混淆矩阵
print("Confusion Matrix:\n", confusion)

# 生成分类报告
class_report = classification_report(y_true, y_pred, target_names=iris.target_names)

# 打印分类报告
print("Classification Report:\n", class_report)