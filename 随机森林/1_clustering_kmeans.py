import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 读取数据
data = pd.read_csv('seed.csv')
#data= pd.read_csv("iris.csv") #sepal萼片 和 petal 花瓣 长宽数据

# 打印列名
print(data.columns)
# 提取特征列
X = data.iloc[:, :-1].values



# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# 使用K-means进行聚类
k = 3  # 假设我们要分成3个簇，可以根据需求调整
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)
cluster_labels = kmeans.labels_

# 将聚类标签添加到数据框中
data['Cluster'] = cluster_labels

# 使用PCA进行降维以便于可视化
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# 可视化聚类结果
plt.figure(figsize=(10, 7))
for cluster in range(k):
    plt.scatter(reduced_data[cluster_labels == cluster, 0], reduced_data[cluster_labels == cluster, 1], label=f'Cluster {cluster+1}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering')
plt.legend()
plt.show(dpi=300)
