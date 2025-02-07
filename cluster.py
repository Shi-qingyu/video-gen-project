import numpy as np
from PIL import Image

from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import torch


case = "car-turn"
trajectories = torch.load(f"cache/DAVIS/Trajectories/{case}/trajectories.pth") # [T, N, 2]
h, w = np.array(Image.open(f"cache/DAVIS/Annotations/{case}/00000.png")).shape
print(h, w)
X = trajectories.permute(1, 0, 2).numpy()
N, T, _ = X.shape
X[:, :, 0] *= w
X[:, :, 1] *= h
X = X[:, 1:] - X[:, :-1]

silhouette_avg = []
k_range = range(3, 15)

for k in k_range:
    model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", verbose=True, random_state=42)
    y_pred = model.fit_predict(X)
    
    # 计算轮廓系数
    silhouette_avg.append(silhouette_score(X.reshape(N, (T-1) * 2), y_pred))

# 可视化轮廓系数 (Silhouette Score vs k)
plt.plot(k_range, silhouette_avg, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.savefig("test.jpg")

best_k_silhouette = k_range[np.argmax(silhouette_avg)]
print(f"best K: {best_k_silhouette}")

best_k = best_k_silhouette
model = TimeSeriesKMeans(n_clusters=best_k, metric="euclidean", verbose=True, random_state=42)
y_pred = model.fit_predict(X)

print("Cluster assignments:", y_pred)

for cluster_num in range(best_k):
    cluster_trajectories = X[y_pred == cluster_num]
    