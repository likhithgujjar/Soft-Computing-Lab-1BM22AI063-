from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data  # features

# Number of clusters (species)
k = 3

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# Cluster assignments (crisp partition)
labels = kmeans.labels_

# Display results
for i in range(k):
    cluster_points = X[labels == i]
    print(f"Cluster {i+1} has {len(cluster_points)} points.")
    print(cluster_points[:3], "...")  # print first 3 points in cluster

# Optional: Show cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)


#optional
# Plotting (same as earlier)
x_index = 0
y_index = 2
colors = ['red', 'green', 'blue']

plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(X[labels == i, x_index], X[labels == i, y_index],
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, x_index], centers[:, y_index],
            c='black', marker='x', s=100, label='Centroids')

plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.title("KMeans Clustering on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
