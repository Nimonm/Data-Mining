"""
=========== ========================================================
date: 2019/10/05
author: zhu li
=========== ========================================================
Shorthand    full name
=========== ========================================================
homo         homogeneity score
compl        completeness score
v-meas       V measure
ARI          adjusted Rand index
AMI          adjusted mutual information
silhouette   silhouette coefficient
=========== ========================================================

"""
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import kneighbors_graph
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

X = PCA(n_components=2).fit_transform(data)

# quantile : [0,1] ,这里我尝试了0.01到0.9，发现大于0.2的时候聚类结果只有一个簇
bandwidth = cluster.estimate_bandwidth(X, quantile=0.04)

connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)

# clusters
mean_shift = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
ten_means = cluster.MiniBatchKMeans(n_clusters=10)
spectral = cluster.SpectralClustering(n_clusters=10,eigen_solver='arpack',
                                    affinity="nearest_neighbors")
dbscan = cluster.DBSCAN(eps=0.2, min_samples=5)
affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)
# linkage: "ward", "complete", "average"，再这个数据集中，ward效果最好
ward = cluster.AgglomerativeClustering(n_clusters=10, linkage='ward',
                                      connectivity=connectivity)
average_linkage = cluster.AgglomerativeClustering(linkage="average",
                                                  affinity="euclidean", n_clusters=10,
                                                  connectivity=connectivity)
gmm = GaussianMixture(n_components=10)
clustering_algorithms0 = [ten_means]
clustering_algorithms1 = [affinity_propagation, mean_shift, spectral, ward, average_linkage, dbscan]
clustering_names0 = ['MB_KMeans']
clustering_names1 = ['AffinityPro', 'MeanShift', 'Spectral', 'Ward', 'Agglom', 'DBSCAN']

# evaluation for k-means with different init
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


# evaluation for clusters
def evaluate0(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t\t%.2fs\t%i\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),))


def evaluate1(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t\t%.2fs\t%i\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), 0000,
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),))


print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI-\tAMI-\tsilhouette')
bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)
bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

print()


"""
:param
nor: Normalized Mutual Information 
homo: Homogeneity
comp: Completeness
"""

print(82 * '_')
print('clusters\ttime\tinertia\tnor\t\thomo\tcomp')
# evaluate0(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#               name="k-means++", data=data)
for name, algorithm in zip(clustering_names0, clustering_algorithms0):
    algorithm.fit(X)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    evaluate0(algorithm, name, data=X)

for name, algorithm in zip(clustering_names1, clustering_algorithms1):
    algorithm.fit(X)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    evaluate1(algorithm, name, data=X)
    
print(82 * '_')


# Visualize the results on PCA-reduced data
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
# point in the mesh [x_min, x_max]x[y_min, y_max].
h = .02
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
