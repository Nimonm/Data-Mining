"""
=========== ========================================================
date: 2019/10/05
author: zhu li
=========== ========================================================
Clustering text documents using k-means
====================================================================
"""
print(__doc__)
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation

import logging
from optparse import OptionParser
import sys
from time import time
from sklearn.neighbors import kneighbors_graph
import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

# 我觉得这个输出太干扰了，没什么用所以注释掉了
# print(__doc__)
# op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)
# X = PCA(n_components=5).fit_transform(X.toarray())    # n_components=[2,3,4,5]
X = LatentDirichletAllocation(n_components=2, random_state=0).fit_transform(X.toarray())

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

clustering_algorithms0 = [ten_means]
clustering_algorithms1 = [affinity_propagation, mean_shift, spectral, ward, average_linkage, dbscan]
clustering_names0 = ['MB_KMeans']
clustering_names1 = ['AffinityPro', 'MeanShift', 'Spectral', 'Ward', 'Agglom', 'DBSCAN']

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
print()

# -------------------------------------------------------------------------
# evaluation for clusters
def evaluate0(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),))


def evaluate1(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), 0000,
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),))
# --------------------------------------------------------------------------

print(82 * '_')
print('clusters\ttime\tinertia\tnorm\thomo\tcomp')
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


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
