import numpy as np
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

def gmm_thresholds(image, k):
    data = image.ravel().reshape(-1, 1)
    gmm = GaussianMixture(n_components=k, covariance_type='full').fit(data)
    thresholds = np.sort(gmm.means_.flatten())
    return thresholds[:-1]

def kmeans_thresholds(image, k):
    data = image.ravel().reshape(-1, 1)  # Flatten image to a single column
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300).fit(data)
    thresholds = np.sort(kmeans.cluster_centers_.flatten())
    return thresholds[:-1]

def percentile_thresholds(image, k):
    percentiles = np.linspace(0, 100, k + 1)[1:-1]

    if len(image.shape)>2:
        thresholds = np.percentile(image, percentiles,axis=0)
    else:
        thresholds = np.percentile(image, percentiles)
    return thresholds


def equal_partition_thresholds(image, k):
    min_val, max_val = np.min(image), np.max(image)
    thresholds = np.linspace(min_val, max_val, k + 1)[1:-1]
    return thresholds


def percentile_thresholds(image, k):
    percentiles = np.linspace(0, 100, k + 1)[1:-1]
    thresholds = np.percentile(image, percentiles)
    return thresholds