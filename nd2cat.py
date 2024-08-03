
#
# nd2cat (n-dimensional 2 categorical)
# Author: Johan Ofverstedt
#

import numpy as np
import skimage.color

#from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def relabel_clusters(clusters):
    k = clusters.shape[0]
    ch = clusters.shape[1]

    def dist(a, b):
        return np.sqrt(np.sum(np.square(a-b)))

    # used will be a list of tuples of (centroid_sum, n)
    used = []
    unused = list(range(k))
    prev_cluster = np.zeros((ch,))
    while len(unused) > 0:
        best_i = 0
        best_d = dist(prev_cluster, clusters[unused[0], :])
        for i in range(1, len(unused)):
            d = dist(prev_cluster, clusters[unused[i], :])
            if d < best_d:
                best_d = d
                best_i = i
        prev_cluster = clusters[unused[best_i], :]
        used.append(unused[best_i])
        unused.pop(best_i)
    return np.array(used)

def quantize_equally_spaced(I, k):
    original_shape = tuple(I.shape)
    sz = np.prod(original_shape[:-1])
    d = original_shape[-1]
    ktot = k**d
    image_array = np.reshape(I, (sz, d))
    image_out = np.zeros((sz,))
    for i in range(d):
        im_q = np.floor(0.5 + image_array[:, i] * (k-1))
        image_out[:] *= k
        image_out[:] += im_q

    return image_out.reshape(original_shape[:-1])

def image2cat_mean_shift(I, bw=None, subset_size=10000, k=16):
    total_shape = I.shape
    spatial_shape = total_shape[:-1]
    channels = total_shape[-1]
    I_lin = I.reshape(-1, channels)
    inds = np.arange(I_lin.shape[0])
    chosen_inds = np.random.choice(inds, subset_size, replace=True)
    I_lin_train = I_lin[chosen_inds, :]
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(I_lin_train)
    clusters = ms.cluster_centers_
    if clusters.shape[0] > k:
        cluster_labels = AgglomerativeClustering(n_clusters=k, linkage='complete').fit_predict(clusters)
    I_res = ms.predict(I_lin)
    if clusters.shape[0] > k:
        I_res = cluster_labels[I_res]
    return I_res.reshape(spatial_shape)

def image2cat_kmeans(I, k, batch_size=100, max_iter=1000, random_seed=1000):
    total_shape = I.shape
    spatial_shape = total_shape[:-1]
    channels = total_shape[-1]
    if k == 1:
        return np.zeros(spatial_shape, dtype='int')
    I_lin = I.reshape(-1, channels)
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter = max_iter, batch_size = batch_size, random_state=random_seed).fit(I_lin)
    centers = kmeans.cluster_centers_
                
    I_res = kmeans.labels_

    labs = relabel_clusters(centers)
    I_res = labs[I_res]

    return I_res.reshape(spatial_shape)

def image2cat_kmeans_masked(I, M, k, batch_size=100, max_iter=1000, random_seed=1000):
    if M is None:
        return image2cat_kmeans(I, k, batch_size, max_iter, random_seed)
    total_shape = I.shape
    spatial_shape = total_shape[:-1]
    channels = total_shape[-1]
    if k == 1:
        return np.zeros(spatial_shape, dtype='int')
    I_lin = I.reshape(-1, channels)
    M_lin = M.reshape((M.size,))
    I_lin_masked = I_lin[M_lin, :]
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter = max_iter, batch_size = batch_size, random_state=random_seed).fit(I_lin_masked)
    centers = kmeans.cluster_centers_
    I_res = kmeans.predict(I_lin)
                
    labs = relabel_clusters(centers)
    I_res = labs[I_res]

    return I_res.reshape(spatial_shape)

def image2cat_pca(I, k, sigmas=None):
    total_shape = I.shape
    spatial_shape = total_shape[:-1]
    channels = total_shape[-1]
    I_lin = I.reshape(-1, channels)
    I_res = PCA(n_components=1).fit_transform(I_lin)
    if sigmas is not None:
        mn = np.mean(I_res, axis=None)
        stddev = np.std(I_res, axis=None, ddof=1)
        I_res = (I_res-mn) / (sigmas*stddev)

    I_res = np.clip(0.5 * (I_res + 1.0), a_min=0.0, a_max=1.0)

    return quantize_equally_spaced(I_res.reshape(spatial_shape+(1,)), k)

def apply_pca(I):
    if I.ndim == 3:
        total_shape = I.shape
        spatial_shape = total_shape[:-1]
        channels = total_shape[-1]
        I_lin = I.reshape(-1, channels)
        I_res = PCA(n_components=1).fit_transform(I_lin)
        return I_res.reshape(spatial_shape)
    else:
        return I

def cat_to_colors(I, c):
    I_hsv = np.zeros(I.shape+(3,))
    I_hsv[:, :, 1:] = 1.0
    I_h = np.zeros(I.shape)
    I_v = np.zeros(I.shape)
    for i in range(c):
        ind = np.where(I == i)
        I_h[ind] = i/float(c)
        I_v[ind] = 1.0
    I_hsv[:, :, 0] = I_h
    I_hsv[:, :, 2] = I_v
    res = skimage.color.hsv2rgb(I_hsv)
    return res
