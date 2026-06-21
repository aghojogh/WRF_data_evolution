import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score


def dataset_descriptor(X, sample_size=600, random_state=0):
    rng = np.random.default_rng(random_state)

    n, d = X.shape

    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    Z = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-12)

    D = pairwise_distances(Z)
    dist_vals = D[np.triu_indices_from(D, k=1)]

    pca = PCA(n_components=min(5, Z.shape[1]))
    pca.fit(Z)
    eig = pca.explained_variance_ratio_

    eig_pad = np.zeros(5)
    eig_pad[:len(eig)] = eig

    cov = np.cov(Z.T)
    cov = np.atleast_2d(cov)

    sil_scores = []
    for k in [2, 3, 4, 5]:
        if len(Z) > k:
            labels = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=random_state
            ).fit_predict(Z)

            try:
                sil = silhouette_score(Z, labels)
            except Exception:
                sil = 0.0
        else:
            sil = 0.0

        sil_scores.append(sil)

    desc = np.array([
        np.log(n),                                  # sample-size descriptor
        d,                                          # dimension descriptor
        np.mean(dist_vals),
        np.std(dist_vals),
        np.quantile(dist_vals, 0.10),
        np.quantile(dist_vals, 0.25),
        np.quantile(dist_vals, 0.50),
        np.quantile(dist_vals, 0.75),
        np.quantile(dist_vals, 0.90),
        np.trace(cov),                              # total variance
        np.linalg.cond(cov + 1e-6 * np.eye(d)),      # anisotropy
        *eig_pad,                                   # PCA spectrum
        *sil_scores                                 # cluster tendency
    ], dtype=float)

    return desc


def supervised_dataset_descriptor(X, y, sample_size=600, random_state=0):
    rng = np.random.default_rng(random_state)

    n = X.shape[0]

    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        Xs = X[idx]
        ys = y[idx]
    else:
        Xs = X
        ys = y

    Z = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-12)

    classes, counts = np.unique(ys, return_counts=True)
    proportions = counts / counts.sum()

    n_classes = len(classes)
    class_entropy = -np.sum(proportions * np.log(proportions + 1e-12))
    class_imbalance = proportions.max() - proportions.min()

    within_dists = []
    centroids = []

    for c in classes:
        Zc = Z[ys == c]

        if len(Zc) >= 2:
            Dc = pairwise_distances(Zc)
            vals = Dc[np.triu_indices_from(Dc, k=1)]
            within_dists.append(np.mean(vals))

        centroids.append(Zc.mean(axis=0))

    if len(within_dists) > 0:
        mean_within = np.mean(within_dists)
    else:
        mean_within = 0.0

    centroids = np.vstack(centroids)

    if len(centroids) >= 2:
        Dc = pairwise_distances(centroids)
        vals = Dc[np.triu_indices_from(Dc, k=1)]
        mean_between = np.mean(vals)
    else:
        mean_between = 0.0

    separability = mean_between / (mean_within + 1e-12)

    desc = np.array([
        n_classes,
        class_entropy,
        class_imbalance,
        mean_within,
        mean_between,
        separability,
    ], dtype=float)

    return desc


def descriptor_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    scale = np.abs(a) + np.abs(b) + 1e-6
    return np.mean(((a - b) / scale) ** 2)


def predict_next_descriptor(descriptors):
    F = np.asarray(descriptors)

    if len(F) == 1:
        return F[-1]

    if len(F) == 2:
        return F[-1] + (F[-1] - F[-2])

    v1 = F[-1] - F[-2]
    v0 = F[-2] - F[-3]
    acceleration = v1 - v0

    return F[-1] + v1 + 0.5 * acceleration


def weighted_history_descriptor(descriptors, decay=0.75):
    T = len(descriptors)
    weights = np.array([decay ** (T - 1 - i) for i in range(T)])
    weights = weights / weights.sum()

    return np.sum([w * d for w, d in zip(weights, descriptors)], axis=0)