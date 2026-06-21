import numpy as np
from sklearn.decomposition import PCA


def align_dataset_to_target_dim(X, target_d, random_state=0):
    """
    Convert X to target_d using only X itself.

    If d == target_d: return copy.
    If d > target_d: PCA reduction, but only when PCA is valid.
    If d < target_d: pad with small noise.

    For image data like MNIST, it is better to avoid dimension changes
    and call generate_candidate_pool(..., allow_shape_jitter=False).
    """
    rng = np.random.default_rng(random_state)

    n, d = X.shape

    if d == target_d:
        return X.copy()

    if d > target_d:
        max_valid_components = min(n, d)

        if target_d > max_valid_components:
            raise ValueError(
                f"Cannot reduce from d={d} to target_d={target_d} "
                f"because PCA requires target_d <= min(n, d)={max_valid_components}."
            )

        pca = PCA(n_components=target_d, random_state=random_state)
        return pca.fit_transform(X)

    feature_std = np.mean(np.std(X, axis=0)) + 1e-12

    extra = rng.normal(
        loc=0.0,
        scale=0.05 * feature_std,
        size=(n, target_d - d)
    )

    return np.hstack([X, extra])