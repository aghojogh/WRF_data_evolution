import numpy as np
from utils import map_dataset_dimensionality


def predict_next_shape(X_history, min_n=200, max_n=5000, min_d=1, max_d=20):
    ns = np.array([X.shape[0] for X in X_history], dtype=float)
    ds = np.array([X.shape[1] for X in X_history], dtype=float)

    def extrapolate(vals):
        if len(vals) == 1:
            return vals[-1]
        if len(vals) == 2:
            return vals[-1] + (vals[-1] - vals[-2])

        v1 = vals[-1] - vals[-2]
        v0 = vals[-2] - vals[-3]
        a = v1 - v0
        return vals[-1] + v1 + 0.5 * a

    pred_n = int(np.clip(round(extrapolate(ns)), min_n, max_n))
    pred_d = int(np.clip(round(extrapolate(ds)), min_d, max_d))

    return pred_n, pred_d


def generate_balanced_family_candidate(
    X_history,
    y_history,
    target_n,
    target_d,
    random_state=0,
    preserve_class_balance=True
):
    """
    Generate a candidate using roughly equal contribution from all previous datasets.

    This is the most important generator for Wittgensteinian family resemblance:
        X_next should contain characteristics of X1, X2, ..., XT,
        not only the last dataset.
    """
    rng = np.random.default_rng(random_state)

    T = len(X_history)

    # Equal number of samples from each previous dataset.
    base = target_n // T
    remainder = target_n % T

    X_parts = []
    y_parts = []

    for t, X_t in enumerate(X_history):
        n_take = base + (1 if t < remainder else 0)

        X_aligned = map_dataset_dimensionality.align_dataset_to_target_dim(
            X_t,
            target_d=target_d,
            random_state=random_state + 10 * t
        )

        y_t = None if y_history is None else y_history[t]

        if y_t is None or not preserve_class_balance:
            X_sampled, y_sampled = sample_rows_with_labels(
                X_aligned,
                y_t,
                n=n_take,
                random_state=random_state + 100 * t
            )
        else:
            X_sampled, y_sampled = fair_sample_rows_by_label(
                X_aligned,
                y_t,
                n=n_take,
                random_state=random_state + 100 * t
            )

        X_parts.append(X_sampled)

        if y_history is not None:
            y_parts.append(y_sampled)

    X = np.vstack(X_parts)

    if y_history is None:
        y = None
    else:
        y = np.concatenate(y_parts)

    # Shuffle after balanced construction.
    idx = rng.permutation(len(X))
    X = X[idx]

    if y is not None:
        y = y[idx]

    # Add very small perturbation so it is not merely a copy.
    noise_scale = estimate_noise_scale_from_history(X_history)
    X = X + rng.normal(0, 0.01 * noise_scale, size=X.shape)

    name = "balanced_family_all_X"
    return X, y, name


def sample_rows_with_labels(X, y=None, n=1000, random_state=0):
    rng = np.random.default_rng(random_state)

    replace = n > len(X)
    idx = rng.choice(len(X), size=n, replace=replace)

    X_sampled = X[idx]

    if y is None:
        y_sampled = None
    else:
        y_sampled = y[idx]

    return X_sampled, y_sampled


def fair_sample_rows_by_label(X, y, n, random_state=0):
    """
    Sample rows fairly from all labels/classes present in y.
    This prevents one class from dominating a supervised candidate.
    """
    rng = np.random.default_rng(random_state)

    labels = np.unique(y)
    n_classes = len(labels)

    base = n // n_classes
    remainder = n % n_classes

    selected_idx = []

    for i, label in enumerate(labels):
        label_idx = np.where(y == label)[0]

        n_take = base + (1 if i < remainder else 0)

        replace = n_take > len(label_idx)

        chosen = rng.choice(
            label_idx,
            size=n_take,
            replace=replace
        )

        selected_idx.extend(list(chosen))

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)

    return X[selected_idx], y[selected_idx]


def estimate_noise_scale_from_history(X_history):
    scales = []

    for X in X_history:
        scales.append(np.mean(np.std(X, axis=0)))

    return float(np.mean(scales) + 1e-12)


def generate_candidate_pool(
    X_history,
    y_history,
    target_n,
    target_d,
    n_candidates=300,
    random_state=0,
    allow_shape_jitter=True,
    include_transition_candidates=False,
    balanced_candidate_fraction=0.85
):
    """
    Generate candidate datasets using only the previous sequence:

        X_history = [X1, X2, ..., Xt]
        y_history = [y1, y2, ..., yt] or None

    Important design choice:
    This version does NOT include one-source resampling candidates such as
    history_resample_X3, because those make the generated dataset look like
    only the last dataset.

    Candidate types:
    1. Balanced family candidates:
       roughly equal contribution from all previous datasets.

    2. Bounded mixture candidates:
       random mixture, but every previous dataset contributes nontrivially.

    Optional:
    transition candidates are disabled by default.
    """
    rng = np.random.default_rng(random_state)

    candidates = []

    n_balanced = int(round(n_candidates * balanced_candidate_fraction))
    n_balanced = max(1, min(n_balanced, n_candidates))

    # --------------------------------------------------------
    # 1. Mostly balanced family candidates.
    # --------------------------------------------------------
    for i in range(n_balanced):
        if allow_shape_jitter and i >= n_balanced // 2:
            n = int(np.clip(
                target_n + rng.normal(0, 0.10 * target_n),
                50,
                10000
            ))

            d = int(np.clip(
                target_d + rng.integers(-1, 2),
                1,
                max(1, target_d + 3)
            ))
        else:
            n = target_n
            d = target_d

        X, y, name = generate_balanced_family_candidate(
            X_history=X_history,
            y_history=y_history,
            target_n=n,
            target_d=d,
            random_state=random_state + i,
            preserve_class_balance=True
        )

        candidates.append((X, y, name))

    # --------------------------------------------------------
    # 2. Add bounded mixtures only.
    #    No one-source resampling.
    #    No interpolation by default.
    # --------------------------------------------------------
    remaining = n_candidates - len(candidates)

    for j in range(remaining):
        i = n_balanced + j

        if allow_shape_jitter:
            if j < remaining // 2:
                n = target_n
                d = target_d
            else:
                n = int(np.clip(
                    target_n + rng.normal(0, 0.10 * target_n),
                    50,
                    10000
                ))

                d = int(np.clip(
                    target_d + rng.integers(-1, 2),
                    1,
                    max(1, target_d + 3)
                ))
        else:
            n = target_n
            d = target_d

        X, y, name = generate_history_mixture_candidate(
            X_history=X_history,
            y_history=y_history,
            target_n=n,
            target_d=d,
            random_state=random_state + i,
            min_history_fraction=0.25
        )

        candidates.append((X, y, name))

    # --------------------------------------------------------
    # 3. Optional transition candidates.
    #    Keep disabled unless you explicitly want last-transition behavior.
    # --------------------------------------------------------
    if include_transition_candidates:
        n_extra = max(1, n_candidates // 10)

        for k in range(n_extra):
            X, y, name = generate_history_transition_candidate(
                X_history=X_history,
                y_history=y_history,
                target_n=target_n,
                target_d=target_d,
                random_state=random_state + 10000 + k
            )

            candidates.append((X, y, name))

    return candidates


def generate_history_resample_candidate(
    X_history,
    y_history,
    target_n,
    target_d,
    random_state=0
):
    rng = np.random.default_rng(random_state)

    source_idx = rng.integers(0, len(X_history))

    X_source = map_dataset_dimensionality.align_dataset_to_target_dim(
        X_history[source_idx],
        target_d=target_d,
        random_state=random_state
    )

    y_source = None if y_history is None else y_history[source_idx]

    X, y = sample_rows_with_labels(
        X_source,
        y_source,
        n=target_n,
        random_state=random_state + 1
    )

    noise_scale = estimate_noise_scale_from_history(X_history)
    X = X + rng.normal(0, 0.02 * noise_scale, size=X.shape)

    name = f"history_resample_X{source_idx + 1}"
    return X, y, name


def generate_history_mixture_candidate(
    X_history,
    y_history,
    target_n,
    target_d,
    random_state=0,
    min_history_fraction=0.15
):
    """
    Mixture candidate with a lower bound on contribution from every previous dataset.

    Unlike a pure Dirichlet mixture, this avoids candidates such as:
        5% X1 + 5% X2 + 90% X3
    """
    rng = np.random.default_rng(random_state)

    T = len(X_history)

    if T * min_history_fraction >= 1.0:
        min_history_fraction = 0.8 / T

    remaining_mass = 1.0 - T * min_history_fraction
    random_extra = rng.dirichlet(np.ones(T)) * remaining_mass
    weights = min_history_fraction + random_extra

    ns = rng.multinomial(target_n, weights)

    X_parts = []
    y_parts = []

    for i, ni in enumerate(ns):
        if ni == 0:
            continue

        Xi = map_dataset_dimensionality.align_dataset_to_target_dim(
            X_history[i],
            target_d=target_d,
            random_state=random_state + 10 * i
        )

        yi = None if y_history is None else y_history[i]

        if yi is None:
            X_sampled, y_sampled = sample_rows_with_labels(
                Xi,
                yi,
                n=int(ni),
                random_state=random_state + 100 * i
            )
        else:
            X_sampled, y_sampled = fair_sample_rows_by_label(
                Xi,
                yi,
                n=int(ni),
                random_state=random_state + 100 * i
            )

        X_parts.append(X_sampled)

        if y_history is not None:
            y_parts.append(y_sampled)

    X = np.vstack(X_parts)

    if y_history is None:
        y = None
    else:
        y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    X = X[idx]

    if y is not None:
        y = y[idx]

    noise_scale = estimate_noise_scale_from_history(X_history)
    X = X + rng.normal(0, 0.015 * noise_scale, size=X.shape)

    name = "bounded_history_mixture_all"
    return X, y, name


def generate_history_interpolation_candidate(
    X_history,
    y_history,
    target_n,
    target_d,
    random_state=0
):
    rng = np.random.default_rng(random_state)

    T = len(X_history)

    if T < 2:
        return generate_history_resample_candidate(
            X_history,
            y_history,
            target_n,
            target_d,
            random_state=random_state
        )

    i, j = rng.choice(T, size=2, replace=False)

    Xi = map_dataset_dimensionality.align_dataset_to_target_dim(
        X_history[i],
        target_d=target_d,
        random_state=random_state + 10
    )

    Xj = map_dataset_dimensionality.align_dataset_to_target_dim(
        X_history[j],
        target_d=target_d,
        random_state=random_state + 20
    )

    yi = None if y_history is None else y_history[i]
    yj = None if y_history is None else y_history[j]

    Xi_s, yi_s = sample_rows_with_labels(
        Xi,
        yi,
        n=target_n,
        random_state=random_state + 30
    )

    Xj_s, yj_s = sample_rows_with_labels(
        Xj,
        yj,
        n=target_n,
        random_state=random_state + 40
    )

    alpha = rng.uniform(0.2, 0.8)

    X = (1.0 - alpha) * Xi_s + alpha * Xj_s

    # Choose labels from one side. Labels are not optimized;
    # they are fixed metadata for supervised descriptors.
    if y_history is None:
        y = None
    else:
        if alpha < 0.5:
            y = yi_s
        else:
            y = yj_s

    noise_scale = estimate_noise_scale_from_history(X_history)
    X = X + rng.normal(0, 0.02 * noise_scale, size=X.shape)

    name = f"history_interpolation_X{i + 1}_X{j + 1}"
    return X, y, name


def generate_history_transition_candidate(
    X_history,
    y_history,
    target_n,
    target_d,
    random_state=0
):
    """
    A rough rule-following candidate:
    X_next ≈ X_last + (X_last - X_previous),
    but without assuming row correspondence.
    """
    rng = np.random.default_rng(random_state)

    if len(X_history) < 2:
        return generate_history_resample_candidate(
            X_history,
            y_history,
            target_n,
            target_d,
            random_state=random_state
        )

    X_prev = map_dataset_dimensionality.align_dataset_to_target_dim(
        X_history[-2],
        target_d=target_d,
        random_state=random_state + 10
    )

    X_last = map_dataset_dimensionality.align_dataset_to_target_dim(
        X_history[-1],
        target_d=target_d,
        random_state=random_state + 20
    )

    y_last = None if y_history is None else y_history[-1]

    X_prev_s, _ = sample_rows_with_labels(
        X_prev,
        None,
        n=target_n,
        random_state=random_state + 30
    )

    X_last_s, y = sample_rows_with_labels(
        X_last,
        y_last,
        n=target_n,
        random_state=random_state + 40
    )

    beta = rng.uniform(0.3, 1.0)

    X = X_last_s + beta * (X_last_s - X_prev_s)

    noise_scale = estimate_noise_scale_from_history(X_history)
    X = X + rng.normal(0, 0.02 * noise_scale, size=X.shape)

    name = "history_transition_last"
    return X, y, name