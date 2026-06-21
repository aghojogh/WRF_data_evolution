import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.datasets import make_moons, make_circles, make_blobs, make_s_curve
from sklearn.decomposition import PCA
from utils import data_evolution, general


def make_dataset(kind, n=1000, d=2, noise=0.05, random_state=0):
    rng = np.random.default_rng(random_state)

    if kind == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=random_state)

    elif kind == "circles":
        X, y = make_circles(
            n_samples=n,
            noise=noise,
            factor=0.45,
            random_state=random_state
        )

    elif kind == "blobs":
        X, y = make_blobs(
            n_samples=n,
            centers=3,
            cluster_std=0.7,
            random_state=random_state
        )

    elif kind == "anisotropic_blobs":
        X, y = make_blobs(
            n_samples=n,
            centers=3,
            cluster_std=0.65,
            random_state=random_state
        )
        X = X @ np.array([[1.0, 0.9], [-0.45, 1.25]])

    elif kind == "spiral":
        theta = np.linspace(0, 4 * np.pi, n)
        r = np.linspace(0.2, 2.0, n)
        X = np.c_[r * np.cos(theta), r * np.sin(theta)]
        X += rng.normal(0, noise, size=X.shape)
        y = (theta > 2 * np.pi).astype(int)

    elif kind == "gaussian":
        X = rng.normal(size=(n, 2))
        y = np.zeros(n, dtype=int)

    elif kind == "s_curve":
        X_3D, t = make_s_curve(n_samples=n, noise=0.05, random_state=0)
        y = (t>0).astype(int)
        X = X_3D[:, [0, 2]]

    else:
        raise ValueError(f"Unknown kind: {kind}")

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # Allow dimensions to differ.
    if d > 2:
        extra = rng.normal(0, 0.15, size=(n, d - 2))
        X = np.hstack([X, extra])
    elif d == 1:
        X = X[:, :1]

    return X, y


def plot_sequence(Xs, ys, names, generated_result):
    X_gen = generated_result["X"]
    y_gen = generated_result["y"]
    kind_gen = generated_result["kind"]
    score = generated_result["score"]

    if ys is None:
        ys = [None] * len(Xs)

    X_before = generated_result.get("X_before_refinement", None)

    if X_before is not None:
        total = len(Xs) + 2
    else:
        total = len(Xs) + 1

    fig, axes = plt.subplots(1, total, figsize=(4 * total, 4))

    for i, (X, y, name) in enumerate(zip(Xs, ys, names)):
        Xp = project_for_plot(X)

        if y is not None:
            axes[i].scatter(Xp[:, 0], Xp[:, 1], c=y, s=8)
        else:
            axes[i].scatter(Xp[:, 0], Xp[:, 1], c='b', s=8)
        axes[i].set_title(f"Observed X{i+1}\n{name}, shape={X.shape}")
        axes[i].axis("equal")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    col = len(Xs)

    if X_before is not None:
        Xp_before = project_for_plot(X_before)

        axes[col].scatter(Xp_before[:, 0], Xp_before[:, 1], c=y_gen, s=8)
        axes[col].set_title(
            # f"Generated before PyTorch\n{kind_gen}, shape={X_before.shape}"
            f"Before refinement\nShape={X_before.shape}"
        )
        axes[col].axis("equal")
        axes[col].set_xticks([])
        axes[col].set_yticks([])

        col += 1

    Xp = project_for_plot(X_gen)

    # title = f"Generated after PyTorch\n{kind_gen}, shape={X_gen.shape}\nscore={score:.4f}"
    # if generated_result.get("best_refinement_epoch", None) is not None:
    #     title += f"\nbest epoch={generated_result['best_refinement_epoch']}"
    title = f"After refinement\nShape={X_gen.shape}"

    axes[col].scatter(Xp[:, 0], Xp[:, 1], c=y_gen, s=8)
    axes[col].set_title(title)
    axes[col].axis("equal")
    axes[col].set_xticks([])
    axes[col].set_yticks([])

    plt.tight_layout()
    plt.show()


def project_for_plot(X):
    if X.shape[1] == 1:
        return np.c_[X[:, 0], np.zeros(len(X))]

    if X.shape[1] == 2:
        return X

    return PCA(n_components=2).fit_transform(X)


def run_experiments_on_toy_dataset(
    dataset_sequences=[("moons", "circles", "blobs")],
    n_candidates=300,
    random_state=10,
    supervised=True
):
    all_results = {}
    for perm_index, perm in enumerate(dataset_sequences):
        X_history = []
        y_history = []

        for i, kind in enumerate(perm):
            X, y = make_dataset(
                kind,
                n=800 + 300 * i,
                d=2 + i,
                noise=0.04 + 0.01 * i,
                random_state=random_state + 100 * perm_index + i
            )

            X_history.append(X)
            y_history.append(y)

        if not supervised:
            y_history = None

        generated_result, rule_target_desc, family_target_desc, descriptors = (
            data_evolution.generate_next_by_rule_following(
                X_history,
                y_history=y_history,
                n_candidates=n_candidates,
                target_dimensionality=None,
                random_state=random_state + 1000 * perm_index,
                use_labels=True,
                use_pytorch_refinement=True,
                pytorch_num_steps=500,
                pytorch_lr=1e-2,
                pytorch_device="cpu",

                # main unsupervised preferences
                weight_rule=0.50,
                weight_family=1.00,
                weight_last=0.00,
                weight_shape=0.70,
                weight_collapse=0.05,

                # main supervised preferences
                weight_supervised_rule=0.30,
                weight_supervised_family=1.00,
                weight_supervised_last=0.00
            )
        )

        print("\n" + "=" * 80)
        print(f"Permutation {perm_index + 1}: {' -> '.join(perm)}")
        print("=" * 80)
        print("Generated family:", generated_result["kind"])
        print("Generated shape:", generated_result["X"].shape)
        print("Total score:", generated_result["score"])
        print("Rule loss:", generated_result["rule_loss"])
        print("Family resemblance loss:", generated_result["family_resemblance_loss"])
        print("Last dataset loss:", generated_result["last_dataset_loss"])
        print("Collapse loss:", generated_result["collapse_loss"])

        plot_sequence(
            X_history,
            y_history,
            list(perm),
            generated_result
        )

        general.plot_refinement_loss(generated_result)

        all_results[perm] = {
            "X_history": X_history,
            "y_history": y_history,
            "generated_result": generated_result,
            "rule_target_descriptor": rule_target_desc,
            "family_target_descriptor": family_target_desc,
            "history_descriptors": descriptors,
        }

    return all_results


if __name__ == '__main__':
    datasets = [("moons", "circles", "blobs"), 
                ("moons", "blobs", "circles"), ("moons", "spiral", "circles"),
                ("spiral", "moons", "circles"), 
                ("s_curve", "moons", "blobs")]
    experiment = 2
    supervised = True
    if experiment == 1:
        dataset_sequences = permutations(datasets[0])
    elif experiment == 2:
        dataset_sequences = datasets[4]
    if not isinstance(dataset_sequences, list):
        dataset_sequences = [dataset_sequences]

    permutation_results = run_experiments_on_toy_dataset(
        dataset_sequences=dataset_sequences,
        n_candidates=300,
        random_state=10,
        supervised=supervised
    )
