import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_digits
from utils import candidate_generation, data_evolution, general


def load_mnist_or_digits(max_samples_per_digit=500, random_state=0):
    rng = np.random.default_rng(random_state)

    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)
        image_shape = (28, 28)
        source = "MNIST"
    except Exception as e:
        print("Could not load MNIST. Falling back to sklearn digits.")
        print("Reason:", e)

        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(int)
        image_shape = (8, 8)
        source = "sklearn_digits"

    X_parts, y_parts = [], []

    for digit in range(10):
        idx = np.where(y == digit)[0]

        if len(idx) > max_samples_per_digit:
            idx = rng.choice(idx, size=max_samples_per_digit, replace=False)

        X_parts.append(X[idx])
        y_parts.append(y[idx])

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    return X[idx], y[idx], image_shape, source


def make_mnist_group_dataset(
    X_all,
    y_all,
    digit_group,
    n_per_digit=300,
    random_state=0
):
    rng = np.random.default_rng(random_state)

    X_parts, y_parts = [], []

    for digit in digit_group:
        idx = np.where(y_all == digit)[0]
        size = min(n_per_digit, len(idx))
        idx = rng.choice(idx, size=size, replace=False)

        X_parts.append(X_all[idx])
        y_parts.append(y_all[idx])

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def plot_mnist_datasets(
    X_history,
    y_history,
    names,
    generated_result,
    image_shape=(28, 28),
    n_images=12,
    random_state=0
):
    def fair_class_sample_indices(y, n_images, rng):
        labels = np.unique(y)
        n_classes = len(labels)

        base = n_images // n_classes
        remainder = n_images % n_classes

        selected_idx = []

        for i, label in enumerate(labels):
            label_idx = np.where(y == label)[0]

            n_take = base + (1 if i < remainder else 0)
            n_take = min(n_take, len(label_idx))

            chosen = rng.choice(
                label_idx,
                size=n_take,
                replace=False
            )

            selected_idx.extend(list(chosen))

        rng.shuffle(selected_idx)
        return selected_idx

    rng = np.random.default_rng(random_state)

    datasets = []

    for X, y, name in zip(X_history, y_history, names):
        datasets.append((X, y, name))

    if generated_result.get("X_before_refinement") is not None:
        datasets.append((
            generated_result["X_before_refinement"],
            generated_result["y"],
            "Generated before PyTorch"
        ))

    datasets.append((
        generated_result["X"],
        generated_result["y"],
        "Generated after PyTorch"
    ))

    n_rows = len(datasets)

    fig, axes = plt.subplots(
        n_rows,
        n_images,
        figsize=(1.05 * n_images, 1.05 * n_rows),
        squeeze=False,
        gridspec_kw={
            "wspace": 0.02,
            "hspace": 0.02
        }
    )

    fig.subplots_adjust(
        left=0.02,
        right=1.0,
        top=1.0,
        bottom=0.0,
        wspace=0.0,
        hspace=0.0
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (X, y, title) in enumerate(datasets):
        row_rng = np.random.default_rng(random_state + row)

        if y is not None:
            if row < 4:  # do not select indices for last (fourth) row which is after dataset refinement (so that the same corresponding images are shown in rows of before and after refinement)
                selected_idx = fair_class_sample_indices(
                    y=y,
                    n_images=n_images,
                    rng=row_rng
                )
        else:
            if row < 4:  # do not select indices for last (fourth) row which is after dataset refinement (so that the same corresponding images are shown in rows of before and after refinement)
                random.seed(random_state)
                selected_idx = random.sample([i for i in range(X.shape[0])], n_images)

        for col in range(n_images):
            ax = axes[row, col]
            ax.axis("off")

            if col < len(selected_idx):
                idx = selected_idx[col]

                img = X[idx].reshape(image_shape)
                img = np.clip(img, 0, 1)

                ax.imshow(img, cmap="gray")
                # ax.set_title(str(y[idx]), fontsize=8)

        axes[row, 0].set_ylabel(
            title,
            rotation=0,
            labelpad=25,
            fontsize=9,
            va="center",
            ha="right"
        )

    # plt.tight_layout()
    plt.show()


def run_experiments_on_mnist_dataset(
    digit_groups=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
    n_per_digit=300,
    n_candidates=80,
    random_state=0,
    supervised=True
):
    X_all, y_all, image_shape, source = load_mnist_or_digits(
        max_samples_per_digit=max(500, n_per_digit),
        random_state=random_state
    )

    print("Dataset source:", source)

    X_history = []
    y_history = []
    names = []

    for i, group in enumerate(digit_groups):
        X, y = make_mnist_group_dataset(
            X_all,
            y_all,
            digit_group=group,
            n_per_digit=n_per_digit,
            random_state=random_state + i
        )

        X_history.append(X)
        y_history.append(y)
        names.append("digits " + ",".join(map(str, group)))

    image_dim = image_shape[0] * image_shape[1]

    # Force MNIST image dimension.
    target_n, _ = candidate_generation.predict_next_shape(X_history)
    target_d = image_dim

    image_dim = image_shape[0] * image_shape[1]

    if not supervised:
        y_history_for_wrf = None
    else:
        y_history_for_wrf = y_history

    generated_result, rule_target_desc, family_target_desc, descriptors = (
        data_evolution.generate_next_by_rule_following(
            X_history,
            y_history=y_history_for_wrf,
            n_candidates=n_candidates,
            target_dimensionality=image_dim,
            random_state=random_state + 1000,
            use_labels=supervised,
            use_pytorch_refinement=True,
            pytorch_num_steps=200,
            pytorch_lr=1e-2,
            pytorch_device="cpu",

            # main unsupervised preferences
            weight_rule=1.00,
            weight_family=0.80,
            weight_last=0.20,
            weight_shape=1.00,
            weight_collapse=0.05,

            # main supervised preferences
            weight_supervised_rule=0.80,
            weight_supervised_family=0.70,
            weight_supervised_last=0.20
        )
    )

    print("\nMNIST rule-following experiment without PCA")
    print("------------------------------------------")
    for i, (name, X) in enumerate(zip(names, X_history), start=1):
        print(f"X{i}: {name}, shape={X.shape}")

    print("Generated kind:", generated_result["kind"])
    print("Generated shape:", generated_result["X"].shape)
    print("Target n:", generated_result["target_n"])
    print("Target d:", generated_result["target_d"])
    print("Best PyTorch epoch:", generated_result["best_refinement_epoch"])

    plot_mnist_datasets(
        X_history,
        y_history,
        names,
        generated_result,
        image_shape=image_shape,
        n_images=10,
        random_state=random_state
    )

    general.plot_refinement_loss(generated_result)

    return {
        "X_history": X_history,
        "y_history": y_history,
        "names": names,
        "generated_result": generated_result,
        "rule_target_descriptor": rule_target_desc,
        "family_target_descriptor": family_target_desc,
        "history_descriptors": descriptors,
        "source": source,
    }


if __name__ == '__main__':
    experiment = 2
    supervised = True
    if experiment == 1:
        digit_groups = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    elif experiment == 2:
        digit_groups = ((0, 1), (9, 4), (6, 7, 8))
    mnist_results = run_experiments_on_mnist_dataset(
        digit_groups=digit_groups,
        n_per_digit=300,
        n_candidates=80,
        random_state=0,
        supervised=supervised
    )