import numpy as np
from sklearn.metrics import pairwise_distances
from utils import dataset_descriptors, candidate_generation, dataset_refinement
from tqdm import tqdm


def generate_next_by_rule_following(
    X_history,
    y_history=None,
    n_candidates=300,
    target_dimensionality=None,
    random_state=0,
    use_labels=True,
    use_pytorch_refinement=True,
    pytorch_num_steps=500,
    pytorch_lr=1e-2,
    pytorch_device="cpu",
    weight_rule=1.00,
    weight_family=0.60,
    weight_last=0.20,
    weight_shape=0.80,
    weight_collapse=0.05,
    weight_supervised_rule=0.80,
    weight_supervised_family=0.60,
    weight_supervised_last=0.20
):
    weights = {
        "rule": np.clip(weight_rule, 0, 1),
        "family": np.clip(weight_family, 0, 1),
        "last": np.clip(weight_last, 0, 1),
        "shape": np.clip(weight_shape, 0, 1),
        "collapse": np.clip(weight_collapse, 0, 1),
        "supervised_rule": np.clip(weight_supervised_rule, 0, 1),
        "supervised_family": np.clip(weight_supervised_family, 0, 1),
        "supervised_last": np.clip(weight_supervised_last, 0, 1),
    }

    descriptors = [
        dataset_descriptors.dataset_descriptor(X, random_state=random_state + i)
        for i, X in enumerate(X_history)
    ]

    rule_target_desc = dataset_descriptors.predict_next_descriptor(descriptors)
    # family_target_desc = weighted_history_descriptor(descriptors, decay=0.75)
    family_target_desc = dataset_descriptors.weighted_history_descriptor(descriptors, decay=1.00)
    last_desc = descriptors[-1]

    supervised_enabled = (
        use_labels
        and y_history is not None
        and len(y_history) == len(X_history)
    )

    if supervised_enabled:
        supervised_descriptors = [
            dataset_descriptors.supervised_dataset_descriptor(
                X,
                y,
                random_state=random_state + 100 + i
            )
            for i, (X, y) in enumerate(zip(X_history, y_history))
        ]

        supervised_rule_target_desc = dataset_descriptors.predict_next_descriptor(supervised_descriptors)
        # supervised_family_target_desc = weighted_history_descriptor(
        #     supervised_descriptors,
        #     decay=0.75
        # )
        supervised_family_target_desc = dataset_descriptors.weighted_history_descriptor(
            supervised_descriptors,
            decay=1.00
        )
        supervised_last_desc = supervised_descriptors[-1]
    else:
        supervised_descriptors = None
        supervised_rule_target_desc = None
        supervised_family_target_desc = None
        supervised_last_desc = None

        weights["supervised_rule"] = 0.0
        weights["supervised_family"] = 0.0
        weights["supervised_last"] = 0.0

    target_n, target_d = candidate_generation.predict_next_shape(X_history)

    target_n, target_d_predicted = candidate_generation.predict_next_shape(X_history)
    if target_dimensionality is None:
        target_d = target_d_predicted
    else:
        target_d = target_dimensionality

    candidates = candidate_generation.generate_candidate_pool(
        X_history=X_history,
        y_history=y_history if supervised_enabled else None,
        target_n=target_n,
        target_d=target_d,
        n_candidates=n_candidates,
        random_state=random_state + 1000,
        allow_shape_jitter=False,
        include_transition_candidates=False,
        balanced_candidate_fraction=0.90
    )

    # candidate scoring:
    raw_results = []
    for i, (X, y, kind) in enumerate(tqdm(candidates, desc="Candidate scroing")):
        desc = dataset_descriptors.dataset_descriptor(X, random_state=random_state + 2000 + i)

        rule_loss = dataset_descriptors.descriptor_distance(desc, rule_target_desc)
        family_loss = dataset_descriptors.descriptor_distance(desc, family_target_desc)
        last_loss = dataset_descriptors.descriptor_distance(desc, last_desc)

        n_loss = ((X.shape[0] - target_n) / (target_n + 1e-12)) ** 2
        d_loss = ((X.shape[1] - target_d) / (target_d + 1e-12)) ** 2
        shape_loss = n_loss + d_loss

        X_small = X[:min(len(X), 600)]
        D = pairwise_distances(X_small)
        collapse_loss = 1.0 / (np.std(D) + 1e-6)

        result = {
            "X": X,
            "y": y,
            "kind": kind,
            "descriptor": desc,
            "rule_loss": rule_loss,
            "family_resemblance_loss": family_loss,
            "last_dataset_loss": last_loss,
            "shape_loss": shape_loss,
            "collapse_loss": collapse_loss,
        }

        if supervised_enabled:
            sdesc = dataset_descriptors.supervised_dataset_descriptor(
                X,
                y,
                random_state=random_state + 3000 + i
            )

            result["supervised_descriptor"] = sdesc
            result["supervised_rule_loss"] = dataset_descriptors.descriptor_distance(
                sdesc,
                supervised_rule_target_desc
            )
            result["supervised_family_loss"] = dataset_descriptors.descriptor_distance(
                sdesc,
                supervised_family_target_desc
            )
            result["supervised_last_loss"] = dataset_descriptors.descriptor_distance(
                sdesc,
                supervised_last_desc
            )
        else:
            result["supervised_descriptor"] = None
            result["supervised_rule_loss"] = 0.0
            result["supervised_family_loss"] = 0.0
            result["supervised_last_loss"] = 0.0

        raw_results.append(result)

    loss_names = [
        "rule_loss",
        "family_resemblance_loss",
        "last_dataset_loss",
        "shape_loss",
        "collapse_loss",
        "supervised_rule_loss",
        "supervised_family_loss",
        "supervised_last_loss",
    ]

    # normalize the candidate scores:
    for loss_name in loss_names:
        values = np.array([r[loss_name] for r in raw_results])
        lo = np.quantile(values, 0.05)
        hi = np.quantile(values, 0.95)
        denom = hi - lo + 1e-12

        for r in raw_results:
            r[loss_name + "_normalized"] = np.clip(
                (r[loss_name] - lo) / denom,
                0,
                1
            )

    # combine the scores:
    best_score = np.inf
    best = None
    active_weight_sum = sum(weights.values()) + 1e-12
    for r in raw_results:
        score = (
            weights["rule"] * r["rule_loss_normalized"]
            + weights["family"] * r["family_resemblance_loss_normalized"]
            + weights["last"] * r["last_dataset_loss_normalized"]
            + weights["shape"] * r["shape_loss_normalized"]
            + weights["collapse"] * r["collapse_loss_normalized"]
            + weights["supervised_rule"] * r["supervised_rule_loss_normalized"]
            + weights["supervised_family"] * r["supervised_family_loss_normalized"]
            + weights["supervised_last"] * r["supervised_last_loss_normalized"]
        ) / active_weight_sum

        r["score"] = score
        r["target_n"] = target_n
        r["target_d"] = target_d

        if score < best_score:
            best_score = score
            best = r

    # dataset refinement:
    if use_pytorch_refinement:
        X_refined, refinement_loss_history, best_refinement_loss, best_refinement_epoch = (
            dataset_refinement.refine_candidate_by_pytorch(
                X_init=best["X"],
                y_init=best["y"],
                X_history=X_history,
                y_history=y_history,
                use_labels=supervised_enabled,
                num_steps=pytorch_num_steps,
                lr=pytorch_lr,
                weight_rule=weight_rule,
                weight_family=weight_family,
                weight_last=weight_last,
                weight_supervised_rule=weight_supervised_rule,
                weight_supervised_family=weight_supervised_family,
                weight_supervised_last=weight_supervised_last,
                weight_collapse=weight_collapse,
                random_state=random_state,
                device=pytorch_device
            )
        )

        best["X_before_refinement"] = best["X"].copy()
        best["X"] = X_refined
        best["refinement_loss_history"] = refinement_loss_history
        best["best_refinement_loss"] = best_refinement_loss
        best["best_refinement_epoch"] = best_refinement_epoch
        best["used_pytorch_refinement"] = True
    else:
        best["X_before_refinement"] = None
        best["refinement_loss_history"] = None
        best["best_refinement_loss"] = None
        best["best_refinement_epoch"] = None
        best["used_pytorch_refinement"] = False

    return best, rule_target_desc, family_target_desc, descriptors
