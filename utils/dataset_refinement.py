import torch
import numpy as np
from tqdm import tqdm
from utils import dataset_descriptors


def torch_standardize(Y):
    return (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-8)


def torch_descriptor(Y):
    """
    Differentiable descriptor.
    Similar to dataset_descriptor, but avoids KMeans/silhouette/quantiles.
    """
    Z = torch_standardize(Y)

    D = torch.cdist(Z, Z)
    mask = torch.triu(torch.ones_like(D), diagonal=1).bool()
    vals = D[mask]

    cov = (Z.T @ Z) / max(Z.shape[0] - 1, 1)

    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.sort(eigvals, descending=True).values
    eigvals = eigvals / (eigvals.sum() + 1e-8)

    eig_pad = torch.zeros(5, device=Y.device, dtype=Y.dtype)
    k = min(5, eigvals.numel())
    eig_pad[:k] = eigvals[:k]

    head = torch.stack([
        vals.mean(),
        vals.std(),
        torch.trace(cov),
        eigvals[0],
    ])

    desc = torch.cat([head, eig_pad])
    return desc


def numpy_torch_descriptor(X):
    Y = torch.tensor(X, dtype=torch.float32)
    return torch_descriptor(Y).detach()


def torch_supervised_descriptor(Y, y):
    """
    Differentiable supervised descriptor.
    Labels are fixed; only point coordinates are optimized.
    """
    Z = torch_standardize(Y)
    y_torch = torch.tensor(y, device=Y.device)

    classes = torch.unique(y_torch)

    centroids = []
    within_terms = []
    proportions = []

    for c in classes:
        mask = y_torch == c
        Zc = Z[mask]

        proportions.append(Zc.shape[0] / Z.shape[0])

        centroid = Zc.mean(dim=0)
        centroids.append(centroid)

        if Zc.shape[0] >= 2:
            Dc = torch.cdist(Zc, Zc)
            m = torch.triu(torch.ones_like(Dc), diagonal=1).bool()
            within_terms.append(Dc[m].mean())

    centroids = torch.stack(centroids)

    if len(within_terms) > 0:
        mean_within = torch.stack(within_terms).mean()
    else:
        mean_within = torch.tensor(0.0, device=Y.device, dtype=Y.dtype)

    if centroids.shape[0] >= 2:
        Dc = torch.cdist(centroids, centroids)
        m = torch.triu(torch.ones_like(Dc), diagonal=1).bool()
        mean_between = Dc[m].mean()
    else:
        mean_between = torch.tensor(0.0, device=Y.device, dtype=Y.dtype)

    proportions = torch.tensor(proportions, device=Y.device, dtype=Y.dtype)
    entropy = -torch.sum(proportions * torch.log(proportions + 1e-8))
    imbalance = proportions.max() - proportions.min()
    separability = mean_between / (mean_within + 1e-8)

    return torch.stack([
        torch.tensor(float(len(classes)), device=Y.device, dtype=Y.dtype),
        entropy,
        imbalance,
        mean_within,
        mean_between,
        separability
    ])


def torch_descriptor_distance(a, b):
    scale = torch.abs(a) + torch.abs(b) + 1e-6
    return torch.mean(((a - b) / scale) ** 2)


def refine_candidate_by_pytorch(
    X_init,
    y_init,
    X_history,
    y_history=None,
    use_labels=True,
    num_steps=500,
    lr=1e-2,
    weight_rule=1.00,
    weight_family=0.60,
    weight_last=0.20,
    weight_supervised_rule=0.80,
    weight_supervised_family=0.60,
    weight_supervised_last=0.20,
    weight_collapse=0.05,
    random_state=0,
    device="cpu"
):
    torch.manual_seed(random_state)

    Y = torch.tensor(
        X_init,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    hist_desc = [
        numpy_torch_descriptor(X).to(device)
        for X in X_history
    ]

    rule_target = dataset_descriptors.predict_next_descriptor([d.cpu().numpy() for d in hist_desc])
    family_target = dataset_descriptors.weighted_history_descriptor([d.cpu().numpy() for d in hist_desc], decay=0.75)

    rule_target = torch.tensor(rule_target, dtype=torch.float32, device=device)
    family_target = torch.tensor(family_target, dtype=torch.float32, device=device)
    last_target = hist_desc[-1]

    supervised_enabled = use_labels and y_history is not None and y_init is not None

    if supervised_enabled:
        hist_sdesc = [
            torch_supervised_descriptor(
                torch.tensor(X, dtype=torch.float32, device=device),
                y
            ).detach()
            for X, y in zip(X_history, y_history)
        ]

        s_rule_target = dataset_descriptors.predict_next_descriptor([d.cpu().numpy() for d in hist_sdesc])
        s_family_target = dataset_descriptors.weighted_history_descriptor([d.cpu().numpy() for d in hist_sdesc], decay=0.75)

        s_rule_target = torch.tensor(s_rule_target, dtype=torch.float32, device=device)
        s_family_target = torch.tensor(s_family_target, dtype=torch.float32, device=device)
        s_last_target = hist_sdesc[-1]
    else:
        weight_supervised_rule = 0.0
        weight_supervised_family = 0.0
        weight_supervised_last = 0.0

    optimizer = torch.optim.Adam([Y], lr=lr)

    loss_history = []
    best_loss = np.inf
    best_X = X_init.copy()
    best_epoch = 0

    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()

        desc = torch_descriptor(Y)

        rule_loss = torch_descriptor_distance(desc, rule_target)
        family_loss = torch_descriptor_distance(desc, family_target)
        last_loss = torch_descriptor_distance(desc, last_target)

        Z = torch_standardize(Y)
        D = torch.cdist(Z, Z)
        collapse_loss = 1.0 / (D.std() + 1e-6)

        loss = (
            weight_rule * rule_loss
            + weight_family * family_loss
            + weight_last * last_loss
            + weight_collapse * collapse_loss
        )

        if supervised_enabled:
            sdesc = torch_supervised_descriptor(Y, y_init)

            s_rule_loss = torch_descriptor_distance(sdesc, s_rule_target)
            s_family_loss = torch_descriptor_distance(sdesc, s_family_target)
            s_last_loss = torch_descriptor_distance(sdesc, s_last_target)

            loss = loss + (
                weight_supervised_rule * s_rule_loss
                + weight_supervised_family * s_family_loss
                + weight_supervised_last * s_last_loss
            )

        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu())
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_X = Y.detach().cpu().numpy().copy()
            best_epoch = step

    return best_X, loss_history, best_loss, best_epoch