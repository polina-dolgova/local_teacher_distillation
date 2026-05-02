import numpy as np
from typing import Dict
from regression.src.utils import mse_loss


def ripple_coordinate(X: np.ndarray, x_a: np.ndarray, normalize=False) -> np.ndarray:
    """
    rho(x; x_a) =(<x, x_a> / <x_a, x_a>)^2
    """
    denom = float(x_a @ x_a)
    if denom <= 1e-15:
        raise ValueError(
            "Deleted point has near-zero norm; ripple coordinate is unstable."
        )

    if normalize:
        X_norm = float(np.linalg.norm(X, axis=1)[0])
        return (X @ x_a) / denom
    else:
        return (X @ x_a) / denom


def bin_by_rho(rho: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Assign each rho value to a bin index in [0, n_bins-2].
    Points outside the range get index -1.
    """
    idx = np.digitize(rho, bin_edges) - 1
    valid = (idx >= 0) & (idx < len(bin_edges) - 1)
    idx = np.where(valid, idx, -1)
    return idx


def aggregate_by_bin(
    values: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate mean and count by bin.
    """
    means = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            means[b] = float(np.mean(values[mask]))

    return means, counts


def evaluate_models_by_ripple(
    X_test: np.ndarray,
    y_test: np.ndarray,
    x_a: np.ndarray,
    models: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    task_type: str,
) -> Dict[str, np.ndarray]:
    """
    Evaluate models by ripple bins.

    For linear:
        y_test is the regression target.

    For logistic:
        y_test can still be passed for auxiliary loss computation,
        but the main ripple quantities are based on logits and label flips.
    """
    rho = ripple_coordinate(X_test, x_a, normalize=True)
    bin_idx = bin_by_rho(rho, bin_edges)
    n_bins = len(bin_edges) - 1

    # print("rho range: ", rho.min(), rho.max())

    result = {
        "rho": rho,
        "bin_idx": bin_idx,
    }

    logits = {}

    for name, theta in models.items():
        logits[name] = X_test @ theta

        losses = mse_loss(y_test, logits[name])
        mean_loss, counts = aggregate_by_bin(losses, bin_idx, n_bins)

        result[f"{name}_pointwise_loss"] = losses
        result[f"{name}_mean_loss_by_bin"] = mean_loss
        result[f"{name}_counts_by_bin"] = counts
        result[f"{name}_logits"] = logits[name]

    retrain_logits = logits["retrain"]

    for name in models.keys():
        if name == "retrain":
            continue

        delta_logit = logits[name] - retrain_logits
        sq_delta_logit = delta_logit**2

        var_delta_logit, _ = aggregate_var_by_bin(delta_logit, bin_idx, n_bins)
        mean_sq_delta_logit, _ = aggregate_by_bin(sq_delta_logit, bin_idx, n_bins)
        mean_delta_logit, _ = aggregate_by_bin(delta_logit, bin_idx, n_bins)

        result[f"{name}_delta_logit"] = delta_logit
        result[f"{name}_sq_delta_logit"] = sq_delta_logit
        result[f"{name}_sq_logit_diff_by_bin"] = mean_sq_delta_logit
        result[f"{name}_delta_logit_by_bin"] = mean_delta_logit
        result[f"{name}_delta_logit_var_by_bin"] = var_delta_logit

        if task_type == "classification":
            retrain_labels = (retrain_logits >= 0.0).astype(np.float64)
            model_labels = (logits[name] >= 0.0).astype(np.float64)
            flip_indicator = (model_labels != retrain_labels).astype(np.float64)

            flip_rate_by_bin, _ = aggregate_by_bin(flip_indicator, bin_idx, n_bins)

            result[f"{name}_flip_indicator"] = flip_indicator
            result[f"{name}_flip_rate_by_bin"] = flip_rate_by_bin

    return result


def aggregate_var_by_bin(
    values: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate variance and count by bin.
    Variance is computed across points within each bin.
    """
    variances = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 1:
            variances[b] = float(np.var(values[mask]))
        elif counts[b] == 1:
            variances[b] = 0.0

    return variances, counts
