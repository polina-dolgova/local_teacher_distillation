from typing import List
import numpy as np

from regression.src.utils import sigmoid


def random_labeling_unlearn(
    theta_init: np.ndarray,
    x_del: np.ndarray,
    y_del: float | np.ndarray,
    steps: int,
    lr: float,
    weight_decay: float,
    x_ft: np.ndarray | None = None,
    y_ft: float | np.ndarray | None = None,
    theta_support: np.ndarray | None = None,
    w_true=None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """
    Unlearn points using random labeling.

    Supports:
        x_del: (d,)  or (k, d)
    """

    if rng is None:
        rng = np.random.default_rng()

    theta = theta_init.copy()
    trajectory = []
    del_losses = []

    # --- unify shape ---
    if x_del.ndim == 1:
        x_del = x_del[None, :]  # (1, d)

    k_del = x_del.shape[0]

    # --- unify x_ft / y_ft shapes if provided ---
    if x_ft is not None:
        x_ft = np.asarray(x_ft)
        if x_ft.ndim == 1:
            x_ft = x_ft[None, :]

        if y_ft is None:
            raise ValueError("y_ft must be provided when x_ft is not None.")

        y_ft = np.asarray(y_ft, dtype=np.float64)
        y_ft = np.atleast_1d(y_ft)

        if x_ft.shape[0] != y_ft.shape[0]:
            raise ValueError(
                f"Mismatched shapes: x_ft has {x_ft.shape[0]} rows, "
                f"but y_ft has shape {y_ft.shape}."
            )

        k_ft = x_ft.shape[0]
    else:
        k_ft = 0

    total_count = k_del + k_ft

    norms = np.sum(x_del * x_del, axis=1)
    mean_norm = np.mean(norms)
    if mean_norm > 1e-15:
        lr = lr / mean_norm

    for step in range(steps):
        # --- random labels for deleted batch ---

        if theta_support is None:
            y_random = rng.binomial(n=1, p=0.5, size=k_del)
        else:
            support_logits = x_del @ theta_support
            y_random = sigmoid(support_logits)

            # print(support_logits, x_del @ w_true)

        logits_del = x_del @ theta
        probs_del = sigmoid(logits_del)

        loss_del = np.mean((y_random - probs_del) ** 2)
        grad_del = x_del * (y_random - probs_del)[:, None]

        if x_ft is not None:
            logits_ft = x_ft @ theta
            probs_ft = sigmoid(logits_ft)

            loss_ft = np.mean((y_ft - probs_ft) ** 2)
            grad_ft = x_ft * (y_ft - probs_ft)[:, None]

            # Average over the whole mixed batch
            loss = (k_del * loss_del + k_ft * loss_ft) / total_count
            grad = (grad_del.sum(axis=0) + grad_ft.sum(axis=0)) / total_count
        else:
            loss = loss_del
            grad = grad_del.mean(axis=0)

        del_losses.append(float(loss))

        # --- weight decay ---
        theta *= 1 - weight_decay

        # --- update ---
        theta -= lr * grad

        trajectory.append(theta.copy())

    return theta, trajectory, del_losses


def gradient_ascent_unlearn(
    theta_init: np.ndarray,
    x_del: np.ndarray,
    y_del: float | np.ndarray,
    steps: int,
    lr: float,
    x_ft: np.ndarray | None = None,
    y_ft: float | np.ndarray | None = None,
) -> tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Perform mixed updates:
    - gradient ascent on the deleted-set loss
    - gradient descent on the fine-tuning set loss

    If x_del has shape (d,), this is treated as a single deleted point.
    If x_del has shape (k_del, d), this is treated as a deleted batch.

    If x_ft is provided, it is treated analogously as a batch of retained / fine-tuning points.

    Deleted loss:
        L_del(theta) = (1 / (2k_del)) * sum_i (x_i^T theta - y_i)^2

    Fine-tuning loss:
        L_ft(theta) = (1 / (2k_ft)) * sum_j (x_j^T theta - y_j)^2

    Update:
        theta_{t+1} = theta_t + lr * grad L_del(theta_t) - lr * grad L_ft(theta_t)

    Returns:
        final theta,
        parameter trajectory,
        mixed losses along the trajectory
    """
    theta = theta_init.copy()

    # --- unify deleted batch shape ---
    x_del = np.asarray(x_del, dtype=np.float64)
    if x_del.ndim == 1:
        x_del = x_del[None, :]

    if np.ndim(y_del) == 0:
        y_del = np.array([y_del], dtype=np.float64)
    else:
        y_del = np.asarray(y_del, dtype=np.float64)

    if x_del.shape[0] != y_del.shape[0]:
        raise ValueError(
            f"Mismatched shapes: x_del has {x_del.shape[0]} rows, "
            f"but y_del has shape {y_del.shape}."
        )

    k_del = x_del.shape[0]

    # --- unify fine-tuning batch shape ---
    if x_ft is not None:
        x_ft = np.asarray(x_ft, dtype=np.float64)
        if x_ft.ndim == 1:
            x_ft = x_ft[None, :]

        if y_ft is None:
            raise ValueError("y_ft must be provided when x_ft is not None.")

        if np.ndim(y_ft) == 0:
            y_ft = np.array([y_ft], dtype=np.float64)
        else:
            y_ft = np.asarray(y_ft, dtype=np.float64)

        if x_ft.shape[0] != y_ft.shape[0]:
            raise ValueError(
                f"Mismatched shapes: x_ft has {x_ft.shape[0]} rows, "
                f"but y_ft has shape {y_ft.shape}."
            )

        k_ft = x_ft.shape[0]
    else:
        k_ft = 0

    trajectory = [theta.copy()]

    # Initial mixed loss
    residual_del = x_del @ theta - y_del
    loss_del = 0.5 * float(np.mean(residual_del**2))

    if x_ft is not None:
        residual_ft = x_ft @ theta - y_ft
        loss_ft = 0.5 * float(np.mean(residual_ft**2))
        total_count = k_del + k_ft
        mixed_loss = (k_del * loss_del + k_ft * loss_ft) / total_count
    else:
        mixed_loss = loss_del

    del_losses = [mixed_loss]

    # Normalize learning rate by average squared norm over the full mixed batch
    row_norm_sq = np.sum(x_del * x_del, axis=1)
    if x_ft is not None:
        row_norm_sq = np.concatenate([row_norm_sq, np.sum(x_ft * x_ft, axis=1)])

    mean_norm_sq = float(np.mean(row_norm_sq))
    if mean_norm_sq <= 1e-15:
        raise ValueError("All points have near-zero norm; gradient update is unstable.")

    lr = lr / mean_norm_sq

    for _ in range(steps):
        # Deleted part: ascent
        residual_del = x_del @ theta - y_del  # shape: (k_del,)
        grad_del = (x_del.T @ residual_del) / k_del  # shape: (d,)

        if x_ft is not None:
            # Fine-tuning part: descent
            residual_ft = x_ft @ theta - y_ft  # shape: (k_ft,)
            grad_ft = (x_ft.T @ residual_ft) / k_ft  # shape: (d,)

            grad = grad_del - grad_ft
        else:
            grad = grad_del

        theta = theta + lr * grad
        trajectory.append(theta.copy())

        # Log mixed loss after update
        residual_del = x_del @ theta - y_del
        loss_del = 0.5 * float(np.mean(residual_del**2))

        if x_ft is not None:
            residual_ft = x_ft @ theta - y_ft
            loss_ft = 0.5 * float(np.mean(residual_ft**2))
            total_count = k_del + k_ft
            mixed_loss = (k_del * loss_del + k_ft * loss_ft) / total_count
        else:
            mixed_loss = loss_del

        del_losses.append(mixed_loss)

    return theta, trajectory, del_losses


def unlearn(cfg, setting, unlearn_seed: int | None = None):

    x_a, y_a = setting["x_a"], setting["y_a"]
    x_s, y_s = setting["x_s"], setting["y_s"]
    x_ft, y_ft = setting["x_ft"], setting["y_ft"]

    if len(y_s) > 0:
        x_del, y_del = np.vstack([x_s, x_a]), np.hstack([y_s, y_a])
    else:
        x_del, y_del = x_a, y_a

    if cfg.unlearning_method == "ga":
        # GA unlearning from the full-data optimum
        theta_unlearn, trajectory, del_losses = gradient_ascent_unlearn(
            theta_init=setting["theta_full"],
            x_del=x_del,
            y_del=y_del,
            steps=cfg.unlearn_steps,
            lr=cfg.unlearn_lr,
            x_ft=x_ft,
            y_ft=y_ft,
        )

    elif cfg.unlearning_method == "rl":
        assert (
            cfg.model_type == "logistic"
        ), "Random labeling unlearning is only implemented for logistic regression."

        rng = np.random.default_rng(unlearn_seed)

        theta_unlearn, trajectory, del_losses = random_labeling_unlearn(
            theta_init=setting["theta_full"],
            x_del=x_del,
            y_del=y_del,
            steps=cfg.unlearn_steps,
            lr=cfg.unlearn_lr,
            weight_decay=cfg.ridge,
            x_ft=x_ft,
            y_ft=y_ft,
            theta_support=setting["theta_support"],
            w_true=setting["w_true"],
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown unlearning method: {cfg.unlearning_method}")

    return theta_unlearn, trajectory, del_losses
