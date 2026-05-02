import numpy as np

from regression.src.dataset.regression_dataset import generate_dataset
from regression.src.train.train_regression import (
    train_logistic_regression,
    train_ridge_regression,
    train_min_norm_on_logits
)
from regression.src.utils import sigmoid, prob_to_logit


CONFIDENCE_TO_PROB_INTERVAL = {
    "low": (0.50, 0.65),
    "mid": (0.65, 0.80),
    "high": (0.80, 0.99),
}


def choose_target_radius(d: int) -> float:
    """
    Choose target norm for deleted point.

    For x ~ N(0, I_d), a typical norm is about sqrt(d).
    """
    return float(np.sqrt(d))


def choose_target_logit(
    rng: np.random.Generator,
    confidence_level: str | None,
    max_abs_logit: float,
) -> float:
    """
    Sample a feasible target logit based on confidence constraints.
    """
    feasible_prob_min = sigmoid(-max_abs_logit)
    feasible_prob_max = sigmoid(max_abs_logit)

    if confidence_level is None:
        prob_min, prob_max = feasible_prob_min, feasible_prob_max
    else:
        prob_min, prob_max = CONFIDENCE_TO_PROB_INTERVAL[confidence_level]

    eff_prob_min = max(prob_min, feasible_prob_min)
    eff_prob_max = min(prob_max, feasible_prob_max)

    if eff_prob_min > eff_prob_max:
        raise ValueError("Requested confidence interval is infeasible.")

    target_prob = rng.uniform(eff_prob_min, eff_prob_max)
    return float(prob_to_logit(target_prob))


def sample_unit_vector_orthogonal_to(
    rng: np.random.Generator,
    u: np.ndarray,
    max_retries: int = 100,
) -> np.ndarray:
    """
    Sample a random unit vector orthogonal to u.
    """
    d = u.shape[0]

    for _ in range(max_retries):
        v = rng.normal(size=d)
        v = v - (v @ u) * u
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            return v / v_norm

    raise ValueError("Failed to sample a vector orthogonal to w_true direction.")


def construct_deleted_point_from_target_logit(
    rng: np.random.Generator,
    w_true: np.ndarray,
    target_logit: float,
    target_radius: float,
) -> np.ndarray:
    """
    Construct x_a such that:
        ||x_a|| = target_radius
        x_a @ w_true = target_logit
    """
    w_norm = np.linalg.norm(w_true)
    if w_norm <= 1e-12:
        raise ValueError("w_true has near-zero norm.")

    u = w_true / w_norm
    max_abs_logit = target_radius * w_norm

    if abs(target_logit) > max_abs_logit + 1e-10:
        raise ValueError("Target logit is infeasible for the chosen radius.")

    alpha = target_logit / (target_radius * w_norm)
    alpha = np.clip(alpha, -1.0, 1.0)

    d = w_true.shape[0]
    if d == 1:
        if abs(abs(alpha) - 1.0) > 1e-10:
            raise ValueError("Cannot realize a non-extreme target logit in 1D.")
        q = np.sign(alpha) * u
        return (target_radius * q).astype(np.float64)

    v = sample_unit_vector_orthogonal_to(rng=rng, u=u)

    beta_sq = max(1.0 - alpha**2, 0.0)
    beta = np.sqrt(beta_sq)

    q = alpha * u + beta * v
    q = q / np.linalg.norm(q)

    x_a = target_radius * q
    return x_a.astype(np.float64)


def generate_label_for_deleted_point(
    rng: np.random.Generator,
    x_a: np.ndarray,
    w_true: np.ndarray,
    task_type: str,
    noise_std: float,
) -> float:
    """
    Generate label for deleted point.
    """
    logit = float(x_a @ w_true)

    if task_type == "classification":
        prob = sigmoid(logit)
        return prob
        #return float(rng.binomial(n=1, p=prob))

    noise = rng.normal(loc=0.0, scale=noise_std)
    return float(logit + noise)


def generate_deleted_point_with_logit_control(
    cfg,
    rng,
    w_true: np.ndarray,
    task_type: str,
    confidence_level: str | None,
) -> tuple[np.ndarray, float]:
    """
    Generate a deleted point x_a such that:
        ||x_a|| ~= typical norm of x ~ N(0, I)
        x_a @ w_true is controlled
    """
    d = w_true.shape[0]
    w_norm = np.linalg.norm(w_true)

    if w_norm <= 1e-12:
        raise ValueError("w_true has near-zero norm.")

    target_radius = choose_target_radius(d=cfg.d)
    max_abs_logit = target_radius * w_norm

    target_logit = choose_target_logit(
        rng=rng,
        confidence_level=confidence_level,
        max_abs_logit=max_abs_logit,
    )

    x_a = construct_deleted_point_from_target_logit(
        rng=rng,
        w_true=w_true,
        target_logit=target_logit,
        target_radius=target_radius,
    )

    y_a = generate_label_for_deleted_point(
        rng=rng,
        x_a=x_a,
        w_true=w_true,
        task_type=task_type,
        noise_std=cfg.train_noise_std,
    )

    return x_a, y_a


def generate_deleted_point(cfg, rng, task_type):
    """
    Generate the deleted point in a reproducible way.
    """

    w_true = rng.normal(loc=0.0, scale=1.0, size=cfg.d)

    if cfg.model_type == "linear":

        X_del, y_del = generate_dataset(
            rng=rng,
            n=1,
            d=cfg.d,
            w_true=w_true,
            task_type=task_type,
            normalize_x=True,
        )

        x_a = X_del[0]
        y_a = float(y_del[0])

    elif cfg.model_type == "logistic":
        x_a, y_a = generate_deleted_point_with_logit_control(
            cfg=cfg,
            rng=rng,
            w_true=w_true,
            task_type=task_type,
            confidence_level=cfg.confidence_level,
        )

    return w_true, x_a, y_a


def generate_softlabel_test_from_train_span(
    rng,
    X_train: np.ndarray,
    w_true: np.ndarray,
    n_test: int,
    normalize_x: bool = True,
):
    """
    Generate test points from span(X_train) and define soft labels
    by applying the same linear combinations to train logits.
    """
    n_train = X_train.shape[0]

    coeffs = rng.normal(size=(n_test, n_train))

    # Normalize coefficients to avoid very large linear combinations.
    coeff_norms = np.linalg.norm(coeffs, axis=1, keepdims=True)
    coeffs = coeffs / np.maximum(coeff_norms, 1e-12)

    X_test = coeffs @ X_train

    z_train = X_train @ w_true
    z_test = coeffs @ z_train

    if normalize_x:
        # If X_test is rescaled, logits must be rescaled in the same way.
        x_norms = np.linalg.norm(X_test, axis=1, keepdims=True)
        safe_norms = np.maximum(x_norms, 1e-12)

        X_test = X_test / safe_norms
        z_test = z_test / safe_norms.squeeze(axis=1)

    y_test = sigmoid(z_test).astype(np.float64)

    return X_test.astype(np.float64), y_test

def generate_regression_dataset(cfg, rng, w_true, task_type="regression"):
    # Generate train and test data
    X_train, y_train = generate_dataset(
        rng=rng,
        n=cfg.n_train - 1,
        d=cfg.d,
        w_true=w_true,
        task_type=task_type,
    )

    # X_test, y_test = generate_dataset(
    #     rng=rng,
    #     n=cfg.n_test,
    #     d=cfg.d,
    #     w_true=w_true,
    #     task_type=task_type,
    #     normalize_x=True,
    # )

    # if task_type == "regression":
    #     # For regression, use the training dataset as the test dataset.
    #     X_test = X_train.copy()
    #     y_test = y_train.copy()

    # else:
    #     X_test, y_test = generate_softlabel_test_from_train_span(
    #         rng=rng,
    #         X_train=X_train,
    #         w_true=w_true,
    #         n_test=cfg.n_test,
    #         normalize_x=True,
    #     )

    return X_train, y_train


def generate_test_set(cfg, rng, w_true, X, y, task_type="regression"):
    if task_type == "regression":
        # For regression, use the training dataset as the test dataset.
        X_test = X.copy()
        y_test = y.copy()

    else:
        X_test, y_test = generate_softlabel_test_from_train_span(
            rng=rng,
            X_train=X,
            w_true=w_true,
            n_test=cfg.n_test,
            normalize_x=True,
        )

    return X_test, y_test

def split_nearest_by_cosine(
    X: np.ndarray,
    y: np.ndarray,
    x_a: np.ndarray,
    n_select: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split X into:
    - X_s: n_select points with the highest cosine similarity to x_a
    - X_ret_new: all remaining points

    Returns:
        X_s, y_s, X_ret_new, y_ret_new
    """
    if n_select < 0:
        raise ValueError("n_select must be non-negative.")
    if n_select > X.shape[0]:
        raise ValueError(
            f"n_select={n_select} is larger than the dataset size {X.shape[0]}."
        )

    if n_select == 0:
        return np.array([]), np.array([]), X, y

    x_a_norm = np.linalg.norm(x_a)
    if x_a_norm <= 1e-12:
        raise ValueError("x_a has near-zero norm.")

    X_norms = np.linalg.norm(X, axis=1)
    if np.any(X_norms <= 1e-12):
        raise ValueError("Some rows in X have near-zero norm.")

    cosine_sim = np.abs(X @ x_a) / X_norms

    print(np.max(cosine_sim), np.min(cosine_sim))

    # Descending order: largest cosine similarity first
    sorted_idx = np.argsort(-cosine_sim)

    s_idx = sorted_idx[:n_select]
    ret_idx = sorted_idx[n_select:]

    X_s = X[s_idx]
    y_s = y[s_idx]
    X_ret_new = X[ret_idx]
    y_ret_new = y[ret_idx]

    return X_s, y_s, X_ret_new, y_ret_new


def get_random_subset(rng, X, y, select_n):
    if select_n > 0:
        if select_n > X.shape[0]:
            raise ValueError(
                f"n={select_n} is larger than available pool size {X.shape[0]}"
            )

        ft_idx = rng.choice(X.shape[0], size=select_n, replace=False)

        x_r = X[ft_idx]
        y_r = y[ft_idx]
    else:
        x_r, y_r = None, None

    return x_r, y_r


def get_support_model(cfg, rng, X_ret, y_ret):

    X_support, y_support = get_random_subset(rng, X_ret, y_ret, cfg.n_support)

    if X_support is not None:
        theta_support = train_logistic_regression(X_support, y_support)
    else:
        theta_support = None

    return theta_support


def generate_regression_training_data(cfg, rng, task_type="regression"):

    w_true, x_a, y_a = generate_deleted_point(cfg=cfg, rng=rng, task_type=task_type)

    # Generate shared latent basis H and true parameter w_true
    X_pool, y_pool = generate_regression_dataset(
        cfg, rng, w_true, task_type=task_type
    )

    X_delete, y_delete, X_ret, y_ret = split_nearest_by_cosine(
        X=X_pool,
        y=y_pool,
        x_a=x_a,
        n_select=cfg.n_delete - 1,
    )

    X_test, y_test = generate_test_set(cfg, rng, w_true, X_ret, y_ret, task_type=task_type)

    x_ft, y_ft = get_random_subset(rng, X_ret, y_ret, select_n=cfg.n_ft)
    X_train, y_train = np.vstack([X_pool, x_a]), np.hstack([y_pool, y_a])

    # Fit models
    if task_type == "regression":
        theta_full = train_ridge_regression(X_train, y_train, ridge=cfg.ridge)
        theta_retrain = train_ridge_regression(X_ret, y_ret, ridge=cfg.ridge)
        theta_support = None
    else:
        theta_full = train_min_norm_on_logits(X_train, y_train)
        theta_retrain = train_min_norm_on_logits(X_ret, y_ret)
        theta_support = None #train_min_norm_on_logits(cfg, rng, X_ret, y_ret)

    setting = {
        "w_true": w_true,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "theta_full": theta_full,
        "x_a": x_a,
        "y_a": y_a,
        "x_s": X_delete,
        "y_s": y_delete,
        "theta_retrain": theta_retrain,
        "y_test_logits": X_test @ w_true,
        "deleted_point_logit": float(x_a @ w_true),
        "x_ft": x_ft,
        "y_ft": y_ft,
        "theta_support": theta_support,
    }

    return setting
