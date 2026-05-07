import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.special import logit

def train_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> np.ndarray:

    model = Ridge(alpha=ridge, fit_intercept=False)
    model.fit(X, y)
    return model.coef_.flatten()


def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit logistic regression by SGD.
    """
    model = LogisticRegression(
        penalty=None,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X, y)
    return model.coef_.flatten()


def train_min_norm_on_logits(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Return the minimum-norm solution w such that X @ w = logit(y).
    """
    eps = 1e-12
    y = np.clip(y.astype(np.float64), eps, 1.0 - eps)

    z = logit(y)

    w = X.T @ np.linalg.solve(X @ X.T, z)

    return w