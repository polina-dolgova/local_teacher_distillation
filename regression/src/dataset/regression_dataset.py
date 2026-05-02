import numpy as np
from regression.src.utils import sigmoid


def generate_dataset(
    rng: np.random.Generator,
    n: int,
    d: int,
    w_true: np.ndarray,
    task_type: str = "regression",
    normalize_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dataset:
        X ~ N(0, I)
        y = X w_true + noise.

    Returns:
        X: shape (n, d)
        y: shape (n,)
        M: shape (n, h)
    """
    X = rng.normal(loc=0.0, scale=1.0, size=(n, d))

    if normalize_x:
        X_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / X_norms
        X = X * X_norms.mean()

    if task_type == "classification":
        #y = rng.binomial(n=1, p=sigmoid(X @ w_true)).astype(np.float64)
        y = sigmoid(X @ w_true).astype(np.float64)
    else:
        y = X @ w_true
    return X, y


