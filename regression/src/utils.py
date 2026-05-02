import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Pointwise squared loss.
    """
    return (y_true - y_pred) ** 2


def prob_to_logit(p: float, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))
