from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cifar.src.models.features import extract_cifar_resnet56_penultimate
from cifar.src.datasets import (
    split_indices_by_forget_class,
)
from cifar.src.custom_types import *


def _compute_resnet56_embeddings_for_indices(
    dataset,
    indices: list[int] | np.ndarray,
    model: torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    feature_extractor_fn=None,
) -> torch.Tensor:
    """
    Compute penultimate embeddings for the given dataset indices.

    Returns:
        Tensor of shape [len(indices), dim]
    """
    if feature_extractor_fn is None:
        feature_extractor_fn = extract_cifar_resnet56_penultimate

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            emb = feature_extractor_fn(model, x)
            all_embeddings.append(emb.cpu())

    if not all_embeddings:
        return torch.empty((0, 0), dtype=torch.float32)

    return torch.cat(all_embeddings, dim=0)


def _compute_similarity_to_forget_class(
    dataset,
    retain_indices: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    model: torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    feature_extractor_fn=None,
) -> np.ndarray:
    """
    Compute cosine similarity of each retain point to the mean embedding
    of the forget subset.

    Returns:
        similarities: np.ndarray of shape [len(retain_indices)]
    """
    retain_embeddings = _compute_resnet56_embeddings_for_indices(
        dataset=dataset,
        indices=retain_indices,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_extractor_fn=feature_extractor_fn,
    )
    forget_embeddings = _compute_resnet56_embeddings_for_indices(
        dataset=dataset,
        indices=forget_indices,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_extractor_fn=feature_extractor_fn,
    )

    if retain_embeddings.shape[0] == 0:
        return np.array([], dtype=float)

    if forget_embeddings.shape[0] == 0:
        raise ValueError("Forget subset is empty, cannot compute similarities.")

    retain_embeddings = F.normalize(retain_embeddings, p=2, dim=1)
    forget_center = forget_embeddings.mean(dim=0, keepdim=True)
    forget_center = F.normalize(forget_center, p=2, dim=1)

    similarities = retain_embeddings @ forget_center.T
    return similarities.squeeze(1).cpu().numpy()


def _sample_retain_indices_random(
    retain_indices: list[int] | np.ndarray,
    retain_fraction: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Randomly sample a fraction of retain indices.
    """
    retain_indices = np.asarray(retain_indices)
    num_to_keep = int(len(retain_indices) * retain_fraction)

    rng = np.random.default_rng(seed)
    selected = rng.choice(retain_indices, size=num_to_keep, replace=False)
    return selected


def _sample_retain_indices_by_similarity(
    dataset,
    retain_indices: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    retain_fraction: float,
    model: torch.nn.Module,
    mode: Literal["closest", "farthest"],
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    feature_extractor_fn=None,
) -> np.ndarray:
    """
    Select retain points by cosine similarity to the forget-class center.

    - closest: keep the most similar points
    - farthest: keep the least similar points
    """
    retain_indices = np.asarray(retain_indices)
    num_to_keep = int(len(retain_indices) * retain_fraction)

    similarities = _compute_similarity_to_forget_class(
        dataset=dataset,
        retain_indices=retain_indices,
        forget_indices=forget_indices,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_extractor_fn=feature_extractor_fn,
    )

    if mode == "closest":
        order = np.argsort(-similarities)
    elif mode == "farthest":
        order = np.argsort(similarities)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    selected = retain_indices[order[:num_to_keep]]
    return selected


def select_retain_indices(
    dataset,
    retain_indices: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    retain_fraction: float,
    retain_selection_mode: RetainSelectionMode = "random",
    seed: int = 42,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    feature_extractor_fn=None,
) -> np.ndarray:
    """
    Select a subset of retain indices according to the specified mode.
    """
    if not (0.0 <= retain_fraction <= 1.0):
        raise ValueError("retain_fraction must be in the interval [0, 1].")

    if retain_fraction == 1.0:
        return np.asarray(retain_indices)

    if retain_fraction == 0.0:
        return np.array([])

    if retain_selection_mode == "random":
        return _sample_retain_indices_random(
            retain_indices=retain_indices,
            retain_fraction=retain_fraction,
            seed=seed,
        )

    if retain_selection_mode in {"closest", "farthest"}:
        if model is None:
            raise ValueError(
                "model must be provided when retain_selection_mode "
                "is 'closest' or 'farthest'."
            )

        return _sample_retain_indices_by_similarity(
            dataset=dataset,
            retain_indices=retain_indices,
            forget_indices=forget_indices,
            retain_fraction=retain_fraction,
            model=model,
            mode=retain_selection_mode,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_extractor_fn=feature_extractor_fn,
        )

    raise ValueError(
        f"Unknown retain_selection_mode: {retain_selection_mode}. "
        f"Expected one of ['random', 'closest', 'farthest']."
    )


def get_subsets(
    dataset,
    class_to_forget: int,
    class_fraction_to_forget: float = 1.0,
    retain_fraction: float = 1.0,
    seed: int = 42,
    retain_selection_mode: RetainSelectionMode = "random",
    model: torch.nn.Module | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    feature_extractor_fn=None,
):
    """
    Split dataset into retain and forget subsets.

    If retain_fraction < 1, retain subset can be selected in one of three modes:
    - random
    - closest
    - farthest
    """
    forget_indices, retain_indices = split_indices_by_forget_class(
        dataset,
        class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
    )

    retain_indices = select_retain_indices(
        dataset=dataset,
        retain_indices=retain_indices,
        forget_indices=forget_indices,
        retain_fraction=retain_fraction,
        retain_selection_mode=retain_selection_mode,
        seed=seed,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_extractor_fn=feature_extractor_fn,
    )

    # targets = np.asarray(dataset.targets)
    # selected_labels = targets[retain_indices]
    # unique_classes, counts = np.unique(selected_labels, return_counts=True)
    # print(f"Retain selection mode: {retain_selection_mode}")
    # for cls, count in zip(unique_classes, counts):
    #     ratio = count / len(retain_indices)
    #     print(f"Class {cls}: count={count}, ratio={ratio:.4f}")

    retain_subset = Subset(dataset, retain_indices.tolist())
    forget_subset = Subset(dataset, forget_indices)

    return retain_subset, forget_subset
