from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from mnist.src.dataset import split_dataset_indices_by_class
from mnist.src.unlearn.support_model import select_support_indices


class DistillationTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, soft_targets: torch.Tensor):
        self.images = images
        self.soft_targets = soft_targets

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.soft_targets[idx]


def _flatten_images_from_indices(
    dataset, indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract images and labels for given dataset indices.

    Returns:
        images_flat: [N, D]
        labels: [N]
    """
    images = []
    labels = []

    for idx in indices:
        x, y = dataset[idx]
        images.append(x.view(-1))
        labels.append(y)

    images = torch.stack(images, dim=0).float()
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def _compute_similarity_matrix(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """
    Compute similarity matrix between query and candidate features.

    Returns:
        sim: [num_queries, num_candidates]
    """
    if mode == "cosine":
        query_features = F.normalize(query_features, dim=1)
        candidate_features = F.normalize(candidate_features, dim=1)
        sim = query_features @ candidate_features.T
        return sim

    if mode == "dot":
        return query_features @ candidate_features.T

    raise ValueError(f"Unknown neighbor mode: {mode}")


def build_neighbor_softlabel_loader(
    dataset,
    forget_indices: list[int],
    retain_indices: list[int],
    batch_size: int,
    num_classes: int = 10,
    k_neighbors: int = 50,
    neighbor_mode: str = "cosine",
    label_mode: str = "per_sample",
    smoothing: float = 1e-8,
) -> DataLoader:
    """
    Build a loader over forget samples with soft targets obtained from
    the label distribution of nearest retain neighbors.

    label_mode:
        - "per_sample": each forget sample gets its own neighbor-label distribution
        - "class_fixed": average the neighbor-label distributions over all forget samples
                         and assign the same distribution to all forget samples
    """
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive.")

    if k_neighbors > len(retain_indices):
        raise ValueError(
            f"k_neighbors={k_neighbors} is larger than retain set size {len(retain_indices)}."
        )

    forget_images_flat, _ = _flatten_images_from_indices(dataset, forget_indices)
    retain_images_flat, retain_labels = _flatten_images_from_indices(
        dataset, retain_indices
    )

    sim = _compute_similarity_matrix(
        query_features=forget_images_flat,
        candidate_features=retain_images_flat,
        mode=neighbor_mode,
    )

    topk_indices = torch.topk(sim, k=k_neighbors, dim=1).indices
    neighbor_labels = retain_labels[topk_indices]  # [N_forget, k]

    soft_targets = torch.zeros(len(forget_indices), num_classes, dtype=torch.float32)

    for i in range(len(forget_indices)):
        counts = torch.bincount(neighbor_labels[i], minlength=num_classes).float()
        probs = counts / counts.sum().clamp_min(1.0)
        soft_targets[i] = probs

    if label_mode == "class_fixed":
        class_distribution = soft_targets.mean(dim=0, keepdim=True)
        class_distribution = class_distribution / class_distribution.sum(
            dim=1, keepdim=True
        ).clamp_min(smoothing)
        soft_targets = class_distribution.repeat(len(forget_indices), 1)

    elif label_mode != "per_sample":
        raise ValueError(f"Unknown label_mode: {label_mode}")

    soft_targets = soft_targets + smoothing
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

    forget_images = []
    for idx in forget_indices:
        x, _ = dataset[idx]
        forget_images.append(x)

    forget_images = torch.stack(forget_images, dim=0)

    distill_dataset = DistillationTensorDataset(
        images=forget_images,
        soft_targets=soft_targets,
    )

    return DataLoader(
        distill_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def unlearn_class_with_neighbor_softlabels(
    model: torch.nn.Module,
    class_to_forget: int,
    support_size: int,
    support_selection: str = "cosine",
    k_neighbors: int = 50,
    neighbor_mode: str = "cosine",
    label_mode: str = "per_sample",
    unlearn_steps: int = 200,
    unlearn_batch_size: int = 128,
    unlearn_lr: float = 1e-4,
    unlearn_weight_decay: float = 1e-6,
    temperature: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.nn.Module, List[float], list[int]]:
    """
    Unlearn one class using neighbor-based soft labels.

    Pipeline:
    1) Select support subset from retain data.
    2) For each forget sample, build a soft target from labels of its nearest support neighbors.
    3) Fine-tune the original model on forget samples using KL loss.

    Returns:
        unlearned_model,
        losses,
        support_indices
    """
    device = torch.device(device)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    forget_indices, retain_indices = split_dataset_indices_by_class(
        dataset=train_dataset,
        class_to_forget=class_to_forget,
    )

    support_indices = select_support_indices(
        dataset=train_dataset,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
        support_size=support_size,
        mode=support_selection,
        seed=seed,
    )

    forget_distill_loader = build_neighbor_softlabel_loader(
        dataset=train_dataset,
        forget_indices=forget_indices,
        retain_indices=support_indices,
        batch_size=unlearn_batch_size,
        num_classes=10,
        k_neighbors=min(k_neighbors, len(support_indices)),
        neighbor_mode=neighbor_mode,
        label_mode=label_mode,
    )

    unlearned_model = deepcopy(model).to(device)
    unlearned_model.train()

    optimizer = torch.optim.AdamW(
        unlearned_model.parameters(),
        lr=unlearn_lr,
        weight_decay=unlearn_weight_decay,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")

    losses: List[float] = []
    loader_iter = iter(forget_distill_loader)

    for _ in range(unlearn_steps):
        try:
            x_forget, soft_targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(forget_distill_loader)
            x_forget, soft_targets = next(loader_iter)

        x_forget = x_forget.to(device)
        soft_targets = soft_targets.to(device)

        student_logits = unlearned_model(x_forget)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

        loss = criterion(student_log_probs, soft_targets) * (temperature**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return unlearned_model, losses, support_indices
