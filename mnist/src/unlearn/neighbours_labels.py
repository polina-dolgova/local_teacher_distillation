from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from mnist.src.dataset import split_dataset_indices_by_class, select_support_indices


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

    raise ValueError(f"Unknown neighbor mode: {mode}")


def build_neighbor_softlabel_loader(
    dataset,
    forget_indices: list[int],
    support_indices: list[int],
    batch_size: int,
    num_classes: int = 10,
    k_neighbors: int = 50,
    support_selection: str = "cosine",
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

    if k_neighbors > len(support_indices):
        raise ValueError(
            f"k_neighbors={k_neighbors} is larger than support set size {len(support_indices)}."
        )

    forget_images_flat, _ = _flatten_images_from_indices(dataset, forget_indices)
    support_images_flat, support_labels = _flatten_images_from_indices(
        dataset, support_indices
    )

    sim = _compute_similarity_matrix(
        query_features=forget_images_flat,
        candidate_features=support_images_flat,
        mode=support_selection,
    )

    topk_indices = torch.topk(sim, k=k_neighbors, dim=1).indices
    neighbor_labels = support_labels[topk_indices]  # [N_forget, k]

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


def get_forget_dataloader_with_neighbor_softlabels(
    class_to_forget: int,
    support_size: int,
    support_selection: str = "cosine",
    support_feature_space: str = "raw",
    feature_model: torch.nn.Module | None = None,
    k_neighbors: int | None = None,
    label_mode: str = "per_sample",
    unlearn_batch_size: int = 128,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.nn.Module, List[float], list[int]]:

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
        feature_space=support_feature_space,
        feature_model=feature_model,
        device=str(device),
        seed=seed,
    )

    if k_neighbors is None:
        k_neighbors = support_size

    forget_distill_loader = build_neighbor_softlabel_loader(
        dataset=train_dataset,
        forget_indices=forget_indices,
        support_indices=support_indices,
        batch_size=unlearn_batch_size,
        num_classes=10,
        k_neighbors=min(k_neighbors, len(support_indices)),
        support_selection=support_selection,
        label_mode=label_mode,
    )

    return forget_distill_loader, support_indices
