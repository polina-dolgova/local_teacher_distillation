from __future__ import annotations

from typing import Any
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cifar.retrain.config import DatasetName, TrainConfig


def get_dataset_info(dataset_name: DatasetName) -> dict[str, Any]:
    if dataset_name == "cifar10":
        return {
            "num_classes": 10,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "dataset_cls": datasets.CIFAR10,
        }
    if dataset_name == "cifar100":
        return {
            "num_classes": 100,
            "mean": (0.5070, 0.4865, 0.4409),
            "std": (0.2673, 0.2564, 0.2761),
            "dataset_cls": datasets.CIFAR100,
        }
    if dataset_name == "svhn":
        return {
            "num_classes": 10,
            "mean": (0.4377, 0.4438, 0.4728),
            "std": (0.1980, 0.2010, 0.1970),
            "dataset_cls": datasets.SVHN,
        }
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_transforms(
    dataset_name: DatasetName,
) -> tuple[transforms.Compose, transforms.Compose]:
    info = get_dataset_info(dataset_name)
    mean = info["mean"]
    std = info["std"]

    normalize = transforms.Normalize(mean=mean, std=std)

    if dataset_name == "svhn":
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    return train_transform, test_transform


def _extract_targets(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)

    targets = []
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        targets.append(int(y))
    return targets


def split_indices_by_forget_class(
    dataset,
    class_to_forget: int,
    class_fraction_to_forget: float = 1.0,
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices into forget and retain parts.

    The "forget" part is the first k% of the chosen class in the original dataset order.
    """
    if not (0.0 <= class_fraction_to_forget <= 1.0):
        raise ValueError("class_fraction_to_forget must be in [0, 1].")

    targets = _extract_targets(dataset)

    forget_indices = []
    retain_indices = []

    for idx, y in enumerate(targets):
        if y == class_to_forget:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)

    forget_size = int(len(forget_indices) * class_fraction_to_forget)

    forget_indices_in_forget = forget_indices[:forget_size]
    forget_indices_in_retain = forget_indices[forget_size:]
    retain_indices.extend(forget_indices_in_retain)

    return forget_indices_in_forget, retain_indices


def _make_dataset(dataset_name: DatasetName, dataset_cls, data_root: str, train: bool, transform):
    if dataset_name in ("cifar10", "cifar100"):
        return dataset_cls(root=data_root, train=train, download=True, transform=transform)
    elif dataset_name == "svhn":
        return dataset_cls(
            root=data_root,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")


def build_dataloaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, list[int], list[int], int]:
    info = get_dataset_info(config.dataset_name)
    dataset_cls = info["dataset_cls"]
    num_classes = info["num_classes"]

    train_transform, test_transform = build_transforms(config.dataset_name)

    train_dataset = _make_dataset(config.dataset_name, dataset_cls, config.data_root, True, train_transform)
    test_dataset = _make_dataset(config.dataset_name, dataset_cls, config.data_root, False, test_transform)

    forget_indices, retain_indices = split_indices_by_forget_class(
        dataset=train_dataset,
        class_to_forget=config.class_to_forget,
        class_fraction_to_forget=config.class_fraction_to_forget,
    )

    retain_train_dataset = Subset(train_dataset, retain_indices)

    pin_memory = config.device.startswith("cuda")
    persistent_workers = config.num_workers > 0

    train_loader = DataLoader(
        retain_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, test_loader, forget_indices, retain_indices, num_classes
