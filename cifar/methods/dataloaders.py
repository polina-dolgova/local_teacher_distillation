import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from cifar.src.datasets import split_indices_by_forget_class
from cifar.src.custom_types import *


def make_random_labels(
    labels: torch.Tensor,
    num_classes: int = 10,
    mode: RandomLabelMode = "any",
) -> torch.Tensor:
    """
    Generate random labels.

    - mode="any": sample uniformly from all classes.
    - mode="wrong": sample uniformly from all classes except the true label.
    """
    if mode == "any":
        return torch.randint(0, num_classes, labels.shape, device=labels.device)

    if mode == "wrong":
        rand = torch.randint(
            low=0,
            high=num_classes - 1,
            size=labels.shape,
            device=labels.device,
        )
        return rand + (rand >= labels).long()

    raise ValueError(f"Unknown random-label mode: {mode}")


def build_separate_dataloaders(
    dataset, class_to_forget, class_fraction_to_forget, batch_size, num_workers=4
):
    forget_indices, retain_indices = split_indices_by_forget_class(
        dataset, class_to_forget, class_fraction_to_forget
    )

    retain_subset = Subset(dataset, retain_indices)
    forget_subset = Subset(dataset, forget_indices)

    retain_loader = DataLoader(
        retain_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    forget_loader = DataLoader(
        forget_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return retain_loader, forget_loader


def extract_targets_from_subset(subset: Dataset) -> torch.Tensor:
    targets = []
    for idx in range(len(subset)):
        _, target = subset[idx]
        targets.append(int(target))
    return torch.tensor(targets, dtype=torch.long)


class RelabeledSubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, new_targets: torch.Tensor) -> None:
        self.base_dataset = base_dataset
        self.new_targets = new_targets.detach().cpu().long()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, _ = self.base_dataset[idx]
        return image, int(self.new_targets[idx].item())


def make_relabeled_forget_dataset(
    forget_subset: Dataset,
    num_classes: int,
    mode: str,
    seed: int,
) -> Dataset:
    original_targets = extract_targets_from_subset(forget_subset)
    torch.manual_seed(seed)
    new_targets = make_random_labels(
        original_targets,
        num_classes=num_classes,
        mode=mode,
    )
    return RelabeledSubsetDataset(forget_subset, new_targets)


def build_random_mixed_loader(
    forget_subset,
    retain_subset,
    num_classes,
    random_label_mode="any",
    batch_size=64,
    num_workers=4,
    seed=42,
) -> DataLoader:
    relabeled_forget_dataset = make_relabeled_forget_dataset(
        forget_subset=forget_subset,
        num_classes=num_classes,
        mode=random_label_mode,
        seed=seed,
    )
    train_dataset = ConcatDataset([relabeled_forget_dataset, retain_subset])
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )


class SoftRelabeledSubsetDataset(Dataset):
    """
    Dataset wrapper for forget samples with soft targets.
    Returns:
        image, soft_target, is_forget
    """

    def __init__(self, base_dataset: Dataset, soft_targets: torch.Tensor) -> None:
        if len(base_dataset) != len(soft_targets):
            raise ValueError("base_dataset and soft_targets must have the same length.")

        self.base_dataset = base_dataset
        self.soft_targets = soft_targets.detach().cpu().float()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, _ = self.base_dataset[idx]
        return image, self.soft_targets[idx], 1


class HardLabeledAsSoftSubsetDataset(Dataset):
    """
    Dataset wrapper for retain samples.
    Converts hard labels to one-hot vectors so retain and forget
    can be mixed in a single DataLoader.
    Returns:
        image, one_hot_target, is_forget
    """

    def __init__(self, base_dataset: Dataset, num_classes: int) -> None:
        self.base_dataset = base_dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        soft_target = torch.zeros(self.num_classes, dtype=torch.float32)
        soft_target[int(target)] = 1.0
        return image, soft_target, 0


def build_soft_mixed_loader(
    forget_subset,
    forget_soft_targets: torch.Tensor,
    retain_subset,
    num_classes: int,
    class_to_forget: int,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    """
    Build and return three loaders:
    1) mixed loader:
       - forget subset with soft targets
       - retain subset with one-hot targets
    2) retain_same_class_loader:
       - only retain samples whose true label equals class_to_forget
       - targets are one-hot
    3) forget_soft_loader:
       - only forget subset with soft targets
    """
    soft_forget_dataset = SoftRelabeledSubsetDataset(
        base_dataset=forget_subset,
        soft_targets=forget_soft_targets,
    )
    hard_retain_dataset = HardLabeledAsSoftSubsetDataset(
        base_dataset=retain_subset,
        num_classes=num_classes,
    )

    train_dataset = ConcatDataset([soft_forget_dataset, hard_retain_dataset])

    # Build retain subset where true label == class_to_forget
    retain_same_class_indices = []

    for i in range(len(retain_subset)):
        _, label = retain_subset[i]
        if int(label) == class_to_forget:
            retain_same_class_indices.append(i)

    retain_same_class_subset = Subset(retain_subset, retain_same_class_indices)
    retain_same_class_dataset = HardLabeledAsSoftSubsetDataset(
        base_dataset=retain_same_class_subset,
        num_classes=num_classes,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        generator=g,
        pin_memory=True,
    )

    retain_same_class_loader = DataLoader(
        retain_same_class_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    forget_soft_loader = DataLoader(
        soft_forget_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, retain_same_class_loader, forget_soft_loader


def build_selected_class_loader(dataset, class_to_eval, batch_size, num_workers):
    eval_indices = []

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if target == class_to_eval:
            eval_indices.append(idx)
    eval_dataset = Subset(dataset, eval_indices)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return eval_loader


def build_eval_loaders(
    forget_subset,
    retain_subset,
    class_to_forget,
    batch_size,
    num_workers,
    test_dataset=None,
):
    eval_loaders = {}

    eval_loaders["forget"] = DataLoader(
        forget_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    retain_same_class_indices = [
        i for i, (_, y) in enumerate(retain_subset) if y == class_to_forget
    ]

    retain_same_class_dataset = Subset(retain_subset, retain_same_class_indices)

    eval_loaders["retain_same_class"] = DataLoader(
        retain_same_class_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if test_dataset is not None:
        eval_loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_same_class_indices = [
            i for i, (_, y) in enumerate(test_dataset) if y == class_to_forget
        ]

        test_same_class_dataset = Subset(test_dataset, test_same_class_indices)

        eval_loaders["test_same_class"] = DataLoader(
            test_same_class_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return eval_loaders


def get_loader(dataset, batch_size=256, num_workers=4, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


class FlaggedSubsetDataset(Dataset):
    """
    Dataset wrapper that returns:
        image, hard_target, is_forget
    """

    def __init__(self, base_dataset: Dataset, is_forget: bool) -> None:
        self.base_dataset = base_dataset
        self.is_forget = bool(is_forget)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        return image, int(target), self.is_forget


def build_online_teacher_mixed_loader(
    forget_subset,
    retain_subset,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Build and return three loaders:
    1) mixed train loader:
       - forget subset with hard labels and is_forget=1
       - retain subset with hard labels and is_forget=0
    2) retain_same_class_eval_loader:
       - only retain samples whose true label equals class_to_forget
       - returns standard (image, target)
    3) forget_eval_loader:
       - only forget subset
       - returns standard (image, target)
    """
    flagged_forget_dataset = FlaggedSubsetDataset(
        base_dataset=forget_subset,
        is_forget=True,
    )
    flagged_retain_dataset = FlaggedSubsetDataset(
        base_dataset=retain_subset,
        is_forget=False,
    )

    train_dataset = ConcatDataset([flagged_forget_dataset, flagged_retain_dataset])

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        generator=g,
        pin_memory=True,
    )

    return train_loader
