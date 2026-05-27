from torchvision import datasets, models, transforms
from torch.utils.data import Subset

from cifar.src.custom_types import *


def get_dataset_class_names(dataset_name: DatasetName) -> list[str]:
    if dataset_name == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    if dataset_name == "cifar100":
        return datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
        ).classes
    if dataset_name == "svhn":
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def get_num_classes(dataset_name: DatasetName) -> int:
    return len(get_dataset_class_names(dataset_name))


def get_dataset_info(dataset_name: DatasetName):
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
    dataset_cls = info["dataset_cls"]

    normalize = transforms.Normalize(mean=mean, std=std)

    if dataset_name == "svhn":
        # SVHN images are already tightly cropped; no spatial augmentation needed.
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

    return train_transform, test_transform, dataset_cls


def _make_dataset(dataset_name: DatasetName, dataset_cls, train: bool, transform):
    if dataset_name in ("cifar10", "cifar100"):
        return dataset_cls(
            root="./data",
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "svhn":
        return dataset_cls(
            root="./data",
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")


def build_cifar_dataset(
    dataset_name: DatasetName,
    train: bool = True,
    do_transforms=True,
    selected_classes: list[int] | None = None,
):
    train_transform, test_transform, dataset_cls = build_transforms(dataset_name)

    if train and do_transforms:
        transform = train_transform
    else:
        transform = test_transform

    dataset = _make_dataset(dataset_name, dataset_cls, train, transform)

    if selected_classes is not None:
        dataset = get_class_subset(dataset, selected_classes=selected_classes)

    return dataset


def build_cifar_dataset_for_imagenet_backbone(
    dataset_name: DatasetName,
    train: bool = True,
):
    weights = models.ResNet18_Weights.DEFAULT
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean,
                std=weights.transforms().std,
            ),
        ]
    )

    info = get_dataset_info(dataset_name)
    dataset_cls = info["dataset_cls"]
    return _make_dataset(dataset_name, dataset_cls, train, transform)


def split_indices_by_forget_class(
    dataset,
    class_to_forget: int,
    class_fraction_to_forget: float = 1.0,
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices into forget and retain parts.
    """
    forget_indices = []
    retain_indices = []

    for idx in range(len(dataset)):
        _, y = dataset[idx]
        # We keep adding forget-class samples until we reach the desired fraction to forget.
        if y == class_to_forget:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)

    forget_size = int(len(forget_indices) * class_fraction_to_forget)

    forget_indices_in_forget = forget_indices[:forget_size]
    forget_indices_in_retain = forget_indices[forget_size:]
    retain_indices.extend(forget_indices_in_retain)

    return forget_indices_in_forget, retain_indices


def get_class_subset(dataset, selected_classes: list[int]):
    """
    Build a dataset subset containing only samples from selected classes.
    """
    selected_set = set(selected_classes)

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise ValueError("Dataset does not have 'targets' or 'labels' attribute.")

    indices = [i for i, y in enumerate(targets) if int(y) in selected_set]
    return Subset(dataset, indices)


def get_dataset_targets(dataset) -> list[int]:
    """
    Return dataset targets as a plain Python list of ints.
    """
    if hasattr(dataset, "targets"):
        return [int(y) for y in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(y) for y in dataset.labels]
    raise ValueError("Dataset does not have 'targets' or 'labels'.")
