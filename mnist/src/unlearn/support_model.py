from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from mnist.src.model import TinyMNISTCNN, SmallMNISTCNN
from mnist.src.dataset import (
    split_dataset_indices_by_class,
    select_support_indices,
    stack_subset_tensors,
)


def train_small_model_on_subset(
    small_model: torch.nn.Module,
    dataset,
    support_indices: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 0.0,
) -> torch.nn.Module:
    """
    Train a small model on a subset using true labels.
    """
    small_model = small_model.to(device)
    small_model.train()

    loader = DataLoader(
        Subset(dataset, support_indices),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        small_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = small_model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return small_model


def build_forget_distillation_loader_by_model(
    teacher_model: torch.nn.Module,
    dataset,
    forget_indices: list[int],
    class_to_forget: int,
    batch_size: int,
    device: str,
    temperature: float = 1.0,
    zero_forget_class_prob: bool = True,
) -> DataLoader:
    """
    Build a loader over forget samples with teacher soft targets.

    Returns batches of:
        x_forget, teacher_probs
    """
    device = torch.device(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    X_forget, _ = stack_subset_tensors(dataset, forget_indices)

    all_probs = []

    with torch.no_grad():
        for start in range(0, len(X_forget), batch_size):
            x_batch = X_forget[start : start + batch_size].to(device)

            teacher_logits = teacher_model(x_batch)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

            if zero_forget_class_prob:
                teacher_probs[:, class_to_forget] = 0.0
                teacher_probs = teacher_probs / torch.clamp(
                    teacher_probs.sum(dim=1, keepdim=True),
                    min=1e-12,
                )

            all_probs.append(teacher_probs.cpu())

    teacher_probs_all = torch.cat(all_probs, dim=0)

    distill_dataset = TensorDataset(X_forget, teacher_probs_all)

    return DataLoader(
        distill_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def get_forget_dataloader_with_distill_softlabels(
    class_to_forget: int,
    support_size: int,
    support_selection: str = "cosine",
    support_feature_space: str = "raw",
    feature_model: torch.nn.Module | None = None,
    teacher_epochs: int = 3,
    teacher_batch_size: int = 128,
    teacher_lr: float = 1e-3,
    teacher_weight_decay: float = 0.0,
    unlearn_batch_size: int = 128,
    temperature: float = 1.0,
    zero_forget_class_prob: bool = False,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.nn.Module, List[float], torch.nn.Module, list[int]]:
    """
    Unlearn one class using support-set distillation.

    Pipeline:
    1) Select support subset from retain data.
    2) Train a small teacher model on that subset.
    3) Infer teacher soft targets on forget-class samples.
    4) Fine-tune the original model on forget-class samples using KL distillation loss.

    Returns:
        unlearned_model,
        losses,
        teacher_model,
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
        feature_space=support_feature_space,
        feature_model=feature_model,
        device=str(device),
        seed=seed,
    )

    teacher_model = TinyMNISTCNN()
    teacher_model = train_small_model_on_subset(
        small_model=teacher_model,
        dataset=train_dataset,
        support_indices=support_indices,
        epochs=teacher_epochs,
        batch_size=teacher_batch_size,
        lr=teacher_lr,
        device=device,
        weight_decay=teacher_weight_decay,
    )

    forget_distill_loader = build_forget_distillation_loader_by_model(
        teacher_model=teacher_model,
        dataset=train_dataset,
        forget_indices=forget_indices,
        class_to_forget=class_to_forget,
        batch_size=unlearn_batch_size,
        device=str(device),
        temperature=temperature,
        zero_forget_class_prob=zero_forget_class_prob,
    )

    return forget_distill_loader, support_indices
