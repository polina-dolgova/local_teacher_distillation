import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from copy import deepcopy

from mnist.src.dataset import mnist_dataset


def sample_random_labels(
    batch_size: int,
    num_classes: int,
    forbidden_class: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    y_random = torch.randint(
        0,
        num_classes - 1,
        (batch_size,),
        device=device,
        generator=generator,
    )

    # y_random = y_random + (y_random >= forbidden_class).long()
    return y_random


def unlearn_class_with_random_labels(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    class_to_forget: int,
    steps: int,
    lr: float,
    device: str = "cpu",
    weight_decay: float = 0.0,
    seed: int = 42,
) -> tuple[torch.nn.Module, List[float]]:
    """
    Unlearn one class by training on its samples with random incorrect labels.

    Args:
        model: A classification model returning logits of shape (B, C).
        forget_loader: Loader containing only samples of the class to forget.
        class_to_forget: The class index to forget.
        steps: Number of optimization steps.
        lr: Learning rate.
        device: Device string, e.g. "cpu", "cuda", "mps".
        weight_decay: Weight decay for AdamW.

    Returns:
        Updated model and list of losses.
    """
    device = torch.device(device)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()

    losses: List[float] = []
    loader_iter = iter(forget_loader)

    for _ in range(steps):
        try:
            x_forget, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(forget_loader)
            x_forget, _ = next(loader_iter)

        x_forget = x_forget.to(device)

        logits = model(x_forget)
        num_classes = logits.shape[1]

        y_random = sample_random_labels(
            batch_size=x_forget.shape[0],
            num_classes=num_classes,
            forbidden_class=class_to_forget,
            device=device,
            seed=seed,
        )

        loss = criterion(logits, y_random)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return model, losses


def unlearn_class(
    model,
    class_to_forget,
    device="cpu",
    steps=200,
    lr=1e-4,
    weight_decay=1e-6,
    seed=42,
):

    forget_loader = mnist_dataset(class_to_forget=class_to_forget, seed=seed)

    unlearned_model, losses = unlearn_class_with_random_labels(
        model=deepcopy(model),
        forget_loader=forget_loader,
        class_to_forget=class_to_forget,
        steps=steps,
        lr=lr,
        device=device,
        weight_decay=weight_decay,
        seed=seed,
    )

    return unlearned_model
