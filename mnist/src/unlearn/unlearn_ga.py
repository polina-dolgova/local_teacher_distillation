import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from copy import deepcopy

from mnist.src.dataset import mnist_dataset


def unlearn_class_with_ga(
    full_model: torch.nn.Module,
    class_to_forget: int,
    steps: int,
    lr: float,
    device: str = "cpu",
    weight_decay: float = 0.0,
    seed: int = 42
) -> tuple[torch.nn.Module, List[float]]:
    """
    Unlearn one class by gradient ascent.

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

    forget_loader = mnist_dataset(class_to_forget=class_to_forget, seed=seed)

    device = torch.device(device)
    model = deepcopy(full_model)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses: List[float] = []
    loader_iter = iter(forget_loader)

    for _ in range(steps):
        try:
            x_forget, y_forget = next(loader_iter)
        except StopIteration:
            loader_iter = iter(forget_loader)
            x_forget, y_forget = next(loader_iter)

        x_forget = x_forget.to(device)
        y_forget = y_forget.to(device)

        logits = model(x_forget)
        num_classes = logits.shape[1]

        loss = -criterion(logits, y_forget)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return model
