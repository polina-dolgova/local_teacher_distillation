from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def save_json(path: str | Path, obj: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    epoch_idx: int,
    num_epochs: int,
    log_interval: int,
) -> dict:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    epoch_train_start = time.perf_counter()

    pbar = tqdm(
        loader,
        desc=f"Train {epoch_idx}/{num_epochs}",
        leave=False,
    )

    for step_idx, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        preds = logits.argmax(dim=1)

        running_loss += loss.item() * batch_size
        running_correct += (preds == y).sum().item()
        running_total += batch_size

        if step_idx % log_interval == 0 or step_idx == len(loader):
            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total
            lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.4f}",
                lr=f"{lr:.5f}",
            )

    epoch_train_time = time.perf_counter() - epoch_train_start

    return {
        "train_loss": running_loss / max(running_total, 1),
        "train_acc": running_correct / max(running_total, 1),
        "train_time_seconds": epoch_train_time,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: str,
    num_classes: int,
    class_to_forget: int,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        correct = preds == y

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct.sum().item()
        total_seen += batch_size

        for cls in range(num_classes):
            mask = y == cls
            if mask.any():
                class_total[cls] += mask.sum().item()
                class_correct[cls] += correct[mask].sum().item()

    per_class_acc = []
    for cls in range(num_classes):
        if class_total[cls] == 0:
            per_class_acc.append(None)
        else:
            per_class_acc.append((class_correct[cls] / class_total[cls]).item())

    forget_class_acc = per_class_acc[class_to_forget]

    retain_class_values = [
        acc
        for cls, acc in enumerate(per_class_acc)
        if cls != class_to_forget and acc is not None
    ]
    retain_classes_mean_acc = (
        float(sum(retain_class_values) / len(retain_class_values))
        if retain_class_values
        else None
    )

    return {
        "val_loss": total_loss / max(total_seen, 1),
        "val_acc": total_correct / max(total_seen, 1),
        "forget_class_acc": forget_class_acc,
        "retain_classes_mean_acc": retain_classes_mean_acc,
        "per_class_acc": per_class_acc,
    }
