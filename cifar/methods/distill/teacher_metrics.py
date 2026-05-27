from __future__ import annotations

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from cifar.src.models.features import extract_cifar_resnet56_penultimate, get_feature_extractor
from cifar.methods.distill.support_teachers import (
    SupportTeacher,
    load_support_teacher_from_checkpoint,
)


def _load_teacher_info_from_unlearning_config(
    unlearning_config_path: str | Path,
) -> dict:
    with open(unlearning_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg


def _compute_teacher_accuracy(
    teacher: SupportTeacher,
    loader,
    feature_extractor: torch.nn.Module,
    device: str,
    class_to_forget: int | None = None,
    feature_extractor_fn=None,
) -> dict[str, float | int | None]:
    """
    Compute teacher accuracy on a dataset.

    If class_to_forget is provided, also compute the fraction of predictions
    equal to class_to_forget.
    """
    if feature_extractor_fn is None:
        feature_extractor_fn = extract_cifar_resnet56_penultimate

    teacher.model.eval()
    if teacher.kind == "embedding":
        feature_extractor.eval()

    correct = 0
    total = 0
    predicted_forget_class = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if teacher.kind == "embedding":
                features = feature_extractor_fn(feature_extractor, x)
                logits = teacher.model(features)
            elif teacher.kind == "image":
                logits = teacher.model(x)
            else:
                raise ValueError(f"Unknown teacher kind: {teacher.kind}")

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.numel()

            if class_to_forget is not None:
                predicted_forget_class += (preds == class_to_forget).sum().item()

    return {
        "accuracy": correct / total if total > 0 else None,
        "count": int(total),
        "predicted_forget_class_rate": (
            predicted_forget_class / total
            if total > 0 and class_to_forget is not None
            else None
        ),
    }


def collect_teacher_metrics(
    teacher_model_path,
    unlearning_config_path: str | Path,
    feature_extractor: torch.nn.Module,
    forget_dataset,
    num_classes: int,
    dataset_name,
    device: str,
    class_to_forget: int | None = None,
) -> dict:
    feature_extractor_fn = get_feature_extractor(dataset_name)
    """
    Load teacher model from checkpoint and compute metrics on D_f.
    """
    forget_loader = loader = DataLoader(
        forget_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    with open(unlearning_config_path, "r", encoding="utf-8") as f:
        teacher_info = json.load(f)

    teacher_model_type = teacher_info["teacher_type"]
    support_hidden_dim = teacher_info["support_hidden_dim"]
    support_dropout = teacher_info["support_dropout"]

    teacher = load_support_teacher_from_checkpoint(
        teacher_model_path=teacher_model_path,
        teacher_type=teacher_model_type,
        num_classes=num_classes,
        feature_extractor=feature_extractor,
        loader=forget_loader,
        device=device,
        hidden_dim=support_hidden_dim,
        dropout=support_dropout,
        dataset_name=dataset_name,
    )

    forget_metrics = _compute_teacher_accuracy(
        teacher=teacher,
        loader=forget_loader,
        feature_extractor=feature_extractor,
        device=device,
        class_to_forget=class_to_forget,
        feature_extractor_fn=feature_extractor_fn,
    )

    return {
        "teacher_model_path": str(teacher_model_path),
        "teacher_model_type": teacher_model_type,
        "teacher_kind": teacher.kind,
        "forget_accuracy": forget_metrics["accuracy"],
        "forget_count": forget_metrics["count"],
        "predicted_forget_class_rate": forget_metrics["predicted_forget_class_rate"],
    }
