from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from cifar.src.custom_types import *
from cifar.src.datasets import (
    build_cifar_dataset,
    get_dataset_class_names,
)

from cifar.methods.utils import (
    build_parser as build_parser_base,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_parser_base()
    parser.description = "Support-based distillation / neighbor-label unlearning"

    parser.add_argument(
        "--mode",
        type=str,
        default="distill",
        choices=["distill", "neighbor"],
    )

    parser.add_argument(
        "--teacher-type",
        type=str,
        default="small_cnn",
        choices=["linear_frozen", "mlp_frozen", "small_cnn", "resnet8", "resnet56"],
    )

    parser.add_argument("--retain-fraction", type=float, default=1.0)
    parser.add_argument(
        "--retain-selection-mode",
        type=str,
        default="random",
        choices=["random", "closest", "farthest"],
    )

    parser.add_argument("--n-support", type=int, default=500)
    parser.add_argument("--k-closest", type=int, default=None)

    parser.add_argument("--support-batch-size", type=int, default=256)
    parser.add_argument("--support-epochs", type=int, default=5)
    parser.add_argument("--support-lr", type=float, default=1e-3)
    parser.add_argument("--support-weight-decay", type=float, default=1e-4)
    parser.add_argument("--support-hidden-dim", type=int, default=256)
    parser.add_argument("--support-dropout", type=float, default=0.0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--retain-loss-weight", type=float, default=1.0)
    parser.add_argument("--forget-loss-weight", type=float, default=1.0)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--zero-forget-class-prob", action="store_true")

    parser.add_argument(
        "--support-selection-mode",
        type=str,
        default="closest",
        choices=["random", "closest", "farthest"],
    )

    parser.add_argument("--teacher-accuracy-threshold", type=float, default=None)

    parser.add_argument("--n-support-analytics", type=int, nargs="*", default=None)
    parser.add_argument("--optimizer-type", type=str, default="adamw")
    parser.add_argument("--retrain-model-path", type=str, default=None)
    parser.add_argument("--trunk-k", type=int, default=5)

    parser.add_argument(
        "--softlabels-target-mode",
        type=str,
        default="soft",
    )

    return parser


def soft_cross_entropy_per_sample(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1)


def get_unlearning_datasets(args):
    class_names = get_dataset_class_names(args.dataset_name)
    num_classes = len(class_names)

    train_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=True)
    test_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=False)
    train_dataset_clean = build_cifar_dataset(
        dataset_name=args.dataset_name, train=True, do_transforms=False
    )

    return (
        train_dataset,
        test_dataset,
        train_dataset_clean,
        num_classes,
    )


def calculate_eval_accuracy(unlearned_model, eval_loader, class_to_forget, device):
    # Compute train accuracy after the epoch.
    unlearned_model.eval()
    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for x, soft_targets, is_forget in eval_loader:
            x = x.to(device)
            soft_targets = soft_targets.to(device)

            logits = unlearned_model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == class_to_forget).sum().item()
            total += soft_targets.size(0)

            loss += soft_cross_entropy_per_sample(logits, soft_targets).sum().item()

    train_acc = correct / total if total > 0 else 0.0
    train_loss = loss / total if total > 0 else 0.0

    return train_acc, train_loss
