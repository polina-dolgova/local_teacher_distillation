import torch
import argparse
import json
from pathlib import Path

from cifar.src.custom_types import *
from cifar.src.datasets import get_dataset_class_names, build_cifar_dataset


def get_unlearning_datasets(args):
    class_names = get_dataset_class_names(args.dataset_name)
    num_classes = len(class_names)
    train_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=True)
    test_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=False)
    return train_dataset, test_dataset, num_classes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random Labels unlearning")

    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="outputs")

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100", "svhn"],
    )
    parser.add_argument("--class-to-forget", type=int, default=25)
    parser.add_argument("--class-fraction-to-forget", type=float, default=1.0)

    parser.add_argument("--model-path", type=str, default=None)

    parser.add_argument("--unlearning-steps", type=int, default=400)
    parser.add_argument("--num-epochs", type=int, default=10)

    parser.add_argument("--unlearning-batch-size", type=int, default=256)
    parser.add_argument("--unlearning-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlearning-time-seconds", type=float, default=0.0)

    return parser


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = (
        Path(args.save_dir) / args.dataset_name / f"class_{args.class_to_forget}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config(args: argparse.Namespace, output_dir: Path, name=None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if name == None:
        name = "config.json"

    if isinstance(args, dict):
        config = args.copy()
    else:
        config = vars(args).copy()

    with open(output_dir / name, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_model(
    model: torch.nn.Module, output_dir: Path, name: str = "unlearned_model.pth"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / name)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eval_accuracy(model: torch.nn.Module, loader, device: str) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    model.train()

    if total == 0:
        return float("nan")

    return correct / total
