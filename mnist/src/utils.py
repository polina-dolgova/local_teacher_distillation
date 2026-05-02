from __future__ import annotations
import torch
from pathlib import Path
import time
import argparse
from dataclasses import dataclass, field, asdict
import json
import random
import numpy as np
import torch

@dataclass
class ExperimentConfig:
    # General
    class_to_forget: int = 9
    seed: int = 42
    n_random_repeats: int = 5

    # Output
    output_root: str = "mnist"
    timestamp: int | None = None

    # Distillation support selection
    support_selections: list[str] = field(default_factory=lambda: ["cosine", "random"])
    support_size: int = 200
    support_feature_space: str = "raw"

    # Teacher training
    teacher_epochs: int = 3
    teacher_batch_size: int = 128
    teacher_lr: float = 1e-3
    teacher_weight_decay: float = 1e-6

    # Unlearning fine-tuning
    unlearn_steps: int = 200
    unlearn_steps_ga: int = 50
    unlearn_batch_size: int = 128
    unlearn_lr: float = 1e-4
    unlearn_lr_ga: float = 1e-4
    unlearn_weight_decay: float = 1e-6

    # Distillation
    temperature: float = 1.0
    zero_forget_class_prob: bool = True
    teacher_selection: str = "model"

    # Distillation with neighbor
    label_mode: str = "per sample"
    k_neighbor: int | None = None


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--class-to-forget", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output-root", type=str, default="mnist/mnist_output")
    parser.add_argument("--timestamp", type=int, default=None)

    # Distillation support selection
    parser.add_argument(
        "--support-selections",
        type=str,
        nargs="+",
        default=["cosine"],
        choices=["cosine", "random"],
        help="One or more support selection modes to run.",
    )
    parser.add_argument("--support-size", type=int, default=2000)

    # Teacher training
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--teacher-weight-decay", type=float, default=1e-6)

    # Unlearning fine-tuning
    parser.add_argument("--unlearn-steps", type=int, default=300)
    parser.add_argument("--unlearn-steps-ga", type=int, default=50)
    parser.add_argument("--unlearn-batch-size", type=int, default=128)
    parser.add_argument("--unlearn-lr", type=float, default=5e-5)
    parser.add_argument("--unlearn-lr-ga", type=float, default=0.0001)
    parser.add_argument("--unlearn-weight-decay", type=float, default=1e-6)

    # Distillation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--zero-forget-class-prob", type=str2bool, default=False)

    # Distillation with neighbor
    parser.add_argument("--teacher-selection", type=str, default="model")
    parser.add_argument("--label-mode", type=str, default="per sample")
    parser.add_argument("--k-neighbor", type=int, default=None)

    parser.add_argument(
        "--support-feature-space",
        type=str,
        default="full_model",
        choices=["raw", "full_model"],
    )

    parser.add_argument(
        "--n-random-repeats",
        type=int,
        default=10,
        help="Number of random repeats for the classical random label unlearning.",
    )

    return parser


def parse_config() -> ExperimentConfig:
    parser = build_parser()
    args = parser.parse_args()
    return ExperimentConfig(**vars(args))


def build_output_dir(cfg: ExperimentConfig) -> Path:
    timestamp = cfg.timestamp if cfg.timestamp is not None else int(time.time())
    output_dir = (
        Path(cfg.output_root)
        / str(timestamp)
        / f"mnist_output_delete_{cfg.class_to_forget}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config(cfg: ExperimentConfig, output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def seed_everything(seed: int) -> None:
    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations more deterministic when possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: stricter determinism, but may fail for some operations.
    # torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    # Make DataLoader workers reproducible.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)