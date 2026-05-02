from __future__ import annotations
import torch
from pathlib import Path
import time

from cifar.src.config import ExperimentConfig


def build_output_dir(cfg: ExperimentConfig) -> Path:
    timestamp = cfg.timestamp if cfg.timestamp is not None else int(time.time())
    output_dir = (
        f"{Path(cfg.save_dir)}/{cfg.unlearning_method}/class_{cfg.class_to_forget}/"
        f"test_on_train={cfg.test_on_train}/{timestamp}"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_default_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
