from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

DatasetName = Literal["cifar10", "cifar100"]


@dataclass
class TrainConfig:
    save_dir: str
    data_root: str = "./data"
    dataset_name: DatasetName = "cifar10"
    class_to_forget: int = 0
    class_fraction_to_forget: float = 0.1

    epochs: int = 200
    batch_size: int = 256
    num_workers: int = 4

    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True

    device: str = "auto"
    seed: int = 42
    log_interval: int = 20
    save_every: int = 10

    def resolved_save_dir(self) -> Path:
        return Path(self.save_dir)

    def to_dict(self) -> dict:
        return asdict(self)
