from dataclasses import asdict, dataclass
from pathlib import Path
import json
import argparse
import shutil

from cifar.src.custom_types import *


@dataclass
class ExperimentConfig:
    output_dir: str | Path = ""
    save_dir: str | None = None
    dataset_name: DatasetName = "cifar100"
    class_to_forget: int = 25
    unlearning_method: UnlearningMethod = "negative_grad"
    similarity_source: SimilaritySource = "full_model"
    unlearning_batch_size: int = 128
    device: str = "cuda"
    selected_classes: list[int] | None = None
    num_classes: int = 100
    unlearned_model_path: str | Path = ""
    timestamp: int | None = None
    seed: int = 42
    class_fraction_to_forget: float = 1.0
    test_on_train: bool = False
    point_topk_fraction: float = 0.1
    metrics: dict | None = None
    full_model_path: str | None = None
    retrain_model_paths: list[str] | None = None
    svc_mia: dict | None = None

    # Teacher evaluation
    teacher_metrics: dict | None = None
    teacher_model_path: str | None = None
    unlearning_config_path: str | None = None

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["output_dir"] = str(data["output_dir"])
        return data


def copy_unlearning_config_dir(src_model_path, dst_dir) -> None:
    src_dir = src_model_path.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        dst_item = dst_dir / item.name

        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)


def save_experiment_config(config: ExperimentConfig, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unlearning experiment.")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where experiment outputs will be stored.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory for saving checkpoints or intermediate artifacts.",
    )

    parser.add_argument(
        "--full-model-path",
        type=str,
        default=None,
        help="Path to fully trained model",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--class-to-forget",
        type=int,
        default=25,
        help="Class index to forget.",
    )
    parser.add_argument(
        "--unlearning-method",
        type=str,
        default="negative_grad",
        help="Unlearning method to run.",
    )
    parser.add_argument(
        "--similarity-source",
        type=str,
        default="imagenet_resnet18",
        choices=["imagenet_resnet18", "full_model", "raw_pixels"],
        help="Feature source used for similarity computations.",
    )
    parser.add_argument(
        "--unlearning-batch-size",
        type=int,
        default=128,
        help="Batch size for unlearning.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use, e.g. cpu, cuda, mps.",
    )
    parser.add_argument(
        "--selected-classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of class indices to keep/use in the experiment.",
    )
    parser.add_argument(
        "--unlearned-model-path",
        type=str,
        default="",
        help="Path to the unlearned model checkpoint for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Timestamp for the experiment.",
    )
    parser.add_argument(
        "--test-on-train",
        action="store_true",
        help="Whether to evaluate the model on the train set.",
    )
    parser.add_argument(
        "--class-fraction-to-forget",
        type=float,
        default=1.0,
        help="Fraction of samples from the forget class to unlearn.",
    )

    parser.add_argument(
        "--point-topk-fraction",
        type=float,
        default=0.1,
        help="Fraction of retain datapoints used for nearest/farthest pointwise metrics.",
    )

    parser.add_argument(
        "--teacher-model-path",
        type=str,
        default=None,
        help="Path to teacher model checkpoint",
    )

    parser.add_argument(
        "--retrain-model-paths",
        type=str,
        nargs="+",
        default=None,
        help="List of paths to retrain model checkpoints",
    )

    parser.add_argument(
        "--unlearning-config-path",
        type=str,
        default=None,
        help="Path to unlearning config",
    )

    return parser


def parse_config() -> ExperimentConfig:
    parser = build_parser()
    args = parser.parse_args()
    return ExperimentConfig(**vars(args))
