from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mix_with_gold(color: str, gold_ratio: float = 0.4):
    rgb = np.array(to_rgb(color))
    gold = np.array(to_rgb("#C9A227"))
    return (1 - gold_ratio) * rgb + gold_ratio * gold


def _save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _plot_metric_group(
    x: np.ndarray,
    series: dict[str, dict[str, list[float]]],
    output_path: str | Path,
    title: str,
    ylabel: str,
    multiply_by_100: bool = True,
    retrain_values: dict[str, float] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_colors = {
        "RA": "#5AA469",
        "RA forget class": "#5AA469",
        "UA": "#B00101",
        "UA forget class": "#B00101",
        "TA": "#024583",
        "TA forget class": "#024583",
    }

    plt.figure(figsize=(20.0, 10.0))

    for metric_name, metric_data in series.items():
        y = np.asarray(metric_data["mean"], dtype=float)
        s = np.asarray(metric_data["std"], dtype=float)

        if multiply_by_100:
            y = 100.0 * y
            s = 100.0 * s

        color = metric_colors.get(metric_name, None)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            label=metric_name,
            color=color,
        )
        plt.fill_between(
            x,
            y - s,
            y + s,
            alpha=0.15,
            color=color,
        )

        print(retrain_values, metric_name)
        if retrain_values is not None and metric_name in retrain_values:
            retrain_y = float(retrain_values[metric_name])
            if multiply_by_100:
                retrain_y = 100.0 * retrain_y

            retrain_color = mix_with_gold(color, gold_ratio=0.6)

            plt.axhline(
                retrain_y,
                linestyle="--",
                linewidth=1.5,
                color=retrain_color,
                alpha=1.0,
                label=f"{metric_name} retrain",
            )

    plt.xlabel("Teacher accuracy threshold", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=30)
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=False,
        fontsize=20,
    )
    plt.tight_layout()
    plt.show()
    # plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def collect_and_plot_by_teacher_accuracy_threshold(
    parent_dir: str | Path,
    method_name: str,
    class_to_forget: int,
    class_fraction_to_forget: float,
    output_dir: str | Path,
    standard_retrain_values: dict[str, float] | None = None,
    forget_class_retrain_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Group runs by teacher_accuracy_threshold and:
    1) compute mean/std for standard metrics: RA, TA, UA
    2) compute mean/std for forget-class metrics:
       RA_forget_class, UA_forget_class, TA_forget_class
    3) compute mean/std for teacher_real_epochs
    4) save one summary json and three plots
    """
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_root = (
        parent_dir / method_name / f"class_{class_to_forget}" / "test_on_train=True"
    )
    test_root = (
        parent_dir / method_name / f"class_{class_to_forget}" / "test_on_train=False"
    )

    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Missing train/test roots:\ntrain_root={train_root}\ntest_root={test_root}"
        )

    train_ts_dirs = sorted(
        [p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name
    )
    test_ts_dirs = sorted(
        [p for p in test_root.iterdir() if p.is_dir()], key=lambda p: p.name
    )

    grouped_values: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {
            "RA": [],
            "TA": [],
            "UA": [],
            "RA_forget_class": [],
            "UA_forget_class": [],
            "TA_forget_class": [],
            "teacher_real_epochs": [],
            "teacher_forget_class_prediction": [],
        }
    )
    grouped_runs: dict[float, list[tuple[str, str]]] = defaultdict(list)

    for train_ts_dir, test_ts_dir in zip(train_ts_dirs, test_ts_dirs):
        train_config_path = train_ts_dir / "config.json"
        test_config_path = test_ts_dir / "config.json"
        unlearning_config_path = test_ts_dir / "unlearning_config" / "config.json"
        teacher_config_path = (
            test_ts_dir / "unlearning_config" / "teacher_model" / "config.json"
        )

        if not (
            train_config_path.exists()
            and test_config_path.exists()
            and unlearning_config_path.exists()
            and teacher_config_path.exists()
        ):
            continue

        try:
            train_config = _load_json(train_config_path)
            test_config = _load_json(test_config_path)
            unlearning_config = _load_json(unlearning_config_path)
            teacher_config = _load_json(teacher_config_path)
        except Exception:
            continue

        train_fraction = train_config.get("class_fraction_to_forget")
        test_fraction = test_config.get("class_fraction_to_forget")
        if (
            train_fraction != class_fraction_to_forget
            or test_fraction != class_fraction_to_forget
        ):
            continue

        teacher_accuracy_threshold = unlearning_config.get("teacher_accuracy_threshold")
        teacher_real_epochs = teacher_config.get("teacher_real_epochs")

        if teacher_accuracy_threshold is None or teacher_real_epochs is None:
            continue

        try:
            teacher_accuracy_threshold = float(teacher_accuracy_threshold)
            teacher_real_epochs = float(teacher_real_epochs)

            ra = float(train_config["metrics"]["retain_accuracy"]["after_unlearning"])
            ua = float(train_config["metrics"]["forget_accuracy"]["after_unlearning"])
            ta = float(test_config["metrics"]["retain_accuracy"]["after_unlearning"])

            ra_forget_class = float(
                train_config["metrics"]["retain_subset_true_label_equals_forget_class"][
                    "unlearn"
                ]
            )
            ua_forget_class = float(
                train_config["metrics"]["forget_accuracy"]["after_unlearning"]
            )
            ta_forget_class = float(
                test_config["metrics"]["retain_subset_true_label_equals_forget_class"][
                    "unlearn"
                ]
            )

            teacher_forget_class_prediction = float(
                train_config["teacher_metrics"]["forget_accuracy"]
            )
        except Exception:
            continue

        grouped_values[teacher_accuracy_threshold]["RA"].append(ra)
        grouped_values[teacher_accuracy_threshold]["TA"].append(ta)
        grouped_values[teacher_accuracy_threshold]["UA"].append(ua)
        grouped_values[teacher_accuracy_threshold]["RA_forget_class"].append(
            ra_forget_class
        )
        grouped_values[teacher_accuracy_threshold]["UA_forget_class"].append(
            ua_forget_class
        )
        grouped_values[teacher_accuracy_threshold]["TA_forget_class"].append(
            ta_forget_class
        )
        grouped_values[teacher_accuracy_threshold]["teacher_real_epochs"].append(
            teacher_real_epochs
        )
        grouped_values[teacher_accuracy_threshold][
            "teacher_forget_class_prediction"
        ].append(teacher_forget_class_prediction)

        grouped_runs[teacher_accuracy_threshold].append(
            (train_ts_dir.name, test_ts_dir.name)
        )

    thresholds = sorted(grouped_values.keys())

    summary: dict[str, Any] = {
        "method_name": method_name,
        "class_to_forget": class_to_forget,
        "class_fraction_to_forget": class_fraction_to_forget,
        "groups": [],
    }

    for thr in thresholds:
        values = grouped_values[thr]
        group_summary = {
            "teacher_accuracy_threshold": thr,
            "num_runs": len(grouped_runs[thr]),
            "used_ts_dirs": grouped_runs[thr],
            "metrics": {
                metric_name: {
                    "values": metric_values,
                    "mean": _mean(metric_values),
                    "std": _std(metric_values),
                }
                for metric_name, metric_values in values.items()
            },
        }
        summary["groups"].append(group_summary)

    summary_json_path = output_dir / f"{method_name}_by_teacher_accuracy_threshold.json"
    _save_json(summary_json_path, summary)

    x = np.asarray(thresholds, dtype=float)

    standard_series = {
        "RA": {
            "mean": [
                grouped_values[t]["RA"] and _mean(grouped_values[t]["RA"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["RA"] and _std(grouped_values[t]["RA"])
                for t in thresholds
            ],
        },
        "TA": {
            "mean": [
                grouped_values[t]["TA"] and _mean(grouped_values[t]["TA"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["TA"] and _std(grouped_values[t]["TA"])
                for t in thresholds
            ],
        },
        "UA": {
            "mean": [
                grouped_values[t]["UA"] and _mean(grouped_values[t]["UA"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["UA"] and _std(grouped_values[t]["UA"])
                for t in thresholds
            ],
        },
    }

    forget_class_series = {
        "RA forget class": {
            "mean": [
                grouped_values[t]["RA_forget_class"]
                and _mean(grouped_values[t]["RA_forget_class"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["RA_forget_class"]
                and _std(grouped_values[t]["RA_forget_class"])
                for t in thresholds
            ],
        },
        "UA forget class": {
            "mean": [
                grouped_values[t]["UA_forget_class"]
                and _mean(grouped_values[t]["UA_forget_class"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["UA_forget_class"]
                and _std(grouped_values[t]["UA_forget_class"])
                for t in thresholds
            ],
        },
        "TA forget class": {
            "mean": [
                grouped_values[t]["TA_forget_class"]
                and _mean(grouped_values[t]["TA_forget_class"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["TA_forget_class"]
                and _std(grouped_values[t]["TA_forget_class"])
                for t in thresholds
            ],
        },
    }

    teacher_epochs_series = {
        "teacher_real_epochs": {
            "mean": [
                grouped_values[t]["teacher_real_epochs"]
                and _mean(grouped_values[t]["teacher_real_epochs"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["teacher_real_epochs"]
                and _std(grouped_values[t]["teacher_real_epochs"])
                for t in thresholds
            ],
        }
    }

    teacher_prediction_series = {
        "teacher_forget_class_prediction": {
            "mean": [
                grouped_values[t]["teacher_forget_class_prediction"]
                and _mean(grouped_values[t]["teacher_forget_class_prediction"])
                for t in thresholds
            ],
            "std": [
                grouped_values[t]["teacher_forget_class_prediction"]
                and _std(grouped_values[t]["teacher_forget_class_prediction"])
                for t in thresholds
            ],
        }
    }

    _plot_metric_group(
        x=x,
        series=standard_series,
        output_path=output_dir
        / f"{method_name}_standard_metrics_vs_teacher_threshold.pdf",
        title=f"RA / TA / UA vs teacher accuracy threshold",
        ylabel="Accuracy (%)",
        multiply_by_100=True,
        retrain_values=standard_retrain_values,
    )

    _plot_metric_group(
        x=x,
        series=forget_class_series,
        output_path=output_dir
        / f"{method_name}_forget_class_metrics_vs_teacher_threshold.pdf",
        title=f"Forget-class metrics vs teacher accuracy threshold",
        ylabel="Accuracy (%)",
        multiply_by_100=True,
        retrain_values=forget_class_retrain_values,
    )

    _plot_metric_group(
        x=x,
        series=teacher_epochs_series,
        output_path=output_dir
        / f"{method_name}_teacher_epochs_vs_teacher_threshold.pdf",
        title=f"Teacher epochs vs teacher accuracy threshold",
        ylabel="Teacher epochs",
        multiply_by_100=False,
    )

    _plot_metric_group(
        x=x,
        series=teacher_prediction_series,
        output_path=output_dir
        / f"{method_name}_teacher_prediction_series_vs_teacher_threshold.pdf",
        title=f"Teacher forget set accuracy vs teacher accuracy threshold",
        ylabel=r"Teacher $D_f$ accuracy",
        multiply_by_100=True,
    )

    return summary
