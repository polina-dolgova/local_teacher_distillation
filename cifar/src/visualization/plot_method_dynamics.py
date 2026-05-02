from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


def _load_run_history(
    ts_dir: Path,
    config_filename: str,
    history_relpath: str,
    class_fraction_to_forget: float | None,
) -> dict[str, Any] | None:
    config_path = ts_dir / config_filename
    history_path = ts_dir / history_relpath

    if not config_path.exists() or not history_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None

    if class_fraction_to_forget is not None:
        if config.get("class_fraction_to_forget") != class_fraction_to_forget:
            return None

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return None

    required_keys = ["forget", "retain_same_class", "test_same_class"]
    if not all(key in history for key in required_keys):
        return None

    forget = np.asarray(history["forget"], dtype=float)
    retain_same_class = np.asarray(history["retain_same_class"], dtype=float)
    test_same_class = np.asarray(history["test_same_class"], dtype=float)

    if len(forget) == 0:
        return None

    if not (len(forget) == len(retain_same_class) == len(test_same_class)):
        return None

    return {
        "ts_dir": str(ts_dir),
        "forget": forget,
        "retain_same_class": retain_same_class,
        "test_same_class": test_same_class,
    }


def _collect_method_histories(
    parent_dir: Path,
    method_name: str,
    class_to_forget: int,
    config_filename: str,
    history_relpath: str,
    class_fraction_to_forget: float | None,
) -> list[dict[str, Any]]:
    method_dir = (
        parent_dir / method_name / f"class_{class_to_forget}" / "test_on_train=True"
    )

    run_histories = []
    if not method_dir.exists() or not method_dir.is_dir():
        return run_histories

    for ts_dir in sorted(method_dir.iterdir()):
        if not ts_dir.is_dir():
            continue

        history = _load_run_history(
            ts_dir=ts_dir,
            config_filename=config_filename,
            history_relpath=history_relpath,
            class_fraction_to_forget=class_fraction_to_forget,
        )
        if history is not None:
            run_histories.append(history)

    return run_histories


def _aggregate_histories(
    run_histories: list[dict[str, Any]],
    multiply_by_100: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_runs": 0,
        "num_epochs": 0,
        "used_ts_dirs": [],
        "epochs": [],
        "metrics": {},
    }

    if len(run_histories) == 0:
        return result

    reference_len = len(run_histories[0]["forget"])
    compatible_runs = []
    for run in run_histories:
        if (
            len(run["forget"]) == reference_len
            and len(run["retain_same_class"]) == reference_len
            and len(run["test_same_class"]) == reference_len
        ):
            compatible_runs.append(run)

    if len(compatible_runs) == 0:
        return result

    epochs = np.arange(1, reference_len + 1, dtype=int)
    metric_keys = ["forget", "retain_same_class", "test_same_class"]

    metrics = {}
    for key in metric_keys:
        matrix = np.stack([run[key] for run in compatible_runs], axis=0)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)

        if multiply_by_100:
            mean = 100.0 * mean
            std = 100.0 * std

        metrics[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    result.update(
        {
            "num_runs": len(compatible_runs),
            "num_epochs": int(reference_len),
            "used_ts_dirs": [run["ts_dir"] for run in compatible_runs],
            "epochs": epochs.tolist(),
            "metrics": metrics,
        }
    )
    return result


def _load_retrain_reference_from_summary(
    summary_json_path: str | Path,
    retrain_method_name: str = "retrain_official",
    multiply_by_100: bool = True,
) -> dict[str, dict[str, float]] | None:
    summary_json_path = Path(summary_json_path)
    if not summary_json_path.exists():
        return None

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if retrain_method_name not in summary:
        return None

    metrics = summary[retrain_method_name]["metrics"]

    mapping = {
        "retain_same_class": "RA_forget_class",
        "forget": "FA_forget_class",
        "test_same_class": "TA_forget_class",
    }

    result = {}
    for curve_name, metric_name in mapping.items():
        metric = metrics.get(metric_name, {})
        mean = metric.get("mean")
        std = metric.get("std")

        if mean is None or std is None:
            result[curve_name] = {"mean": None, "std": None}
            continue

        if multiply_by_100:
            mean = 100.0 * float(mean)
            std = 100.0 * float(std)
        else:
            mean = float(mean)
            std = float(std)

        result[curve_name] = {"mean": mean, "std": std}

    return result


def mix_with_gray(color: str, gray_ratio: float = 0.5):
    """
    gray_ratio:
        0.0 → original color
        1.0 → pure gray
    """
    rgb = np.array(to_rgb(color))
    gray = np.array([0.5, 0.5, 0.5])
    mixed = (1 - gray_ratio) * rgb + gray_ratio * gray
    return mixed


def plot_single_method_curves_with_retrain_reference(
    parent_dir: str | Path,
    method_name: str,
    class_to_forget: int,
    output_path: str | Path,
    retrain_summary_json_path: str | Path,
    class_fraction_to_forget: float | None = None,
    official_method_name: str | None = None,
    retrain_method_name: str = "retrain_official",
    retrain_display_name: str = "Retrain",
    config_filename: str = "config.json",
    history_relpath: str = "unlearning_config/accuracies.json",
    output_path_stats: str | Path | None = None,
    multiply_by_100: bool = True,
    fill_alpha: float = 0.16,
    reference_fill_alpha: float = 0.10,
) -> dict[str, Any]:
    parent_dir = Path(parent_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path_stats is None:
        output_path_stats = output_path.with_name(output_path.stem + "_stats.json")
    output_path_stats = Path(output_path_stats)

    run_histories = _collect_method_histories(
        parent_dir=parent_dir,
        method_name=method_name,
        class_to_forget=class_to_forget,
        config_filename=config_filename,
        history_relpath=history_relpath,
        class_fraction_to_forget=class_fraction_to_forget,
    )
    aggregated = _aggregate_histories(
        run_histories=run_histories,
        multiply_by_100=multiply_by_100,
    )

    retrain_reference = _load_retrain_reference_from_summary(
        summary_json_path=retrain_summary_json_path,
        retrain_method_name=retrain_method_name,
        multiply_by_100=multiply_by_100,
    )

    result = {
        "plot_path": str(output_path),
        "stats_path": str(output_path_stats),
        "method_name": method_name,
        "official_method_name": official_method_name or method_name,
        "class_to_forget": int(class_to_forget),
        "class_fraction_to_forget": class_fraction_to_forget,
        "num_runs": aggregated["num_runs"],
        "num_epochs": aggregated["num_epochs"],
        "used_ts_dirs": aggregated["used_ts_dirs"],
        "epochs": aggregated["epochs"],
        "metrics": aggregated["metrics"],
        "retrain_reference": retrain_reference,
    }

    if aggregated["num_runs"] == 0:
        with open(output_path_stats, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    metric_order = ["retain_same_class", "forget", "test_same_class"]
    metric_labels = {
        "retain_same_class": "Retain Same Class",
        "forget": "Forget",
        "test_same_class": "Test Same Class",
    }
    metric_colors = {
        "retain_same_class": "#5AA469",
        "forget": "#B00101",
        "test_same_class": "#024583",
    }
    reference_linestyles = {
        "retain_same_class": "-",
        "forget": "--",
        "test_same_class": ":",
    }

    epochs = np.asarray(aggregated["epochs"], dtype=float)

    plt.figure(figsize=(20.0, 10.0))

    # Retrain reference: horizontal mean ± std bands
    if retrain_reference is not None:
        for key in metric_order:
            ref_mean = retrain_reference[key]["mean"]
            ref_std = retrain_reference[key]["std"]

            if ref_mean is None or ref_std is None:
                continue

            plt.fill_between(
                epochs,
                np.full_like(epochs, ref_mean - ref_std, dtype=float),
                np.full_like(epochs, ref_mean + ref_std, dtype=float),
                color=mix_with_gray(metric_colors[key], 0.4),
                alpha=reference_fill_alpha,
                zorder=0,
            )
            plt.plot(
                epochs,
                np.full_like(epochs, ref_mean, dtype=float),
                color=mix_with_gray(metric_colors[key], 0.4),
                linestyle=reference_linestyles[key],
                linewidth=2.2,
                alpha=0.95,
                label=f"{metric_labels[key]} ({retrain_display_name})",
                zorder=1,
            )

    # Method curves by epoch
    for key in metric_order:
        mean = np.asarray(aggregated["metrics"][key]["mean"], dtype=float)
        std = np.asarray(aggregated["metrics"][key]["std"], dtype=float)

        plt.plot(
            epochs,
            mean,
            marker="o",
            markersize=4,
            linewidth=2.0,
            label=metric_labels[key],
            color=metric_colors[key],
            zorder=3,
        )
        plt.fill_between(
            epochs,
            mean - std,
            mean + std,
            color=metric_colors[key],
            alpha=fill_alpha,
            zorder=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)" if multiply_by_100 else "Accuracy")
    plt.title(f"{official_method_name or method_name}: unlearning dynamics")
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.show()
    # plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.close()
