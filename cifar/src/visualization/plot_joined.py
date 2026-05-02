from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb

SplitName = str


def _empty_method_result(
    method_name: str,
    official_method_names: dict[str, str],
) -> dict[str, Any]:
    return {
        "official_name": official_method_names.get(method_name, method_name),
        "num_runs": 0,
        "bin_centers": [],
        "mean": [],
        "std": [],
        "bin_counts_mean": [],
        "bin_consensus_counts": [],
        "used_ts_dirs": [],
    }


def _load_run_stats(
    ts_dir: Path,
    stats_filename: str,
    config_filename: str,
    class_fraction_to_forget: float,
) -> dict[str, Any] | None:
    config_path = ts_dir / config_filename
    stats_path = ts_dir / stats_filename

    if not config_path.exists() or not stats_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None

    if config.get("class_fraction_to_forget", None) != class_fraction_to_forget:
        return None

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        return None

    bin_centers = np.asarray(stats["bin_centers"], dtype=float)
    mean_drop = np.asarray(stats["mean_accuracy_drop_per_bin"], dtype=float)

    counts_key = (
        "bin_consensus_counts" if "bin_consensus_counts" in stats else "bin_counts"
    )
    bin_counts = np.asarray(stats[counts_key], dtype=float)

    if len(bin_centers) == 0 or len(mean_drop) == 0:
        return None

    if not (len(bin_centers) == len(mean_drop) == len(bin_counts)):
        return None

    return {
        "ts_dir": str(ts_dir),
        "bin_centers": bin_centers,
        "mean_drop": mean_drop,
        "bin_counts": bin_counts,
    }


def _collect_method_runs(
    parent_dir: Path,
    method_name: str,
    class_to_forget: int,
    class_fraction_to_forget: float,
    test_on_train: bool,
    stats_filename: str,
    config_filename: str,
) -> list[dict[str, Any]]:
    method_dir = (
        parent_dir
        / method_name
        / f"class_{class_to_forget}"
        / f"test_on_train={test_on_train}"
    )

    if not method_dir.exists() or not method_dir.is_dir():
        return []

    run_stats = []
    for ts_dir in sorted(method_dir.iterdir()):
        if not ts_dir.is_dir():
            continue

        stats = _load_run_stats(
            ts_dir=ts_dir,
            stats_filename=stats_filename,
            config_filename=config_filename,
            class_fraction_to_forget=class_fraction_to_forget,
        )

        if stats is not None:
            run_stats.append(stats)

    return run_stats


def _aggregate_runs(
    run_stats: list[dict[str, Any]],
    method_name: str,
    official_method_names: dict[str, str],
) -> dict[str, Any]:
    if len(run_stats) == 0:
        return _empty_method_result(method_name, official_method_names)

    reference_bin_centers = run_stats[0]["bin_centers"]

    compatible_runs = []
    for run in run_stats:
        if len(run["bin_centers"]) != len(reference_bin_centers):
            continue
        if not np.allclose(run["bin_centers"], reference_bin_centers, atol=1e-10):
            continue
        compatible_runs.append(run)

    if len(compatible_runs) == 0:
        return _empty_method_result(method_name, official_method_names)

    mean_matrix = np.stack([run["mean_drop"] for run in compatible_runs], axis=0)
    count_matrix = np.stack([run["bin_counts"] for run in compatible_runs], axis=0)

    aggregated_mean = np.nanmean(mean_matrix, axis=0)
    aggregated_std = np.nanstd(mean_matrix, axis=0)
    aggregated_counts_mean = np.nanmean(count_matrix, axis=0)

    return {
        "official_name": official_method_names.get(method_name, method_name),
        "num_runs": len(compatible_runs),
        "bin_centers": reference_bin_centers.tolist(),
        "mean": aggregated_mean.tolist(),
        "std": aggregated_std.tolist(),
        "bin_counts_mean": aggregated_counts_mean.tolist(),
        "bin_consensus_counts": aggregated_counts_mean.tolist(),
        "used_ts_dirs": [run["ts_dir"] for run in compatible_runs],
    }


def _collect_results_for_split(
    parent_dir: Path,
    method_names: list[str],
    class_to_forget: int,
    class_fraction_to_forget: float,
    official_method_names: dict[str, str],
    test_on_train: bool,
    stats_filename: str,
    config_filename: str,
) -> dict[str, dict[str, Any]]:
    results = {}

    for method_name in method_names:
        run_stats = _collect_method_runs(
            parent_dir=parent_dir,
            method_name=method_name,
            class_to_forget=class_to_forget,
            class_fraction_to_forget=class_fraction_to_forget,
            test_on_train=test_on_train,
            stats_filename=stats_filename,
            config_filename=config_filename,
        )

        results[method_name] = _aggregate_runs(
            run_stats=run_stats,
            method_name=method_name,
            official_method_names=official_method_names,
        )

    return results


def _compute_global_limits(
    per_split_results: dict[SplitName, dict[str, dict[str, Any]]],
) -> tuple[float | None, float | None, float | None, float | None]:
    global_x_min = None
    global_x_max = None
    global_y_min = None
    global_y_max = None

    for split_results in per_split_results.values():
        for result in split_results.values():
            if result["num_runs"] == 0:
                continue

            x = np.asarray(result["bin_centers"], dtype=float)
            y = np.asarray(result["mean"], dtype=float)
            y_std = np.asarray(result["std"], dtype=float)

            valid_mask = ~np.isnan(y)
            if not np.any(valid_mask):
                continue

            x_valid = x[valid_mask]
            y_low_valid = (y - y_std)[valid_mask]
            y_high_valid = (y + y_std)[valid_mask]

            x_min = float(np.min(x_valid))
            x_max = float(np.max(x_valid))
            y_min = float(np.min(y_low_valid))
            y_max = float(np.max(y_high_valid))

            global_x_min = x_min if global_x_min is None else min(global_x_min, x_min)
            global_x_max = x_max if global_x_max is None else max(global_x_max, x_max)
            global_y_min = y_min if global_y_min is None else min(global_y_min, y_min)
            global_y_max = y_max if global_y_max is None else max(global_y_max, y_max)

    return global_x_min, global_x_max, global_y_min, global_y_max


def _plot_result_series(
    ax,
    result: dict[str, Any],
    reference_result,
    label: str,
    color: str,
    fill_alpha: float,
    line_alpha: float,
    merge_group_size: int,
    linestyle: str = "-",
) -> None:
    if result["num_runs"] == 0:
        return

    x = np.asarray(result["bin_centers"], dtype=float)
    y = np.asarray(result["mean"], dtype=float)
    y_ref = np.asarray(reference_result["mean"], dtype=float)
    y = y - y_ref
    y_std = np.asarray(result["std"], dtype=float)
    counts = np.asarray(result["bin_consensus_counts"], dtype=float)

    # x, y, y_std = _merge_bins_with_counts(
    #     x=x,
    #     y=y,
    #     y_std=y_std,
    #     counts=counts,
    #     group_size=merge_group_size,
    # )

    valid_mask = ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    y_std = y_std[valid_mask]

    ax.plot(
        x,
        y,
        marker="o",
        markersize=4,
        label=label,
        color=color,
        alpha=line_alpha,
        linestyle=linestyle,
    )
    ax.fill_between(
        x,
        y - y_std,
        y + y_std,
        color=color,
        alpha=fill_alpha,
    )


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


def _plot_one_method(
    ax,
    method_name: str,
    per_split_results: dict[SplitName, dict[str, dict[str, Any]]],
    global_limits: tuple[float | None, float | None, float | None, float | None],
    fill_alpha: float,
    merge_group_size: int,
    reference_method: str,
    main_method: str,
) -> None:
    test_result = per_split_results["test"][method_name]
    train_result = per_split_results["train"][method_name]

    reference_test_result = per_split_results["test"].get(reference_method)
    reference_train_result = per_split_results["train"].get(reference_method)

    official_name = test_result["official_name"]
    test_runs = test_result["num_runs"]
    train_runs = train_result["num_runs"]

    ax.set_title(f"{official_name}", fontsize=20, fontweight="semibold")

    if test_runs == 0 and train_runs == 0:
        ax.text(0.5, 0.5, "No matching runs", ha="center", va="center")
        ax.grid(True, alpha=0.3)
        return

    test_color = "steelblue"
    train_color = "limegreen"
    # if reference_test_result is not None:
    #     _plot_result_series(
    #         ax=ax,
    #         result=reference_test_result,
    #         label="retrain ref. test",
    #         color=mix_with_gray(test_color, 0.4),
    #         fill_alpha=0.08,
    #         line_alpha=0.75,
    #         merge_group_size=merge_group_size,
    #         linestyle="--",
    #     )

    # if reference_train_result is not None:
    #     _plot_result_series(
    #         ax=ax,
    #         result=reference_train_result,
    #         label="retrain ref. train",
    #         color=mix_with_gray(train_color, 0.4),
    #         fill_alpha=0.05,
    #         line_alpha=0.55,
    #         merge_group_size=merge_group_size,
    #         linestyle=":",
    #     )

    _plot_result_series(
        ax=ax,
        result=test_result,
        reference_result=reference_test_result,
        label="test data mismatch",
        color=test_color,
        fill_alpha=fill_alpha,
        line_alpha=1.0,
        merge_group_size=merge_group_size,
    )

    _plot_result_series(
        ax=ax,
        result=train_result,
        reference_result=reference_train_result,
        label="retain data mismatch",
        color=train_color,
        fill_alpha=fill_alpha,
        line_alpha=1.0,
        merge_group_size=merge_group_size,
    )

    if method_name == main_method:
        ax.set_xlabel("Similarity bin", fontsize=20)
        ax.set_ylabel("Prediction mismatch", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
    else:
        ax.set_xlabel("Similarity bin", fontsize=15)
        ax.set_ylabel("Prediction mismatch", fontsize=15)
        ax.tick_params(axis="both", labelsize=15)

    ax.grid(True, alpha=0.3)
    if method_name == main_method:
        ax.legend(fontsize=20)

    global_x_min, global_x_max, global_y_min, global_y_max = global_limits

    if global_x_min is not None and global_x_max is not None:
        ax.set_xlim(global_x_min, global_x_max)
    if global_y_min is not None and global_y_max is not None:
        ax.set_ylim(global_y_min, global_y_max)


def _create_layout(
    method_names: list[str],
    main_method: str,
    reference_method: str,
    figsize_per_subplot: tuple[float, float],
):
    small_methods = [
        m for m in method_names if m != main_method and m != reference_method
    ]

    figsize = (
        figsize_per_subplot[0] * 5.0,
        figsize_per_subplot[1] * 2.2,
    )

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=2,
        ncols=4,
        figure=fig,
        width_ratios=[1, 1, 1, 2.3],
    )

    small_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    main_ax = fig.add_subplot(gs[:, 3])
    return fig, small_axes, main_ax, small_methods


def plot_binned_similarity_vs_accuracy_drop_across_runs(
    parent_dir: str | Path,
    method_names: list[str],
    class_to_forget: int,
    class_fraction_to_forget: float,
    official_method_names: dict[str, str],
    output_path: str | Path,
    stats_filename: str = "binned_similarity_vs_accuracy_drop_stats.json",
    config_filename: str = "config.json",
    ncols: int = 3,
    figsize_per_subplot: tuple[float, float] = (5.5, 4.0),
    fill_alpha: float = 0.12,
    main_method: str = "distill_labels",
    merge_group_size: int = 2,
    reference_method: str = "retrain",
) -> dict[str, Any]:
    """
    Aggregate binned similarity-vs-prediction-mismatch statistics across runs.

    Each subplot contains two curves:
    - test_on_train=False
    - test_on_train=True

    Expected directory structure:
        parent_dir/
            method_name/
                class_{class_to_forget}/
                    test_on_train=False/
                        <timestamp_dir>/
                            config.json
                            binned_similarity_vs_accuracy_drop_stats.json
                    test_on_train=True/
                        <timestamp_dir>/
                            config.json
                            binned_similarity_vs_accuracy_drop_stats.json
    """
    del ncols

    parent_dir = Path(parent_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_split_results = {
        "test": _collect_results_for_split(
            parent_dir=parent_dir,
            method_names=method_names,
            class_to_forget=class_to_forget,
            class_fraction_to_forget=class_fraction_to_forget,
            official_method_names=official_method_names,
            test_on_train=False,
            stats_filename=stats_filename,
            config_filename=config_filename,
        ),
        "train": _collect_results_for_split(
            parent_dir=parent_dir,
            method_names=method_names,
            class_to_forget=class_to_forget,
            class_fraction_to_forget=class_fraction_to_forget,
            official_method_names=official_method_names,
            test_on_train=True,
            stats_filename=stats_filename,
            config_filename=config_filename,
        ),
    }

    global_limits = _compute_global_limits(per_split_results)

    fig, small_axes, main_ax, small_methods = _create_layout(
        method_names=method_names,
        main_method=main_method,
        reference_method=reference_method,
        figsize_per_subplot=figsize_per_subplot,
    )

    for ax, method_name in zip(small_axes, small_methods):
        _plot_one_method(
            ax=ax,
            method_name=method_name,
            per_split_results=per_split_results,
            global_limits=global_limits,
            fill_alpha=fill_alpha,
            merge_group_size=merge_group_size,
            reference_method=reference_method,
            main_method=main_method,
        )

    for ax in small_axes[len(small_methods) :]:
        ax.axis("off")

    if main_method in method_names:
        _plot_one_method(
            ax=main_ax,
            method_name=main_method,
            per_split_results=per_split_results,
            global_limits=global_limits,
            fill_alpha=fill_alpha,
            merge_group_size=merge_group_size,
            reference_method=reference_method,
            main_method=main_method,
        )
    else:
        main_ax.axis("off")

    fig.suptitle(
        r"Prediction Mismatch with Retrain vs Similarity to $D_f$",
        fontsize=30,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=3.5)
    plt.show()
    # plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.close(fig)

    return {
        "plot_path": str(output_path),
        "parent_dir": str(parent_dir),
        "class_to_forget": class_to_forget,
        "class_fraction_to_forget": class_fraction_to_forget,
        "methods": per_split_results,
    }
