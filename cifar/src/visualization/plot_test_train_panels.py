from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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


def _compute_x_limits(
    per_split_results: dict[SplitName, dict[str, dict[str, Any]]],
) -> tuple[float | None, float | None]:
    global_x_min = None
    global_x_max = None

    for split_results in per_split_results.values():
        for result in split_results.values():
            if result["num_runs"] == 0:
                continue

            x = np.asarray(result["bin_centers"], dtype=float)
            y = np.asarray(result["mean"], dtype=float)

            valid_mask = ~np.isnan(y)
            if not np.any(valid_mask):
                continue

            x_valid = x[valid_mask]
            x_min = float(np.min(x_valid))
            x_max = float(np.max(x_valid))

            global_x_min = x_min if global_x_min is None else min(global_x_min, x_min)
            global_x_max = x_max if global_x_max is None else max(global_x_max, x_max)

    return global_x_min, global_x_max


def _compute_y_limits_for_split(
    split_results: dict[str, dict[str, Any]],
) -> tuple[float | None, float | None]:
    global_y_min = None
    global_y_max = None

    for result in split_results.values():
        if result["num_runs"] == 0:
            continue

        y = np.asarray(result["mean"], dtype=float)
        y_std = np.asarray(result["std"], dtype=float)

        valid_mask = ~np.isnan(y)
        if not np.any(valid_mask):
            continue

        y_low_valid = (y - y_std)[valid_mask]
        y_high_valid = (y + y_std)[valid_mask]

        y_min = float(np.min(y_low_valid))
        y_max = float(np.max(y_high_valid))

        global_y_min = y_min if global_y_min is None else min(global_y_min, y_min)
        global_y_max = y_max if global_y_max is None else max(global_y_max, y_max)

    return global_y_min, global_y_max


# def _make_method_colors(
#     method_names: list[str],
#     reference_method: str,
#     method_colors: dict[str, str] | None = None,
# ) -> dict[str, str]:
#     if method_colors is None:
#         method_colors = {}

#     non_reference_methods = [m for m in method_names if m != reference_method]
#     cmap = plt.get_cmap("tab10")

#     output = dict(method_colors)
#     color_idx = 0
#     for method_name in non_reference_methods:
#         if method_name not in output:
#             output[method_name] = cmap(color_idx % 10)
#             color_idx += 1

#     return output


def _make_method_colors(
    method_names: list[str],
    reference_method: str,
    main_method: str,
    method_colors: dict[str, str] | None = None,
) -> dict[str, str]:
    if method_colors is None:
        method_colors = {}

    pastel_palette = [
        "#A1C9F4",
        "#FFB482",
        "#8DE5A1",
        "#FF9F9B",
        "#D0BBFF",
        "#DEBB9B",
        "#FAB0E4",
        "#CFCFCF",
    ]

    output = dict(method_colors)

    pastel_idx = 0
    for method_name in method_names:
        if method_name == reference_method:
            continue
        if method_name == main_method:
            continue
        if method_name not in output:
            output[method_name] = pastel_palette[pastel_idx % len(pastel_palette)]
            pastel_idx += 1

    if main_method not in output and main_method in method_names:
        output[main_method] = "#D62728"

    return output


def _plot_method_series(
    ax,
    result: dict[str, Any],
    reference_result,
    label: str,
    color,
    fill_alpha: float,
    line_alpha: float,
    linewidth: float,
    linestyle: str = "-",
    zorder: int = 2,
    show_std: bool = True,
) -> None:
    if result["num_runs"] == 0:
        return

    x = np.asarray(result["bin_centers"], dtype=float)
    y = 100 * np.asarray(result["mean"], dtype=float)
    y_std = 100 * np.asarray(result["std"], dtype=float)

    y_ref = 100 * np.asarray(reference_result["mean"], dtype=float)
    y = y - y_ref

    valid_mask = ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    y_std = y_std[valid_mask]

    if len(x) == 0:
        return

    ax.plot(
        x,
        y,
        marker="o",
        markersize=2,
        label=label,
        color=color,
        alpha=line_alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )

    if show_std:
        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=color,
            alpha=fill_alpha,
            zorder=zorder - 1,
        )


def plot_binned_similarity_train_test_panels(
    parent_dir: str | Path,
    method_names: list[str],
    class_to_forget: int,
    class_fraction_to_forget: float,
    official_method_names: dict[str, str],
    output_path: str | Path,
    accuracy_stats_filename: str = "binned_similarity_vs_accuracy_drop_stats.json",
    confidence_stats_filename: str = "binned_similarity_vs_confidence_drop_stats.json",
    config_filename: str = "config.json",
    figsize: tuple[float, float] = (16.0, 10.0),
    reference_method: str = "retrain",
    fill_alpha: float = 0.08,
    reference_fill_alpha: float = 0.14,
    show_std: bool = True,
    share_y: bool = False,
    method_colors: dict[str, str] | None = None,
    main_method: str = "distill_labels",
) -> dict[str, Any]:
    """
    Draw four panels:
    - top-left: accuracy difference on retain/train set
    - top-right: accuracy difference on test set
    - bottom-left: confidence difference on retain/train set
    - bottom-right: confidence difference on test set
    """
    parent_dir = Path(parent_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_specs = {
        "accuracy": {
            "stats_filename": accuracy_stats_filename,
            "ylabel": "Accuracy Difference",
            "row_title": "Accuracy Difference",
        },
        "confidence": {
            "stats_filename": confidence_stats_filename,
            "ylabel": "Confidence Difference",
            "row_title": "Confidence Difference",
        },
    }

    per_metric_results = {}

    for metric_name, metric_spec in metric_specs.items():
        per_metric_results[metric_name] = {
            "test": _collect_results_for_split(
                parent_dir=parent_dir,
                method_names=method_names,
                class_to_forget=class_to_forget,
                class_fraction_to_forget=class_fraction_to_forget,
                official_method_names=official_method_names,
                test_on_train=False,
                stats_filename=metric_spec["stats_filename"],
                config_filename=config_filename,
            ),
            "train": _collect_results_for_split(
                parent_dir=parent_dir,
                method_names=method_names,
                class_to_forget=class_to_forget,
                class_fraction_to_forget=class_fraction_to_forget,
                official_method_names=official_method_names,
                test_on_train=True,
                stats_filename=metric_spec["stats_filename"],
                config_filename=config_filename,
            ),
        }

    method_colors = _make_method_colors(
        method_names=method_names,
        reference_method=reference_method,
        method_colors=method_colors,
        main_method=main_method,
    )

    xmin = None
    xmax = None
    for metric_name in metric_specs:
        metric_xmin, metric_xmax = _compute_x_limits(per_metric_results[metric_name])
        if metric_xmin is None or metric_xmax is None:
            continue
        xmin = metric_xmin if xmin is None else min(xmin, metric_xmin)
        xmax = metric_xmax if xmax is None else max(xmax, metric_xmax)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    split_names = ["train", "test"]
    split_to_col = {
        "train": 0,
        "test": 1,
    }
    split_to_title = {
        "train": "Retain Set",
        "test": "Test Set",
    }

    metric_names = ["accuracy", "confidence"]

    for row_idx, metric_name in enumerate(metric_names):
        metric_results = per_metric_results[metric_name]
        metric_spec = metric_specs[metric_name]

        if share_y:
            all_y_min = None
            all_y_max = None

            for split_name in split_names:
                y_min, y_max = _compute_y_limits_for_split(metric_results[split_name])
                if y_min is None or y_max is None:
                    continue

                all_y_min = y_min if all_y_min is None else min(all_y_min, y_min)
                all_y_max = y_max if all_y_max is None else max(all_y_max, y_max)

            y_limits = {
                "train": (all_y_min, all_y_max),
                "test": (all_y_min, all_y_max),
            }
        else:
            y_limits = {
                "train": _compute_y_limits_for_split(metric_results["train"]),
                "test": _compute_y_limits_for_split(metric_results["test"]),
            }

        for split_name in split_names:
            col_idx = split_to_col[split_name]
            ax = axes[row_idx, col_idx]
            split_results = metric_results[split_name]

            for method_name in method_names:
                if method_name == reference_method:
                    continue

                result = split_results[method_name]
                reference_result = split_results[reference_method]
                is_main = method_name == main_method

                _plot_method_series(
                    ax=ax,
                    result=result,
                    reference_result=reference_result,
                    label=official_method_names.get(method_name, method_name),
                    color=method_colors[method_name],
                    fill_alpha=fill_alpha,
                    line_alpha=1.0,
                    linewidth=2.8 if is_main else 1.5,
                    linestyle="-",
                    zorder=5 if is_main else 3,
                    show_std=show_std,
                )

            ax.hlines(
                y=0,
                xmin=xmin,
                xmax=xmax,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=1.0,
                zorder=1,
            )

            if row_idx == 0:
                ax.set_title(
                    split_to_title[split_name], fontsize=20, fontweight="semibold"
                )

            ax.set_xlabel("Similarity bin", fontsize=18)

            if split_name == "train":
                ax.set_ylabel(metric_spec["ylabel"], fontsize=18)
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.3)

            # if xmin is not None and xmax is not None:
            #     ax.set_xlim(xmin, xmax)

            # y_min, y_max = y_limits[split_name]
            # if y_min is not None and y_max is not None:
            #     margin = 0.05 * max(y_max - y_min, 1e-8)
            #     ax.set_ylim(y_min - margin, y_max + margin)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles,
        labels,
        loc="upper left",
        fontsize=14,
        frameon=False,
    )

    # fig.suptitle(
    #     r"Difference with Retrain vs Similarity to $D_f$",
    #     fontsize=28,
    #     fontweight="bold",
    #     y=1.02,
    # )

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.32, wspace=0.10)

    plt.show()
    # plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot_path": str(output_path),
        "parent_dir": str(parent_dir),
        "class_to_forget": class_to_forget,
        "class_fraction_to_forget": class_fraction_to_forget,
        "methods": per_metric_results,
        "method_colors": {k: str(v) for k, v in method_colors.items()},
    }


# def plot_binned_similarity_train_test_panels(
#     parent_dir: str | Path,
#     method_names: list[str],
#     class_to_forget: int,
#     class_fraction_to_forget: float,
#     official_method_names: dict[str, str],
#     output_path: str | Path,
#     stats_filename: str = "binned_similarity_vs_accuracy_drop_stats.json",
#     config_filename: str = "config.json",
#     figsize: tuple[float, float] = (16.0, 6.2),
#     reference_method: str = "retrain",
#     fill_alpha: float = 0.08,
#     reference_fill_alpha: float = 0.14,
#     show_std: bool = True,
#     share_y: bool = False,
#     method_colors: dict[str, str] | None = None,
#     main_method: str = "distill_labels",
# ) -> dict[str, Any]:
#     """
#     Draw two panels:
#     - left: train / retain-side curve (test_on_train=True)
#     - right: test-side curve (test_on_train=False)

#     Each non-reference method keeps the same color in both panels.
#     The reference method is drawn with a distinct dashed black style.
#     """
#     parent_dir = Path(parent_dir)
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     per_split_results = {
#         "test": _collect_results_for_split(
#             parent_dir=parent_dir,
#             method_names=method_names,
#             class_to_forget=class_to_forget,
#             class_fraction_to_forget=class_fraction_to_forget,
#             official_method_names=official_method_names,
#             test_on_train=False,
#             stats_filename=stats_filename,
#             config_filename=config_filename,
#         ),
#         "train": _collect_results_for_split(
#             parent_dir=parent_dir,
#             method_names=method_names,
#             class_to_forget=class_to_forget,
#             class_fraction_to_forget=class_fraction_to_forget,
#             official_method_names=official_method_names,
#             test_on_train=True,
#             stats_filename=stats_filename,
#             config_filename=config_filename,
#         ),
#     }

#     method_colors = _make_method_colors(
#         method_names=method_names,
#         reference_method=reference_method,
#         method_colors=method_colors,
#         main_method=main_method,
#     )

#     xmin, xmax = _compute_x_limits(per_split_results)

#     if share_y:
#         all_y_min = None
#         all_y_max = None
#         for split_name in ["train", "test"]:
#             y_min, y_max = _compute_y_limits_for_split(per_split_results[split_name])
#             if y_min is None or y_max is None:
#                 continue
#             all_y_min = y_min if all_y_min is None else min(all_y_min, y_min)
#             all_y_max = y_max if all_y_max is None else max(all_y_max, y_max)
#         y_limits = {
#             "train": (all_y_min, all_y_max),
#             "test": (all_y_min, all_y_max),
#         }
#     else:
#         y_limits = {
#             "train": _compute_y_limits_for_split(per_split_results["train"]),
#             "test": _compute_y_limits_for_split(per_split_results["test"]),
#         }

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
#     split_to_ax = {
#         "train": axes[0],
#         "test": axes[1],
#     }
#     split_to_title = {
#         "train": r"Retain Set $D_r$",
#         "test": "Test Set",
#     }

#     for split_name in ["train", "test"]:
#         ax = split_to_ax[split_name]
#         split_results = per_split_results[split_name]

#         # if reference_method in split_results:
#         #     reference_result = split_results[reference_method]
#         #     _plot_method_series(
#         #         ax=ax,
#         #         result=reference_result,
#         #         label=official_method_names.get(reference_method, reference_method),
#         #         color="black",
#         #         fill_alpha=reference_fill_alpha,
#         #         line_alpha=0.95,
#         #         linewidth=2.8,
#         #         linestyle="--",
#         #         zorder=5,
#         #         show_std=show_std,
#         #     )

#         for method_name in method_names:
#             if method_name == reference_method:
#                 continue

#             result = split_results[method_name]
#             _plot_method_series(
#                 ax=ax,
#                 result=result,
#                 reference_result=split_results[reference_method],
#                 label=official_method_names.get(method_name, method_name),
#                 color=method_colors[method_name],
#                 fill_alpha=fill_alpha,
#                 line_alpha=0.95,
#                 linewidth=1.5,
#                 linestyle="-",
#                 zorder=3,
#                 show_std=show_std,
#                 is_main_method=(method_name == main_method),
#             )

#         ax.hlines(
#             y=0,
#             xmin=xmin,
#             xmax=xmax,
#             color="gray",  # "#C9A227",
#             linestyle="--",
#             linewidth=1.0,
#             alpha=1.0,
#             zorder=1,
#         )

#         ax.set_title(split_to_title[split_name], fontsize=20, fontweight="semibold")
#         ax.set_xlabel("Similarity bin", fontsize=20)
#         if split_name == "train":
#             ax.set_ylabel("Confidence Diff. with Retrain", fontsize=15)
#         ax.tick_params(axis="both", labelsize=15)
#         ax.grid(True, alpha=0.3)

#         # if x_min is not None and x_max is not None:
#         # ax.set_xlim(x_min, x_max)

#         # y_min, y_max = y_limits[split_name]
#         # if y_min is not None and y_max is not None:
#         #     ax.set_ylim(y_min, y_max)

#     # handles, labels = axes[1].get_legend_handles_labels()
#     # fig.legend(
#     #     handles,
#     #     labels,
#     #     loc="upper center",
#     #     ncol=min(len(labels), 4),
#     #     fontsize=20,
#     #     frameon=False,
#     #     bbox_to_anchor=(0.5, 1.02),
#     # )
#     handles, labels = axes[0].get_legend_handles_labels()
#     axes[0].legend(
#         handles,
#         labels,
#         loc="upper left",
#         fontsize=15,
#         frameon=False,
#     )

#     # fig.suptitle(
#     #     r"Correct Class Confidence Difference with Retrain vs Similarity to $D_f$",
#     #     fontsize=30,
#     #     fontweight="bold",
#     #     y=1.08,
#     # )
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.78, wspace=0.20)

#     plt.show()
#     # plt.savefig(output_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     return {
#         "plot_path": str(output_path),
#         "parent_dir": str(parent_dir),
#         "class_to_forget": class_to_forget,
#         "class_fraction_to_forget": class_fraction_to_forget,
#         "methods": per_split_results,
#         "method_colors": {k: str(v) for k, v in method_colors.items()},
#     }
