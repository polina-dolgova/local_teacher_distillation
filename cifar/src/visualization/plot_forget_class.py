from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


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


def plot_forget_class_metrics_barplot(
    summary_json_path: str | Path,
    output_path: str | Path,
    method_order: list[str] | None = None,
    method_display_names: dict[str, str] | None = None,
) -> None:
    """
    Plot grouped bar chart for forget-class metrics:
    - RA_forget_class
    - FA_forget_class as UA_forget_class
    - TA_forget_class

    Values are shown in percent.
    """
    summary_json_path = Path(summary_json_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if method_order is None:
        method_order = [
            "rl",
            "ft",
            "ga",
            "iu",
            "salun",
            "amun",
            "advonly",
            "distill_labels2",
        ]

    if method_display_names is None:
        method_display_names = {
            "rl": "RL",
            "ft": "FT",
            "ga": "GA",
            "iu": "IU",
            "salun": "SalUn",
            "amun": "AMUN",
            "advonly": "AdvOnly",
            "distill_labels2": "Distill Labels (Ours)",
        }

    metric_keys = [
        "RA_forget_class",
        "FA_forget_class",
        "TA_forget_class",
    ]
    metric_labels = [
        r"RA$_{\mathrm{forget-cls}}$",
        r"UA$_{\mathrm{forget-cls}}$",
        r"TA$_{\mathrm{forget-cls}}$",
    ]

    metric_colors = [
        "#5AA469",  # retain: soft green
        "#B00101",  # forget: soft red
        "#024583",  # test: muted blue
    ]

    means = []
    stds = []
    labels = []

    for method in method_order:
        if method not in summary:
            continue

        metrics = summary[method]["metrics"]

        means.append([100.0 * metrics[key]["mean"] for key in metric_keys])
        stds.append([100.0 * metrics[key]["std"] for key in metric_keys])
        labels.append(method_display_names.get(method, method))

    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    num_methods = len(labels)
    num_metrics = len(metric_keys)

    x = np.arange(num_methods)
    bar_width = 0.22

    plt.figure(figsize=(1.5 * num_methods + 2, 5.5))

    for metric_idx in range(num_metrics):
        offset = (metric_idx - 1) * bar_width

        plt.bar(
            x + offset,
            means[:, metric_idx],
            width=bar_width,
            yerr=stds[:, metric_idx],
            capsize=3,
            label=metric_labels[metric_idx],
            color=metric_colors[metric_idx],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
        )

    ax = plt.gca()
    retrain_idx = labels.index("Retrain")

    for method_idx in range(num_methods):
        for metric_idx in range(num_metrics):
            offset = (metric_idx - 1) * bar_width
            x_center = x[method_idx] + offset

            ref_mean = means[retrain_idx, metric_idx]
            ref_std = stds[retrain_idx, metric_idx]
            ref_low = ref_mean - ref_std
            ref_high = ref_mean + ref_std

            band_width = bar_width * 0.9

            # semi-transparent band for retrain mean ± std
            rect = plt.Rectangle(
                (x_center - band_width / 2, ref_low),
                band_width,
                ref_high - ref_low,
                facecolor="#D8C27A",
                edgecolor="none",
                alpha=0.3,
                zorder=0,
            )
            ax.add_patch(rect)

            # central retrain mean line
            ax.hlines(
                y=ref_mean,
                xmin=x_center - band_width / 2,
                xmax=x_center + band_width / 2,
                color="#C9A227",
                linestyle="--",
                linewidth=1.2,
                alpha=1.0,
                zorder=1,
            )

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Forget-Class Metrics Across Methods")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_forget_class_metrics_barplot_differences(
    summary_json_path: str | Path,
    output_path: str | Path,
    method_order: list[str] | None = None,
    method_display_names: dict[str, str] | None = None,
    retrain_key: str = "retrain",
    main_method_key: str = "distill_labels",
) -> None:
    """
    Plot grouped bar chart for forget-class metrics as differences from retrain.

    Bars:
        method_mean - retrain_mean

    Error bars:
        method std only

    Shaded bands:
        retrain mean ± retrain std, centered around zero after subtraction
    """
    summary_json_path = Path(summary_json_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metric_keys = [
        "RA_forget_class",
        "FA_forget_class",
        "TA_forget_class",
    ]
    metric_labels = [
        r"RA$_{\mathrm{affected-class}}$",
        r"UA$_{\mathrm{affected-class}}$",
        r"TA$_{\mathrm{affected-class}}$",
    ]

    metric_colors = [
        "#5AA469",  # retain: soft green
        "#B00101",  # forget/unlearning: red
        "#024583",  # test: muted blue
    ]

    retrain_metrics = summary[retrain_key]["metrics"]
    retrain_means = np.asarray(
        [100.0 * retrain_metrics[key]["mean"] for key in metric_keys],
        dtype=float,
    )
    retrain_stds = np.asarray(
        [100.0 * retrain_metrics[key]["std"] for key in metric_keys],
        dtype=float,
    )

    diff_means = []
    method_stds = []
    labels = []

    for method in method_order:
        # if method == retrain_key:
        #     continue
        if method not in summary:
            continue

        metrics = summary[method]["metrics"]

        method_means = np.asarray(
            [100.0 * metrics[key]["mean"] for key in metric_keys],
            dtype=float,
        )
        method_std = np.asarray(
            [100.0 * metrics[key]["std"] for key in metric_keys],
            dtype=float,
        )

        diff_means.append(method_means - retrain_means)
        method_stds.append(method_std)
        labels.append(method_display_names.get(method, method))

    diff_means = np.asarray(diff_means, dtype=float)
    method_stds = np.asarray(method_stds, dtype=float)

    num_methods = len(labels)
    num_metrics = len(metric_keys)

    x = np.arange(num_methods)
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(1.5 * num_methods + 2, 4))

    for metric_idx in range(num_metrics):
        offset = (metric_idx - 1) * bar_width

        # Shaded retrain variability band for this metric.
        # ax.axhspan(
        #     -retrain_stds[metric_idx],
        #     retrain_stds[metric_idx],
        #     xmin=0.0,
        #     xmax=1.0,
        #     color=metric_colors[metric_idx],
        #     alpha=0.08,
        #     zorder=0,
        # )

        ax.bar(
            x + offset,
            diff_means[:, metric_idx],
            width=bar_width,
            yerr=method_stds[:, metric_idx],
            capsize=3,
            label=metric_labels[metric_idx],
            color=metric_colors[metric_idx],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        zorder=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    main_display = method_display_names.get(main_method_key, main_method_key)
    retrain_display = method_display_names.get(retrain_key, retrain_key)
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == main_display:
            tick_label.set_color("#9548B2")
            tick_label.set_fontweight("bold")
        elif tick_label.get_text() == retrain_display:
            tick_label.set_color("#C9A227")
    ax.set_ylabel("Accuracy difference from retrain (pp)")
    ax.set_title("Affected-Class Metrics Relative to Retrain")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    y_min = np.nanmin(diff_means - method_stds)
    y_max = np.nanmax(diff_means + method_stds)
    band_max = np.nanmax(retrain_stds)

    margin = 0.1 * max(abs(y_min), abs(y_max), band_max, 1.0)
    ax.set_ylim(
        min(y_min, -band_max) - margin,
        max(y_max, band_max) + margin,
    )

    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)