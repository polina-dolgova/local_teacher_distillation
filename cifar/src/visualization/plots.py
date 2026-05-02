from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import json


def save_per_class_accuracy_bar_chart(
    retain_per_class_acc,
    forget_per_class_acc,
    class_names: list[str],
    output_path: Path,
    title: str,
    selected_classes: list[int] | None = None,
) -> None:
    """
    Save a per-class accuracy bar chart.
    If forget_per_class_acc is provided, draw retain and forget bars side by side
    for classes present in forget_per_class_acc.
    """

    if selected_classes is None:
        selected_classes = list(range(len(class_names)))

    labels = [class_names[c] for c in selected_classes]
    retain_values = [retain_per_class_acc.get(c, None) for c in selected_classes]
    x = np.arange(len(selected_classes))

    if len(selected_classes) > 30:
        plt.figure(figsize=(30, 15))
    else:
        plt.figure(figsize=(10, 5))

    if len(forget_per_class_acc) == 0:
        retain_pairs = [
            (xi, value) for xi, value in zip(x, retain_values) if value is not None
        ]
        retain_x_single = [p[0] for p in retain_pairs]
        retain_values_single = [p[1] for p in retain_pairs]

        bars = plt.bar(retain_x_single, retain_values_single, label="Retain")
        for bar, value in zip(bars, retain_values_single):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        width = 0.4

        forget_values = [forget_per_class_acc.get(c, None) for c in selected_classes]
        has_forget = [v is not None for v in forget_values]

        retain_x = []
        forget_x = []
        retain_plot_values = []
        forget_plot_values = []

        for xi, retain_value, forget_value in zip(x, retain_values, forget_values):

            has_retain = retain_value is not None
            has_forget = forget_value is not None

            if has_retain and has_forget:
                retain_x.append(xi - width / 2)
                forget_x.append(xi + width / 2)
                retain_plot_values.append(retain_value)
                forget_plot_values.append(forget_value)
            elif has_retain:
                retain_x.append(xi)
                retain_plot_values.append(retain_value)
            elif has_forget:
                forget_x.append(xi)
                forget_plot_values.append(forget_value)

        retain_bars = plt.bar(retain_x, retain_plot_values, width=width, label="Retain")
        forget_bars = plt.bar(
            forget_x, forget_plot_values, width=width, color="red", label="Forget"
        )

        for bar, value in zip(retain_bars, retain_plot_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        for bar, value in zip(forget_bars, forget_plot_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.legend()

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_accuracy_drop_vs_similarity_plot(
    retain_acc,
    forget_acc,
    unlearn_retain_acc,
    unlearn_forget_acc,
    cosine_similarity_retain: torch.Tensor,
    cosine_similarity_forget: torch.Tensor,
    forget_class: int,
    class_names: list[str],
    output_path: Path,
    title: str,
    xlabel: str,
    selected_classes: list[int] | None = None,
) -> None:
    """
    Save scatter plot:
    x = cosine similarity to the forget class,
    y = accuracy drop after unlearning.

    Retain classes are plotted as the main scatter.
    The forget class is highlighted twice:
    1) as part of the retain dataset,
    2) as part of the forget dataset.
    """
    if selected_classes is None:
        selected_classes = list(range(len(class_names)))

    selected_classes = [i for i in selected_classes if i in retain_acc]

    print(f"SELECTED_CLASSES: {selected_classes}", "FULL ACC", retain_acc.keys())
    # Accuracy drops for retain dataset.
    drop_retain = np.array(
        [retain_acc[c] - unlearn_retain_acc[c] for c in selected_classes],
        dtype=float,
    )
    # Similarities for retain dataset.
    sim_retain_all = cosine_similarity_retain.detach().cpu().numpy()
    sim_retain = np.array(
        [sim_retain_all[c] for c in selected_classes],
        dtype=float,
    )

    # Accuracy drops for forget dataset.
    drop_forget = [forget_acc[forget_class] - unlearn_forget_acc[forget_class]]

    # Similarities for forget dataset.
    sim_forget_all = cosine_similarity_forget.detach().cpu().numpy()
    sim_forget = [sim_forget_all[forget_class]]

    if len(selected_classes) < 30:
        plt.figure(figsize=(7, 6))
    else:
        plt.figure(figsize=(20, 10))

    # Mask for all retain classes except the forget class.
    mask_non_forget = np.array([c != forget_class for c in selected_classes])

    # Main scatter: retain dataset except forget class.
    plt.scatter(
        sim_retain[mask_non_forget],
        drop_retain[mask_non_forget],
        s=70,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
        label="Retain classes",
    )

    if forget_class in selected_classes:
        # Index of the forget class inside selected_classes.
        forget_idx = selected_classes.index(forget_class)

        # Highlight forget class from retain dataset.
        plt.scatter(
            [sim_retain[forget_idx]],
            [drop_retain[forget_idx]],
            s=95,
            marker="s",
            facecolors="tab:red",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="Forget class in retain set",
        )

    # Highlight forget class from forget dataset.
    plt.scatter(
        sim_forget,
        drop_forget,
        s=110,
        marker="D",
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
        label="Forget class in forget set",
    )

    # Annotate retain points.
    for idx, c in enumerate(selected_classes):
        color = "red" if c == forget_class else "black"
        fontweight = "bold" if c == forget_class else "normal"

        plt.annotate(
            class_names[c],
            (sim_retain[idx], drop_retain[idx]),
            fontsize=9,
            color=color,
            fontweight=fontweight,
            xytext=(4, 4),
            textcoords="offset points",
        )

    # Annotate the forget-set point separately so it is clearly distinguishable.
    plt.annotate(
        f"{class_names[forget_class]} (forget set)",
        (sim_forget[0], drop_forget[0]),
        fontsize=9,
        color="tab:blue",
        fontweight="bold",
        xytext=(4, -12),
        textcoords="offset points",
    )

    # Correlation over all retain classes.
    if len(sim_retain) >= 2 and not np.allclose(sim_retain, sim_retain[0]):
        corr_all = np.corrcoef(sim_retain, drop_retain)[0, 1]
    else:
        corr_all = np.nan

    # Fit excluding forget class, using retain dataset only.
    if mask_non_forget.sum() >= 2:
        sim_fit = sim_retain[mask_non_forget]
        drop_fit = drop_retain[mask_non_forget]

        if not np.allclose(sim_fit, sim_fit[0]):
            coef = np.polyfit(sim_fit, drop_fit, 1)
            x_line = np.linspace(sim_retain.min(), sim_retain.max(), 100)
            y_line = coef[0] * x_line + coef[1]

            corr_other = np.corrcoef(sim_fit, drop_fit)[0, 1]
            if np.isnan(corr_other):
                fit_label = "Fit excl. forget"
            else:
                fit_label = f"Fit excl. forget (r={corr_other:.2f})"

            plt.plot(
                x_line,
                y_line,
                linestyle="--",
                linewidth=1.8,
                color="black",
                alpha=0.85,
                zorder=2,
                label=fit_label,
            )

    plt.axhline(0.0, linestyle=":", linewidth=1.2, color="gray", alpha=0.8)
    plt.grid(alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel("Accuracy drop")
    plt.title(title)
    plt.legend(frameon=False)

    if not np.isnan(corr_all):
        plt.text(
            0.02,
            0.98,
            f"Pearson r = {corr_all:.2f}",
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_argmax_softlabel_bar_chart(
    soft_targets: torch.Tensor,
    class_names: list[str],
    output_path: str,
    title: str,
    class_to_forget: int = 25,
) -> None:
    argmax_labels = soft_targets.argmax(dim=1).cpu().numpy()
    counts = np.bincount(argmax_labels, minlength=len(class_names))
    ratios = counts / max(len(argmax_labels), 1)

    print(
        f"Class 25 ({class_names[class_to_forget]}): count={counts[class_to_forget]}, ratio={ratios[class_to_forget]:.4f}"
    )

    x = np.arange(len(class_names))

    if len(class_names) > 30:
        plt.figure(figsize=(30, 15))
    else:
        plt.figure(figsize=(10, 5))

    bars = plt.bar(x, ratios)

    for bar, count, ratio in zip(bars, counts, ratios):
        if count == 0:
            continue
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + 0.005,
            f"{count}\n({ratio:.2f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pointwise_topk_scatter_plot(
    selected_similarities,
    selected_drop,
    selected_targets,
    class_to_forget: int,
    output_path: str | Path,
    point_topk_fraction: float,
    title: str | None = None,
) -> None:
    """
    Save scatter plot for the union of:
    - top-k most similar retain points to forget set
    - top-k least similar retain points to forget set

    Points with true label == class_to_forget are colored separately.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    forget_mask = selected_targets == class_to_forget
    other_mask = ~forget_mask

    plt.figure(figsize=(7, 6))

    plt.scatter(
        selected_similarities[other_mask],
        selected_drop[other_mask],
        label="other classes",
        alpha=0.8,
    )
    plt.scatter(
        selected_similarities[forget_mask],
        selected_drop[forget_mask],
        label=f"class {class_to_forget}",
        alpha=0.8,
    )

    plt.xlabel("Similarity to forget set")
    plt.ylabel("Accuracy drop (retrain - unlearn)")
    plt.title(
        title
        if title is not None
        else f"Top-{point_topk_fraction:.3f} closest + farthest points"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_similarity_survival_plots(
    similarities: torch.Tensor,
    output_dir: str | Path,
    prefix: str = "similarity_survival",
    title_suffix: str | None = None,
    num_grid_points: int | None = None,
) -> dict[str, str]:
    """
    Save two plots based on point-wise similarities:
    1) x = similarity threshold, y = fraction of dataset with similarity > x
    2) x = similarity threshold, y = number of points with similarity > x
    """
    if len(similarities) == 0:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = similarities.detach().cpu().numpy().astype(float)
    n = len(sim)

    if num_grid_points is None:
        x_values = np.sort(sim)
    else:
        x_values = np.linspace(sim.min(), sim.max(), num_grid_points)

    counts_greater = np.array([(sim > x).sum() for x in x_values], dtype=float)
    fractions_greater = counts_greater / n

    if title_suffix is None:
        title_suffix = ""

    frac_path = output_dir / f"{prefix}_fraction.png"
    abs_path = output_dir / f"{prefix}_absolute.png"

    plt.figure(figsize=(7, 6))
    plt.plot(x_values, fractions_greater)
    plt.xlabel("Similarity")
    plt.ylabel("Fraction with similarity > x")
    plt.title(f"Similarity survival curve{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(frac_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(x_values, counts_greater)
    plt.xlabel("Similarity")
    plt.ylabel("Number of points with similarity > x")
    plt.title(f"Similarity survival curve (absolute){title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(abs_path, dpi=200)
    plt.close()

    return {
        "fraction_plot_path": str(frac_path),
        "absolute_plot_path": str(abs_path),
    }


def save_flip_similarity_survival_plot(
    similarities: torch.Tensor,
    pointwise_drop: torch.Tensor,
    output_path: str | Path,
    title: str | None = None,
) -> str | None:
    """
    Save survival curve over similarities for points with accuracy drop == 1.

    x-axis: similarity threshold x
    y-axis: fraction of flipped points with similarity > x

    Here "flipped points" means points with pointwise_drop == 1.
    """
    flip_mask = pointwise_drop == 1
    flip_similarities = similarities[flip_mask]

    if len(flip_similarities) == 0:
        return None

    sim = flip_similarities.detach().cpu().numpy().astype(float)
    sim_sorted = np.sort(sim)
    n = len(sim_sorted)

    counts_greater = n - np.arange(1, n + 1)
    fractions_greater = counts_greater / n

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(sim_sorted, fractions_greater)
    plt.xlabel("Similarity")
    plt.ylabel("Fraction of flipped points with similarity > x")
    plt.title(
        title if title is not None else "Similarity survival curve for flipped points"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return str(output_path)


def save_binned_similarity_vs_accuracy_drop_plot(
    similarities: torch.Tensor,
    pointwise_drop: torch.Tensor,
    output_path: str | Path,
    num_bins: int = 20,
    title: str | None = None,
    output_path_stats: str | Path = "",
) -> dict:
    """
    Bin point-wise similarities and plot average accuracy drop per bin.

    x-axis: similarity bin
    y-axis: mean pointwise accuracy drop in the bin

    pointwise_drop is expected to be in {0, 1, NaN}.
    NaN means retrain models disagree and the point is ignored.
    """
    if len(similarities) == 0:
        return {
            "plot_path": None,
            "num_bins": num_bins,
            "bin_edges": [],
            "bin_centers": [],
            "bin_counts": [],
            "mean_accuracy_drop_per_bin": [],
        }

    sim = similarities.detach().cpu().numpy().astype(float)
    drop = pointwise_drop.detach().cpu().numpy().astype(float)

    sim_min = sim.min()
    sim_max = sim.max()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(sim_min, sim_max, num_bins + 1)
    bin_ids = np.digitize(sim, bin_edges[1:-1], right=False)

    bin_centers = []
    bin_counts = []
    mean_drop = []
    widths = []
    bin_total_counts = []

    for bin_idx in range(num_bins):
        left = bin_edges[bin_idx]
        right = bin_edges[bin_idx + 1]
        mask = bin_ids == bin_idx

        bin_centers.append((left + right) / 2.0)
        widths.append(right - left)

        valid_drop_in_bin = drop[mask]
        valid_drop_in_bin = valid_drop_in_bin[~np.isnan(valid_drop_in_bin)]

        bin_counts.append(int(len(valid_drop_in_bin)))
        bin_total_counts.append(int(mask.sum()))

        if len(valid_drop_in_bin) > 0:
            mean_drop.append(float(valid_drop_in_bin.mean()))
        else:
            mean_drop.append(np.nan)

    bin_centers = np.array(bin_centers, dtype=float)
    bin_counts = np.array(bin_counts, dtype=int)
    mean_drop = np.array(mean_drop, dtype=float)
    widths = np.array(widths, dtype=float)
    bin_total_counts = np.array(bin_total_counts, dtype=int)

    plt.figure(figsize=(8, 6))
    valid_mask = ~np.isnan(mean_drop)

    plt.bar(
        bin_centers[valid_mask],
        mean_drop[valid_mask],
        width=widths[valid_mask],
        align="center",
        alpha=0.8,
        edgecolor="black",
    )
    plt.xlabel("Similarity bin")
    plt.ylabel("Disagreement rate among retrain-consensus points")
    plt.title(
        title if title is not None else "Binned similarity vs average accuracy drop"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    result = {
        "plot_path": str(output_path),
        "num_bins": int(num_bins),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "bin_counts": bin_counts.tolist(),
        "mean_accuracy_drop_per_bin": mean_drop.tolist(),
        "bin_total_counts": bin_total_counts.tolist(),
        "bin_consensus_counts": bin_counts.tolist(),
    }

    with open(output_path_stats, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
