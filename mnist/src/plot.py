import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import pandas as pd
from torchvision import datasets, transforms
from scipy.stats import t
import math

from mnist.src.train import load_mnist_class_mean_embeddings
from mnist.src.dataset import load_mnist_class_means


def compute_similarity_matrix(
    class_means: np.ndarray, metric: str = "pearson"
) -> np.ndarray:
    """
    Compute class-to-class similarity matrix from mean images.

    Args:
        class_means: numpy array of shape (10, 28, 28)
        metric: "pearson" or "cosine"

    Returns:
        sim_matrix: numpy array of shape (10, 10)
    """
    X = class_means.reshape(10, -1)

    if metric == "pearson":
        sim_matrix = np.corrcoef(X)

    elif metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / np.clip(norms, 1e-12, None)
        sim_matrix = X_norm @ X_norm.T

    else:
        raise ValueError("metric must be either 'pearson' or 'cosine'")

    return sim_matrix


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    title: str = "MNIST class similarity",
    figsize: tuple = (8, 7),
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    output_path=None,
) -> None:
    """
    Plot a heatmap with numbers inside each cell.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_title(title)

    # Write numbers inside cells
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            value = sim_matrix[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Similarity")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_class_mean_images(class_means: np.ndarray) -> None:
    """
    Plot the mean image for each class.
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for cls in range(10):
        axes[cls].imshow(class_means[cls], cmap="gray")
        axes[cls].set_title(f"Class {cls}")
        axes[cls].axis("off")

    plt.tight_layout()
    plt.show()


def plot_heatmaps(embeddings, emb_type, output_dir):
    # pearson_matrix = compute_similarity_matrix(embeddings, metric="pearson")
    # plot_similarity_heatmap(
    #     pearson_matrix,
    #     title="MNIST class similarity (Pearson correlation)",
    #     output_path=output_dir / f"pearson_{emb_type}_embeggings.pdf",
    # )

    # Cosine similarity between images
    cosine_matrix = compute_similarity_matrix(embeddings, metric="cosine")
    plot_similarity_heatmap(
        cosine_matrix,
        title="MNIST class similarity (Cosine similarity)",
        vmin=0.0,
        vmax=1.0,
        output_path=output_dir / f"cosine_{emb_type}_embeggings.pdf",
    )

    similarity_matrices = {
        #f"{emb_type}-pearson": pearson_matrix,
        f"{emb_type}-cosine": cosine_matrix,
    }

    return similarity_matrices


def plot_all_similarity_heatmaps(output_dir, model, unlearned_model=None, device="cpu"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    similarity_matrices = {}

    # ### RAW DATA ###
    # class_means = load_mnist_class_means(root="./data", train=True)
    # sim_m = plot_heatmaps(embeddings=class_means, emb_type="raw", output_dir=output_dir)
    # similarity_matrices.update(sim_m)

    ### CNN DATA ###
    class_mean_embeddings = load_mnist_class_mean_embeddings(
        model=model,
        root="./data",
        train=True,
        batch_size=256,
        device=device,
    )
    sim_m = plot_heatmaps(
        embeddings=class_mean_embeddings, emb_type="cnn", output_dir=output_dir
    )
    similarity_matrices.update(sim_m)

    # ### UNLEARNED RL DATA ###
    # if unlearned_model is not None:
    #     class_mean_embeddings = load_mnist_class_mean_embeddings(
    #         model=unlearned_model,
    #         root="./data",
    #         train=True,
    #         batch_size=256,
    #         device=device,
    #     )
    #     sim_m = plot_heatmaps(
    #         embeddings=class_mean_embeddings, emb_type="unlearn", output_dir=output_dir
    #     )
    #     similarity_matrices.update(sim_m)

    return similarity_matrices


def plot_accuracy_per_class(
    df_metrics,
    out_path=None,
):
    x = df_metrics["class"].values

    plt.figure(figsize=(8, 5))
    plt.plot(x, df_metrics["acc_model"].values, marker="o", label="model")
    plt.plot(x, df_metrics["acc_unlearned"].values, marker="o", label="unlearned")
    plt.xticks(x)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-class accuracy")
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_accuracy_drop_per_class(
    df_metrics,
    out_path=None,
    forgotten_class=None,
):
    x = df_metrics["class"].values

    plt.figure(figsize=(8, 5))
    plt.bar(x, df_metrics["acc_drop"].values)

    if forgotten_class is not None:
        plt.axvspan(forgotten_class - 0.4, forgotten_class + 0.4, alpha=0.15)

    plt.xticks(x)
    plt.xlabel("Class")
    plt.ylabel("Accuracy drop")
    plt.title("Accuracy drop after unlearning")
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sq_logit_gap_per_class(
    df_metrics,
    out_path=None,
):
    x = df_metrics["class"].values

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        df_metrics["mean_sq_logit_gap_all"].values,
        marker="o",
        label="all logits",
    )
    plt.plot(
        x,
        df_metrics["mean_sq_logit_gap_true"].values,
        marker="o",
        label="true logit",
    )

    plt.xticks(x)
    plt.xlabel("Class")
    plt.ylabel("Squared logit gap")
    plt.title("Per-class logit shift")
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_similarity_vs_accuracy_drop(
    similarity_matrix,
    df_metrics,
    deleted_class: int,
    out_path=None,
):
    """
    Scatter: similarity(deleted_class, j) vs accuracy_drop(j)
    """
    sim_row = similarity_matrix[deleted_class]  # shape: (10,)
    acc_drop = df_metrics["acc_drop"].values

    classes = df_metrics["class"].values

    plt.figure(figsize=(6, 5))

    plt.scatter(sim_row, acc_drop)

    # подписать точки классами
    for i, cls in enumerate(classes):
        plt.text(sim_row[i], acc_drop[i], str(cls), fontsize=9)

    coef = np.polyfit(sim_row, acc_drop, 1)
    x_line = np.linspace(sim_row.min(), sim_row.max(), 100)
    y_line = coef[0] * x_line + coef[1]

    corr = np.corrcoef(sim_row, acc_drop)[0, 1]

    plt.plot(x_line, y_line)

    plt.xlabel(f"Similarity to class {deleted_class}")
    plt.ylabel("Accuracy drop")
    plt.title(f"Similarity vs Accuracy Drop (corr={corr:.2f})")

    plt.grid(True)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# def plot_similarity_vs_accuracy_drop_pretty(
#     similarity_matrix,
#     df_metrics_dict,
#     deleted_class: int,
#     out_path=None,
# ):
#     sim_row = similarity_matrix[deleted_class].copy()

#     # Combine all runs and aggregate acc_drop per class
#     df_all = pd.concat(
#         [df_metrics[["class", "acc_drop"]] for df_metrics in df_metrics_list],
#         ignore_index=True,
#     )

#     df_agg = (
#         df_all.groupby("class")["acc_drop"]
#         .agg(["mean", "std", "count"])
#         .reset_index()
#         .rename(
#             columns={
#                 "mean": "acc_drop_mean",
#                 "std": "acc_drop_std",
#                 "count": "n_runs",
#             }
#         )
#     )

#     # If only one run is present, std will be NaN
#     df_agg["acc_drop_std"] = df_agg["acc_drop_std"].fillna(0.0)

#     # Two-sided 95% CI
#     alpha = 0.05
#     df_agg["t_crit"] = df_agg["n_runs"].apply(
#         lambda n: t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else 0.0
#     )
#     df_agg["acc_drop_ci95"] = (
#         df_agg["t_crit"] * df_agg["acc_drop_std"] / np.sqrt(df_agg["n_runs"])
#     )

#     classes = df_agg["class"].values
#     acc_drop = df_agg["acc_drop_mean"].values
#     acc_ci = df_agg["acc_drop_ci95"].values

#     sim_vals = sim_row[classes]

#     mask_other = classes != deleted_class

#     sim_other = sim_vals[mask_other]
#     drop_other = acc_drop[mask_other]
#     std_other = acc_ci[mask_other]
#     classes_other = classes[mask_other]

#     sim_self = sim_vals[~mask_other]
#     drop_self = acc_drop[~mask_other]
#     std_self = acc_ci[~mask_other]

#     corr_all = np.corrcoef(sim_row, acc_drop)[0, 1]
#     corr_other = np.corrcoef(sim_other, drop_other)[0, 1]

#     plt.figure(figsize=(10, 5))

#     plt.scatter(
#         sim_other,
#         drop_other,
#         s=70,
#         facecolors="white",
#         edgecolors="black",
#         linewidths=1.2,
#         zorder=3,
#         label="Other classes",
#     )

#     plt.errorbar(
#         sim_other,
#         drop_other,
#         yerr=std_other,
#         fmt="none",
#         ecolor=(0, 0, 0, 0.18),
#         elinewidth=0.8,
#         capsize=2,
#         capthick=0.8,
#         zorder=2,
#     )

#     # Deleted class itself: mean ± std
#     plt.scatter(
#         sim_self,
#         drop_self,
#         s=85,
#         marker="s",
#         facecolors="tab:red",
#         edgecolors="black",
#         linewidths=1.0,
#         zorder=4,
#         label=f"Deleted class ({deleted_class})",
#     )

#     plt.errorbar(
#         sim_self,
#         drop_self,
#         yerr=std_self,
#         fmt="none",
#         ecolor=(0.7, 0, 0, 0.5),
#         elinewidth=0.8,
#         capsize=2,
#         capthick=0.8,
#         zorder=3,
#     )

#     # Labels with slight offset
#     for x, y, cls in zip(sim_other, drop_other, classes_other):
#         plt.annotate(
#             str(cls),
#             (x, y),
#             xytext=(4, 4),
#             textcoords="offset points",
#             fontsize=9,
#         )

#     plt.annotate(
#         str(deleted_class),
#         (sim_self[0], drop_self[0]),
#         xytext=(4, 4),
#         textcoords="offset points",
#         fontsize=9,
#         fontweight="bold",
#     )

#     # Regression excluding self-class
#     coef = np.polyfit(sim_other, drop_other, 1)
#     x_line = np.linspace(sim_other.min(), sim_other.max(), 100)
#     y_line = coef[0] * x_line + coef[1]
#     plt.plot(
#         x_line,
#         y_line,
#         linestyle="--",
#         linewidth=1.8,
#         color="black",
#         alpha=0.85,
#         label=f"Fit excl. self (r={corr_other:.2f})",
#     )

#     # Horizontal zero line
#     plt.axhline(0.0, linestyle=":", linewidth=1.2, color="gray", alpha=0.8)

#     plt.xlabel(f"Cosine similarity to class {deleted_class}")
#     plt.ylabel("Accuracy drop")
#     plt.title(f"Impact of unlearning class {deleted_class}")
#     plt.legend(frameon=False)
#     plt.grid(alpha=0.2)
#     plt.tight_layout()

#     # # Optional text for full correlation
#     # plt.text(
#     #     0.02,
#     #     0.98,
#     #     f"Pearson r (all) = {corr_all:.2f}",
#     #     transform=plt.gca().transAxes,
#     #     ha="left",
#     #     va="top",
#     #     fontsize=9,
#     # )

#     if out_path:
#         plt.savefig(out_path, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()


def plot_support_set_class_distribution(
    support_indices: list[int],
    class_to_forget: int,
    out_dir: str | Path,
    prefix: str = "support_set",
) -> pd.DataFrame:
    """
    Plot class distribution inside the selected support set.

    Saves:
    - bar chart with counts
    - bar chart with proportions
    - csv table with counts and proportions

    Returns:
        DataFrame with columns:
            class, count, proportion
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    # Collect labels of support samples
    support_labels = [int(dataset[idx][1]) for idx in support_indices]

    # Count class frequencies
    label_counter = Counter(support_labels)

    # Keep only retain classes on the x-axis
    retain_classes = [cls for cls in range(10) if cls != class_to_forget]

    counts = np.array([label_counter.get(cls, 0) for cls in retain_classes], dtype=int)
    total = counts.sum()

    if total == 0:
        raise ValueError("Support set is empty.")

    proportions = counts / total

    df = pd.DataFrame(
        {
            "class": retain_classes,
            "count": counts,
            "proportion": proportions,
        }
    )
    df.to_csv(out_dir / f"{prefix}_class_distribution.csv", index=False)

    # -------- Counts plot --------
    plt.figure(figsize=(8, 5))
    bars = plt.bar(retain_classes, counts)

    for cls, count in zip(retain_classes, counts):
        plt.text(cls, count, str(count), ha="center", va="bottom", fontsize=9)

    plt.xticks(retain_classes)
    plt.xlabel("Retain class")
    plt.ylabel("Count in support set")
    plt.title(f"Support set class counts (forget class = {class_to_forget})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_class_counts.png", bbox_inches="tight")
    plt.close()

    # -------- Proportions plot --------
    plt.figure(figsize=(8, 5))
    bars = plt.bar(retain_classes, proportions)

    for cls, prop in zip(retain_classes, proportions):
        plt.text(cls, prop, f"{100 * prop:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(retain_classes)
    plt.xlabel("Retain class")
    plt.ylabel("Proportion in support set")
    plt.title(f"Support set class proportions (forget class = {class_to_forget})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_class_proportions.png", bbox_inches="tight")
    plt.close()

    return df


def aggregate_acc_drop(df_metrics_list, alpha: float = 0.05) -> pd.DataFrame:
    df_all = pd.concat(
        [df_metrics[["class", "acc_drop"]] for df_metrics in df_metrics_list],
        ignore_index=True,
    )

    df_agg = (
        df_all.groupby("class")["acc_drop"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "acc_drop_mean",
                "std": "acc_drop_std",
                "count": "n_runs",
            }
        )
    )

    df_agg["acc_drop_std"] = df_agg["acc_drop_std"].fillna(0.0)

    df_agg["t_crit"] = df_agg["n_runs"].apply(
        lambda n: t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else 0.0
    )
    df_agg["acc_drop_ci95"] = (
        df_agg["t_crit"] * df_agg["acc_drop_std"] / np.sqrt(df_agg["n_runs"])
    )

    return df_agg


def plot_similarity_vs_accuracy_drop_on_ax(
    ax,
    similarity_matrix,
    df_metrics_list,
    deleted_class: int,
    method_name: str,
    show_legend: bool = True,
):
    sim_row = similarity_matrix[deleted_class].copy()
    df_agg = aggregate_acc_drop(df_metrics_list)

    classes = df_agg["class"].values
    acc_drop = df_agg["acc_drop_mean"].values
    acc_ci = df_agg["acc_drop_ci95"].values

    sim_vals = sim_row[classes]

    mask_other = classes != deleted_class

    sim_other = sim_vals[mask_other]
    drop_other = acc_drop[mask_other]
    ci_other = acc_ci[mask_other]
    classes_other = classes[mask_other]

    sim_self = sim_vals[~mask_other]
    drop_self = acc_drop[~mask_other]
    ci_self = acc_ci[~mask_other]

    corr_all = np.corrcoef(sim_vals, acc_drop)[0, 1]
    corr_other = np.corrcoef(sim_other, drop_other)[0, 1]

    # ax.scatter(
    #     sim_other,
    #     drop_other,
    #     s=70,
    #     facecolors="white",
    #     edgecolors="black",
    #     linewidths=1.2,
    #     zorder=3,
    #     label="Other classes",
    # )
    ax.scatter(
        sim_other,
        drop_other,
        s=75,
        facecolors="gray",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.45,
        zorder=3,
        label="Other classes",
    )

    ax.errorbar(
        sim_other,
        drop_other,
        yerr=ci_other,
        fmt="none",
        ecolor=(0, 0, 0, 0.5),
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=2,
    )

    ax.scatter(
        sim_self,
        drop_self,
        s=85,
        marker="s",
        facecolors="tab:red",
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
        label=f"Deleted class ({deleted_class})",
    )

    ax.errorbar(
        sim_self,
        drop_self,
        yerr=ci_self,
        fmt="none",
        ecolor=(0.7, 0, 0, 0.5),
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=3,
    )

    for x, y, cls in zip(sim_other, drop_other, classes_other):
        # ax.annotate(
        #     str(cls),
        #     (x, y),
        #     xytext=(4, 4),
        #     textcoords="offset points",
        #     fontsize=9,
        # )
        ax.annotate(
            str(cls),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="0.35",
            alpha=0.85,
            zorder=5,
        )

    ax.annotate(
        str(deleted_class),
        (sim_self[0], drop_self[0]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

    coef = np.polyfit(sim_other, drop_other, 1)
    x_line = np.linspace(sim_other.min(), sim_other.max(), 100)
    y_line = coef[0] * x_line + coef[1]
    # ax.plot(
    #     x_line,
    #     y_line,
    #     linestyle="--",
    #     linewidth=1.8,
    #     color="#55A868",
    #     alpha=0.9,
    #     label=f"Fit excl. deleted class (r={corr_other:.2f})",
    # )
    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=5.0,
        color="#55A868",
        alpha=1.0,
        label=f"Trend excluding deleted class",
    )

    ax.axhline(0.0, linestyle=":", linewidth=1.2, color="gray", alpha=0.8)

    # ax.set_xlabel(f"Cosine similarity to class {deleted_class}")
    # ax.set_ylabel("Accuracy drop")
    # ax.set_title(method_name)

    ax.set_xlabel(f"Similarity to deleted class {deleted_class}")
    ax.set_ylabel("Accuracy drop vs. retraining")
    ax.set_title(method_name, fontweight="bold")
    ax.grid(alpha=0.2)

    # if show_legend:
    #     ax.legend(frameon=False)
    # ax.text(
    #     0.97,
    #     0.90,
    #     f"trend: r={corr_other:.2f}",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=10,
    #     color="#2F7D46",
    #     fontweight="bold",
    # )

    # ax.annotate(
    #     "higher similarity\nlarger drop",
    #     xy=(0.86, 0.72),
    #     xytext=(0.57, 0.43),
    #     xycoords="axes fraction",
    #     textcoords="axes fraction",
    #     fontsize=10,
    #     color="#2F7D46",
    #     fontweight="bold",
    #     arrowprops=dict(
    #         arrowstyle="->",
    #         color="#2F7D46",
    #         linewidth=2.0,
    #     ),
    # )

    return corr_all, corr_other


def plot_similarity_vs_accuracy_drop_pretty(
    similarity_matrix,
    df_metrics_dict,
    deleted_class: int,
    out_path=None,
):
    n_methods = len(df_metrics_dict)

    if n_methods == 0:
        raise ValueError("df_metrics_dict is empty")

    ncols = 1 if n_methods == 1 else 2
    nrows = math.ceil(n_methods / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8, 4),
        squeeze=False,
    )

    axes_flat = axes.flatten()
    items = list(df_metrics_dict.items())

    for i, (method_name, df_metrics_list) in enumerate(items):
        ax = axes_flat[i]
        plot_similarity_vs_accuracy_drop_on_ax(
            ax=ax,
            similarity_matrix=similarity_matrix,
            df_metrics_list=df_metrics_list,
            deleted_class=deleted_class,
            method_name=method_name,
            show_legend=True,
        )

    used_axes = axes_flat[:len(items)]

    # Use common y-limits so that zero is aligned across all subplots.
    y_min = min(ax.get_ylim()[0] for ax in used_axes)
    y_max = max(ax.get_ylim()[1] for ax in used_axes)

    for ax in used_axes:
        ax.set_ylim(y_min, y_max)
        #ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)

    for j in range(len(items), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, wspace=0.22)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
