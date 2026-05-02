from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from cifar.methods.distill.support_teachers import (
    SupportTeacher,
    select_support_indices,
    train_support_teacher,
    predict_soft_targets_with_teacher,
)


def compute_argmax_match_accuracy(
    target_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    k=3,
) -> float:
    # target_labels = target_probs.argmax(dim=1)
    # pred_labels = pred_probs.argmax(dim=1)
    # return float((target_labels == pred_labels).float().mean().item())

    target_topk_labels = target_probs.topk(k=k, dim=1).indices
    pred_labels = pred_probs.argmax(dim=1)

    matches = (target_topk_labels == pred_labels.unsqueeze(1)).any(dim=1)
    return float(matches.float().mean().item())


def compute_argmax_match_accuracy_on_mask(
    target_probs: torch.Tensor, pred_probs: torch.Tensor, mask: torch.Tensor, k=3
) -> float:
    if target_probs.shape != pred_probs.shape:
        raise ValueError(
            f"Shape mismatch: target_probs.shape={target_probs.shape}, "
            f"pred_probs.shape={pred_probs.shape}"
        )

    if mask.ndim != 1 or mask.shape[0] != target_probs.shape[0]:
        raise ValueError(
            f"Mask must have shape [N], got {tuple(mask.shape)} "
            f"for target_probs.shape={tuple(target_probs.shape)}"
        )

    if mask.sum().item() == 0:
        return 0.0

    # target_labels = target_probs.argmax(dim=1)
    target_topk_labels = target_probs.topk(k=k, dim=1).indices
    pred_labels = pred_probs.argmax(dim=1)

    # acc = float((target_labels[mask] == pred_labels[mask]).float().mean().item())
    matches = (target_topk_labels[mask] == pred_labels[mask].unsqueeze(1)).any(dim=1)
    return float(matches.float().mean().item())


def compute_argmax_histogram_kl(
    retrain_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    num_classes: int | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Compute KL between argmax-class histograms:
        KL(hist_retrain || hist_pred)

    retrain_probs: [N, C]
    pred_probs: [N, C]
    """
    if retrain_probs.shape != pred_probs.shape:
        raise ValueError(
            f"Shape mismatch: retrain_probs.shape={retrain_probs.shape}, "
            f"pred_probs.shape={pred_probs.shape}"
        )

    if retrain_probs.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors of shape [N, C], got ndim={retrain_probs.ndim}"
        )

    if retrain_probs.numel() == 0:
        return 0.0

    if num_classes is None:
        num_classes = retrain_probs.shape[1]

    retrain_labels = retrain_probs.argmax(dim=1)
    pred_labels = pred_probs.argmax(dim=1)

    retrain_hist = torch.bincount(retrain_labels, minlength=num_classes).float()
    pred_hist = torch.bincount(pred_labels, minlength=num_classes).float()

    retrain_hist = retrain_hist / retrain_hist.sum().clamp_min(1.0)
    pred_hist = pred_hist / pred_hist.sum().clamp_min(1.0)

    retrain_hist = retrain_hist.clamp_min(eps)
    pred_hist = pred_hist.clamp_min(eps)

    kl = (retrain_hist * (retrain_hist.log() - pred_hist.log())).sum()
    return float(kl.item())


def compute_topk_kl_from_probs(
    target_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    k: int,
    eps: float = 1e-12,
) -> float:
    """
    Compute truncated KL divergence on top-k classes selected from target_probs:

        mean_x sum_{i in top-k(target(x))} target_i(x) * log(target_i(x) / pred_i(x))

    This does not renormalize the selected top-k probabilities.
    It ignores the tail outside top-k and focuses on the main mass of target_probs.

    Args:
        target_probs: Tensor of shape [N, C], reference probability distributions.
        pred_probs: Tensor of shape [N, C], predicted probability distributions.
        k: Number of top classes from target_probs to keep.
        eps: Small constant for numerical stability.

    Returns:
        Scalar float KL value averaged over the batch.
    """
    if target_probs.shape != pred_probs.shape:
        raise ValueError(
            f"Shape mismatch: target_probs.shape={target_probs.shape}, "
            f"pred_probs.shape={pred_probs.shape}"
        )

    if target_probs.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors of shape [N, C], got ndim={target_probs.ndim}"
        )

    if target_probs.numel() == 0:
        return 0.0

    num_classes = target_probs.shape[1]
    if not (1 <= k <= num_classes):
        raise ValueError(f"k must be in [1, {num_classes}], got k={k}")

    topk_idx = torch.topk(target_probs, k=k, dim=1).indices

    target_topk = torch.gather(target_probs, 1, topk_idx).clamp_min(eps)
    pred_topk = torch.gather(pred_probs, 1, topk_idx).clamp_min(eps)

    kl = (target_topk * (target_topk.log() - pred_topk.log())).sum(dim=1).mean()
    return float(kl.item())


def get_argmax_histogram(
    probs: torch.Tensor,
    num_classes: int | None = None,
) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError(f"Expected shape [N, C], got {tuple(probs.shape)}")

    if num_classes is None:
        num_classes = probs.shape[1]

    labels = probs.argmax(dim=1)
    hist = torch.bincount(labels, minlength=num_classes).float()
    return hist


def compute_argmax_histogram_kl_between_mean_probs(
    prev_mean_probs: torch.Tensor,
    current_mean_probs: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Compute KL between argmax histograms of two mean probability tensors:
        KL(hist(prev_mean_probs) || hist(current_mean_probs))
    """
    prev_hist = get_argmax_histogram(prev_mean_probs).clamp_min(eps)
    current_hist = get_argmax_histogram(current_mean_probs).clamp_min(eps)

    prev_hist = prev_hist / prev_hist.sum().clamp_min(eps)
    current_hist = current_hist / current_hist.sum().clamp_min(eps)

    kl = (prev_hist * (prev_hist.log() - current_hist.log())).sum()
    return float(kl.item())


def compute_accuracy_against_forget_class_label(
    pred_probs: torch.Tensor,
    class_to_forget: int,
) -> float:
    """
    Accuracy when all ground-truth labels are class_to_forget.
    """
    pred_labels = pred_probs.argmax(dim=1)
    acc = (pred_labels == class_to_forget).float().mean()
    return float(acc.item())


def save_support_teacher_checkpoint(
    teacher: SupportTeacher,
    output_dir: str | Path,
    n_support: int,
    support_indices: list[int] | np.ndarray,
    threshold: float | None,
    train_accuracy: float,
    epochs_trained: int,
    teacher_type: str,
) -> Path:
    """
    Save teacher model and metadata to:
        output_dir / f"n_support_{n_support}"
    """
    save_dir = Path(output_dir) / f"n_support_{n_support}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "teacher_model.pt"
    meta_path = save_dir / "meta.json"
    indices_path = save_dir / "support_indices.npy"

    torch.save(
        {
            "kind": teacher.kind,
            "model_state_dict": teacher.model.state_dict(),
            "n_support": int(n_support),
            "teacher_type": teacher_type,
        },
        model_path,
    )

    np.save(indices_path, np.asarray(support_indices, dtype=int))

    metadata = {
        "n_support": int(n_support),
        "teacher_type": teacher_type,
        "kind": teacher.kind,
        "train_accuracy": float(train_accuracy),
        "epochs_trained": int(epochs_trained),
        "threshold": None if threshold is None else float(threshold),
        "model_path": str(model_path),
        "support_indices_path": str(indices_path),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return save_dir


def save_delta_vs_n_support_plot(
    n_support_values: list[int],
    kl_to_retrain_mean: list[float],
    kl_to_retrain_std: list[float],
    retrain_hist: torch.Tensor,
    output_path: str | Path,
    collapsed_class: int = 25,
    eps: float = 1e-12,
) -> None:
    """
    Plot mean ± std for KL between predicted-class histograms:
    - KL(hist_retrain || hist_teacher)
    - KL(hist_retrain || uniform)
    - KL(hist_retrain || collapsed)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retrain_hist = retrain_hist.detach().float().cpu().clamp_min(eps)
    retrain_hist = retrain_hist / retrain_hist.sum().clamp_min(eps)

    num_classes = retrain_hist.numel()

    uniform_hist = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)

    collapsed_hist = torch.full((num_classes,), eps, dtype=torch.float32)
    collapsed_hist[collapsed_class] = 1.0
    collapsed_hist = collapsed_hist / collapsed_hist.sum().clamp_min(eps)

    kl_uniform = float(
        (retrain_hist * (retrain_hist.log() - uniform_hist.log())).sum().item()
    )
    kl_collapsed = float(
        (retrain_hist * (retrain_hist.log() - collapsed_hist.log())).sum().item()
    )

    plt.figure(figsize=(8, 5))

    x_full = np.array(n_support_values, dtype=float)
    y_retrain = np.array(kl_to_retrain_mean, dtype=float)
    s_retrain = np.array(kl_to_retrain_std, dtype=float)

    plt.plot(
        x_full,
        y_retrain,
        marker="o",
        label=r"$\mathrm{KL}(\mathrm{hist}_{\mathrm{retrain}} \| \mathrm{hist}_{\mathrm{teacher}})$",
    )
    plt.fill_between(
        x_full,
        y_retrain - s_retrain,
        y_retrain + s_retrain,
        alpha=0.2,
    )

    plt.axhline(
        y=kl_uniform,
        linestyle="--",
        label=r"$\mathrm{KL}(\mathrm{hist}_{\mathrm{retrain}} \| \mathrm{uniform})$",
    )
    plt.axhline(
        y=kl_collapsed,
        linestyle=":",
        label=rf"$\mathrm{{KL}}(\mathrm{{hist}}_{{\mathrm{{retrain}}}} \| \mathrm{{collapsed}}_{{forget class}})$",
    )

    plt.xlabel("Support size")
    plt.ylabel("KL divergence")
    plt.title("Predicted-class histogram mismatch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_argmax_match_vs_n_support_plot(
    n_support_values: list[int],
    argmax_match_mean: list[float],
    argmax_match_std: list[float],
    argmax_match_nonforget_mean,
    argmax_match_nonforget_std,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    x = np.array(n_support_values, dtype=float)
    y = np.array(argmax_match_mean, dtype=float)
    s = np.array(argmax_match_std, dtype=float)

    plt.plot(x, y, marker="o", label="Argmax match with retrain")
    plt.fill_between(x, y - s, y + s, alpha=0.2)

    x = np.array(n_support_values, dtype=float)
    y = np.array(argmax_match_nonforget_mean, dtype=float)
    s = np.array(argmax_match_nonforget_std, dtype=float)

    plt.plot(
        x,
        y,
        marker="o",
        label="Argmax match on retrain-nonforget subset",
    )
    plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Support size")
    plt.ylabel("Argmax match accuracy")
    plt.title("Instance-wise prediction agreement with retrain")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_kl_and_argmax_match_vs_n_support_plot(
    n_support_values: list[int],
    kl_mean: list[float],
    kl_std: list[float],
    argmax_match_mean: list[float],
    argmax_match_std: list[float],
    argmax_match_nonforget_mean,
    argmax_match_nonforget_std,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(n_support_values, dtype=float)
    kl_mean = np.array(kl_mean, dtype=float)
    kl_std = np.array(kl_std, dtype=float)
    argmax_match_mean = np.array(argmax_match_mean, dtype=float)
    argmax_match_std = np.array(argmax_match_std, dtype=float)

    argmax_match_nonforget_mean = np.array(argmax_match_nonforget_mean, dtype=float)
    argmax_match_nonforget_std = np.array(argmax_match_nonforget_std, dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        x,
        kl_mean,
        marker="o",
        label="KL(hist_retrain || hist_teacher)",
    )
    ax1.fill_between(
        x,
        kl_mean - kl_std,
        kl_mean + kl_std,
        alpha=0.2,
    )

    line2 = ax2.plot(
        x,
        argmax_match_mean,
        marker="s",
        linestyle="--",
        label="Argmax match with retrain",
    )
    ax2.fill_between(
        x,
        argmax_match_mean - argmax_match_std,
        argmax_match_mean + argmax_match_std,
        alpha=0.2,
    )

    line3 = ax2.plot(
        x,
        argmax_match_nonforget_mean,
        marker="^",
        linestyle=":",
        label="Argmax match on retrain-nonforget subset",
    )
    ax2.fill_between(
        x,
        argmax_match_nonforget_mean - argmax_match_nonforget_std,
        argmax_match_nonforget_mean + argmax_match_nonforget_std,
        alpha=0.2,
    )

    ax1.set_xlabel("Support size")
    ax1.set_ylabel("KL divergence")
    ax2.set_ylabel("Argmax match accuracy")
    ax1.set_title("Teacher agreement with retrain vs support size")

    ax1.grid(True, alpha=0.3)

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_probs(
    retrain_probs: torch.Tensor,
    class_to_forget: int = 25,
) -> dict[str, float]:
    eps = 1e-12
    probs = retrain_probs.clamp_min(eps)

    max_probs = probs.max(dim=1).values
    entropies = -(probs * probs.log()).sum(dim=1)

    top3_mass = torch.topk(probs, k=3, dim=1).values.sum(dim=1)
    top5_mass = torch.topk(probs, k=5, dim=1).values.sum(dim=1)
    top10_mass = torch.topk(probs, k=10, dim=1).values.sum(dim=1)

    predicted_labels = probs.argmax(dim=1)
    argmax_accuracy_on_forget_class = (
        (predicted_labels == class_to_forget).float().mean()
    )

    num_classes = probs.shape[1]
    max_entropy = float(torch.log(torch.tensor(float(num_classes))).item())

    return {
        "mean_max_prob": float(max_probs.mean().item()),
        "std_max_prob": float(max_probs.std().item()),
        "mean_entropy": float(entropies.mean().item()),
        "mean_entropy_normalized": float((entropies.mean() / max_entropy).item()),
        "mean_top3_mass": float(top3_mass.mean().item()),
        "mean_top5_mass": float(top5_mass.mean().item()),
        "mean_top10_mass": float(top10_mass.mean().item()),
        "argmax_accuracy_on_forget_class": float(
            argmax_accuracy_on_forget_class.item()
        ),
    }


def save_hist_kl_to_previous_vs_n_support_plot(
    n_support_values: list[int],
    hist_kl_to_previous_mean: list[float],
    hist_kl_to_previous_std: list[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(n_support_values[1:], dtype=float)
    y = np.array(hist_kl_to_previous_mean, dtype=float)
    s = np.array(hist_kl_to_previous_std, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", label="KL(hist_previous || hist_current)")
    plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Support size")
    plt.ylabel("Histogram KL to previous")
    plt.title("Predicted-class histogram shift under support expansion")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_argmax_match_to_previous_vs_n_support_plot(
    n_support_values: list[int],
    argmax_match_to_previous_mean: list[float],
    argmax_match_to_previous_std: list[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(n_support_values[1:], dtype=float)
    y = np.array(argmax_match_to_previous_mean, dtype=float)
    s = np.array(argmax_match_to_previous_std, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", label="Argmax match to previous")
    plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel("Support size")
    plt.ylabel("Argmax match accuracy")
    plt.title("Instance-wise agreement with previous support size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_forget_class_accuracy_vs_n_support_plot(
    n_support_values: list[int],
    forget_class_accuracy_mean: list[float],
    forget_class_accuracy_std: list[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(n_support_values, dtype=float)
    y = np.array(forget_class_accuracy_mean, dtype=float)
    s = np.array(forget_class_accuracy_std, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", label="Accuracy on forget class")
    plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel("Support size")
    plt.ylabel("Accuracy")
    plt.title("Prediction accuracy against forget-class label")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_num_epochs_trained_vs_n_support_plot(
    n_support_values: list[int],
    data_mean: list[float],
    data_std: list[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(n_support_values, dtype=float)
    y = np.array(data_mean, dtype=float)
    s = np.array(data_std, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", label="Accuracy on forget class")
    plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel("Support size")
    plt.ylabel("Number trained epochs")
    plt.title("Number of epochs trained")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_topk_kl_to_previous_vs_n_support_plot(
    n_support_values: list[int],
    stability_mean: list[float],
    stability_std: list[float],
    output_path: str | Path,
    k: int,
) -> None:
    """
    Plot top-k KL stability to previous support size.

    Args:
        n_support_values: Full list of support sizes.
        stability_mean: Mean top-k KL values for transitions from previous support.
                        Must have length len(n_support_values) - 1.
        stability_std: Std of top-k KL values for transitions from previous support.
                       Must have length len(n_support_values) - 1.
        output_path: Path to save the figure.
        k: Top-k used in the KL computation.
    """
    if len(n_support_values) < 2:
        raise ValueError("At least two n_support values are required.")

    if len(stability_mean) != len(n_support_values) - 1:
        raise ValueError(
            "stability_mean must have length len(n_support_values) - 1, "
            f"got {len(stability_mean)} vs {len(n_support_values) - 1}"
        )

    if len(stability_std) != len(n_support_values) - 1:
        raise ValueError(
            "stability_std must have length len(n_support_values) - 1, "
            f"got {len(stability_std)} vs {len(n_support_values) - 1}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(n_support_values[1:], dtype=float)
    y = np.array(stability_mean, dtype=float)
    s = np.array(stability_std, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y,
        marker="o",
        label=f"Top-{k} KL to previous support size",
    )
    plt.fill_between(
        x,
        y - s,
        y + s,
        alpha=0.2,
    )

    plt.xlabel("Support size")
    plt.ylabel(f"Top-{k} KL to previous")
    plt.title("Prediction stability under support expansion")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def prepare_retrain_reference(
    forget_dataset,
    feature_extractor: torch.nn.Module,
    retrain_model,
    batch_size: int,
    device: str,
    num_workers: int,
    temperature: float,
    forget_class: int | None,
    class_to_forget: int,
    target_mode,
):
    """
    Prepare retrain probabilities and derived diagnostics on the forget set.
    """
    retrain_probs = predict_soft_targets_with_teacher(
        forget_dataset=forget_dataset,
        teacher=SupportTeacher(kind="image", model=retrain_model),
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        temperature=temperature,
        target_mode=target_mode,
    )

    retrain_argmax = retrain_probs.argmax(dim=1)
    retrain_not_forget_mask = retrain_argmax != class_to_forget
    retrain_stats = summarize_probs(retrain_probs)

    print(
        "Num forget-set points reassigned by retrain:",
        int(retrain_not_forget_mask.sum().item()),
    )
    print(retrain_stats)

    return {
        "retrain_probs": retrain_probs,
        "retrain_not_forget_mask": retrain_not_forget_mask,
        "retrain_stats": retrain_stats,
        "retrain_num_probs": retrain_probs.shape,
    }


def load_or_train_teacher_run(
    dataset,
    forget_dataset,
    retain_indices_full: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    n_support: int,
    run_seed: int,
    teacher_type: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    output_dir: Path,
    num_workers: int,
    hidden_dim: int,
    dropout: float,
    accuracy_threshold: float | None,
    support_selection_mode: str,
    temperature: float,
    forget_class: int | None,
    optimizer_type: str = "adamw",
    dataset_name: str = "cifar100",
    target_mode: str = "top3_renorm",
):
    """
    Load a saved teacher run if available; otherwise train it and save outputs.
    """
    run_output_dir = output_dir / f"n_support={n_support}" / f"seed={run_seed}"
    save_dir = run_output_dir / f"n_support_{n_support}"
    probs_path = save_dir / "forget_probs.pt"
    meta_path = save_dir / "meta.json"

    if probs_path.exists():
        print(f"Loading existing run from {probs_path}")
        probs = torch.load(probs_path, map_location="cpu")
        print(summarize_probs(probs))

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return {
            "probs": probs,
            "metadata": metadata,
            "meaningful_metadata_available": True,
        }

    support_indices, threshold = select_support_indices(
        dataset=dataset,
        retain_indices_full=retain_indices_full,
        forget_indices=forget_indices,
        n_support=n_support,
        feature_extractor=feature_extractor,
        support_selection_mode=support_selection_mode,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=run_seed,
    )

    teacher, train_accuracy, last_epoch = train_support_teacher(
        dataset=dataset,
        support_indices=support_indices,
        feature_extractor=feature_extractor,
        teacher_type=teacher_type,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        num_workers=num_workers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        accuracy_threshold=accuracy_threshold,
        optimizer_type=optimizer_type,
        seed=run_seed,
        dataset_name=dataset_name,
    )

    save_dir = save_support_teacher_checkpoint(
        teacher=teacher,
        output_dir=run_output_dir,
        n_support=n_support,
        support_indices=support_indices,
        threshold=threshold,
        train_accuracy=train_accuracy,
        epochs_trained=last_epoch + 1,
        teacher_type=teacher_type,
    )

    probs = predict_soft_targets_with_teacher(
        forget_dataset=forget_dataset,
        teacher=teacher,
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        temperature=temperature,
        target_mode=target_mode,
    )
    print(summarize_probs(probs))

    probs_path = save_dir / "forget_probs.pt"
    torch.save(probs, probs_path)

    metadata = {
        "seed": int(run_seed),
        "train_accuracy": float(train_accuracy),
        "epochs_trained": int(last_epoch + 1),
        "threshold": None if threshold is None else float(threshold),
        "save_dir": str(save_dir),
        "forget_probs_path": str(probs_path),
    }

    return {
        "probs": probs,
        "metadata": metadata,
        "meaningful_metadata_available": True,
    }


def aggregate_support_run_metrics(
    probs_per_run: list[torch.Tensor],
    run_results: list[dict[str, Any]],
    retrain_probs: torch.Tensor,
    retrain_not_forget_mask: torch.Tensor,
    output_dir: Path,
    n_support: int,
    num_teacher_seeds: int,
    save_base: bool,
    class_to_forget: int,
) -> dict[str, Any]:
    """
    Aggregate metrics across multiple seeds for the same support size.
    """
    kl_to_retrain_per_run = [
        compute_argmax_histogram_kl(retrain_probs, probs) for probs in probs_per_run
    ]
    argmax_match_per_run = [
        compute_argmax_match_accuracy(target_probs=retrain_probs, pred_probs=probs)
        for probs in probs_per_run
    ]
    argmax_match_nonforget_per_run = [
        compute_argmax_match_accuracy_on_mask(
            target_probs=retrain_probs,
            pred_probs=probs,
            mask=retrain_not_forget_mask,
        )
        for probs in probs_per_run
    ]

    forget_class_accuracy_per_run = [
        compute_accuracy_against_forget_class_label(
            pred_probs=probs,
            class_to_forget=class_to_forget,
        )
        for probs in probs_per_run
    ]

    num_epochs_per_run = [run_result["epochs_trained"] for run_result in run_results]

    stacked_probs = torch.stack(probs_per_run, dim=0)
    mean_probs = stacked_probs.mean(dim=0)

    mean_probs_path = output_dir / f"n_support={n_support}" / "mean_forget_probs.pt"
    if save_base:
        mean_probs_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mean_probs, mean_probs_path)

    aggregated = {
        "mean_probs": mean_probs,
        "kl_to_retrain_mean": float(np.mean(kl_to_retrain_per_run)),
        "kl_to_retrain_std": float(np.std(kl_to_retrain_per_run)),
        "argmax_match_mean": float(np.mean(argmax_match_per_run)),
        "argmax_match_std": float(np.std(argmax_match_per_run)),
        "argmax_match_nonforget_mean": float(np.mean(argmax_match_nonforget_per_run)),
        "argmax_match_nonforget_std": float(np.std(argmax_match_nonforget_per_run)),
        "forget_class_accuracy_mean": float(np.mean(forget_class_accuracy_per_run)),
        "forget_class_accuracy_std": float(np.std(forget_class_accuracy_per_run)),
        "num_epochs_trained_mean": float(np.mean(num_epochs_per_run)),
        "num_epochs_trained_std": float(np.std(num_epochs_per_run)),
    }

    if save_base:
        train_accs = [x["train_accuracy"] for x in run_results if x is not None]
        aggregated["result"] = {
            "n_support": int(n_support),
            "num_teacher_seeds": int(num_teacher_seeds),
            "mean_forget_probs_path": str(mean_probs_path),
            "hist_kl_to_retrain_mean": aggregated["kl_to_retrain_mean"],
            "hist_kl_to_retrain_std": aggregated["kl_to_retrain_std"],
            "train_accuracy_mean": float(np.mean(train_accs)) if train_accs else None,
            "train_accuracy_std": float(np.std(train_accs)) if train_accs else None,
            "runs": run_results,
            "argmax_match_mean": aggregated["argmax_match_mean"],
            "argmax_match_std": aggregated["argmax_match_std"],
            "argmax_match_nonforget_mean": aggregated["argmax_match_nonforget_mean"],
            "argmax_match_nonforget_std": aggregated["argmax_match_nonforget_std"],
            "forget_class_accuracy_mean": aggregated["forget_class_accuracy_mean"],
            "forget_class_accuracy_std": aggregated["forget_class_accuracy_std"],
            "num_epochs_trained_mean": aggregated["num_epochs_trained_mean"],
            "num_epochs_trained_std": aggregated["num_epochs_trained_std"],
        }
    else:
        aggregated["result"] = {}

    return aggregated


def compute_stability_to_previous(
    prev_mean_probs: torch.Tensor | None,
    mean_probs: torch.Tensor,
    probs_per_run: list[torch.Tensor],
) -> dict[str, float | None]:
    """
    Compute stability of current support-size predictions w.r.t. the previous one:
    - histogram KL between mean predictions
    - argmax match to previous mean predictions
    """
    if prev_mean_probs is None:
        return {
            "hist_kl_to_previous_mean": None,
            "hist_kl_to_previous_std": None,
            "argmax_match_to_previous_mean": None,
            "argmax_match_to_previous_std": None,
        }

    # hist_kl_to_previous = compute_argmax_histogram_kl_between_mean_probs(
    #     prev_mean_probs=prev_mean_probs,
    #     current_mean_probs=mean_probs,
    # )

    hist_kl_to_previous = [
        compute_argmax_histogram_kl_between_mean_probs(
            prev_mean_probs=prev_mean_probs,
            current_mean_probs=probs,
        )
        for probs in probs_per_run
    ]

    argmax_match_to_previous_per_run = [
        compute_argmax_match_accuracy(
            target_probs=prev_mean_probs,
            pred_probs=probs,
        )
        for probs in probs_per_run
    ]

    return {
        # "hist_kl_to_previous": hist_kl_to_previous,
        "argmax_match_to_previous_mean": float(
            np.mean(argmax_match_to_previous_per_run)
        ),
        "argmax_match_to_previous_std": float(np.std(argmax_match_to_previous_per_run)),
        "hist_kl_to_previous_mean": float(np.mean(hist_kl_to_previous)),
        "hist_kl_to_previous_std": float(np.std(hist_kl_to_previous)),
    }


def process_single_support_size(
    dataset,
    forget_dataset,
    retain_indices_full: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    n_support: int,
    teacher_type: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    output_dir: Path,
    num_workers: int,
    hidden_dim: int,
    dropout: float,
    accuracy_threshold: float | None,
    support_selection_mode: str,
    temperature: float,
    forget_class: int | None,
    seed: int,
    num_teacher_seeds: int,
    retrain_probs: torch.Tensor,
    retrain_not_forget_mask: torch.Tensor,
    prev_mean_probs: torch.Tensor | None,
    trunk_k: int,
    class_to_forget: int = 25,
    optimizer_type: str = "adamw",
    dataset_name: str = "cifar100",
    target_mode: str = "top3_renorm",
):
    """
    Train/load all teacher runs for one support size and aggregate metrics.
    """
    print(f"\n=== Training teachers for n_support={n_support} ===")

    run_results = []
    probs_per_run = []
    save_base = True

    for run_idx in range(num_teacher_seeds):
        run_seed = seed + run_idx
        print(
            f"--- n_support={n_support} | run {run_idx + 1}/{num_teacher_seeds} "
            f"| seed={run_seed}"
        )

        run_data = load_or_train_teacher_run(
            dataset=dataset,
            forget_dataset=forget_dataset,
            retain_indices_full=retain_indices_full,
            forget_indices=forget_indices,
            feature_extractor=feature_extractor,
            n_support=n_support,
            run_seed=run_seed,
            teacher_type=teacher_type,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            output_dir=output_dir,
            num_workers=num_workers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            accuracy_threshold=accuracy_threshold,
            support_selection_mode=support_selection_mode,
            temperature=temperature,
            optimizer_type=optimizer_type,
            forget_class=forget_class,
            dataset_name=dataset_name,
            target_mode=target_mode,
        )

        probs_per_run.append(run_data["probs"])
        if run_data["meaningful_metadata_available"]:
            run_results.append(run_data["metadata"])
        else:
            save_base = False

    aggregated = aggregate_support_run_metrics(
        probs_per_run=probs_per_run,
        run_results=run_results,
        retrain_probs=retrain_probs,
        retrain_not_forget_mask=retrain_not_forget_mask,
        output_dir=output_dir,
        n_support=n_support,
        num_teacher_seeds=num_teacher_seeds,
        save_base=save_base,
        class_to_forget=class_to_forget,
    )

    stability = compute_stability_to_previous(
        prev_mean_probs=prev_mean_probs,
        mean_probs=aggregated["mean_probs"],
        probs_per_run=probs_per_run,
    )
    aggregated["result"].update(stability)

    print(
        f"CE to retrain for n_support={n_support}: "
        f"{aggregated['kl_to_retrain_mean']:.6f} ± {aggregated['kl_to_retrain_std']:.6f}"
    )
    # if stability["kl_to_previous_mean"] is not None:
    #     print(
    #         f"CE to previous for n_support={n_support}: "
    #         f"{stability['kl_to_previous_mean']:.6f} ± "
    #         f"{stability['kl_to_previous_std']:.6f}"
    #     )

    aggregated["save_base"] = save_base

    # if save_base:
    #     torch.save(aggregated, output_dir / f"n_support={n_support}" / "stats.json")

    return aggregated


def save_all_support_plots(
    output_dir: Path,
    n_support_values: list[int],
    retrain_probs: torch.Tensor,
    kl_to_retrain_mean_values: list[float],
    kl_to_retrain_std_values: list[float],
    argmax_match_mean_values: list[float],
    argmax_match_std_values: list[float],
    argmax_match_nonforget_mean_values: list[float],
    argmax_match_nonforget_std_values: list[float],
    hist_kl_to_previous_mean_values,
    hist_kl_to_previous_std_values,
    argmax_match_to_previous_mean_values: list[float],
    argmax_match_to_previous_std_values: list[float],
    forget_class_accuracy_mean_values: list[float],
    forget_class_accuracy_std_values: list[float],
    num_epochs_trained_mean_values,
    num_epochs_trained_std_values,
    class_to_forget: int,
    trunk_k: int,
) -> dict[str, str]:
    """
    Save all plots produced by the support-size sweep.
    """
    paths: dict[str, str] = {}

    retrain_hist = get_argmax_histogram(retrain_probs)

    hist_kl_plot_path = output_dir / "hist_KL_vs_n_support.pdf"
    save_delta_vs_n_support_plot(
        n_support_values=n_support_values,
        kl_to_retrain_mean=kl_to_retrain_mean_values,
        kl_to_retrain_std=kl_to_retrain_std_values,
        output_path=hist_kl_plot_path,
        retrain_hist=retrain_hist,
        collapsed_class=class_to_forget,
    )
    paths["hist_kl_plot_path"] = str(hist_kl_plot_path)

    argmax_plot_path = output_dir / "argmax_match_vs_n_support.pdf"
    save_argmax_match_vs_n_support_plot(
        n_support_values=n_support_values,
        argmax_match_mean=argmax_match_mean_values,
        argmax_match_std=argmax_match_std_values,
        argmax_match_nonforget_mean=argmax_match_nonforget_mean_values,
        argmax_match_nonforget_std=argmax_match_nonforget_std_values,
        output_path=argmax_plot_path,
    )
    paths["argmax_plot_path"] = str(argmax_plot_path)

    combined_plot_path = output_dir / "kl_and_argmax_match_vs_n_support.pdf"
    save_kl_and_argmax_match_vs_n_support_plot(
        n_support_values=n_support_values,
        kl_mean=kl_to_retrain_mean_values,
        kl_std=kl_to_retrain_std_values,
        argmax_match_mean=argmax_match_mean_values,
        argmax_match_std=argmax_match_std_values,
        argmax_match_nonforget_mean=argmax_match_nonforget_mean_values,
        argmax_match_nonforget_std=argmax_match_nonforget_std_values,
        output_path=combined_plot_path,
    )
    paths["combined_plot_path"] = str(combined_plot_path)

    hist_kl_prev_plot_path = output_dir / "hist_kl_to_previous_vs_n_support.pdf"
    save_hist_kl_to_previous_vs_n_support_plot(
        n_support_values=n_support_values,
        hist_kl_to_previous_mean=hist_kl_to_previous_mean_values,
        hist_kl_to_previous_std=hist_kl_to_previous_mean_values,
        output_path=hist_kl_prev_plot_path,
    )
    paths["hist_kl_to_previous_plot_path"] = str(hist_kl_prev_plot_path)

    argmax_prev_plot_path = output_dir / "argmax_match_to_previous_vs_n_support.pdf"
    save_argmax_match_to_previous_vs_n_support_plot(
        n_support_values=n_support_values,
        argmax_match_to_previous_mean=argmax_match_to_previous_mean_values,
        argmax_match_to_previous_std=argmax_match_to_previous_std_values,
        output_path=argmax_prev_plot_path,
    )
    paths["argmax_match_to_previous_plot_path"] = str(argmax_prev_plot_path)

    forget_acc_plot_path = output_dir / "forget_class_accuracy_vs_n_support.pdf"
    save_forget_class_accuracy_vs_n_support_plot(
        n_support_values=n_support_values,
        forget_class_accuracy_mean=forget_class_accuracy_mean_values,
        forget_class_accuracy_std=forget_class_accuracy_std_values,
        output_path=forget_acc_plot_path,
    )
    paths["forget_class_accuracy_plot_path"] = str(forget_acc_plot_path)

    epochs_plot_path = output_dir / "num_epochs_trained_vs_n_support.pdf"
    save_num_epochs_trained_vs_n_support_plot(
        n_support_values=n_support_values,
        data_mean=num_epochs_trained_mean_values,
        data_std=num_epochs_trained_std_values,
        output_path=epochs_plot_path,
    )
    paths["num_epochs_plot_path"] = str(epochs_plot_path)

    return paths


def train_support_teachers_and_measure_stability(
    dataset,
    forget_dataset,
    retain_indices_full: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    n_support_values: list[int],
    teacher_type: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    output_dir: str | Path,
    num_workers: int = 0,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    accuracy_threshold: float | None = None,
    support_selection_mode: str = "closest",
    temperature: float = 1.0,
    forget_class: int | None = None,
    seed: int = 42,
    retrain_model=None,
    num_teacher_seeds: int = 3,
    trunk_k: int = 3,
    class_to_forget: int = 25,
    optimizer_type: str = "adamw",
    dataset_name: str = "cifar100",
    target_mode: str = "top3_renorm",
) -> dict | None:
    """
    Train multiple teacher models for each n_support, aggregate teacher predictions
    on the forget set, and save agreement/stability diagnostics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_support_values = sorted(set(int(x) for x in n_support_values))
    if len(n_support_values) == 0:
        raise ValueError("n_support_values must be non-empty.")

    if num_teacher_seeds <= 0:
        raise ValueError("num_teacher_seeds must be positive.")

    retrain_ref = prepare_retrain_reference(
        forget_dataset=forget_dataset,
        feature_extractor=feature_extractor,
        retrain_model=retrain_model,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        temperature=temperature,
        forget_class=forget_class,
        class_to_forget=class_to_forget,
        target_mode=target_mode,
    )

    retrain_probs = retrain_ref["retrain_probs"]
    retrain_not_forget_mask = retrain_ref["retrain_not_forget_mask"]
    retrain_stats = retrain_ref["retrain_stats"]

    results = []
    prev_mean_probs = None

    kl_to_retrain_mean_values = []
    kl_to_retrain_std_values = []
    hist_kl_to_previous_mean_values = []
    hist_kl_to_previous_std_values = []

    argmax_match_mean_values = []
    argmax_match_std_values = []
    argmax_match_nonforget_mean_values = []
    argmax_match_nonforget_std_values = []

    argmax_match_to_previous_mean_values = []
    argmax_match_to_previous_std_values = []

    forget_class_accuracy_mean_values = []
    forget_class_accuracy_std_values = []

    num_epochs_trained_mean_values = []
    num_epochs_trained_std_values = []

    save_base_globally = True

    for n_support in n_support_values:
        support_data = process_single_support_size(
            dataset=dataset,
            forget_dataset=forget_dataset,
            retain_indices_full=retain_indices_full,
            forget_indices=forget_indices,
            feature_extractor=feature_extractor,
            n_support=n_support,
            teacher_type=teacher_type,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            output_dir=output_dir,
            num_workers=num_workers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            accuracy_threshold=accuracy_threshold,
            support_selection_mode=support_selection_mode,
            temperature=temperature,
            forget_class=forget_class,
            seed=seed,
            num_teacher_seeds=num_teacher_seeds,
            retrain_probs=retrain_probs,
            retrain_not_forget_mask=retrain_not_forget_mask,
            prev_mean_probs=prev_mean_probs,
            trunk_k=trunk_k,
            class_to_forget=class_to_forget,
            optimizer_type=optimizer_type,
            dataset_name=dataset_name,
            target_mode=target_mode,
        )

        save_base_globally = save_base_globally and support_data["save_base"]

        kl_to_retrain_mean_values.append(support_data["kl_to_retrain_mean"])
        kl_to_retrain_std_values.append(support_data["kl_to_retrain_std"])

        argmax_match_mean_values.append(support_data["argmax_match_mean"])
        argmax_match_std_values.append(support_data["argmax_match_std"])

        forget_class_accuracy_mean_values.append(
            support_data["forget_class_accuracy_mean"]
        )
        forget_class_accuracy_std_values.append(
            support_data["forget_class_accuracy_std"]
        )

        argmax_match_nonforget_mean_values.append(
            support_data["argmax_match_nonforget_mean"]
        )
        argmax_match_nonforget_std_values.append(
            support_data["argmax_match_nonforget_std"]
        )

        num_epochs_trained_mean_values.append(support_data["num_epochs_trained_mean"])
        num_epochs_trained_std_values.append(support_data["num_epochs_trained_std"])

        if support_data["result"]["hist_kl_to_previous_mean"] is not None:
            hist_kl_to_previous_mean_values.append(
                support_data["result"]["hist_kl_to_previous_mean"]
            )
            hist_kl_to_previous_std_values.append(
                support_data["result"]["hist_kl_to_previous_std"]
            )

        # if support_data["result"]["hist_kl_to_previous"] is not None:
        #     hist_kl_to_previous_values.append(
        #         support_data["result"]["hist_kl_to_previous"]
        #     )

        if support_data["result"]["argmax_match_to_previous_mean"] is not None:
            argmax_match_to_previous_mean_values.append(
                support_data["result"]["argmax_match_to_previous_mean"]
            )
            argmax_match_to_previous_std_values.append(
                support_data["result"]["argmax_match_to_previous_std"]
            )

        results.append(support_data["result"])
        prev_mean_probs = support_data["mean_probs"].clone()

    plot_paths = {}
    if len(n_support_values) >= 2:
        plot_paths = save_all_support_plots(
            output_dir=output_dir,
            n_support_values=n_support_values,
            retrain_probs=retrain_probs,
            kl_to_retrain_mean_values=kl_to_retrain_mean_values,
            kl_to_retrain_std_values=kl_to_retrain_std_values,
            argmax_match_mean_values=argmax_match_mean_values,
            argmax_match_std_values=argmax_match_std_values,
            argmax_match_nonforget_mean_values=argmax_match_nonforget_mean_values,
            argmax_match_nonforget_std_values=argmax_match_nonforget_std_values,
            hist_kl_to_previous_mean_values=hist_kl_to_previous_mean_values,
            hist_kl_to_previous_std_values=hist_kl_to_previous_std_values,
            argmax_match_to_previous_mean_values=argmax_match_to_previous_mean_values,
            argmax_match_to_previous_std_values=argmax_match_to_previous_std_values,
            forget_class_accuracy_mean_values=forget_class_accuracy_mean_values,
            forget_class_accuracy_std_values=forget_class_accuracy_std_values,
            num_epochs_trained_mean_values=num_epochs_trained_mean_values,
            num_epochs_trained_std_values=num_epochs_trained_std_values,
            class_to_forget=class_to_forget,
            trunk_k=trunk_k,
        )

    if not save_base_globally:
        return None

    summary = {
        "teacher_type": teacher_type,
        "num_teacher_seeds": int(num_teacher_seeds),
        "n_support_values": [int(x) for x in n_support_values],
        "results": results,
        "retrain_stats": retrain_stats,
        **plot_paths,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
