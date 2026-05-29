from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from cifar.src.models.features import extract_cifar_resnet56_penultimate, get_feature_extractor
from cifar.src.visualization.plots import (
    save_pointwise_topk_scatter_plot,
    save_similarity_survival_plots,
    save_binned_similarity_vs_accuracy_drop_plot,
)


def _pin_memory_for_device(device: str) -> bool:
    return str(device).startswith("cuda")


def _compute_accuracy_and_correct_mask(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[float | None, torch.Tensor]:
    """
    Compute dataset accuracy and per-sample correctness mask.
    """
    if dataset is None or len(dataset) == 0:
        return None, torch.empty(0, dtype=torch.bool)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory_for_device(device),
    )

    model.eval()
    all_correct = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)
            probs = F.softmax(logits, dim=1)
            all_correct.append((pred == y).cpu())
            all_probs.append(probs.cpu())

    correct_mask = torch.cat(all_correct, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    if len(correct_mask) == 0:
        return None, correct_mask, torch.empty((0, 0), dtype=torch.float32)

    accuracy = float(correct_mask.float().mean().item())
    return accuracy, correct_mask, all_probs


def _compute_similarity_to_forget_set(
    full_model: torch.nn.Module,
    retain_dataset,
    forget_dataset,
    batch_size: int,
    num_workers: int,
    device: str,
    feature_extractor_fn=None,
) -> torch.Tensor:
    """
    Compute cosine similarity of each retain point to the mean embedding
    of the forget dataset using ResNet-56 penultimate features.
    """
    if feature_extractor_fn is None:
        feature_extractor_fn = extract_cifar_resnet56_penultimate

    if len(retain_dataset) == 0:
        return torch.empty(0, dtype=torch.float32)
    if len(forget_dataset) == 0:
        raise ValueError("Forget dataset is empty, cannot compute similarities.")

    def compute_embeddings(dataset) -> torch.Tensor:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_pin_memory_for_device(device),
        )

        full_model.eval()
        all_embeddings = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                emb = feature_extractor_fn(full_model, x)
                all_embeddings.append(emb.cpu())

        if not all_embeddings:
            return torch.empty((0, 0), dtype=torch.float32)

        return torch.cat(all_embeddings, dim=0)

    retain_embeddings = compute_embeddings(retain_dataset)
    forget_embeddings = compute_embeddings(forget_dataset)

    retain_embeddings = F.normalize(retain_embeddings.float(), dim=1)
    forget_center = forget_embeddings.float().mean(dim=0, keepdim=True)
    forget_center = F.normalize(forget_center, dim=1)

    similarities = (retain_embeddings @ forget_center.T).squeeze(1)
    return similarities.cpu()


def _extract_targets(dataset) -> torch.Tensor:
    """
    Extract targets from an arbitrary dataset or subset.
    """
    targets = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        targets.append(int(y))
    return torch.tensor(targets, dtype=torch.long)


def _compute_pointwise_true_class_confidence_drop(
    full_probs: torch.Tensor,
    unlearned_probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Per-point drop in probability of the true class:
        full_probs[y_true] - unlearned_probs[y_true]

    Output range: [-1, 1]
        > 0  : unlearning reduced confidence in the true class
        = 0  : no change
        < 0  : unlearning increased confidence in the true class
    """
    idx = torch.arange(targets.shape[0], device=targets.device)
    full_true = full_probs[idx, targets.long()]
    unlearn_true = unlearned_probs[idx, targets.long()]
    return full_true - unlearn_true


def _compute_pointwise_accuracy_drop(
    retrain_corrects: list[torch.Tensor],
    unlearned_correct: torch.Tensor,
) -> torch.Tensor:
    """
    Return pointwise drop over retrain-consensus points:
        NaN : retrain models disagree
        0   : retrain models agree and unlearned_correct is the same
        1   : retrain models agree and unlearned_correct is different
    """
    retrain_stack = torch.stack(
        [x.bool() for x in retrain_corrects],
        dim=0,
    )

    num_retrains = retrain_stack.shape[0]
    num_correct = retrain_stack.sum(dim=0)

    all_agree = (num_correct == 0) | (num_correct == num_retrains)
    consensus_correct = retrain_stack[0]

    drop = (consensus_correct != unlearned_correct.bool()).float()
    drop[~all_agree] = torch.nan

    return drop


def get_retain_same_class_metrics(retain_targets, class_to_forget, accuracies):

    retain_same_forget_class_idx = (
        (retain_targets == class_to_forget).nonzero(as_tuple=True)[0].cpu().numpy()
    )

    if len(retain_same_forget_class_idx) == 0:
        retain_same_class_metrics = {
            "count": 0,
            "before_unlearning": None,
            "after_unlearning": None,
            "accuracy_drop": None,
        }
    else:
        retain_same_class_metrics = {"count": int(len(retain_same_forget_class_idx))}

        for name in accuracies:
            retain_same_class_metrics[name] = float(
                accuracies[name]["retain_correct"][retain_same_forget_class_idx]
                .float()
                .mean()
                .item()
            )

        # retain_same_class_metrics["accuracy_drop"] = float(
        #     retain_same_class_metrics["retrain"] - retain_same_class_metrics["unlearn"]
        # )

        # retain_same_class_metrics["accuracy_drop_vs_full"] = float(
        #     retain_same_class_metrics["full"] - retain_same_class_metrics["unlearn"]
        # )

    return retain_same_class_metrics


def compute_stats(
    models,
    datasets,
    batch_size,
    num_workers,
    device,
):
    accuracies = {}
    for name, model in models.items():
        accuracies[name] = {}

        for dataset_name, dataset in datasets.items():
            acc, correct, probs = _compute_accuracy_and_correct_mask(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
            accuracies[name].update(
                {
                    f"{dataset_name}_acc": acc,
                    f"{dataset_name}_correct": correct,
                    f"{dataset_name}_probs": probs,
                }
            )

    return accuracies


def collect_extra_metrics(
    unlearned_model: torch.nn.Module,
    full_model: torch.nn.Module,
    retain_dataset,
    forget_dataset,
    test_dataset,
    class_to_forget: int,
    point_topk_fraction: float,
    batch_size: int = 256,
    num_workers: int = 2,
    device: str = "cpu",
    point_topk_fractions=[0.005, 0.01, 0.05, 0.1],
    save_plots: bool = True,
    plots_dir=None,
    feature_extractor_fn=None,
) -> dict:
    """
    Compute additional evaluation metrics:
    1) retain accuracy
    2) forget accuracy
    3) accuracy on retain points whose true label equals the forget class
    4) pointwise metrics on the union of:
       - top k% closest retain points to forget set
       - top k% farthest retain points from forget set

       Correlation is computed between:
       - pointwise accuracy drop in {-1, 0, 1}
       - pointwise similarity to forget set
    5) test accuracy
    """
    if not (0.0 < point_topk_fraction <= 0.5):
        raise ValueError("point_topk_fraction must be in the interval (0, 0.5].")

    models = {"full": full_model, "unlearn": unlearned_model}
    # retrain_models_dict = {
    #     f"retrain_{i}": retrain_model for i, retrain_model in enumerate(retrain_models)
    # }
    # models.update(retrain_models_dict)

    datasets = {
        "retain": retain_dataset,
        "forget": forget_dataset,
        "test": test_dataset,
    }
    accuracies = compute_stats(
        models,
        datasets,
        batch_size,
        num_workers,
        device,
    )

    retain_targets = _extract_targets(retain_dataset)
    retain_same_class_metrics = get_retain_same_class_metrics(
        retain_targets, class_to_forget, accuracies
    )

    # similarities by embedding from the full model
    similarities = _compute_similarity_to_forget_set(
        full_model=full_model,
        retain_dataset=retain_dataset,
        forget_dataset=forget_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        feature_extractor_fn=feature_extractor_fn,
    )
    # if save_plots and plots_dir is not None:
    #     save_similarity_survival_plots(
    #         similarities=similarities,
    #         output_dir=plots_dir,
    #         title_suffix="pointwise_similarity_survival",
    #         num_grid_points=500,
    #     )

    # accuracy difference between retrain and unlearning
    # retrain_corrects = [
    #     accuracies[retrain_name]["retain_correct"]
    #     for retrain_name in retrain_models_dict
    # ]

    pointwise_drop = _compute_pointwise_true_class_confidence_drop(
        full_probs=accuracies["full"]["retain_probs"],
        unlearned_probs=accuracies["unlearn"]["retain_probs"],
        targets=retain_targets,
    )

    n_points = len(retain_dataset)
    if n_points == 0:
        pointwise_metrics = {
            "topk_fraction_each_side": float(point_topk_fraction),
            "k_points_each_side": 0,
            "num_selected_points_total": 0,
            "mean_accuracy_drop": None,
            "corr_drop_vs_similarity": None,
            "mean_similarity": None,
        }
    else:

        pointwise_metrics = []

        # order_desc = torch.argsort(similarities, descending=True).cpu().numpy()
        # for point_topk_fraction in point_topk_fractions:
        #     k = max(1, int(n_points * point_topk_fraction))

        #     closest_idx = order_desc[:k]
        #     farthest_idx = order_desc[-k:]

        #     selected_idx = np.concatenate([closest_idx, farthest_idx])

        #     selected_similarities = similarities[selected_idx].cpu().numpy()
        #     selected_drop = pointwise_drop[selected_idx].cpu().numpy()
        #     selected_targets = retain_targets[selected_idx].cpu().numpy()

        #     closest_drop = pointwise_drop[closest_idx].cpu().numpy()
        #     farthest_drop = pointwise_drop[farthest_idx].cpu().numpy()

        #     pointwise_metrics.append(
        #         {
        #             "topk_fraction_each_side": float(point_topk_fraction),
        #             "k_points_each_side": int(k),
        #             "num_selected_points_total": int(len(selected_idx)),
        #             "corr_drop_vs_similarity": _safe_corr(
        #                 selected_similarities,
        #                 selected_drop,
        #             ),
        #             "mean_similarity": float(selected_similarities.mean()),
        #             "closest_accuracy_drop": {
        #                 "mean": float(closest_drop.mean()),
        #                 "std": float(closest_drop.std()),
        #             },
        #             "farthest_accuracy_drop": {
        #                 "mean": float(farthest_drop.mean()),
        #                 "std": float(farthest_drop.std()),
        #             },
        #         }
        #     )

        #     if save_plots and plots_dir is not None:
        #         save_pointwise_topk_scatter_plot(
        #             selected_similarities=selected_similarities,
        #             selected_drop=selected_drop,
        #             selected_targets=selected_targets,
        #             class_to_forget=class_to_forget,
        #             output_path=Path(plots_dir)
        #             / f"pointwise_scatter_topk_{point_topk_fraction:.4f}.png",
        #             point_topk_fraction=point_topk_fraction,
        #         )

        if save_plots and plots_dir is not None:

            for num_bins in [10]:
                save_binned_similarity_vs_accuracy_drop_plot(
                    similarities=similarities,
                    pointwise_drop=pointwise_drop,
                    output_path=Path(plots_dir)
                    / f"binned_similarity_vs_confidence_drop_{num_bins}bins.pdf",
                    output_path_stats=Path(plots_dir)
                    / f"binned_similarity_vs_confidence_drop_stats_{num_bins}bins.json",
                    num_bins=num_bins,
                )

                save_binned_similarity_vs_accuracy_drop_plot(
                    similarities=similarities,
                    pointwise_drop=1 - accuracies["unlearn"]["retain_correct"].int(),
                    output_path=Path(plots_dir)
                    / f"binned_similarity_vs_accuracy_drop_{num_bins}bins.pdf",
                    output_path_stats=Path(plots_dir)
                    / f"binned_similarity_vs_accuracy_drop_stats_{num_bins}bins.json",
                    num_bins=num_bins,
                )

    results = {
        "retain_accuracy": {},
        "forget_accuracy": {},
        "retain_subset_true_label_equals_forget_class": retain_same_class_metrics,
    }

    official_names = {"full": "before_unlearning", "unlearn": "after_unlearning"}
    # official_names.update({name: f"after_{name}" for name in retrain_models_dict})

    for model_name in accuracies:
        official_name = official_names[model_name]
        results["retain_accuracy"][official_name] = accuracies[model_name]["retain_acc"]
        results["forget_accuracy"][official_name] = accuracies[model_name]["forget_acc"]

    return results
