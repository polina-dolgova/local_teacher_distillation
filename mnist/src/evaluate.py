import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path

from mnist.src.plot import (
    plot_accuracy_drop_per_class,
    plot_accuracy_per_class,
    plot_sq_logit_gap_per_class,
    plot_similarity_vs_accuracy_drop_pretty,
)


def build_loader_for_single_class(
    dataset,
    target_class: int,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    """
    Build a DataLoader containing only samples of one class.
    """
    indices = [i for i, (_, y) in enumerate(dataset) if int(y) == target_class]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model_pair_per_class(
    model: torch.nn.Module,
    unlearned_model: torch.nn.Module,
    dataset,
    batch_size: int = 256,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Evaluate original and unlearned models on each class separately.

    Metrics per class:
    - original accuracy
    - unlearned accuracy
    - accuracy drop
    - mean squared logit gap over all logits
    - mean squared logit gap on the true-class logit only

    Returns:
        DataFrame with one row per class.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    unlearned_model = unlearned_model.to(device).eval()

    rows = []

    with torch.no_grad():
        for cls in range(10):
            loader = build_loader_for_single_class(
                dataset=dataset,
                target_class=cls,
                batch_size=batch_size,
                shuffle=False,
            )

            total = 0
            correct_model = 0
            correct_unlearned = 0

            sq_gap_all_sum = 0.0
            sq_gap_true_sum = 0.0

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                logits_model = model(x)
                logits_unlearned = unlearned_model(x)

                pred_model = logits_model.argmax(dim=1)
                pred_unlearned = logits_unlearned.argmax(dim=1)

                correct_model += (pred_model == y).sum().item()
                correct_unlearned += (pred_unlearned == y).sum().item()
                total += x.shape[0]

                # Mean squared gap over all logits
                sq_gap_all = ((logits_unlearned - logits_model) ** 2).mean(dim=1)
                sq_gap_all_sum += sq_gap_all.sum().item()

                # Squared gap on the true-class logit
                true_logits_model = logits_model.gather(1, y.unsqueeze(1)).squeeze(1)
                true_logits_unlearned = logits_unlearned.gather(
                    1, y.unsqueeze(1)
                ).squeeze(1)
                sq_gap_true = (true_logits_unlearned - true_logits_model) ** 2
                sq_gap_true_sum += sq_gap_true.sum().item()

            if total == 0:
                raise ValueError(f"No samples found for class {cls}.")

            acc_model = correct_model / total
            acc_unlearned = correct_unlearned / total

            rows.append(
                {
                    "class": cls,
                    "n_samples": total,
                    "acc_model": acc_model,
                    "acc_unlearned": acc_unlearned,
                    "acc_drop": acc_model - acc_unlearned,
                    "mean_sq_logit_gap_all": sq_gap_all_sum / total,
                    "mean_sq_logit_gap_true": sq_gap_true_sum / total,
                }
            )

    return pd.DataFrame(rows)


def compare_models(
    model,
    unlearned_models_dict,
    class_to_forget,
    output_dir,
    similarity_matrices={},
    device="mps",
    tag="",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    df_metrics = {}
    for name, unlearned_models in unlearned_models_dict.items():
        df_metrics[name] = []
        for unlearned_model in unlearned_models:
            df_metrics_per_model = evaluate_model_pair_per_class(
                model=model,
                unlearned_model=unlearned_model,
                dataset=test_dataset,
                batch_size=256,
                device=device,
            )
            df_metrics[name].append(df_metrics_per_model)

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # df_metrics_per_model.to_csv(
            #     output_dir / f"{tag}per_class_metrics.csv", index=False
            # )

            # --- save plots ---
        plot_accuracy_per_class(
            df_metrics[name][0],
            out_path=output_dir / f"{tag}-{name}-accuracy_per_class.png",
        )

        # plot_accuracy_drop_per_class(
        #     df_metrics[name][0],
        #     out_path=output_dir / f"{tag}-{name}-accuracy_drop.png",
        #     forgotten_class=class_to_forget,
        # )

        # plot_sq_logit_gap_per_class(
        #     df_metrics[name][0],
        #     out_path=output_dir / f"{tag}-{name}-logit_gap.png",
        # )

    for name, similarity_matrix in similarity_matrices.items():
        plot_similarity_vs_accuracy_drop_pretty(
            similarity_matrix,
            df_metrics,
            class_to_forget,
            out_path=output_dir / f"{tag}similarity_vs_acc_drop_{name}.pdf",
        )
