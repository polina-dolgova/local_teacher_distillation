import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List

from mnist.src.evaluate import compare_models
from mnist.src.plot import plot_support_set_class_distribution
from mnist.src.unlearn.support_model import (
    get_forget_dataloader_with_distill_softlabels,
)
from mnist.src.unlearn.neighbours_labels import (
    get_forget_dataloader_with_neighbor_softlabels,
)


def unlearn_class_with_distill(
    model: torch.nn.Module,
    class_to_forget: int,
    support_size: int,
    support_selection: str = "cosine",
    teacher_selection: str = "model",
    support_feature_space: str = "raw",
    label_mode: str = "per_sample",
    teacher_epochs: int = 3,
    teacher_batch_size: int = 128,
    teacher_lr: float = 1e-3,
    teacher_weight_decay: float = 0.0,
    unlearn_steps: int = 200,
    unlearn_batch_size: int = 128,
    unlearn_lr: float = 1e-4,
    unlearn_weight_decay: float = 1e-6,
    temperature: float = 1.0,
    zero_forget_class_prob: bool = False,
    k_neighbors: int | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.nn.Module, List[float], torch.nn.Module, list[int]]:
    """
    Unlearn one class using support-set distillation.

    Pipeline:
    1) Select support subset from retain data.
    2) Train a small teacher model on that subset.
    3) Infer teacher soft targets on forget-class samples.
    4) Fine-tune the original model on forget-class samples using KL distillation loss.

    Returns:
        unlearned_model,
        losses,
        teacher_model,
        support_indices
    """
    device = torch.device(device)

    if teacher_selection == "model":
        forget_distill_loader, support_indices = (
            get_forget_dataloader_with_distill_softlabels(
                class_to_forget=class_to_forget,
                support_size=support_size,
                support_selection=support_selection,
                support_feature_space=support_feature_space,
                feature_model=model,
                teacher_epochs=teacher_epochs,
                teacher_batch_size=teacher_batch_size,
                teacher_lr=teacher_lr,
                teacher_weight_decay=teacher_weight_decay,
                unlearn_batch_size=unlearn_batch_size,
                temperature=temperature,
                zero_forget_class_prob=zero_forget_class_prob,
                device=device,
                seed=seed,
            )
        )
    elif teacher_selection == "neighborhood":
        forget_distill_loader, support_indices = (
            get_forget_dataloader_with_neighbor_softlabels(
                class_to_forget=class_to_forget,
                support_size=support_size,
                support_selection=support_selection,
                support_feature_space=support_feature_space,
                feature_model=model,
                k_neighbors=k_neighbors,
                label_mode=label_mode,
                unlearn_batch_size=unlearn_batch_size,
                device=device,
                seed=seed,
            )
        )

    else:
        raise NotImplementedError(
            f"teacher selection {teacher_selection} not implemented"
        )

    unlearned_model = deepcopy(model).to(device)
    unlearned_model.train()

    optimizer = torch.optim.AdamW(
        unlearned_model.parameters(),
        lr=unlearn_lr,
        weight_decay=unlearn_weight_decay,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")

    losses: List[float] = []
    loader_iter = iter(forget_distill_loader)

    for _ in range(unlearn_steps):
        try:
            x_forget, teacher_probs = next(loader_iter)
        except StopIteration:
            loader_iter = iter(forget_distill_loader)
            x_forget, teacher_probs = next(loader_iter)

        x_forget = x_forget.to(device)
        teacher_probs = teacher_probs.to(device)

        student_logits = unlearned_model(x_forget)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

        loss = criterion(student_log_probs, teacher_probs) * (temperature**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return unlearned_model, support_indices


def unlearn_with_distilled_labels(
    model: torch.nn.Module,
    retrain_model: torch.nn.Module,
    similarity_matrices: dict,
    class_to_forget: int,
    support_size: int,
    support_selection: str = "cosine",
    teacher_selection: str = "model",
    support_feature_space: str = "raw",
    label_mode: str = "per_sample",
    teacher_epochs: int = 3,
    teacher_batch_size: int = 128,
    teacher_lr: float = 1e-3,
    teacher_weight_decay: float = 0.0,
    unlearn_steps: int = 200,
    unlearn_batch_size: int = 128,
    unlearn_lr: float = 1e-4,
    unlearn_weight_decay: float = 1e-6,
    temperature: float = 1.0,
    zero_forget_class_prob: bool = False,
    k_neighbor: int | None = None,
    device: str = "cpu",
    seed: int = 42,
    output_dir: str = "",
):
    unlearned_distill_model, support_indices = unlearn_class_with_distill(
        model=model,
        class_to_forget=class_to_forget,
        support_size=support_size,
        support_selection=support_selection,
        support_feature_space=support_feature_space,
        teacher_selection=teacher_selection,
        teacher_epochs=teacher_epochs,
        teacher_batch_size=teacher_batch_size,
        teacher_lr=teacher_lr,
        teacher_weight_decay=teacher_weight_decay,
        unlearn_steps=unlearn_steps,
        unlearn_batch_size=unlearn_batch_size,
        unlearn_lr=unlearn_lr,
        unlearn_weight_decay=unlearn_weight_decay,
        temperature=temperature,
        zero_forget_class_prob=zero_forget_class_prob,
        label_mode=label_mode,
        k_neighbors=k_neighbor,
        device=device,
        seed=seed,
    )

    run_name = f"{support_selection}_{support_feature_space}"

    compare_models(
        model=retrain_model,
        unlearned_models_dict={"distill": [unlearned_distill_model]},
        class_to_forget=class_to_forget,
        similarity_matrices=similarity_matrices,
        output_dir=output_dir / run_name,
        device=device,
    )

    plot_support_set_class_distribution(
        support_indices=support_indices,
        class_to_forget=class_to_forget,
        out_dir=output_dir / run_name / "support_distribution",
        prefix=f"support_set",
    )
