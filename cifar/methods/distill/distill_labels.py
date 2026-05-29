from __future__ import annotations

import time
from copy import deepcopy

import torch
import torch.nn.functional as F

from cifar.src.custom_types import *
from cifar.src.datasets import split_indices_by_forget_class
from cifar.src.models.backbones import get_model
from cifar.src.models.features import get_feature_extractor
from cifar.src.utils import get_default_device
from cifar.methods.utils import (
    resolve_output_dir,
    save_config,
    save_model,
    seed_everything,
    compute_eval_accuracy,
    warmup_cuda,
)
from cifar.methods.retain_selection import get_subsets
from cifar.methods.dataloaders import (
    build_online_teacher_mixed_loader,
    get_loader,
    build_eval_loaders,
)
from cifar.methods.distill.support_teachers import (
    select_support_indices,
    train_support_teacher,
    predict_soft_targets_with_teacher_on_batch,
    extract_cifar_resnet56_penultimate,
    NeighborSoftLabeler
)
from cifar.methods.distill.support_analytics import (
    train_support_teachers_and_measure_stability,
)
from cifar.methods.distill.utils import (
    build_parser,
    soft_cross_entropy_per_sample,
    get_unlearning_datasets,
)


def _build_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=weight_decay,
            weight_decay=weight_decay,
        )

    print("optimizer:", optimizer_type)
    return optimizer


def _prepare_subsets_and_support(
    model: torch.nn.Module,
    dataset,
    dataset_clean,
    class_to_forget: int,
    class_fraction_to_forget: float,
    retain_fraction: float,
    retain_selection_mode: str,
    support_selection_mode: str,
    n_support: int,
    batch_size: int,
    support_batch_size: int,
    num_workers: int,
    device: str,
    seed: int,
    feature_extractor_fn=None,
):
    forget_indices, retain_indices_full = split_indices_by_forget_class(
        dataset,
        class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
    )

    retain_subset, forget_subset = get_subsets(
        dataset=dataset,
        class_to_forget=class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
        retain_fraction=retain_fraction,
        seed=seed,
        retain_selection_mode=retain_selection_mode,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_extractor_fn=feature_extractor_fn,
    )

    retain_subset_clean, forget_subset_clean = get_subsets(
        dataset=dataset_clean,
        class_to_forget=class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
        retain_fraction=retain_fraction,
        seed=seed,
        retain_selection_mode=retain_selection_mode,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_extractor_fn=feature_extractor_fn,
    )

    start_time = time.time()
    support_indices, support_threshold = select_support_indices(
        dataset=dataset,
        retain_indices_full=retain_indices_full,
        forget_indices=forget_indices,
        n_support=n_support,
        feature_extractor=model,
        support_selection_mode=support_selection_mode,
        device=device,
        batch_size=support_batch_size,
        num_workers=num_workers,
        seed=seed,
        feature_extractor_fn=feature_extractor_fn,
    )
    support_time = time.time() - start_time

    return {
        "forget_indices": forget_indices,
        "retain_indices_full": retain_indices_full,
        "retain_subset": retain_subset,
        "forget_subset": forget_subset,
        "support_indices": support_indices,
        "support_threshold": support_threshold,
        "retain_subset_clean": retain_subset_clean,
        "forget_subset_clean": forget_subset_clean,
        "support_time": support_time,
    }


def _maybe_run_support_analytics(
    n_support_analytics,
    dataset,
    forget_dataset,
    retain_indices_full,
    forget_indices,
    model: torch.nn.Module,
    teacher_type: str,
    num_classes: int,
    support_epochs: int,
    support_batch_size: int,
    support_lr: float,
    support_weight_decay: float,
    teacher_accuracy_threshold: float,
    support_selection_mode: str,
    class_to_forget: int,
    seed: int,
    output_dir,
    num_workers: int,
    device: str,
    retrain_model,
    optimizer_type: str,
    trunk_k: int,
    dataset_name: str,
    target_mode: str,
) -> bool:
    if n_support_analytics is None:
        return False

    train_support_teachers_and_measure_stability(
        dataset=dataset,
        forget_dataset=forget_dataset,
        retain_indices_full=retain_indices_full,
        forget_indices=forget_indices,
        feature_extractor=model,
        n_support_values=n_support_analytics,
        teacher_type=teacher_type,
        num_classes=num_classes,
        epochs=support_epochs,
        batch_size=support_batch_size,
        lr=support_lr,
        weight_decay=support_weight_decay,
        device=device,
        output_dir=output_dir / "support_teacher_stability",
        num_workers=num_workers,
        accuracy_threshold=teacher_accuracy_threshold,
        support_selection_mode=support_selection_mode,
        forget_class=class_to_forget,
        seed=seed,
        retrain_model=retrain_model,
        optimizer_type=optimizer_type,
        trunk_k=trunk_k,
        dataset_name=dataset_name,
        target_mode=target_mode,
    )

    print("done")
    return True


def _compute_teacher_forget_class_prediction_rate(
    teacher,
    forget_eval_loader,
    feature_extractor: torch.nn.Module,
    class_to_forget: int,
    device: str,
) -> float | None:
    if teacher is None:
        return None

    start_time = time.time()
    teacher.model.eval()
    if teacher.kind == "embedding":
        feature_extractor.eval()

    total = 0
    num_predicted_forget_class = 0

    with torch.no_grad():
        for x, _ in forget_eval_loader:
            x = x.to(device)

            if teacher.kind == "embedding":
                x_features = extract_cifar_resnet56_penultimate(feature_extractor, x)
                logits = teacher.model(x_features)
            elif teacher.kind == "image":
                logits = teacher.model(x)
            else:
                raise ValueError(f"Unknown teacher kind: {teacher.kind}")

            preds = logits.argmax(dim=1)
            num_predicted_forget_class += (preds == class_to_forget).sum().item()
            total += preds.numel()

    if total == 0:
        return None

    forget_class_teacher_acc = num_predicted_forget_class / total
    print(f"Teacher Prediction Acc Forget class: {forget_class_teacher_acc}")
    return time.time() - start_time


def _prepare_teacher(
    mode: str,
    dataset,
    support_indices,
    model: torch.nn.Module,
    teacher_type: str,
    num_classes: int,
    support_epochs: int,
    support_batch_size: int,
    support_lr: float,
    support_weight_decay: float,
    support_hidden_dim: int,
    support_dropout: float,
    teacher_accuracy_threshold: float,
    optimizer_type: str,
    dataset_name: str,
    device: str,
    num_workers: int,
    seed: int,
    k_neighbors=None,
    feature_extractor_fn=None,
):
    teacher = None
    final_train_accuracy = None
    final_epoch = None
    teacher_kind = None

    if mode == "distill":
        teacher, final_train_accuracy, final_epoch = train_support_teacher(
            dataset=dataset,
            support_indices=support_indices,
            feature_extractor=model,
            teacher_type=teacher_type,
            num_classes=num_classes,
            epochs=support_epochs,
            batch_size=support_batch_size,
            lr=support_lr,
            weight_decay=support_weight_decay,
            device=device,
            num_workers=num_workers,
            hidden_dim=support_hidden_dim,
            accuracy_threshold=teacher_accuracy_threshold,
            dropout=support_dropout,
            optimizer_type=optimizer_type,
            dataset_name=dataset_name,
            seed=seed,
        )

        teacher_kind = teacher.kind

    elif mode == "neighbor":
        teacher = NeighborSoftLabeler(
            dataset=dataset,
            support_indices=support_indices,
            feature_extractor=model,
            num_classes=num_classes,
            k_neighbors=k_neighbors,
            batch_size=support_batch_size,
            device=device,
            num_workers=num_workers,
            feature_extractor_fn=feature_extractor_fn,
        )
        teacher_kind = teacher.kind
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return teacher, teacher_kind, final_train_accuracy, final_epoch


def _save_teacher_artifacts(
    teacher,
    teacher_kind: str,
    final_train_accuracy,
    final_epoch,
    support_threshold,
    output_dir,
) -> float:
    extra_time = 0.0

    if teacher is not None:
        extra_time_item = time.time()
        save_model(
            teacher.model,
            output_dir / "teacher_model",
            name="teacher_model.pth",
        )
        extra_time += time.time() - extra_time_item

    extra_time_item = time.time()
    save_config(
        {
            "teacher_final_train_accuracy": (
                None if final_train_accuracy is None else float(final_train_accuracy)
            ),
            "support_threshold": (
                None if support_threshold is None else float(support_threshold)
            ),
            "teacher_real_epochs": final_epoch,
            "teacher_kind": teacher_kind,
        },
        output_dir=output_dir / "teacher_model",
    )

    # save_argmax_softlabel_bar_chart(
    #     soft_targets=forget_soft_targets,
    #     class_names=(
    #         get_dataset_class_names("cifar100")
    #         if num_classes == 100
    #         else get_dataset_class_names("cifar10")
    #     ),
    #     output_path=output_dir / "argmax_softlabel_distribution_teacher.pdf",
    #     title="Argmax distribution of teacher soft labels on forget set",
    #     class_to_forget=class_to_forget,
    # )
    extra_time += time.time() - extra_time_item

    return extra_time


def _build_online_soft_targets(
    x: torch.Tensor,
    y: torch.Tensor,
    is_forget: torch.Tensor,
    num_classes: int,
    mode: str,
    teacher,
    feature_extractor: torch.nn.Module,
    temperature: float,
    softlabels_target_mode: str,
) -> torch.Tensor:
    soft_targets = F.one_hot(y, num_classes=num_classes).float()

    if not is_forget.any():
        return soft_targets

    if mode == "distill":
        soft_targets[is_forget] = predict_soft_targets_with_teacher_on_batch(
            x_batch=x[is_forget],
            teacher=teacher,
            feature_extractor=feature_extractor,
            temperature=temperature,
            target_mode=softlabels_target_mode,
        )
        return soft_targets

    if mode == "neighbor":
        soft_targets[is_forget] = teacher.predict_proba(
            x_batch=x[is_forget],
        ).to(soft_targets.device)
        return soft_targets

    raise ValueError(f"Unknown mode: {mode}")


def _run_unlearning_epoch(
    loader,
    unlearned_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    teacher,
    model: torch.nn.Module,
    num_classes: int,
    mode: str,
    temperature: float,
    retain_loss_weight: float,
    forget_loss_weight: float,
    softlabels_target_mode: str,
    device: str,
) -> None:

    unlearned_model.train()

    loss_cum_epoch = 0.0
    num_samples_epoch = 0

    for x, y, is_forget in loader:
        x = x.to(device)
        y = y.to(device)
        is_forget = is_forget.to(device).bool()

        soft_targets = _build_online_soft_targets(
            x=x,
            y=y,
            is_forget=is_forget,
            num_classes=num_classes,
            mode=mode,
            teacher=teacher,
            feature_extractor=model,
            temperature=temperature,
            softlabels_target_mode=softlabels_target_mode,
        )
        soft_targets = soft_targets.to(device)

        logits = unlearned_model(x)

        # losses = soft_cross_entropy_per_sample(
        #     logits=logits,
        #     soft_targets=soft_targets,
        # )
        # loss = losses.mean()

        loss = 0.0

        retain_mask = is_forget == 0
        forget_mask = is_forget == 1

        if retain_mask.any():
            retain_losses = soft_cross_entropy_per_sample(
                logits=logits[retain_mask],
                soft_targets=soft_targets[retain_mask],
            )
            loss = loss + retain_loss_weight * retain_losses.sum()

        if forget_mask.any():
            forget_losses = soft_cross_entropy_per_sample(
                logits=logits[forget_mask] / temperature,
                soft_targets=soft_targets[forget_mask],
            )
            loss = loss + forget_loss_weight * forget_losses.sum()

        loss = loss / is_forget.numel()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_cum_epoch += loss.item() * is_forget.numel()
        num_samples_epoch += is_forget.numel()

    mean_loss_epoch = loss_cum_epoch / max(num_samples_epoch, 1)
    print(f"Mean train loss this epoch: {mean_loss_epoch:.6f}")


def _run_unlearning_training(
    unlearned_model: torch.nn.Module,
    teacher,
    model: torch.nn.Module,
    loader,
    eval_loaders,
    num_classes: int,
    mode: str,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    retain_loss_weight: float,
    forget_loss_weight: float,
    class_to_forget: int,
    softlabels_target_mode: str,
    device: str,
    output_dir,
):
    distill_metrics = {eval_name: [] for eval_name in eval_loaders}

    best_score = float("inf")
    best_epoch = None
    extra_time = 0.0

    for epoch in range(num_epochs):
        print(f"Unlearning epoch {epoch + 1}/{num_epochs}")

        _run_unlearning_epoch(
            loader=loader,
            unlearned_model=unlearned_model,
            optimizer=optimizer,
            teacher=teacher,
            model=model,
            num_classes=num_classes,
            mode=mode,
            temperature=temperature,
            retain_loss_weight=retain_loss_weight,
            forget_loss_weight=forget_loss_weight,
            softlabels_target_mode=softlabels_target_mode,
            device=device,
        )

        extra_time_item = time.time()
        print(f"Epoch {epoch + 1}: ")
        for eval_name, eval_loader in eval_loaders.items():
            distill_metrics[eval_name].append(
                compute_eval_accuracy(unlearned_model, eval_loader, device)
            )
            print(f"{eval_name}={distill_metrics[eval_name][-1]:.4f}")

        # best_score, best_epoch = _update_best_model(
        #     current_score=current_score,
        #     best_score=best_score,
        #     best_epoch=best_epoch,
        #     epoch=epoch,
        #     unlearned_model=unlearned_model,
        #     output_dir=output_dir,
        # )
        extra_time += time.time() - extra_time_item

    # distill_metrics["best_score"] = best_score
    # distill_metrics["best_epoch"] = best_epoch
    return distill_metrics, extra_time


def unlearn_one_class(
    model: torch.nn.Module,
    dataset,
    dataset_clean,
    class_to_forget: int,
    class_fraction_to_forget: float,
    num_classes: int,
    num_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    retain_fraction: float,
    retain_selection_mode: str,
    mode: str,
    teacher_type: str,
    n_support: int,
    support_epochs: int,
    support_batch_size: int,
    support_lr: float,
    support_weight_decay: float,
    support_hidden_dim: int,
    support_dropout: float,
    temperature: float,
    retain_loss_weight: float,
    forget_loss_weight: float,
    support_selection_mode: str,
    teacher_accuracy_threshold: float,
    device: str,
    seed: int,
    num_workers: int,
    output_dir,
    test_dataset,
    softlabels_target_mode,
    n_support_analytics=None,
    optimizer_type="adamw",
    retrain_model=None,
    dataset_name: str = "cifar100",
    k_neighbors: int | None = None,
    trunk_k=100,
):
    torch.manual_seed(seed)

    feature_extractor_fn = get_feature_extractor(dataset_name)

    prepared = _prepare_subsets_and_support(
        model=model,
        dataset=dataset,
        dataset_clean=dataset_clean,
        class_to_forget=class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
        retain_fraction=retain_fraction,
        retain_selection_mode=retain_selection_mode,
        support_selection_mode=support_selection_mode,
        n_support=n_support,
        batch_size=batch_size,
        support_batch_size=support_batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
        feature_extractor_fn=feature_extractor_fn,
    )

    # for unlearning
    retain_indices_full = prepared["retain_indices_full"]
    retain_subset = prepared["retain_subset"]
    forget_subset = prepared["forget_subset"]
    support_indices = prepared["support_indices"]
    support_threshold = prepared["support_threshold"]

    # for eval
    retain_subset_clean = prepared["retain_subset_clean"]
    forget_subset_clean = prepared["forget_subset_clean"]

    eval_loaders = build_eval_loaders(
        forget_subset=forget_subset_clean,
        retain_subset=retain_subset_clean,
        test_dataset=test_dataset,
        class_to_forget=class_to_forget,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    unlearned_model = deepcopy(model).to(device)
    unlearned_model.train()

    optimizer = _build_optimizer(
        model=unlearned_model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_cuda(unlearned_model, device)
    start_time = time.time()
    extra_time = prepared["support_time"]

    # for support size analytics
    if _maybe_run_support_analytics(
        n_support_analytics=n_support_analytics,
        dataset=dataset,
        forget_dataset=forget_subset_clean,
        retain_indices_full=retain_indices_full,
        forget_indices=prepared["forget_indices"],
        model=model,
        teacher_type=teacher_type,
        num_classes=num_classes,
        support_epochs=support_epochs,
        support_batch_size=support_batch_size,
        support_lr=support_lr,
        support_weight_decay=support_weight_decay,
        teacher_accuracy_threshold=teacher_accuracy_threshold,
        support_selection_mode=support_selection_mode,
        class_to_forget=class_to_forget,
        seed=seed,
        output_dir=output_dir,
        num_workers=num_workers,
        device=device,
        retrain_model=retrain_model,
        optimizer_type=optimizer_type,
        trunk_k=trunk_k,
        dataset_name=dataset_name,
        target_mode=softlabels_target_mode,
    ):
        return None

    teacher, teacher_kind, final_train_accuracy, final_epoch = _prepare_teacher(
        mode=mode,
        dataset=dataset,
        support_indices=support_indices,
        model=model,
        teacher_type=teacher_type,
        num_classes=num_classes,
        support_epochs=support_epochs,
        support_batch_size=support_batch_size,
        support_lr=support_lr,
        support_weight_decay=support_weight_decay,
        support_hidden_dim=support_hidden_dim,
        support_dropout=support_dropout,
        teacher_accuracy_threshold=teacher_accuracy_threshold,
        optimizer_type=optimizer_type,
        dataset_name=dataset_name,
        device=device,
        num_workers=num_workers,
        seed=seed,
        feature_extractor_fn=feature_extractor_fn,
        k_neighbors=k_neighbors
    )
    if mode != "neighbor":
        extra_time += _compute_teacher_forget_class_prediction_rate(
            teacher=teacher,
            forget_eval_loader=eval_loaders["forget"],
            feature_extractor=model,
            class_to_forget=class_to_forget,
            device=device,
        )

        extra_time += _save_teacher_artifacts(
            teacher=teacher,
            teacher_kind=teacher_kind,
            final_train_accuracy=final_train_accuracy,
            final_epoch=final_epoch,
            support_threshold=support_threshold,
            output_dir=output_dir,
        )

    loader = build_online_teacher_mixed_loader(
        forget_subset=forget_subset,
        retain_subset=retain_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    distill_metrics, extra_eval_time = _run_unlearning_training(
        unlearned_model=unlearned_model,
        teacher=teacher,
        model=model,
        loader=loader,
        eval_loaders=eval_loaders,
        num_classes=num_classes,
        mode=mode,
        num_epochs=num_epochs,
        optimizer=optimizer,
        temperature=temperature,
        retain_loss_weight=retain_loss_weight,
        forget_loss_weight=forget_loss_weight,
        class_to_forget=class_to_forget,
        device=device,
        output_dir=output_dir,
        softlabels_target_mode=softlabels_target_mode,
    )
    extra_time += extra_eval_time

    rte = time.time() - start_time - extra_time

    print("rte: ", rte)
    distill_metrics["rte"] = rte
    save_config(distill_metrics, output_dir=output_dir, name="accuracies.json")

    return unlearned_model


def _load_retrain_model_if_needed(args, device: str):

    print(args.n_support_analytics, args.retrain_model_path)
    if args.n_support_analytics is None or args.retrain_model_path is None:
        return None

    retrain_model = get_model(
        dataset_name=args.dataset_name,
        device=device,
        model_path=args.retrain_model_path,
    )

    print(f"retrain is from {args.retrain_model_path}")
    return retrain_model


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = get_default_device()
    args.device = device
    seed_everything(args.seed)

    output_dir = resolve_output_dir(args)

    train_dataset, test_dataset, train_dataset_clean, num_classes = (
        get_unlearning_datasets(args)
    )

    model = get_model(
        dataset_name=args.dataset_name,
        device=args.device,
        model_path=args.model_path,
    )

    retrain_model = _load_retrain_model_if_needed(args, device)

    unlearned_model = unlearn_one_class(
        model=model,
        dataset=train_dataset,
        dataset_clean=train_dataset_clean,
        class_to_forget=args.class_to_forget,
        class_fraction_to_forget=args.class_fraction_to_forget,
        num_classes=num_classes,
        num_epochs=args.num_epochs,
        batch_size=args.unlearning_batch_size,
        lr=args.unlearning_lr,
        weight_decay=args.weight_decay,
        retain_fraction=args.retain_fraction,
        retain_selection_mode=args.retain_selection_mode,
        mode=args.mode,
        teacher_type=args.teacher_type,
        n_support=args.n_support,
        support_epochs=args.support_epochs,
        support_batch_size=args.support_batch_size,
        support_lr=args.support_lr,
        support_weight_decay=args.support_weight_decay,
        support_hidden_dim=args.support_hidden_dim,
        support_dropout=args.support_dropout,
        temperature=args.temperature,
        retain_loss_weight=args.retain_loss_weight,
        forget_loss_weight=args.forget_loss_weight,
        support_selection_mode=args.support_selection_mode,
        teacher_accuracy_threshold=args.teacher_accuracy_threshold,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=output_dir,
        n_support_analytics=args.n_support_analytics,
        optimizer_type=args.optimizer_type,
        retrain_model=retrain_model,
        trunk_k=args.trunk_k,
        dataset_name=args.dataset_name,
        test_dataset=test_dataset,
        softlabels_target_mode=args.softlabels_target_mode,
        k_neighbors=args.k_neighbors
    )

    save_config(args, output_dir)

    if unlearned_model is not None:
        save_model(unlearned_model, output_dir, name="unlearned_model.pth")


if __name__ == "__main__":
    main()
