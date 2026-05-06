from __future__ import annotations

import time
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cifar.methods.utils import (
    build_parser as build_parser_base,
    save_config,
    save_model,
    seed_everything,
    resolve_output_dir,
    compute_eval_accuracy,
)
from cifar.methods.dataloaders import (
    build_random_mixed_loader,
    build_eval_loaders,
    get_loader,
)
from cifar.methods.retain_selection import get_subsets

from cifar.src.datasets import (
    build_cifar_dataset,
    get_dataset_class_names,
    split_indices_by_forget_class,
)
from cifar.src.models.backbones import get_cifar_resnet56
from cifar.src.utils import get_default_device
from cifar.src.custom_types import *


### UTILS ###
def build_parser() -> argparse.ArgumentParser:
    parser = build_parser_base()
    parser.description = "SALUN unlearning"

    parser.add_argument(
        "--random-label-mode",
        type=str,
        default="any",
        choices=["wrong", "any"],
    )
    parser.add_argument("--retain-fraction", type=float, default=1.0)
    parser.add_argument("--mask-path", type=str, default=None)
    parser.add_argument("--mask-topk-ratio", type=float, default=0.5)
    parser.add_argument("--mask-batch-size", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--retain-selection-mode", type=str, default="random")

    return parser


def save_mask(mask: dict[str, torch.Tensor], output_dir: Path, ratio: float) -> None:
    torch.save(mask, output_dir / f"with_{ratio}.pt")


### METHOD ###


def build_salun_mask(
    model: torch.nn.Module,
    dataset,
    class_to_forget: int,
    class_fraction_to_forget: float,
    batch_size: int,
    topk_ratio: float,
    device: str,
    num_workers: int,
) -> dict[str, torch.Tensor]:

    forget_indices, _ = split_indices_by_forget_class(
        dataset, class_to_forget, class_fraction_to_forget
    )
    forget_subset = Subset(dataset, forget_indices)
    forget_loader = DataLoader(
        forget_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()

    gradients: dict[str, torch.Tensor] = {}
    for name, param in reference_model.named_parameters():
        gradients[name] = torch.zeros_like(param, device=device)

    for image, target in tqdm(forget_loader, desc="Building SALUN mask"):
        image = image.to(device)
        target = target.to(device)

        output_clean = reference_model(image)
        loss = -criterion(output_clean, target)

        reference_model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            for name, param in reference_model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])
        threshold_index = int(len(all_elements) * topk_ratio)

        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        hard_dict: dict[str, torch.Tensor] = {}

        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index : start_index + num_elements]

            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)

            hard_dict[key] = threshold_tensor.cpu()
            start_index += num_elements

    return hard_dict


def apply_mask_to_grads(
    model: torch.nn.Module,
    mask: dict[str, torch.Tensor],
) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name not in mask:
            continue

        mask_tensor = mask[name].to(device=param.device, dtype=param.grad.dtype)
        param.grad.mul_(mask_tensor)


def restore_masked_params(
    model: torch.nn.Module,
    mask: dict[str, torch.Tensor],
    theta0: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue

            mask_tensor = mask[name].to(device=param.device, dtype=param.dtype)
            inv_mask_tensor = 1 - mask_tensor

            if torch.count_nonzero(inv_mask_tensor) == 0:
                continue

            param.data.mul_(mask_tensor).add_(
                theta0[name].to(device=param.device, dtype=param.dtype)
                * inv_mask_tensor
            )

            state = optimizer.state.get(param, None)
            if state is not None and "momentum_buffer" in state:
                state["momentum_buffer"].mul_(mask_tensor)


def unlearn_one_class_salun(
    model: torch.nn.Module,
    dataset,
    dataset_clean,
    class_to_forget: int,
    class_fraction_to_forget: float,
    num_classes: int,
    num_epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    random_label_mode: str,
    retain_fraction: float,
    device: str,
    seed: int,
    num_workers: int,
    test_dataset,
    retain_selection_mode: RetainSelectionMode = "random",
    mask_batch_size: int = 256,
    mask_topk_ratio: float = 0.1,
    output_dir="",
) -> tuple[torch.nn.Module, list[float]]:

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
    )

    eval_loaders = build_eval_loaders(
        forget_subset=forget_subset_clean,
        retain_subset=retain_subset_clean,
        class_to_forget=class_to_forget,
        batch_size=batch_size,
        num_workers=4,
        test_dataset=test_dataset,
    )

    unlearned_model = copy.deepcopy(model).to(device)
    unlearned_model.train()

    optimizer = torch.optim.SGD(
        unlearned_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history = {eval_name: [] for eval_name in eval_loaders}

    start_time = time.time()
    extra_time = 0.0

    mask = build_salun_mask(
        model=model,
        dataset=dataset,
        class_to_forget=class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
        batch_size=mask_batch_size,
        topk_ratio=mask_topk_ratio,
        device=device,
        num_workers=num_workers,
    )

    theta0 = None
    if mask:
        with torch.no_grad():
            theta0 = {
                name: param.detach().clone()
                for name, param in unlearned_model.named_parameters()
                if name in mask
            }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        forget_loader = build_random_mixed_loader(
            forget_subset,
            retain_subset,
            num_classes,
            random_label_mode=random_label_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=epoch + seed,
        )
        for image, target in forget_loader:

            image = image.to(device)
            target = target.to(device)

            output_clean = unlearned_model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if mask:
                apply_mask_to_grads(unlearned_model, mask)

            optimizer.step()

            if mask:
                restore_masked_params(unlearned_model, mask, theta0, optimizer)

        if (epoch + 1) % 5 == 0:
            extra_time_item = time.time()

            print(f"Epoch {epoch + 1}: ")
            for eval_name, eval_loader in eval_loaders.items():
                history[eval_name].append(
                    compute_eval_accuracy(unlearned_model, eval_loader, device)
                )
                print(f"{eval_name}={history[eval_name][-1]:.4f}")

            extra_time += time.time() - extra_time_item

    history["rte"] = time.time() - start_time - extra_time

    print(
        "forget_loader_acc:",
        compute_eval_accuracy(unlearned_model, forget_loader, device),
    )
    print(
        f"Unlearning completed in {history['rte']:.2f} seconds not including {extra_time:.2f} seconds of extra evaluation time."
    )

    save_config(history, output_dir=output_dir, name="accuracies.json")
    return unlearned_model


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = get_default_device()
    args.device = device
    seed_everything(args.seed)

    output_dir = resolve_output_dir(args)

    class_names = get_dataset_class_names(args.dataset_name)
    num_classes = len(class_names)

    train_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=True)
    test_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=False)
    train_dataset_clean = build_cifar_dataset(
        dataset_name=args.dataset_name, train=True, do_transforms=False
    )

    model = get_cifar_resnet56(
        dataset_name=args.dataset_name, device=args.device, model_path=args.model_path
    )

    unlearned_model = unlearn_one_class_salun(
        model=model,
        dataset=train_dataset,
        class_to_forget=args.class_to_forget,
        class_fraction_to_forget=args.class_fraction_to_forget,
        num_classes=num_classes,
        num_epochs=args.num_epochs,
        batch_size=args.unlearning_batch_size,
        lr=args.unlearning_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        random_label_mode=args.random_label_mode,
        retain_fraction=args.retain_fraction,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
        retain_selection_mode=args.retain_selection_mode,
        mask_batch_size=args.mask_batch_size,
        mask_topk_ratio=args.mask_topk_ratio,
        output_dir=output_dir,
        test_dataset=test_dataset,
        dataset_clean=train_dataset_clean,
    )

    save_config(args, output_dir)
    save_model(unlearned_model, output_dir, name="unlearned_model.pth")


if __name__ == "__main__":
    main()
