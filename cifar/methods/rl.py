import torch.nn.functional as F
import torch
import time
from copy import deepcopy
from tqdm import tqdm
import argparse


from cifar.src.custom_types import *
from cifar.src.utils import get_default_device
from cifar.src.models.backbones import get_model
from cifar.methods.utils import (
    build_parser as build_parser_base,
    resolve_output_dir,
    save_config,
    save_model,
    compute_eval_accuracy,
    get_unlearning_datasets,
)
from cifar.methods.dataloaders import build_random_mixed_loader, build_eval_loaders
from cifar.methods.retain_selection import get_subsets

### UTILS ###


def build_parser() -> argparse.ArgumentParser:
    parser = build_parser_base()

    parser.add_argument(
        "--random-label-mode",
        type=str,
        default="any",
        choices=["wrong", "any"],
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--retain-fraction", type=float, default=1.0)
    return parser


### METHOD ###
def unlearn_one_class(
    model: torch.nn.Module,
    dataset,
    class_to_forget: int,
    class_fraction_to_forget: float = 1.0,
    num_classes: int = 100,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    random_label_mode: RandomLabelMode = "wrong",
    retain_fraction: float = 1.0,
    momentum: float = 0.9,
    num_workers: int = 4,
    device: str = "cpu",
    seed: int = 42,
    output_dir="",
):
    torch.manual_seed(seed)

    retain_subset, forget_subset = get_subsets(
        dataset,
        class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
        retain_fraction=retain_fraction,
        seed=seed,
    )

    eval_loaders = build_eval_loaders(
        forget_subset,
        retain_subset,
        class_to_forget,
        batch_size,
        2,
    )

    unlearned_model = deepcopy(model).to(device)
    unlearned_model.train()

    optimizer = torch.optim.SGD(
        unlearned_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    history = {eval_name: [] for eval_name in eval_loaders}

    start_time = time.time()
    extra_time = 0.0

    for epoch in tqdm(range(num_epochs), desc="Unlearning epochs"):
        # generate random labels on the fly for each epoch
        loader = build_random_mixed_loader(
            forget_subset,
            retain_subset,
            num_classes,
            random_label_mode=random_label_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed + epoch,
        )
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = unlearned_model(x)

            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            extra_time_item = time.time()

            print(f"Epoch {epoch + 1}: ")
            for eval_name, eval_loader in eval_loaders.items():
                history[eval_name].append(
                    compute_eval_accuracy(unlearned_model, eval_loader, device)
                )
                print(f"{eval_name}={history[eval_name][-1]:.4f}\n")

            extra_time += time.time() - extra_time_item

    history["rte"] = time.time() - start_time - extra_time
    print(
        f"Unlearning completed in {history['rte']:.2f} seconds not including {extra_time:.2f} seconds of extra evaluation time."
    )

    save_config(history, output_dir=output_dir, name="accuracies.json")

    return unlearned_model


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = get_default_device()
    args.device = device

    output_dir = resolve_output_dir(args)

    ### DATASETS ###
    train_dataset, test_dataset, num_classes = get_unlearning_datasets(args)

    ### MODEL SETUP ###
    model = get_model(
        dataset_name=args.dataset_name, device=args.device, model_path=args.model_path
    )

    ### UNLEARNING ###
    unlearned_model = unlearn_one_class(
        model=model,
        dataset=train_dataset,
        class_to_forget=args.class_to_forget,
        class_fraction_to_forget=args.class_fraction_to_forget,
        num_classes=num_classes,
        num_epochs=args.num_epochs,
        batch_size=args.unlearning_batch_size,
        lr=args.unlearning_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        num_workers=args.num_workers,
        random_label_mode=args.random_label_mode,
        retain_fraction=args.retain_fraction,
        device=device,
        seed=args.seed,
        output_dir=output_dir,
    )

    save_config(args, output_dir)
    save_model(unlearned_model, output_dir, name="unlearned_model.pth")


if __name__ == "__main__":
    main()
