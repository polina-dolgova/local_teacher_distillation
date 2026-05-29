import torch.nn.functional as F
import torch
from copy import deepcopy
from tqdm import tqdm

from cifar.src.custom_types import *
from cifar.src.datasets import build_cifar_dataset
from cifar.src.utils import get_default_device
from cifar.src.models.backbones import get_model
from cifar.methods.utils import (
    build_parser,
    resolve_output_dir,
    save_config,
    save_model,
    compute_eval_accuracy,
    get_unlearning_datasets,
    warmup_cuda,
)
from cifar.methods.retain_selection import get_subsets
from cifar.methods.dataloaders import build_separate_dataloaders, build_eval_loaders
import time

### METHOD ###


def unlearn_one_class(
    model: torch.nn.Module,
    dataset,
    dataset_clean,
    test_dataset,
    class_to_forget: int,
    class_fraction_to_forget: float = 1.0,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    device: str = "cpu",
    seed: int = 42,
    num_workers=4,
    output_dir="",
):

    torch.manual_seed(seed)

    retain_loader, forget_loader = build_separate_dataloaders(
        dataset, class_to_forget, class_fraction_to_forget, batch_size
    )

    retain_subset_clean, forget_subset_clean = get_subsets(
        dataset=dataset_clean,
        class_to_forget=class_to_forget,
        class_fraction_to_forget=class_fraction_to_forget,
        retain_fraction=1.0,
        seed=seed,
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
        num_workers=num_workers,
        test_dataset=test_dataset,
    )

    unlearned_model = deepcopy(model).to(device)
    unlearned_model.train()

    optimizer = torch.optim.SGD(
        unlearned_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    history = {eval_name: [] for eval_name in eval_loaders}

    warmup_cuda(unlearned_model, device)
    start_time = time.time()
    extra_time = 0.0

    for epoch in tqdm(range(num_epochs), desc="Unlearning epochs"):
        print(f"Epoch {epoch + 1}/num_epochs")
        for x, y in retain_loader:

            x = x.to(device)
            y = y.to(device)

            logits = unlearned_model(x)

            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        extra_time_item = time.time()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: ")
            for eval_name, eval_loader in eval_loaders.items():
                history[eval_name].append(
                    compute_eval_accuracy(unlearned_model, eval_loader, device)
                )
                print(f"{eval_name}={history[eval_name][-1]:.4f}")

        extra_time += time.time() - extra_time_item

    history["rte"] = time.time() - start_time - extra_time
    print(f"Unlearning completed in {history['rte']:.2f} seconds.")

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
    train_dataset_clean = build_cifar_dataset(
        dataset_name=args.dataset_name, train=True, do_transforms=False
    )

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
        num_epochs=args.num_epochs,
        batch_size=args.unlearning_batch_size,
        lr=args.unlearning_lr,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed,
        output_dir=output_dir,
        test_dataset=test_dataset,
        dataset_clean=train_dataset_clean,
    )

    save_config(args, output_dir)
    save_model(unlearned_model, output_dir, name="unlearned_model.pth")


if __name__ == "__main__":
    main()
