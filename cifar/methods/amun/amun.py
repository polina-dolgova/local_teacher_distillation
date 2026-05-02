from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from cifar.methods.amun.helper import compute_advset
from cifar.methods.utils import (
    build_parser as build_parser_base,
    resolve_output_dir,
    save_config,
    save_model,
    compute_eval_accuracy,
)
from cifar.src.custom_types import *
from cifar.src.datasets import (
    get_dataset_class_names,
    split_indices_by_forget_class,
    build_cifar_dataset,
)
from cifar.src.models.backbones import get_cifar_resnet56
from cifar.src.utils import get_default_device
from cifar.methods.dataloaders import build_separate_dataloaders, build_eval_loaders


def build_parser() -> argparse.ArgumentParser:
    parser = build_parser_base()
    parser.description = "AMUN unlearning"

    parser.add_argument("--unlearn-method", default="amun", type=str)
    parser.add_argument("--remain", default="use", type=str)
    parser.add_argument("--attack", default="pgdl2", type=str)
    parser.add_argument("--lr-steps", default=5, type=int)

    parser.set_defaults(
        unlearning_lr=0.05,
        num_epochs=10,
        unlearning_batch_size=256,
        weight_decay=5e-4,
    )
    return parser


def get_official_transforms(dataset_name: DatasetName):
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_cls = torchvision.datasets.CIFAR10
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        dataset_cls = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_adv = transforms.Compose([transforms.ToTensor()])
    return dataset_cls, transform_train, transform_test, transform_adv


def get_unlearning_datasets(args):
    class_names = get_dataset_class_names(args.dataset_name)
    num_classes = len(class_names)

    dataset_cls, transform_train, transform_test, _ = get_official_transforms(
        args.dataset_name
    )

    trainset = dataset_cls(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    trainset_clean = dataset_cls(
        root="./data",
        train=True,
        download=True,
        transform=transform_test,
    )
    testset = dataset_cls(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )
    return trainset, trainset_clean, testset, num_classes


def train_one_epoch(
    epoch: int,
    net: torch.nn.Module,
    trainset,
    forgetset,
    unlearn_idx: list[int],
    optimizer,
    scheduler,
    criterion,
    args,
    device: str,
    advset=None,
):
    print(f"\nEpoch: {epoch}")

    net.train()
    batch_idx = -1

    if args.use_remain:
        trainset_combined = torch.utils.data.ConcatDataset([trainset, advset])
    else:
        if args.unlearn_method == "advonly":
            print("only advset is being used!")
            trainset_combined = advset
        else:
            trainset_combined = torch.utils.data.ConcatDataset([forgetset, advset])

    trainloader = torch.utils.data.DataLoader(
        trainset_combined,
        shuffle=True,
        batch_size=args.unlearning_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if epoch == 0 and batch_idx == 0:
            print("inputs shape: ", inputs.shape)

        inputs, targets = inputs.float().to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    scheduler.step()

    net.eval()

    return
    # return train_loss / (batch_idx + 1), 100.0 * correct / total, advset


def get_eval_loaders(
    class_to_forget, class_fraction_to_forget, dataset_name="cifar100", batch_size=128
):
    dataset = build_cifar_dataset(dataset_name=dataset_name, train=True)
    retain_loader, forget_loader = build_separate_dataloaders(
        dataset, class_to_forget, class_fraction_to_forget, batch_size
    )

    eval_loaders = build_eval_loaders(
        forget_loader.dataset,
        retain_loader.dataset,
        class_to_forget,
        batch_size,
        2,
    )

    return eval_loaders


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = get_default_device()
    args.device = device

    args.use_remain = args.remain == "use"
    print("use remain flag: ", args.use_remain)

    if device == "cuda":
        print("chosen: ", device)
        cudnn.benchmark = True

    output_dir = resolve_output_dir(args)

    seed_val = args.seed
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    trainset, trainset_clean, testset, _ = get_unlearning_datasets(args)

    forget_indices, _ = split_indices_by_forget_class(
        trainset_clean,
        class_to_forget=args.class_to_forget,
        class_fraction_to_forget=args.class_fraction_to_forget,
    )
    unlearn_idx = [int(i) for i in forget_indices]
    removed_classes = [trainset[i][1] for i in unlearn_idx]
    pd.DataFrame(
        {
            "unlearn_idx": unlearn_idx,
            "removed_classes": removed_classes,
        }
    ).to_csv(Path(output_dir) / "forget_indices.csv", index=False)

    trainset_filtered = torch.utils.data.Subset(
        trainset,
        list(set(range(len(trainset))) - set(unlearn_idx)),
    )
    forgetset = torch.utils.data.Subset(trainset_clean, unlearn_idx)

    if args.unlearn_method == "advonly":
        trainset = trainset_filtered

    net = get_cifar_resnet56(
        dataset_name=args.dataset_name, device=args.device, model_path=args.model_path
    )
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    eval_loaders = get_eval_loaders(
        class_to_forget=args.class_to_forget,
        class_fraction_to_forget=args.class_fraction_to_forget,
        batch_size=128,
    )

    start_time = time.time()

    print("===> Computing the adversarial set...")
    net.eval()
    percentile = 50
    _, _, _, transform_adv = get_official_transforms(args.dataset_name)

    indices_sample = list(range(min(50, len(forgetset))))
    advset_init, df_adv_init = compute_advset(
        args,
        forgetset,
        net,
        initial_eps=0.01,
        outdir=str(output_dir),
        transform=transform_adv,
        indices=indices_sample,
    )
    # print("adv df: ", df_adv_init)

    smallest_eps = df_adv_init["smallest_eps"].values
    eps_percentile = np.percentile(smallest_eps, percentile)
    # print(f"{percentile}th percentile of smallest epsilons: ", eps_percentile)

    indices_others = list(set(range(len(forgetset))) - set(indices_sample))
    if len(indices_others) > 0:
        advset_others, df_adv_others = compute_advset(
            args,
            forgetset,
            net,
            initial_eps=eps_percentile,
            outdir=str(output_dir),
            transform=transform_adv,
            indices=indices_others,
        )
        advset = torch.utils.data.ConcatDataset([advset_init, advset_others])
    else:
        advset = advset_init

    print("length of final advset: ", len(advset))

    print("===> Fine-tuning the model...")
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.unlearning_lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_steps,
        gamma=0.1,
    )

    t_max = args.num_epochs
    # print("Tmax: ", t_max)

    history = {eval_name: [] for eval_name in eval_loaders}
    extra_time = 0

    for epoch in range(t_max):
        train_one_epoch(
            epoch=epoch,
            net=net,
            trainset=trainset,
            forgetset=forgetset,
            unlearn_idx=unlearn_idx,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            args=args,
            device=device,
            advset=advset,
        )

        if (epoch + 1) % 5 == 0:

            extra_time_item = time.time()

            print(f"Epoch {epoch + 1}: ")

            for eval_name, eval_loader in eval_loaders.items():
                history[eval_name].append(
                    compute_eval_accuracy(net, eval_loader, device)
                )
                print(f"{eval_name}={history[eval_name][-1]:.4f}")

            extra_time += time.time() - extra_time_item

    history["rte"] = time.time() - start_time - extra_time

    print("unlearning time: ", history["rte"])
    save_config(history, output_dir=output_dir, name="accuracies.json")

    save_config(args, output_dir)
    save_model(net, output_dir, name="unlearned_model.pth")


if __name__ == "__main__":
    main()
