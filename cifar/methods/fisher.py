import torch
from torch.autograd import grad
from tqdm import tqdm
import time
import torch.nn.functional as F
import torch
from copy import deepcopy
from tqdm import tqdm
import argparse

from cifar.src.custom_types import *
from cifar.src.utils import get_default_device
from cifar.src.models.backbones import get_cifar_resnet56
from cifar.methods.utils import (
    build_parser as build_parser_base,
    resolve_output_dir,
    save_config,
    save_model,
    compute_eval_accuracy,
)
from cifar.methods.dataloaders import build_separate_dataloaders, build_eval_loaders
from cifar.src.datasets import (
    get_dataset_class_names,
    build_cifar_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_parser_base()

    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--n-woodfisher", type=int, default=1000)
    return parser


def get_unlearning_datasets(args):
    class_names = get_dataset_class_names(args.dataset_name)
    num_classes = len(class_names)

    train_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=True)
    test_dataset = build_cifar_dataset(dataset_name=args.dataset_name, train=False)

    return train_dataset, test_dataset, num_classes


def get_require_grad_params(model: torch.nn.Module, named=False):
    if named:
        return [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
    else:
        return [param for param in model.parameters() if param.requires_grad]


def sam_grad(model, loss):
    names = []
    params = []

    for param in get_require_grad_params(model, named=False):
        params.append(param)

    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]

    return torch.cat(sample_grad)


def apply_perturb(model, v, mask=None):
    curr = 0

    if mask:
        for name, param in get_require_grad_params(model, named=True):
            length = param.view(-1).shape[0]
            param.view(-1).data += v[curr : curr + length].data * mask[name].view(-1)
            curr += length

    else:
        for param in get_require_grad_params(model, named=False):
            length = param.view(-1).shape[0]
            param.view(-1).data += v[curr : curr + length].data
            curr += length


def woodfisher(model, train_dl, device, criterion, v, N=1000):
    model.eval()
    k_vec = torch.clone(v)
    o_vec = None
    for idx, (data, label) in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)

        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec


def Wfisher(
    retain_loader,
    forget_loader,
    model,
    criterion,
    batch_size,
    alpha,
    n_woodfisher,
    device,
):
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    params_flat = []

    for param in model.parameters():
        if param.requires_grad:
            params_flat.append(param.view(-1))

    forget_grad = torch.zeros_like(torch.cat(params_flat)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params_flat)).to(device)

    total = 0
    model.eval()

    for x, y in tqdm(forget_loader):
        model.zero_grad()
        real_num = x.shape[0]
        x = x.to(device)
        y = y.to(device)
        output = model(x)

        loss = criterion(output, y)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for x, y in tqdm(retain_grad_loader):
        model.zero_grad()
        real_num = x.shape[0]
        x = x.to(device)
        y = y.to(device)
        output = model(x)

        loss = criterion(output, y)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2

    perturb = woodfisher(
        model,
        retain_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad,
        N=n_woodfisher,
    )

    apply_perturb(model, alpha * perturb)

    return model


### METHOD ###
def unlearn_one_class(
    model: torch.nn.Module,
    dataset,
    class_to_forget: int,
    class_fraction_to_forget: float = 1.0,
    batch_size: int = 128,
    alpha: float = 1.0,
    n_woodfisher: int = 1000,
    device: str = "cpu",
    seed: int = 42,
    output_dir="",
):

    torch.manual_seed(seed)

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

    unlearned_model = deepcopy(model).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.time()

    unlearned_model = Wfisher(
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        model=unlearned_model,
        criterion=criterion,
        batch_size=batch_size,
        alpha=alpha,
        n_woodfisher=n_woodfisher,
        device=device,
    )

    history = {eval_name: [] for eval_name in eval_loaders}

    history["rte"] = time.time() - start_time

    for eval_name, eval_loader in eval_loaders.items():
        history[eval_name].append(
            compute_eval_accuracy(unlearned_model, eval_loader, device)
        )
        print(f"{eval_name}={history[eval_name][-1]:.4f}")

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

    ### MODEL SETUP ###
    model = get_cifar_resnet56(
        dataset_name=args.dataset_name, device=args.device, model_path=args.model_path
    )

    ### UNLEARNING ###

    unlearned_model = unlearn_one_class(
        model=model,
        dataset=train_dataset,
        class_to_forget=args.class_to_forget,
        class_fraction_to_forget=args.class_fraction_to_forget,
        batch_size=args.unlearning_batch_size,
        alpha=args.alpha,
        n_woodfisher=args.n_woodfisher,
        device=device,
        seed=args.seed,
        output_dir=output_dir,
    )

    save_config(args, output_dir)
    save_model(unlearned_model, output_dir, name="unlearned_model.pth")


if __name__ == "__main__":
    main()
