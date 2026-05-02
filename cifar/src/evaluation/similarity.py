import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifar.src.custom_types import *
from cifar.src.models.backbones import get_imagenet_resnet18
from cifar.src.models.features import (
    extract_imagenet_resnet18_penultimate,
    extract_cifar_resnet56_penultimate,
)


def compute_imagenet_class_mean_embeddings_cifar(
    dataset,
    num_classes: int,
    device: str = "cpu",
    batch_size: int = 256,
    selected_classes: list[int] | None = None,
) -> torch.Tensor:
    """
    Compute class-mean embeddings for CIFAR-10 using ImageNet-pretrained ResNet-18.

    Returns:
        Tensor of shape [10, 512]
    """
    model, _ = get_imagenet_resnet18(device=device)

    features_per_class = {cls: [] for cls in selected_classes}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    total_classes = set()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            emb = extract_imagenet_resnet18_penultimate(model, x)
            embedding_dim = emb.shape[1]

            for cls in selected_classes:
                mask = y == cls
                if mask.any():
                    features_per_class[cls].append(emb[mask].cpu())
                    total_classes.add(cls)

    means = torch.full((num_classes, embedding_dim), torch.nan, dtype=torch.float32)
    for cls in selected_classes:
        if cls not in total_classes:
            continue
        cls_emb = torch.cat(features_per_class[cls], dim=0)
        means[cls] = cls_emb.mean(dim=0)

    return means


def compute_full_model_class_mean_embeddings_cifar(
    model: torch.nn.Module,
    dataset,
    num_classes: int,
    device: str = "cpu",
    batch_size: int = 256,
    selected_classes: list[int] | None = None,
) -> torch.Tensor:
    """
    Compute class-mean embeddings for CIFAR-10 using the full CIFAR-10 pretrained ResNet-56.

    Returns:
        Tensor of shape [10, D]
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    if selected_classes is None:
        selected_classes = list(range(num_classes))
    features_per_class = {cls: [] for cls in selected_classes}
    total_classes = set()

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            emb = extract_cifar_resnet56_penultimate(model, x)
            embedding_dim = emb.shape[1]

            for cls in selected_classes:
                mask = y == cls
                if mask.any():
                    features_per_class[cls].append(emb[mask].cpu())
                    total_classes.add(cls)

    means = torch.full((num_classes, embedding_dim), torch.nan, dtype=torch.float32)
    for cls in selected_classes:
        if cls not in total_classes:
            continue
        cls_emb = torch.cat(features_per_class[cls], dim=0)
        means[cls] = cls_emb.mean(dim=0)

    return means


def compute_cosine_similarity_to_forget_class(
    class_mean_embeddings_retain: torch.Tensor,
    class_mean_embeddings_forget: torch.Tensor,
    class_to_forget: int,
) -> torch.Tensor:
    """
    Compute cosine similarity between the forget class embedding and all class embeddings.

    Returns:
        Tensor of shape [10]
    """
    emb_retain = F.normalize(class_mean_embeddings_retain.float(), dim=1)
    emb_forget = F.normalize(class_mean_embeddings_forget.float(), dim=1)

    forget_vec = emb_forget[class_to_forget].unsqueeze(0)
    sim_retain = (emb_retain @ forget_vec.T).squeeze(1)

    sim_forget = (emb_forget @ forget_vec.T).squeeze(1)

    return sim_retain, sim_forget


def _get_gradient_parameter_list(
    model: torch.nn.Module,
    gradient_layer_scope: GradientLayerScope,
) -> list[torch.nn.Parameter]:
    """
    Select parameters for gradient-based similarity.
    """
    classifier_prefixes = ("fc.", "linear.", "classifier.")

    selected_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if gradient_layer_scope == "last_layer":
            use_param = name.startswith(classifier_prefixes)

        elif gradient_layer_scope == "last_block_and_head":
            use_param = name.startswith("layer3.") or name.startswith(
                classifier_prefixes
            )

        else:
            raise ValueError(f"Unknown gradient_layer_scope: {gradient_layer_scope}")

        if use_param:
            selected_params.append(param)

    if not selected_params:
        available_names = [name for name, _ in model.named_parameters()]
        raise ValueError(
            "Could not find parameters for gradient similarity. "
            f"Available parameter names: {available_names}"
        )

    return selected_params
