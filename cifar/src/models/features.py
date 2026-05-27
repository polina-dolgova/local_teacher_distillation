import torch
from typing import Callable


def extract_cifar_resnet56_penultimate(
    model: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """
    Extract penultimate embeddings from CIFAR-style ResNet-56.
    """
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = torch.relu(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x


def extract_vit_tiny_penultimate(
    model: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """
    Extract CLS-token embedding from a timm ViT model.
    """
    with torch.no_grad():
        features = model.forward_features(x)  # [B, num_tokens, embed_dim]
        return features[:, 0]  # CLS token


def get_feature_extractor(dataset_name: str) -> Callable:
    if dataset_name == "svhn":
        return extract_vit_tiny_penultimate
    return extract_cifar_resnet56_penultimate


def extract_imagenet_resnet18_penultimate(
    model: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """
    Extract penultimate embeddings from torchvision ResNet-18.
    """
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x
