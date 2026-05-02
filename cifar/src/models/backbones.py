import torch
from torchvision import models, transforms
from pathlib import Path

from cifar.src.custom_types import *


def get_cifar_resnet56(
    dataset_name: DatasetName,
    model_path=None,
    device: str = "cpu",
) -> torch.nn.Module:

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        f"{dataset_name}_resnet56",
        pretrained=True,
    ).to(device)

    if model_path is not None:
        if not Path(model_path).exists():
            raise ValueError(
                f"Output directory {model_path} does not exist. Please run the experiment first to generate the unlearned model."
            )
        checkpoint = torch.load(model_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint

        model.load_state_dict(model_state_dict)

    model.float()
    model.eval()
    return model


def get_imagenet_resnet18(
    device: str = "cpu",
) -> tuple[torch.nn.Module, transforms.Compose]:
    """
    Load ImageNet-pretrained ResNet-18 and preprocessing.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean,
                std=weights.transforms().std,
            ),
        ]
    )
    return model, preprocess
