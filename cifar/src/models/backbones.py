import torch
import timm
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


def get_vit_tiny(
    num_classes: int = 10,
    model_path=None,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    ViT-Tiny configured for 32x32 inputs (patch_size=4, 64 tokens).
    """
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=32,
        patch_size=4,
    ).to(device)

    if model_path is not None:
        if not Path(model_path).exists():
            raise ValueError(f"Model path {model_path} does not exist.")
        checkpoint = torch.load(model_path, map_location=device)
        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(model_state_dict)

    model.float()
    model.eval()
    return model


def get_model(
    dataset_name: DatasetName,
    num_classes: int | None = None,
    model_path=None,
    device: str = "cpu",
) -> torch.nn.Module:
    if dataset_name in ("cifar10", "cifar100"):
        return get_cifar_resnet56(
            dataset_name=dataset_name, model_path=model_path, device=device
        )
    elif dataset_name == "svhn":
        _num_classes = num_classes if num_classes is not None else 10
        return get_vit_tiny(
            num_classes=_num_classes, model_path=model_path, device=device
        )
    else:
        raise NotImplementedError(f"No model defined for dataset '{dataset_name}'.")


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
