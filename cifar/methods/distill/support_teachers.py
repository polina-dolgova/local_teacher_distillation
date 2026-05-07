from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
from pathlib import Path

from cifar.src.models.features import extract_cifar_resnet56_penultimate


@dataclass
class SupportTeacher:
    """
    kind:
        - "embedding": teacher consumes frozen ResNet-56 embeddings
        - "image": teacher consumes raw images
    """

    kind: str
    model: nn.Module


class SupportLinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SupportMLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCIFARCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class CIFARResNet(nn.Module):
    def __init__(
        self, block: type[CIFARBasicBlock], num_blocks: list[int], num_classes: int
    ):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[CIFARBasicBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for cur_stride in strides:
            layers.append(block(self.in_planes, planes, cur_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class NeighborSoftLabeler:
    def __init__(
        self,
        dataset,
        support_indices,
        feature_extractor: torch.nn.Module,
        num_classes: int,
        k_neighbors: int,
        batch_size: int,
        device: str,
        num_workers: int = 0,
    ):
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive.")

        self.kind = "neighbor"
        self.dataset = dataset
        self.support_indices = np.asarray(support_indices)
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.k_neighbors = min(k_neighbors, len(self.support_indices))
        self.device = device

        self.support_features = _compute_resnet56_embeddings_for_indices(
            dataset=dataset,
            indices=self.support_indices,
            model=feature_extractor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.support_features = F.normalize(
            self.support_features.to(device),
            p=2,
            dim=1,
        )

        targets = np.asarray(dataset.targets)
        self.support_labels = torch.as_tensor(
            targets[self.support_indices],
            dtype=torch.long,
            device=device,
        )

    @torch.no_grad()
    def predict_proba(self, x_batch: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()

        query_features = _compute_resnet56_embeddings_for_batch(
            x_batch=x_batch,
            model=self.feature_extractor,
            device=self.device,
        )

        query_features = F.normalize(query_features, p=2, dim=1)

        similarities = query_features @ self.support_features.T

        topk_indices = torch.topk(
            similarities,
            k=self.k_neighbors,
            dim=1,
            largest=True,
        ).indices

        neighbor_labels = self.support_labels[topk_indices]

        weights = torch.full(
            neighbor_labels.shape,
            fill_value=1.0 / self.k_neighbors,
            dtype=torch.float32,
            device=self.device,
        )

        soft_targets = torch.zeros(
            x_batch.size(0),
            self.num_classes,
            dtype=torch.float32,
            device=self.device,
        )

        soft_targets.scatter_add_(
            dim=1,
            index=neighbor_labels,
            src=weights,
        )

        return soft_targets

def build_resnet8(num_classes: int) -> nn.Module:
    """
    CIFAR ResNet-8:
    depth = 6n + 2 with n = 1.
    """
    return CIFARResNet(CIFARBasicBlock, [1, 1, 1], num_classes=num_classes)


def build_resnet56(dataset_name) -> nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        f"{dataset_name}_resnet56",
        pretrained=False,
    )

    return model


def _pin_memory_for_device(device: str) -> bool:
    return str(device).startswith("cuda")


@torch.no_grad()
def _compute_resnet56_embeddings_for_batch(
    x_batch: torch.Tensor,
    model: torch.nn.Module,
    device: str,
) -> torch.Tensor:
    model.eval()
    x_batch = x_batch.to(device)

    features = extract_cifar_resnet56_penultimate(model, x_batch)
    features = features.flatten(start_dim=1)

    return features

def _compute_resnet56_embeddings_for_indices(
    dataset,
    indices: list[int] | np.ndarray,
    model: torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
) -> torch.Tensor:

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory_for_device(device),
    )

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            emb = extract_cifar_resnet56_penultimate(model, x)
            all_embeddings.append(emb.cpu())

    if not all_embeddings:
        return torch.empty((0, 0), dtype=torch.float32)

    return torch.cat(all_embeddings, dim=0)


def _compute_similarity_to_forget_center(
    dataset,
    candidate_indices: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
) -> np.ndarray:
    candidate_embeddings = _compute_resnet56_embeddings_for_indices(
        dataset=dataset,
        indices=candidate_indices,
        model=feature_extractor,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    forget_embeddings = _compute_resnet56_embeddings_for_indices(
        dataset=dataset,
        indices=forget_indices,
        model=feature_extractor,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if candidate_embeddings.shape[0] == 0:
        return np.array([], dtype=float)

    if forget_embeddings.shape[0] == 0:
        raise ValueError("Forget subset is empty.")

    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    forget_center = forget_embeddings.mean(dim=0, keepdim=True)
    forget_center = F.normalize(forget_center, p=2, dim=1)

    similarities = candidate_embeddings @ forget_center.T
    return similarities.squeeze(1).cpu().numpy()


def select_support_indices(
    dataset,
    retain_indices_full: list[int] | np.ndarray,
    forget_indices: list[int] | np.ndarray,
    n_support: int,
    feature_extractor: torch.nn.Module,
    support_selection_mode: str = "closest",
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """
    Select support samples as the closest retain samples to the forget-set center.
    """
    retain_indices_full = np.asarray(retain_indices_full)
    n_support = min(n_support, len(retain_indices_full))
    n_retain = len(retain_indices_full)

    threshold = None

    if support_selection_mode == "closest":

        similarities = _compute_similarity_to_forget_center(
            dataset=dataset,
            candidate_indices=retain_indices_full,
            forget_indices=forget_indices,
            feature_extractor=feature_extractor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        top_pos = np.argpartition(similarities, n_retain - n_support)[-n_support:]
        selected = retain_indices_full[top_pos]
        
        # just for history
        threshold = float(similarities[top_pos].min())

    # ablations
    elif support_selection_mode == "random":

        np.random.seed(seed)
        selected = np.random.choice(
            retain_indices_full,
            size=n_support,
            replace=False,
        )
    # ablations
    elif support_selection_mode == "farthest":

        similarities = _compute_similarity_to_forget_center(
            dataset=dataset,
            candidate_indices=retain_indices_full,
            forget_indices=forget_indices,
            feature_extractor=feature_extractor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        top_pos = np.argpartition(similarities, n_support - 1)[:n_support]
        selected = retain_indices_full[top_pos]

        threshold = float(similarities[top_pos].max())

    else:
        print(support_selection_mode, "is not implemented")
        raise NotImplementedError

    return selected, threshold


def _build_teacher_model(
    teacher_type: str,
    num_classes: int,
    embedding_dim: int | None = None,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    dataset_name: str = "cifar100",
) -> tuple[nn.Module, str]:
    """
    Returns:
        model, kind
    """
    if teacher_type == "linear_frozen":
        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided for linear_frozen.")
        return SupportLinearClassifier(embedding_dim, num_classes), "embedding"

    if teacher_type == "mlp_frozen":
        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided for mlp_frozen.")
        return (
            SupportMLPClassifier(
                in_dim=embedding_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            ),
            "embedding",
        )

    if teacher_type == "small_cnn":
        return SmallCIFARCNN(num_classes=num_classes), "image"

    if teacher_type == "resnet8":
        return build_resnet8(num_classes=num_classes), "image"

    if teacher_type == "resnet56":
        return build_resnet56(dataset_name=dataset_name), "image"

    raise ValueError(
        f"Unknown teacher_type: {teacher_type}. "
        "Expected one of ['linear_frozen', 'mlp_frozen', 'small_cnn', 'resnet8', 'resnet56']."
    )


def get_support_teacher(
    teacher_type: str,
    num_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    dataset_name: str = "cifar100",
    embedding_dim: int | None = None,
) -> SupportTeacher:

    if teacher_type in {"linear_frozen", "mlp_frozen"}:

        teacher_model, kind = _build_teacher_model(
            teacher_type=teacher_type,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            dataset_name=dataset_name,
        )
    else:

        teacher_model, kind = _build_teacher_model(
            teacher_type=teacher_type,
            num_classes=num_classes,
            embedding_dim=None,
            hidden_dim=hidden_dim,
            dropout=dropout,
            dataset_name=dataset_name,
        )

    return teacher_model, kind


def get_train_dataset(
    dataset,
    support_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    teacher_type: str,
    batch_size: int,
    device: str,
    num_workers: int = 4,
):
    if teacher_type in {"linear_frozen", "mlp_frozen"}:
        support_embeddings = _compute_resnet56_embeddings_for_indices(
            dataset=dataset,
            indices=support_indices,
            model=feature_extractor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        embedding_dim = int(support_embeddings.shape[1])

        targets = np.asarray(dataset.targets)
        support_labels = torch.tensor(targets[support_indices], dtype=torch.long)

        train_dataset = TensorDataset(support_embeddings, support_labels)
    else:
        train_dataset = Subset(dataset, support_indices)
        embedding_dim = None

    return train_dataset, embedding_dim


def prepare_teacher_for_training(
    dataset,
    support_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    teacher_type: str,
    num_classes: int,
    batch_size: int,
    device: str,
    num_workers: int = 4,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    dataset_name: str = "cifar100",
):

    train_dataset, embedding_dim = get_train_dataset(
        dataset=dataset,
        support_indices=support_indices,
        feature_extractor=feature_extractor,
        teacher_type=teacher_type,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    teacher_model, kind = get_support_teacher(
        teacher_type=teacher_type,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        dataset_name=dataset_name,
        embedding_dim=embedding_dim,
    )

    return teacher_model, kind, train_dataset


def train_support_teacher(
    dataset,
    support_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    teacher_type: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    num_workers: int = 4,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    accuracy_threshold: float | None = None,
    optimizer_type: str = "adamw",
    dataset_name: str = "cifar100",
    seed: int = 42,
) -> SupportTeacher:
    """
    Train a support teacher using one of four options:
    - linear_frozen
    - mlp_frozen
    - small_cnn
    - resnet8
    """

    g = torch.Generator()
    g.manual_seed(seed)

    teacher_model, kind, train_dataset = prepare_teacher_for_training(
        teacher_type=teacher_type,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        dataset_name=dataset_name,
        dataset=dataset,
        support_indices=support_indices,
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    teacher_model = teacher_model.to(device)
    teacher_model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )

    # print(f"optimizer: {optimizer_type}")
    # if optimizer_type == "adamw":
    #     optimizer = torch.optim.AdamW(
    #         teacher_model.parameters(),
    #         lr=lr,
    #         weight_decay=weight_decay,
    #     )
    # else:
    optimizer = torch.optim.SGD(
        teacher_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=200,
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        teacher_model.train()

        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(loader, desc=f"Support Teacher: {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)

            logits = teacher_model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = epoch_loss / total
        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"loss: {avg_loss:.4f}, "
            f"train acc: {accuracy:.4f}"
        )

        scheduler.step()

        if accuracy_threshold is not None and accuracy >= accuracy_threshold:
            break

    return SupportTeacher(kind=kind, model=teacher_model), accuracy, epoch


def predict_soft_targets_with_teacher(
    forget_dataset,
    teacher: SupportTeacher,
    feature_extractor: torch.nn.Module,
    batch_size: int,
    device: str,
    num_workers: int,
    temperature: float = 1.0,
    target_mode="top3_renorm",
    prob_threshold: float = 0.02,
) -> torch.Tensor:
    teacher.model.eval()

    all_probs = []

    if teacher.kind == "embedding":
        forget_features = _compute_resnet56_embeddings_for_indices(
            dataset=forget_dataset,
            indices=range(len(forget_dataset)),
            model=feature_extractor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        with torch.no_grad():
            for start in range(0, len(forget_features), batch_size):
                x_batch = forget_features[start : start + batch_size].to(device)
                logits = teacher.model(x_batch)
                probs = F.softmax(logits / temperature, dim=1)

                all_probs.append(probs.cpu())

    elif teacher.kind == "image":
        loader = DataLoader(
            forget_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_pin_memory_for_device(device),
        )

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = teacher.model(x)
                probs = F.softmax(logits / temperature, dim=1)
                all_probs.append(probs.cpu())
    else:
        raise ValueError(f"Unknown teacher kind: {teacher.kind}")

    if not all_probs:
        return torch.empty((0, 0), dtype=torch.float32)

    probs = torch.cat(all_probs, dim=0)

    targets = modify_probs(
        probs, target_mode=target_mode, prob_threshold=prob_threshold
    )

    return targets


def modify_probs(probs, target_mode="soft", prob_threshold=0.02):
    if target_mode == "soft":
        targets = probs

    elif target_mode == "threshold_renorm":
        targets = probs.clone()
        targets[targets < prob_threshold] = 0.0

        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    elif target_mode == "hard":
        pred_labels = probs.argmax(dim=1)
        targets = F.one_hot(pred_labels, num_classes=probs.shape[1]).float()

    elif target_mode == "top3_renorm":
        topk_values, topk_indices = torch.topk(probs, k=3, dim=1)
        targets = torch.zeros_like(probs)
        targets.scatter_(1, topk_indices, topk_values)

        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)

    elif target_mode == "top5_renorm":
        topk_values, topk_indices = torch.topk(probs, k=3, dim=1)

        targets = torch.zeros_like(probs)
        targets.scatter_(1, topk_indices, topk_values)

        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)

    else:
        raise ValueError(
            f"Unknown target_mode: {target_mode}. "
            "Expected one of ['soft', 'onehot']."
        )

    return targets


def predict_soft_targets_with_teacher_on_batch(
    x_batch: torch.Tensor,
    teacher: SupportTeacher,
    feature_extractor: torch.nn.Module,
    temperature: float = 1.0,
    target_mode: str = "soft",
    prob_threshold: float = 0.02,
) -> torch.Tensor:
    teacher.model.eval()

    with torch.no_grad():
        if teacher.kind == "embedding":
            feature_extractor.eval()
            features = extract_cifar_resnet56_penultimate(feature_extractor, x_batch)
            logits = teacher.model(features)
        elif teacher.kind == "image":
            logits = teacher.model(x_batch)
        else:
            raise ValueError(f"Unknown teacher kind: {teacher.kind}")

        probs = F.softmax(logits / temperature, dim=1)

        targets = modify_probs(
            probs, target_mode=target_mode, prob_threshold=prob_threshold
        )
    return targets.detach()


def build_neighbor_soft_targets(
    dataset,
    forget_indices: list[int] | np.ndarray,
    support_indices: list[int] | np.ndarray,
    feature_extractor: torch.nn.Module,
    num_classes: int,
    k_neighbors: int,
    batch_size: int,
    device: str,
    num_workers: int = 0,
    forget_class: int | None = None,
) -> torch.Tensor:
    """
    Build per-sample soft labels from the label distribution of k nearest
    support samples in frozen ResNet-56 embedding space.
    """
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive.")

    forget_features = _compute_resnet56_embeddings_for_indices(
        dataset=dataset,
        indices=forget_indices,
        model=feature_extractor,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    support_features = _compute_resnet56_embeddings_for_indices(
        dataset=dataset,
        indices=support_indices,
        model=feature_extractor,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    forget_features = F.normalize(forget_features, p=2, dim=1)
    support_features = F.normalize(support_features, p=2, dim=1)

    targets = np.asarray(dataset.targets)
    support_labels = torch.tensor(targets[support_indices], dtype=torch.long)

    k_neighbors = min(k_neighbors, len(support_indices))
    sim = forget_features @ support_features.T
    topk_indices = torch.topk(sim, k=k_neighbors, dim=1).indices
    neighbor_labels = support_labels[topk_indices]

    soft_targets = torch.zeros(
        len(forget_indices),
        num_classes,
        dtype=torch.float32,
    )

    for i in range(len(forget_indices)):
        counts = torch.bincount(
            neighbor_labels[i],
            minlength=num_classes,
        ).float()
        probs = counts / counts.sum().clamp_min(1.0)

        soft_targets[i] = probs

    return soft_targets


def _infer_embedding_dim(
    loader,
    feature_extractor: torch.nn.Module,
    device: str,
) -> int:
    """
    Infer penultimate embedding dimension for frozen teachers.
    """
    feature_extractor.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            emb = extract_cifar_resnet56_penultimate(feature_extractor, x)
            return int(emb.shape[1])

    raise ValueError("Dataset is empty, cannot infer embedding dimension.")


def load_support_teacher_from_checkpoint(
    teacher_model_path: str | Path,
    teacher_type: str,
    num_classes: int,
    feature_extractor: torch.nn.Module,
    loader,
    device: str,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    dataset_name: str = "cifar100",
) -> SupportTeacher:
    """
    Rebuild teacher architecture and load its checkpoint.
    """
    teacher_model_path = Path(teacher_model_path)
    checkpoint = torch.load(teacher_model_path, map_location=device)

    embedding_dim = None
    if teacher_type in {"linear_frozen", "mlp_frozen"}:
        embedding_dim = _infer_embedding_dim(
            loader=loader,
            feature_extractor=feature_extractor,
            device=device,
        )

    model, kind = _build_teacher_model(
        teacher_type=teacher_type,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        dataset_name=dataset_name,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return SupportTeacher(kind=kind, model=model)
