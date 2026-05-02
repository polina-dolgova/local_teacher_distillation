from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset, Subset


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract inputs and labels from a batch.

    Supported formats:
    - (x, y)
    - (x, y, ...)
    - {"image": x, "label": y}
    - {"x": x, "y": y}
    """
    if isinstance(batch, dict):
        if "image" in batch and "label" in batch:
            return batch["image"], batch["label"]
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        raise ValueError(f"Unsupported batch dict keys: {list(batch.keys())}")

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list must have at least 2 elements: (x, y)")
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Standard prediction entropy.
    """
    safe_probs = torch.clamp(probs, min=1e-30)
    return -(safe_probs * safe_probs.log()).sum(dim=dim)


def _modified_entropy(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Modified entropy used in MIA literature.

    For the true class y, replaces p_y with (1 - p_y) inside the entropy-like score.
    """
    safe_probs = torch.clamp(probs, min=1e-30)
    reverse_probs = torch.clamp(1.0 - probs, min=1e-30)

    log_probs = safe_probs.log()
    log_reverse_probs = reverse_probs.log()

    modified_probs = probs.clone()
    modified_log_probs = log_reverse_probs.clone()

    row_idx = torch.arange(probs.size(0), device=probs.device)
    modified_probs[row_idx, labels] = reverse_probs[row_idx, labels]
    modified_log_probs[row_idx, labels] = log_probs[row_idx, labels]

    return -(modified_probs * modified_log_probs).sum(dim=-1)


@torch.no_grad()
def _collect_probs_and_labels(
    loader: DataLoader | None,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Collect softmax probabilities and labels for a loader.
    """
    if loader is None:
        return None, None

    probs_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    model.eval()

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=-1)

        probs_list.append(probs.detach().cpu())
        labels_list.append(y.detach().cpu())

    if not probs_list:
        return None, None

    return torch.cat(probs_list, dim=0), torch.cat(labels_list, dim=0)


def _svc_fit_predict(
    shadow_train_feat: torch.Tensor,
    shadow_test_feat: torch.Tensor,
    target_train_feat: torch.Tensor | None,
    target_test_feat: torch.Tensor | None,
) -> float:
    """
    Train an SVC on shadow-member vs shadow-nonmember features and
    evaluate it on target-member / target-nonmember features.

    Returns average attack accuracy over available target splits.
    """
    n_shadow_train = shadow_train_feat.shape[0]
    n_shadow_test = shadow_test_feat.shape[0]

    x_shadow = torch.cat([shadow_train_feat, shadow_test_feat], dim=0).cpu().numpy()
    x_shadow = x_shadow.reshape(n_shadow_train + n_shadow_test, -1)

    y_shadow = np.concatenate(
        [
            np.ones(n_shadow_train, dtype=np.int64),
            np.zeros(n_shadow_test, dtype=np.int64),
        ]
    )

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(x_shadow, y_shadow)

    accs: list[float] = []

    if target_train_feat is not None and target_train_feat.numel() > 0:
        x_target_train = (
            target_train_feat.cpu().numpy().reshape(target_train_feat.shape[0], -1)
        )
        pred_train = clf.predict(x_target_train)
        accs.append(float(pred_train.mean()))

    if target_test_feat is not None and target_test_feat.numel() > 0:
        x_target_test = (
            target_test_feat.cpu().numpy().reshape(target_test_feat.shape[0], -1)
        )
        pred_test = clf.predict(x_target_test)
        accs.append(float(1.0 - pred_test.mean()))

    if not accs:
        raise ValueError(
            "At least one of target_train_feat or target_test_feat must be non-empty."
        )

    return float(np.mean(accs))


def _build_shadow_train_loader(
    dataset: Dataset,
    shadow_size: int,
    num_workers: int = 2,
    batch_size: int = 256,
    seed: int = 42,
) -> DataLoader:
    """
    Sample a random subset from retain data and rebuild a loader
    with the same loading parameters as retain_loader.
    """
    n_total = len(dataset)
    n_sample = min(shadow_size, n_total)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_sample, replace=False).tolist()

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def compute_svc_mia(
    model: torch.nn.Module,
    shadow_train_loader: DataLoader,
    shadow_test_loader: DataLoader,
    target_train_loader: DataLoader | None,
    target_test_loader: DataLoader | None,
    device: torch.device | str,
) -> dict[str, float]:
    """
    Generic SVC-MIA computation.

    Returns scores in [0, 1].
    """
    device = torch.device(device)

    shadow_train_prob, shadow_train_labels = _collect_probs_and_labels(
        shadow_train_loader, model, device
    )
    shadow_test_prob, shadow_test_labels = _collect_probs_and_labels(
        shadow_test_loader, model, device
    )
    target_train_prob, target_train_labels = _collect_probs_and_labels(
        target_train_loader, model, device
    )
    target_test_prob, target_test_labels = _collect_probs_and_labels(
        target_test_loader, model, device
    )

    if shadow_train_prob is None or shadow_test_prob is None:
        raise ValueError(
            "shadow_train_loader and shadow_test_loader must be non-empty."
        )

    shadow_train_corr = (
        (torch.argmax(shadow_train_prob, dim=1) == shadow_train_labels)
        .float()
        .unsqueeze(1)
    )
    shadow_test_corr = (
        (torch.argmax(shadow_test_prob, dim=1) == shadow_test_labels)
        .float()
        .unsqueeze(1)
    )

    target_train_corr = None
    if target_train_prob is not None:
        target_train_corr = (
            (torch.argmax(target_train_prob, dim=1) == target_train_labels)
            .float()
            .unsqueeze(1)
        )

    target_test_corr = None
    if target_test_prob is not None:
        target_test_corr = (
            (torch.argmax(target_test_prob, dim=1) == target_test_labels)
            .float()
            .unsqueeze(1)
        )

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])

    target_train_conf = None
    if target_train_prob is not None:
        target_train_conf = torch.gather(
            target_train_prob, 1, target_train_labels[:, None]
        )

    target_test_conf = None
    if target_test_prob is not None:
        target_test_conf = torch.gather(
            target_test_prob, 1, target_test_labels[:, None]
        )

    shadow_train_ent = _entropy(shadow_train_prob).unsqueeze(1)
    shadow_test_ent = _entropy(shadow_test_prob).unsqueeze(1)

    target_train_ent = None
    if target_train_prob is not None:
        target_train_ent = _entropy(target_train_prob).unsqueeze(1)

    target_test_ent = None
    if target_test_prob is not None:
        target_test_ent = _entropy(target_test_prob).unsqueeze(1)

    shadow_train_ment = _modified_entropy(
        shadow_train_prob, shadow_train_labels
    ).unsqueeze(1)
    shadow_test_ment = _modified_entropy(
        shadow_test_prob, shadow_test_labels
    ).unsqueeze(1)

    target_train_ment = None
    if target_train_prob is not None:
        target_train_ment = _modified_entropy(
            target_train_prob, target_train_labels
        ).unsqueeze(1)

    target_test_ment = None
    if target_test_prob is not None:
        target_test_ment = _modified_entropy(
            target_test_prob, target_test_labels
        ).unsqueeze(1)

    return {
        "correctness": 100
        * _svc_fit_predict(
            shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
        ),
        "confidence": 100
        * _svc_fit_predict(
            shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
        ),
        "entropy": 100
        * _svc_fit_predict(
            shadow_train_ent, shadow_test_ent, target_train_ent, target_test_ent
        ),
        "m_entropy": 100
        * _svc_fit_predict(
            shadow_train_ment, shadow_test_ment, target_train_ment, target_test_ment
        ),
        "prob": 100
        * _svc_fit_predict(
            shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
        ),
    }


def compute_svc_mia_forget_efficacy(
    model: torch.nn.Module,
    retain_dataset,
    forget_dataset,
    test_dataset,
    num_workers: int,
    batch_size: int,
    device: torch.device | str,
    seed: int = 42,
) -> dict[str, float]:
    """
    SVC-MIA in the same spirit as AMUN / SalUn forget-efficacy evaluation.

    - shadow_train: random subset from retain data, size = |test|
    - shadow_test: test set
    - target_train: None
    - target_test: forget set

    Returns raw scores in [0, 1].
    """
    shadow_size = len(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    shadow_train_loader = _build_shadow_train_loader(
        dataset=retain_dataset,
        shadow_size=shadow_size,
        num_workers=num_workers,
        batch_size=batch_size,
        seed=seed,
    )

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return compute_svc_mia(
        model=model,
        shadow_train_loader=shadow_train_loader,
        shadow_test_loader=test_loader,
        target_train_loader=None,
        target_test_loader=forget_loader,
        device=device,
    )
