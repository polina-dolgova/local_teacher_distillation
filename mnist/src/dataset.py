import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

from mnist.src.utils import seed_everything, seed_worker

def load_mnist_class_means(root: str = "./data", train: bool = True) -> np.ndarray:
    """
    Load MNIST and compute the mean image for each class.

    Returns:
        class_means: numpy array of shape (10, 28, 28)
    """
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)

    sums = np.zeros((10, 28, 28), dtype=np.float64)
    counts = np.zeros(10, dtype=np.int64)

    for img, label in dataset:
        img_np = img.squeeze(0).numpy()
        sums[label] += img_np
        counts[label] += 1

    class_means = sums / counts[:, None, None]
    return class_means


def build_class_subset_loader(
    dataset,
    class_to_forget: int,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42
) -> DataLoader:
    """
    Build a DataLoader containing only samples of the selected class.
    """
    indices = [i for i, (_, y) in enumerate(dataset) if int(y) == class_to_forget]
    subset = Subset(dataset, indices)

    seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            worker_init_fn=seed_worker,
            num_workers=0,
        )
    
    return dataloader


def mnist_dataset(class_to_forget, seed=42):

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    forget_loader = build_class_subset_loader(
        dataset=train_dataset,
        class_to_forget=class_to_forget,
        batch_size=64,
        shuffle=True,
        seed=seed
    )

    return forget_loader


def split_dataset_indices_by_class(
    dataset,
    class_to_forget: int,
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices into:
    - forget indices: samples with label == class_to_forget
    - retain indices: all other samples
    """
    forget_indices = []
    retain_indices = []

    for idx in range(len(dataset)):
        _, y = dataset[idx]
        if int(y) == class_to_forget:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)

    return forget_indices, retain_indices


def stack_subset_tensors(
    dataset,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Materialize a subset into tensors.

    Returns:
        X: shape (n, C, H, W)
        y: shape (n,)
    """
    xs = []
    ys = []

    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(int(y))

    X = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


def flatten_for_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten image tensor batch to shape (n, d).
    """
    return x.view(x.shape[0], -1)


def compute_cosine_similarity_scores_to_forget_prototype(
    dataset,
    forget_indices: list[int],
    candidate_indices: list[int],
    feature_space: str = "raw",
    feature_model: torch.nn.Module | None = None,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute cosine similarity of each candidate sample to the forget-class prototype
    in raw pixel space.

    Prototype = mean raw image over forget_indices.
    """

    if feature_space == "raw":
        X_forget, _ = stack_subset_tensors(dataset, forget_indices)
        X_candidates, _ = stack_subset_tensors(dataset, candidate_indices)

        X_forget = flatten_for_similarity(X_forget).float()
        X_candidates = flatten_for_similarity(X_candidates).float()

    elif feature_space == "full_model":
        if feature_model is None:
            raise ValueError(
                "feature_model must be provided when feature_space='penultimate'."
            )

        X_forget = compute_penultimate_embeddings(
            model=feature_model,
            dataset=dataset,
            indices=forget_indices,
            batch_size=batch_size,
            device=device,
        ).float()

        X_candidates = compute_penultimate_embeddings(
            model=feature_model,
            dataset=dataset,
            indices=candidate_indices,
            batch_size=batch_size,
            device=device,
        ).float()

    else:
        raise ValueError(f"Unknown feature_space: {feature_space}")

    forget_proto = X_forget.mean(dim=0, keepdim=True)

    forget_proto = forget_proto / torch.clamp(
        forget_proto.norm(dim=1, keepdim=True), min=1e-12
    )
    X_candidates = X_candidates / torch.clamp(
        X_candidates.norm(dim=1, keepdim=True), min=1e-12
    )

    sim = X_candidates @ forget_proto.T
    return sim.squeeze(1).cpu().numpy()


def select_support_indices(
    dataset,
    forget_indices: list[int],
    retain_indices: list[int],
    support_size: int,
    mode: str = "cosine",
    feature_space: str = "raw",
    feature_model: torch.nn.Module | None = None,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> list[int]:
    """
    Select a support subset from retain data.

    mode:
        - "random": random subset from retain
        - "cosine": top-k retain samples most similar to forget prototype
    """
    if support_size <= 0:
        raise ValueError("support_size must be positive.")

    if support_size > len(retain_indices):
        raise ValueError(
            f"support_size={support_size} is larger than retain set size {len(retain_indices)}."
        )

    rng = np.random.default_rng(seed)

    if mode == "random":
        chosen = rng.choice(retain_indices, size=support_size, replace=False)
        return chosen.tolist()

    if mode == "cosine":
        sim_scores = compute_cosine_similarity_scores_to_forget_prototype(
            dataset=dataset,
            forget_indices=forget_indices,
            candidate_indices=retain_indices,
            feature_space=feature_space,
            feature_model=feature_model,
            batch_size=batch_size,
            device=device,
        )
        order = np.argsort(-sim_scores)
        chosen = [retain_indices[i] for i in order[:support_size]]
        return chosen

    raise ValueError(f"Unknown support selection mode: {mode}")


def compute_penultimate_embeddings(
    model: torch.nn.Module,
    dataset,
    indices: list[int],
    batch_size: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute penultimate-layer embeddings for a subset.
    """
    if len(indices) == 0:
        return torch.empty(0, dtype=torch.float32)

    device = torch.device(device)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            emb = model.get_embedding(x)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)
