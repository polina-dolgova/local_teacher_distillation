from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import DataLoader, Subset

from mnist.src.model import SmallMNISTCNN
from mnist.src.utils import seed_everything, seed_worker

def train_mnist_cnn(
    root: str = "./data",
    train: bool = True,
    batch_size: int = 128,
    epochs: int = 3,
    lr: float = 1e-3,
    class_to_forget: int | None = None,
    device: str | None = None,
    seed: int = 42
):
    """
    Train a small CNN on MNIST and return the trained model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)

    if class_to_forget is None:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker,
            num_workers=0,
        )
    else:
        retain_indices = [
            idx
            for idx, (_, label) in enumerate(dataset)
            if int(label) != class_to_forget
        ]

        retain_dataset = Subset(dataset, retain_indices)
        loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker,
            num_workers=0,
        )

    model = SmallMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += imgs.size(0)

        avg_loss = total_loss / total_count
        avg_acc = total_correct / total_count
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    return model


def load_mnist_class_mean_embeddings(
    model: nn.Module,
    root: str = "./data",
    train: bool = True,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Compute the mean embedding for each MNIST class.

    Returns:
        class_mean_embeddings: numpy array of shape (10, embedding_dim)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    class_sums = None
    class_counts = np.zeros(10, dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = model.get_embedding(imgs).cpu().numpy()
            labels_np = labels.numpy()

            if class_sums is None:
                embedding_dim = emb.shape[1]
                class_sums = np.zeros((10, embedding_dim), dtype=np.float64)

            for cls in range(10):
                mask = labels_np == cls
                if np.any(mask):
                    class_sums[cls] += emb[mask].sum(axis=0)
                    class_counts[cls] += mask.sum()

    class_mean_embeddings = class_sums / class_counts[:, None]
    return class_mean_embeddings


def train_models(class_to_forget, device="mps", seed=42):
    model = train_mnist_cnn(
        root="./data",
        train=True,
        batch_size=128,
        epochs=4,
        lr=1e-3,
        device=device,
        seed=seed
    )

    retrain_model = train_mnist_cnn(
        root="./data",
        train=True,
        batch_size=128,
        epochs=4,
        lr=1e-3,
        device=device,
        class_to_forget=class_to_forget,
        seed=seed
    )

    return model, retrain_model
