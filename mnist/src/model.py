import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


class SmallMNISTCNN(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        x = self.features(x)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x):
        x = self.features(x)
        emb = self.embedding(x)
        return emb


class TinyMNISTCNN(nn.Module):
    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        x = self.features(x)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x):
        x = self.features(x)
        emb = self.embedding(x)
        return emb


class MediumMNISTCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 x 28 -> 14 x 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14 -> 7 x 7
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        x = self.features(x)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x):
        x = self.features(x)
        emb = self.embedding(x)
        return emb

def load_mnist_class_mean_embeddings(
    model: nn.Module,
    root: str = "./data",
    train: bool = True,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute the mean embedding for each MNIST class.

    Returns:
        class_mean_embeddings: numpy array of shape (10, embedding_dim)
    """
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
