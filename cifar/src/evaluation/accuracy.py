import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_per_class_accuracy(
    model: torch.nn.Module,
    dataset,
    num_classes: int,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict[int, float]:
    """
    Compute per-class accuracy on a dataset.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()

    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            for cls in range(num_classes):
                mask = y == cls
                if mask.any():
                    total[cls] += mask.sum().item()
                    correct[cls] += (pred[mask] == y[mask]).sum().item()

    acc = {}
    for cls in range(num_classes):
        if total[cls].item() == 0:
            continue
        acc[cls] = float(correct[cls].item() / total[cls].item())
    return acc
