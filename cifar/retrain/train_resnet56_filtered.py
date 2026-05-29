from __future__ import annotations

import argparse
import time
import torch

from cifar.retrain.config import DatasetName, TrainConfig
from cifar.retrain.data import build_dataloaders
from cifar.retrain.engine import (
    evaluate,
    format_seconds,
    resolve_device,
    save_json,
    seed_everything,
    train_one_epoch,
)


def get_cifar_resnet56(
    dataset_name: DatasetName,
    device: str = "cpu",
) -> torch.nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        f"{dataset_name}_resnet56",
        pretrained=False,
    ).to(device)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")

    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["cifar10", "cifar100"],
        default="cifar10",
    )
    parser.add_argument("--class-to-forget", type=int, required=True)
    parser.add_argument("--class-fraction-to-forget", type=float, required=True)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true", default=False)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)

    return parser


def make_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        save_dir=args.save_dir,
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        class_to_forget=args.class_to_forget,
        class_fraction_to_forget=args.class_fraction_to_forget,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
        save_every=args.save_every,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    if not (0.0 <= config.class_fraction_to_forget <= 1.0):
        raise ValueError("--class-fraction-to-forget must be in [0, 1].")

    device = resolve_device(config.device)
    config.device = device

    save_dir = config.resolved_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    save_json(save_dir / "config.json", config.to_dict())

    print("Building dataloaders...")
    train_loader, test_loader, forget_indices, retain_indices, num_classes = (
        build_dataloaders(config)
    )

    split_info = {
        "dataset_name": config.dataset_name,
        "class_to_forget": config.class_to_forget,
        "class_fraction_to_forget": config.class_fraction_to_forget,
        "num_forget_indices": len(forget_indices),
        "num_retain_indices": len(retain_indices),
        "forget_indices": forget_indices,
    }
    save_json(save_dir / "split_info.json", split_info)

    print("Building model...")
    model = get_cifar_resnet56(
        dataset_name=config.dataset_name,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=0.0,
    )

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1

    total_train_time_seconds = 0.0
    total_wall_time_seconds = 0.0

    if torch.cuda.is_available():
        with torch.no_grad():
            _ = model(torch.randn(1, 3, 32, 32, device=device))
        torch.cuda.synchronize()

    global_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_wall_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
            num_epochs=config.epochs,
            log_interval=config.log_interval,
        )

        eval_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            class_to_forget=config.class_to_forget,
        )

        scheduler.step()

        epoch_wall_time_seconds = time.perf_counter() - epoch_wall_start
        total_train_time_seconds += train_metrics["train_time_seconds"]
        total_wall_time_seconds += epoch_wall_time_seconds

        avg_train_epoch_time = total_train_time_seconds / epoch
        avg_wall_epoch_time = total_wall_time_seconds / epoch
        remaining_epochs = config.epochs - epoch

        eta_train_only_seconds = avg_train_epoch_time * remaining_epochs
        eta_wall_seconds = avg_wall_epoch_time * remaining_epochs

        current_metrics = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **eval_metrics,
            "epoch_wall_time_seconds": epoch_wall_time_seconds,
            "eta_train_only_seconds": eta_train_only_seconds,
            "eta_wall_seconds": eta_wall_seconds,
            "best_val_acc_so_far": max(best_val_acc, eval_metrics["val_acc"]),
        }
        history.append(current_metrics)

        is_best = eval_metrics["val_acc"] > best_val_acc
        if is_best:
            best_val_acc = eval_metrics["val_acc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config.to_dict(),
                },
                save_dir / "best_model.pt",
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "config": config.to_dict(),
            },
            save_dir / "last_model.pt",
        )

        if config.save_every > 0 and epoch % config.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config.to_dict(),
                },
                save_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )

        save_json(save_dir / "history.json", history)

        print(
            f"[Epoch {epoch:03d}/{config.epochs:03d}] "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"train_acc={train_metrics['train_acc']:.4f} "
            f"val_loss={eval_metrics['val_loss']:.4f} "
            f"val_acc={eval_metrics['val_acc']:.4f} "
            f"forget_acc={eval_metrics['forget_class_acc']:.4f} "
            f"retain_mean_acc={eval_metrics['retain_classes_mean_acc']:.4f} "
            f"train_time={format_seconds(train_metrics['train_time_seconds'])} "
            f"epoch_wall={format_seconds(epoch_wall_time_seconds)} "
            f"eta_train_only={format_seconds(eta_train_only_seconds)} "
            f"eta_wall={format_seconds(eta_wall_seconds)} "
            f"{'[BEST]' if is_best else ''}"
        )

    total_runtime = time.perf_counter() - global_start

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_time_seconds_excluding_metrics": total_train_time_seconds,
        "total_wall_time_seconds_including_metrics": total_runtime,
        "avg_train_epoch_time_seconds": total_train_time_seconds / config.epochs,
        "avg_wall_epoch_time_seconds": total_wall_time_seconds / config.epochs,
    }
    save_json(save_dir / "summary.json", summary)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(
        "Pure training time (excluding metrics): "
        f"{format_seconds(total_train_time_seconds)}"
    )
    print("Total wall time (including metrics): " f"{format_seconds(total_runtime)}")


if __name__ == "__main__":
    main()
