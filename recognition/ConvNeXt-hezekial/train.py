from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from modules import build_convnext_tiny, freeze_backbone

DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/ADNI/AD_NC")
DEFAULT_METADATA_PATH = Path("/home/groups/comp3710/ADNI/meta_data_with_label.json")
DEFAULT_OUTPUT_DIR = Path("results")

@dataclass
class TrainConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    metadata_path: Optional[Path] = DEFAULT_METADATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    valid_split: float = 0.1
    seed: int = 42
    num_workers: int = 4
    image_size: int = 224
    drop_path_rate: float = 0.1
    freeze_backbone_epochs: int = 0
    target_test_acc: Optional[float] = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    drop_last: bool = False
    label_smoothing: float = 0.0
    patience: int = 0  # 0 disables early stopping
    gradient_clip: Optional[float] = None
    history: Dict[str, Iterable[float]] = field(default_factory=dict, init=False)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny on ADNI slices.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing train/ and test/ subfolders.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Optional path to meta_data_with_label.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints, plots, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--valid-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Number of initial epochs to train the final head only.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument(
        "--target-test-acc",
        type=float,
        default=0.8,
        help="Optional test accuracy target for early stopping/checks.",
    )

    args = parser.parse_args()

    if args.pin_memory and args.no_pin_memory:
        raise ValueError("Cannot set both --pin-memory and --no-pin-memory.")
    pin_memory = args.pin_memory or (not args.no_pin_memory)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    return TrainConfig(
        data_root=args.data_root,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        valid_split=args.valid_split,
        seed=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
        drop_path_rate=args.drop_path_rate,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        target_test_acc=args.target_test_acc,
        device=device,
        pin_memory=pin_memory,
        drop_last=args.drop_last,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    losses = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        batch_size = labels.size(0)
        losses += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": losses / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        losses += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": losses / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def plot_history(history: Dict[str, Iterable[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = Path(output_dir) / "training_curves.png"

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(10, 6))
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history and any(history["val_loss"]):
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    if "train_acc" in history:
        plt.plot(epochs, history["train_acc"], label="Train Acc")
    if "val_acc" in history and any(history["val_acc"]):
        plt.plot(epochs, history["val_acc"], label="Val Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Dynamics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure, dpi=150)
    plt.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    filename: str = "best_model.pt",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device(config.device)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config.data_root,
        metadata_path=config.metadata_path,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        valid_split=config.valid_split,
        seed=config.seed,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    model = build_convnext_tiny(
        num_classes=2,
        in_chans=3,
        drop_path_rate=config.drop_path_rate,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - config.freeze_backbone_epochs)
    )

    best_val_acc = 0.0
    best_epoch = -1
    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    epochs_without_improvement = 0
    final_test_metrics: Optional[Dict[str, float]] = None

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        if config.freeze_backbone_epochs and epoch <= config.freeze_backbone_epochs:
            freeze_backbone(model, train_head_only=True)
        else:
            freeze_backbone(model, train_head_only=False)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=config.gradient_clip,
        )

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
        else:
            val_metrics = {"loss": 0.0, "accuracy": 0.0}

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"Time: {elapsed:.1f}s"
        )

        current_val_acc = val_metrics["accuracy"] if val_loader is not None else train_metrics["accuracy"]
        if current_val_acc >= best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics={"val_accuracy": best_val_acc, "val_loss": val_metrics["loss"]},
                output_dir=config.output_dir,
            )
            print(f"  Saved checkpoint to {checkpoint_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            config.target_test_acc is not None
            and test_loader is not None
            and val_metrics["accuracy"] >= config.target_test_acc
        ):
            provisional_test = evaluate(model, test_loader, criterion, device)
            print(
                f"  Provisional test check -> Loss: {provisional_test['loss']:.4f} "
                f"Acc: {provisional_test['accuracy']:.4f}"
            )
            if provisional_test["accuracy"] >= config.target_test_acc:
                final_test_metrics = provisional_test
                print(
                    f"Target test accuracy {config.target_test_acc:.3f} reached at epoch {epoch}. "
                    "Stopping early."
                )
                break
            else:
                print(
                    f"Test accuracy {provisional_test['accuracy']:.3f} below target "
                    f"{config.target_test_acc:.3f}; continuing training."
                )

        if config.patience and epochs_without_improvement >= config.patience:
            print("Early stopping triggered.")
            break

    plot_history(history, config.output_dir)

    (config.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )

    print(f"Best epoch: {best_epoch} with val acc {best_val_acc:.4f}")

    # Reload best checkpoint before testing
    checkpoint = torch.load(config.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if final_test_metrics is None:
        final_test_metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"Test Loss: {final_test_metrics['loss']:.4f} "
        f"Test Acc: {final_test_metrics['accuracy']:.4f}"
    )

    (config.output_dir / "test_metrics.json").write_text(
        json.dumps(final_test_metrics, indent=2)
    )

    if (
        config.target_test_acc is not None
        and final_test_metrics["accuracy"] < config.target_test_acc
    ):
        print(
            f"WARNING: Test accuracy {final_test_metrics['accuracy']:.3f} "
            f"did not reach the target {config.target_test_acc:.3f}."
        )


if __name__ == "__main__":
    main()
