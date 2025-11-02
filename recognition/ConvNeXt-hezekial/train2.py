from __future__ import annotations

import argparse
import json
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
from modules import build_convnext_small

DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/ADNI/AD_NC")
DEFAULT_METADATA_PATH = Path("/home/groups/comp3710/ADNI/meta_data_with_label.json")
DEFAULT_OUTPUT_DIR = Path("results")


@dataclass
class TrainConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    metadata_path: Optional[Path] = DEFAULT_METADATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 5e-3
    valid_split: float = 0.1
    seed: int = 42
    num_workers: int = 0
    image_size: int = 224
    drop_path_rate: float = 0.1
    head_dropout: float = 0.2
    label_smoothing: float = 0.05
    patience: int = 15
    gradient_clip: Optional[float] = 1.0
    augment_fraction: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    drop_last: bool = False
    history: Dict[str, Iterable[float]] = field(default_factory=dict, init=False)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="ConvNeXt-Small training with LR/WD grid search.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--valid-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--augment-fraction", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--drop-last", action="store_true")

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
        image_size=args.image_size,
        valid_split=args.valid_split,
        seed=args.seed,
        num_workers=args.num_workers,
        drop_path_rate=args.drop_path_rate,
        head_dropout=args.head_dropout,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        augment_fraction=args.augment_fraction,
        device=device,
        pin_memory=pin_memory,
        drop_last=args.drop_last,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: Optional[float] = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
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
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total if total else 0.0, running_correct / total if total else 0.0


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        logits_sum = None
        loss_sum = 0.0
        for flip in (False, True):
            augmented = torch.flip(images, dims=[3]) if flip else images
            outputs = model(augmented)
            loss_sum += criterion(outputs, labels).item()
            logits_sum = outputs if logits_sum is None else logits_sum + outputs

        logits = logits_sum / 2.0
        loss = loss_sum / 2.0

        batch_size = labels.size(0)
        running_loss += loss * batch_size
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total if total else 0.0, running_correct / total if total else 0.0


def plot_history(history: Dict[str, Iterable[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(10, 6))
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    if "train_acc" in history:
        plt.plot(epochs, history["train_acc"], label="Train Acc")
    if "val_acc" in history:
        plt.plot(epochs, history["val_acc"], label="Val Acc")
    if "test_acc" in history:
        plt.plot(epochs[: len(history["test_acc"])], history["test_acc"], label="Test Acc (running)")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training / Validation / Test Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def save_checkpoint(model: nn.Module, path: Path, epoch: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()},
        "metrics": metrics,
    }
    try:
        torch.save(state, path)
    except RuntimeError as exc:
        from tempfile import NamedTemporaryFile

        print(f"Warning: checkpoint save failed ({exc}); retrying with legacy serialization.")
        with NamedTemporaryFile(dir=str(path.parent), suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        torch.save(state, tmp_path, _use_new_zipfile_serialization=False)
        tmp_path.replace(path)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device(config.device)

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
        augment_fraction=config.augment_fraction,
    )

    best_lr = config.learning_rate
    best_wd = config.weight_decay
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    model = build_convnext_small(
        num_classes=2,
        in_chans=3,
        drop_path_rate=config.drop_path_rate,
        head_dropout=config.head_dropout,
        pretrained=False,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: Dict[str, list] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc": []}
    best_val_acc = 0.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=config.gradient_clip,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device) if val_loader else (0.0, 0.0)
        _, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"Test Acc: {test_acc:.4f} "
            f"Time: {elapsed:.1f}s"
        )

        if val_loader:
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                save_checkpoint(
                    model,
                    config.output_dir / "best_model.pt",
                    epoch,
                    {"val_accuracy": val_acc, "val_loss": val_loss, "test_accuracy": test_acc},
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if config.patience and epochs_without_improvement >= config.patience:
                print("Early stopping triggered.")
                break

    plot_history(history, config.output_dir)
    (config.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    print(f"Best epoch: {best_epoch} with val acc {best_val_acc:.4f}")

    checkpoint_path = config.output_dir / "best_model.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])

    final_test_loss, final_test_acc = evaluate_tta(model, test_loader, criterion, device)
    print(f"Final Test Loss (TTA): {final_test_loss:.4f} Test Acc: {final_test_acc:.4f}")
    (config.output_dir / "test_metrics.json").write_text(
        json.dumps({"test_loss": final_test_loss, "test_accuracy": final_test_acc}, indent=2)
    )


if __name__ == "__main__":
    main()
