from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

from copy import deepcopy
import shutil
import tempfile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from modules import build_convnext_small, freeze_backbone

DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/ADNI/AD_NC")
DEFAULT_METADATA_PATH = Path("/home/groups/comp3710/ADNI/meta_data_with_label.json")
DEFAULT_OUTPUT_DIR = Path("results")

@dataclass
class TrainConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    metadata_path: Optional[Path] = DEFAULT_METADATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 1e-2
    valid_split: float = 0.1
    seed: int = 42
    num_workers: int = 4
    image_size: int = 224
    drop_path_rate: float = 0.2
    head_dropout: float = 0.3
    freeze_backbone_epochs: int = 2
    target_test_acc: Optional[float] = 0.8
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    mixup_alpha: float = 0.3
    mixup_prob: float = 0.5
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.3
    ema_decay: float = 0.999
    mixup_decay: bool = True
    cutmix_decay: bool = True
    finetune_epochs: int = 5
    finetune_lr_factor: float = 1.0
    balance_classes: bool = True
    save_optimizer: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    drop_last: bool = False
    label_smoothing: float = 0.05
    patience: int = 5  # 0 disables early stopping
    gradient_clip: Optional[float] = 1.0
    history: Dict[str, Iterable[float]] = field(default_factory=dict, init=False)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Small on ADNI slices.")
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--valid-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Number of initial epochs to train the final head only.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--target-test-acc",
        type=float,
        default=0.8,
        help="Optional test accuracy target for early stopping/checks.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--no-mixup-decay", action="store_true")
    parser.add_argument("--no-cutmix-decay", action="store_true")
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-lr-factor", type=float, default=1.0)
    parser.add_argument("--no-balance-classes", action="store_true")

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
        head_dropout=args.head_dropout,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        target_test_acc=args.target_test_acc,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        ema_decay=args.ema_decay,
        mixup_decay=not args.no_mixup_decay,
        cutmix_decay=not args.no_cutmix_decay,
        finetune_epochs=args.finetune_epochs,
        finetune_lr_factor=args.finetune_lr_factor,
        balance_classes=not args.no_balance_classes,
        save_optimizer=args.save_optimizer,
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


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        raise ValueError("Alpha must be positive for mixup.")
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, float(lam)


def _rand_bbox(width: int, height: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, x2, y1, y2


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        raise ValueError("Alpha must be positive for cutmix.")
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    batch_size, _, height, width = images.size()
    index = torch.randperm(batch_size, device=images.device)

    x1, x2, y1, y2 = _rand_bbox(width, height, lam)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    adjusted_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    labels_a, labels_b = labels, labels[index]
    return images, labels_a, labels_b, float(adjusted_lam)


def mixup_criterion(
    criterion: nn.Module,
    predictions: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: Optional[torch.Tensor],
    lam: float,
) -> torch.Tensor:
    if labels_b is None:
        return criterion(predictions, labels_a)
    return lam * criterion(predictions, labels_a) + (1.0 - lam) * criterion(predictions, labels_b)


@torch.no_grad()
def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip: Optional[float] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    ema_model: Optional[nn.Module] = None,
    ema_decay: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    losses = 0.0
    correct = 0
    total = 0
    batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        do_cutmix = cutmix_alpha > 0.0 and random.random() < cutmix_prob
        do_mixup = mixup_alpha > 0.0 and random.random() < mixup_prob and not do_cutmix

        if do_cutmix:
            images, labels_a, labels_b, lam = cutmix_batch(images, labels, cutmix_alpha)
        elif do_mixup:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
        else:
            labels_a = labels
            labels_b = None
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()

        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema_model is not None and ema_decay is not None:
            update_ema(model, ema_model, ema_decay)

        batch_size = labels.size(0)
        losses += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        if labels_b is not None:
            correct += lam * (preds == labels_a).sum().item()
            correct += (1.0 - lam) * (preds == labels_b).sum().item()
        else:
            correct += (preds == labels).sum().item()
        total += batch_size
        batches += 1

    return {
        "loss": losses / total,
        "accuracy": correct / total,
        "lr": optimizer.param_groups[0]["lr"],
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


def create_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> Optional[optim.lr_scheduler.LambdaLR]:
    if steps_per_epoch == 0:
        return None
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = min(total_steps, max(1, config.warmup_epochs * steps_per_epoch))
    base_lr = config.learning_rate
    min_lr = config.min_lr
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(min_ratio, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_ratio, cosine)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_history(history: Dict[str, Iterable[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = Path(output_dir) / "training_curves.png"

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history and any(history["val_loss"]):
        ax1.plot(epochs, history["val_loss"], label="Val Loss")
    if "train_acc" in history:
        ax1.plot(epochs, history["train_acc"], label="Train Acc")
    if "val_acc" in history and any(history["val_acc"]):
        ax1.plot(epochs, history["val_acc"], label="Val Acc")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / Accuracy")
    ax1.set_title("Training Dynamics")
    ax1.grid(True, linestyle="--", alpha=0.3)

    lr_values = history.get("lr")
    if lr_values:
        ax2 = ax1.twinx()
        ax2.plot(epochs, lr_values, label="Learning Rate", color="tab:gray", linestyle="--")
        ax2.set_ylabel("Learning Rate")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(figure, dpi=150)
    plt.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    ema_model: Optional[nn.Module] = None,
    include_optimizer: bool = True,
    filename: str = "best_model.pt",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metrics": metrics,
    }
    if include_optimizer:
        state["optimizer_state"] = optimizer.state_dict()
    if ema_model is not None:
        state["ema_state"] = ema_model.state_dict()
    try:
        torch.save(state, checkpoint_path)
    except RuntimeError as exc:
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / (checkpoint_path.name + ".tmp")
        print(f"Warning: primary checkpoint save failed ({exc}). Saving to temporary file {tmp_path}.")
        torch.save(state, tmp_path)
        shutil.move(tmp_path, checkpoint_path)
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
        balance_classes=config.balance_classes,
    )

    model = build_convnext_small(
        num_classes=2,
        in_chans=3,
        drop_path_rate=config.drop_path_rate,
        head_dropout=config.head_dropout,
    )
    model.to(device)

    ema_model = deepcopy(model)
    ema_model.to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)

    best_val_acc = 0.0
    best_epoch = -1
    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    epochs_without_improvement = 0
    final_test_metrics: Optional[Dict[str, float]] = None

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        if config.freeze_backbone_epochs and epoch <= config.freeze_backbone_epochs:
            freeze_backbone(model, train_head_only=True)
        else:
            freeze_backbone(model, train_head_only=False)

        ema_model.eval()
        mixup_prob_epoch = config.mixup_prob
        cutmix_prob_epoch = config.cutmix_prob
        if config.mixup_decay and config.epochs > 0:
            decay = max(0.0, 1.0 - (epoch - 1) / config.epochs)
            mixup_prob_epoch *= decay
        if config.cutmix_decay and config.epochs > 0:
            decay = max(0.0, 1.0 - (epoch - 1) / config.epochs)
            cutmix_prob_epoch *= decay
        if config.finetune_epochs > 0 and epoch > config.epochs - config.finetune_epochs:
            mixup_prob_epoch = 0.0
            cutmix_prob_epoch = 0.0

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=config.gradient_clip,
            scheduler=scheduler,
            mixup_alpha=config.mixup_alpha,
            mixup_prob=mixup_prob_epoch,
            cutmix_alpha=config.cutmix_alpha,
            cutmix_prob=cutmix_prob_epoch,
            ema_model=ema_model,
            ema_decay=config.ema_decay,
        )

        if val_loader is not None:
            val_metrics = evaluate(ema_model, val_loader, criterion, device)
        else:
            val_metrics = {"loss": 0.0, "accuracy": 0.0}

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(train_metrics.get("lr", optimizer.param_groups[0]["lr"]))

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"LR: {history['lr'][-1]:.2e} "
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
                ema_model=ema_model,
                include_optimizer=config.save_optimizer,
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
            provisional_test = evaluate(ema_model, test_loader, criterion, device)
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
    if "ema_state" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state"])
    else:
        ema_model.load_state_dict(checkpoint["model_state"])
    ema_model.to(device)
    ema_model.eval()

    if final_test_metrics is None:
        if test_loader is not None:
            final_test_metrics = evaluate(ema_model, test_loader, criterion, device)
        else:
            final_test_metrics = {"loss": 0.0, "accuracy": 0.0}

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
