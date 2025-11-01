from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 3e-3
    valid_split: float = 0.1
    seed: int = 42
    num_workers: int = 0
    image_size: int = 224
    drop_path_rate: float = 0.2
    head_dropout: float = 0.3
    freeze_backbone_epochs: int = 2
    label_smoothing: float = 0.05
    patience: int = 8
    gradient_clip: Optional[float] = 1.0
    augment_fraction: float = 0.3
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    mixup_alpha: float = 0.3
    mixup_prob: float = 0.5
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.3
    ema_decay: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    drop_last: bool = False
    history: Dict[str, Iterable[float]] = field(default_factory=dict, init=False)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Small on ADNI slices.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--valid-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--augment-fraction", type=float, default=0.3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
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
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        valid_split=args.valid_split,
        seed=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
        drop_path_rate=args.drop_path_rate,
        head_dropout=args.head_dropout,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        augment_fraction=args.augment_fraction,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        ema_decay=args.ema_decay,
        device=device,
        pin_memory=pin_memory,
        drop_last=args.drop_last,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], float(lam)


def _rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
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
    images: torch.Tensor, labels: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    shuffled = images[index]

    _, _, height, width = images.size()
    x1, x2, y1, y2 = _rand_bbox(width, height, lam)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = shuffled[:, :, y1:y2, x1:x2]

    adjusted_lam = 1.0 - (x2 - x1) * (y2 - y1) / (width * height)
    return images, labels, labels[index], float(adjusted_lam)


def mixup_criterion(
    criterion: nn.Module, preds: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float
) -> torch.Tensor:
    return lam * criterion(preds, targets_a) + (1.0 - lam) * criterion(preds, targets_b)


@torch.no_grad()
def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


def create_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> Optional[optim.lr_scheduler.LambdaLR]:
    if steps_per_epoch == 0:
        return None

    total_steps = config.epochs * steps_per_epoch
    warmup_steps = min(total_steps, max(1, config.warmup_epochs * steps_per_epoch))
    base_lr = config.learning_rate
    min_ratio = config.min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(min_ratio, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_ratio, cosine)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        apply_cutmix = cutmix_alpha > 0 and random.random() < cutmix_prob
        apply_mixup = mixup_alpha > 0 and random.random() < mixup_prob and not apply_cutmix

        if apply_cutmix:
            images, targets_a, targets_b, lam = cutmix_batch(images, labels, cutmix_alpha)
        elif apply_mixup:
            images, targets_a, targets_b, lam = mixup_batch(images, labels, mixup_alpha)
        else:
            targets_a, targets_b, lam = labels, None, 1.0

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        if targets_b is not None:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets_a)
        loss.backward()

        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema_model is not None and ema_decay is not None:
            update_ema(model, ema_model, ema_decay)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        if targets_b is not None:
            running_correct += lam * (preds == targets_a).sum().item()
            running_correct += (1.0 - lam) * (preds == targets_b).sum().item()
        else:
            running_correct += (preds == labels).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "accuracy": running_correct / total,
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

    return {
        "loss": running_loss / total if total else 0.0,
        "accuracy": running_correct / total if total else 0.0,
    }


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)

        logits_accum = None
        loss_accum = 0.0
        for transform in ("identity", "hflip"):
            if transform == "hflip":
                augmented = torch.flip(images, dims=[3])
            else:
                augmented = images
            outputs = model(augmented)
            loss_accum += criterion(outputs, labels).item()
            if logits_accum is None:
                logits_accum = outputs
            else:
                logits_accum += outputs

        logits_accum /= 2.0
        loss_accum /= 2.0

        batch_size = labels.size(0)
        running_loss += loss_accum * batch_size
        running_correct += (logits_accum.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total if total else 0.0,
        "accuracy": running_correct / total if total else 0.0,
    }


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
    if "test_acc_history" in history and history["test_acc_history"]:
        plt.plot(
            epochs[: len(history["test_acc_history"])],
            history["test_acc_history"],
            label="Test Acc (running)",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training / Validation / Test Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def _state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor for key, tensor in state_dict.items()}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    ema_model: Optional[nn.Module] = None,
    filename: str = "best_model.pt",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    state = {
        "epoch": epoch,
        "model_state": _state_dict_to_cpu(model.state_dict()),
        "metrics": metrics,
    }
    if ema_model is not None:
        state["ema_state"] = _state_dict_to_cpu(ema_model.state_dict())

    try:
        torch.save(state, checkpoint_path)
    except RuntimeError as exc:
        from tempfile import NamedTemporaryFile

        print(f"Warning: checkpoint save failed ({exc}); retrying with legacy serialization.")
        with NamedTemporaryFile(dir=str(output_dir), suffix=".pt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        torch.save(state, tmp_path, _use_new_zipfile_serialization=False)
        tmp_path.replace(checkpoint_path)

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
        augment_fraction=config.augment_fraction,
    )

    model = build_convnext_small(
        num_classes=2,
        in_chans=3,
        drop_path_rate=config.drop_path_rate,
        head_dropout=config.head_dropout,
        pretrained=True,
    ).to(device)

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

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc_history": [],
    }
    best_val_acc = 0.0
    best_epoch = -1
    epochs_without_improvement = 0
    last_test_acc = None

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        if epoch <= config.freeze_backbone_epochs:
            freeze_backbone(model, train_head_only=True)
        else:
            freeze_backbone(model, train_head_only=False)

        # decay augmentation intensities near the end
        mixup_prob = config.mixup_prob
        cutmix_prob = config.cutmix_prob
        if epoch > config.epochs * 0.7:
            mixup_prob *= 0.2
            cutmix_prob *= 0.2

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=config.gradient_clip,
            scheduler=scheduler,
            mixup_alpha=config.mixup_alpha,
            mixup_prob=mixup_prob,
            cutmix_alpha=config.cutmix_alpha,
            cutmix_prob=cutmix_prob,
            ema_model=ema_model,
            ema_decay=config.ema_decay,
        )

        val_metrics = (
            evaluate(ema_model, val_loader, criterion, device) if val_loader else {"loss": 0.0, "accuracy": 0.0}
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"LR: {train_metrics['lr']:.2e} "
            f"Time: {elapsed:.1f}s"
        )

        if val_loader:
            if val_metrics["accuracy"] >= best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    {"val_accuracy": best_val_acc, "val_loss": val_metrics["loss"]},
                    config.output_dir,
                    ema_model=ema_model,
                )
                test_metrics = evaluate_tta(ema_model, test_loader, criterion, device)
                last_test_acc = test_metrics["accuracy"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if config.patience and epochs_without_improvement >= config.patience:
                print("Early stopping triggered.")
                break

        history["test_acc_history"].append(last_test_acc if last_test_acc is not None else 0.0)

    plot_history(history, config.output_dir)
    (config.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    print(f"Best epoch: {best_epoch} with val acc {best_val_acc:.4f}")

    checkpoint_path = config.output_dir / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "ema_state" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state"])
        else:
            ema_model.load_state_dict(checkpoint["model_state"])
    else:
        ema_model.load_state_dict(model.state_dict())

    ema_model.to(device)
    ema_model.eval()
    test_metrics = evaluate_tta(ema_model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f} Test Acc: {test_metrics['accuracy']:.4f}")
    (config.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
