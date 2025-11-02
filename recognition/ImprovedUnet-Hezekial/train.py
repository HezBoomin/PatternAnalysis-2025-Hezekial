from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset import DataConfig, create_dataloaders
from modules import DiceLoss, ImprovedUNet, UNetConfig, build_improved_unet, dice_coefficient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Improved U-Net on OASIS brain dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory of the OASIS PNG dataset. "
        "If omitted, tries $OASIS_DATA_ROOT or /home/groups/comp3710/OASIS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/improved-unet"),
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for all splits.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--image-size", type=int, default=256, help="Square crop/resize side length.")
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentation.")
    parser.add_argument(
        "--deep-supervision-weight", type=float, default=0.4, help="Weight for each auxiliary decoder head loss."
    )
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Scale factor for Dice loss component.")
    parser.add_argument("--ce-weight", type=float, default=1.0, help="Scale factor for CrossEntropy loss component.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def resolve_data_root(cli_value: Path | None) -> Path:
    """
    Locate the dataset root using CLI input, environment variable, or default path.
    """
    candidates: List[Path] = []
    if cli_value is not None:
        candidates.append(cli_value.expanduser())
    env_value = os.environ.get("OASIS_DATA_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(Path("/home/groups/comp3710/OASIS"))

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate the OASIS dataset. "
        "Pass --data-root, configure $OASIS_DATA_ROOT, or place the data under /home/groups/comp3710/OASIS."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep some stochasticity for better generalisation
    torch.backends.cudnn.benchmark = True


def prepare_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(in_channels: int, num_classes: int) -> ImprovedUNet:
    config = UNetConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=48,  # Slightly wider encoder helps boost Dice without exploding params.
        depth=4,
        dropout=0.1,
        use_attention=True,
        deep_supervision=True,
    )
    return build_improved_unet(config)


def compute_loss(
    outputs,
    targets: torch.Tensor,
    *,
    dice_loss: DiceLoss,
    ce_loss_fn: nn.Module,
    dice_weight: float,
    ce_weight: float,
    aux_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combine main and auxiliary predictions into a single loss scalar.
    """
    if isinstance(outputs, tuple):
        main_logits, aux_logits = outputs
    else:
        main_logits, aux_logits = outputs, []

    losses = {}

    ce_main = ce_loss_fn(main_logits, targets)
    dice_main = dice_loss(main_logits, targets)
    total = ce_weight * ce_main + dice_weight * dice_main
    losses["ce/main"] = float(ce_main.detach())
    losses["dice/main"] = float(dice_main.detach())

    if aux_logits:
        aux_loss = 0.0
        for aux_idx, aux_logit in enumerate(aux_logits, start=1):
            ce_aux = ce_loss_fn(aux_logit, targets)
            dice_aux = dice_loss(aux_logit, targets)
            aux_term = ce_weight * ce_aux + dice_weight * dice_aux
            aux_loss = aux_loss + aux_term
            losses[f"ce/aux{aux_idx}"] = float(ce_aux.detach())
            losses[f"dice/aux{aux_idx}"] = float(dice_aux.detach())
        total = total + aux_weight * aux_loss / len(aux_logits)

    losses["total"] = float(total.detach())
    return total, losses


@torch.no_grad()
def accumulate_dice_stats(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return per-class intersection and cardinality tensors for Dice computation.
    """
    probs = torch.softmax(logits, dim=1)
    if targets.ndim == probs.ndim:
        target_one_hot = targets.float()
    else:
        target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersect = torch.sum(probs * target_one_hot, dim=dims)
    cardinality = torch.sum(probs + target_one_hot, dim=dims)
    return intersect, cardinality


def update_history(history: Dict[str, List[float]], values: Dict[str, float]) -> None:
    for key, value in values.items():
        history.setdefault(key, []).append(value)


def plot_history(history: Dict[str, List[float]], output_dir: Path) -> None:
    """
    Save loss and Dice coefficient curves to disk for quick inspection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train/loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train/loss"], label="Train")
    plt.plot(epochs, history["val/loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train/dice"], label="Train")
    plt.plot(epochs, history["val/dice"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.title("Dice Coefficient")
    plt.legend()

    plt.tight_layout()
    fig_path = output_dir / "training_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()


def save_checkpoint(
    state: Dict,
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    dice_loss: DiceLoss,
    ce_loss_fn: nn.Module,
    dice_weight: float,
    ce_weight: float,
    aux_weight: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss, _ = compute_loss(
                outputs,
                labels,
                dice_loss=dice_loss,
                ce_loss_fn=ce_loss_fn,
                dice_weight=dice_weight,
                ce_weight=ce_weight,
                aux_weight=aux_weight,
            )

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            batch_dice = dice_coefficient(logits, labels, apply_softmax=True).mean()

        total_loss += loss.detach().item()
        total_dice += batch_dice.item()
        batches += 1

    return total_loss / batches, total_dice / batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    dice_loss: DiceLoss,
    ce_loss_fn: nn.Module,
    dice_weight: float,
    ce_weight: float,
    aux_weight: float,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0

    # Accumulate intersection/cardinality to compute dataset-level Dice per class.
    num_classes = model.classifier.out_channels
    sum_intersect = torch.zeros(num_classes, device=device)
    sum_cardinality = torch.zeros(num_classes, device=device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss, _ = compute_loss(
            outputs,
            labels,
            dice_loss=dice_loss,
            ce_loss_fn=ce_loss_fn,
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            aux_weight=aux_weight,
        )

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        batch_dice = dice_coefficient(logits, labels, apply_softmax=True).mean()

        intersect, cardinality = accumulate_dice_stats(logits, labels, num_classes)
        sum_intersect += intersect
        sum_cardinality += cardinality

        total_loss += loss.detach().item()
        total_dice += batch_dice.item()
        batches += 1

    mean_loss = total_loss / batches
    mean_dice = total_dice / batches
    dice_per_class = (2 * sum_intersect / (sum_cardinality + 1e-6)).detach().cpu().numpy()
    return mean_loss, mean_dice, dice_per_class


def log_epoch(epoch: int, phase: str, loss: float, dice: float, dice_per_class: np.ndarray) -> None:
    dice_str = ", ".join([f"class{i}:{score:.3f}" for i, score in enumerate(dice_per_class)])
    print(f"[Epoch {epoch:03d}] {phase:<5} | loss={loss:.4f} | mean_dice={dice:.4f} | {dice_str}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = prepare_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_root = resolve_data_root(args.data_root)

    data_config = DataConfig(
        root=data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_classes=4,
        normalise=True,
        mean=0.5,
        std=0.5,
        train_augment=not args.no_augment,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        preload_train=False,
    )

    loaders = create_dataloaders(data_config)

    model = build_model(in_channels=1, num_classes=data_config.num_classes).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(args.epochs // 6, 10), T_mult=2, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    best_val_dice = -math.inf
    history = {
        "train/loss": [],
        "train/dice": [],
        "val/loss": [],
        "val/dice": [],
    }

    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint.get("best_val_dice", best_val_dice)
        history = checkpoint.get("history", history)
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    best_path = args.output_dir / "best_model.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scheduler,
            device,
            dice_loss_fn,
            ce_loss_fn,
            args.dice_weight,
            args.ce_weight,
            args.deep_supervision_weight,
            scaler,
        )

        val_loss, val_dice, val_dice_per_class = evaluate(
            model,
            loaders["val"],
            device,
            dice_loss_fn,
            ce_loss_fn,
            args.dice_weight,
            args.ce_weight,
            args.deep_supervision_weight,
        )

        history["train/loss"].append(train_loss)
        history["train/dice"].append(train_dice)
        history["val/loss"].append(val_loss)
        history["val/dice"].append(val_dice)

        log_epoch(epoch, "Train", train_loss, train_dice, np.zeros_like(val_dice_per_class))
        log_epoch(epoch, "Val", val_loss, val_dice, val_dice_per_class)

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_path = args.output_dir / "best_model.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "best_val_dice": best_val_dice,
                    "history": history,
                    "args": {**vars(args), "data_root": str(data_root)},
                },
                best_path,
            )

        latest_path = args.output_dir / "latest_model.pt"
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_val_dice": best_val_dice,
                "history": history,
                "args": {**vars(args), "data_root": str(data_root)},
            },
            latest_path,
        )

        # Early stopping once the network consistently exceeds the 0.9 Dice threshold.
        if val_dice_per_class.min() >= 0.9:
            print(f"All classes reached Dice >= 0.9 on validation at epoch {epoch}. Stopping early.")
            break

    plot_history(history, args.output_dir)

    # Evaluate the best checkpoint on the test set for final reporting.
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])

    test_loss, test_dice, test_dice_per_class = evaluate(
        model,
        loaders["test"],
        device,
        dice_loss_fn,
        ce_loss_fn,
        args.dice_weight,
        args.ce_weight,
        args.deep_supervision_weight,
    )
    log_epoch(0, "Test", test_loss, test_dice, test_dice_per_class)

    metrics = {
        "train": {
            "loss": history["train/loss"],
            "dice": history["train/dice"],
        },
        "val": {
            "loss": history["val/loss"],
            "dice": history["val/dice"],
        },
        "test": {
            "loss": float(test_loss),
            "dice": float(test_dice),
            "dice_per_class": test_dice_per_class.tolist(),
        },
        "best_val_dice": float(best_val_dice),
        "args": {**vars(args), "data_root": str(data_root)},
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Training complete. Metrics saved to {metrics_path}.")


if __name__ == "__main__":
    main()
