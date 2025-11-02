from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import DataConfig, OASISBrainDataset, create_dataloaders
from modules import UNetConfig, build_improved_unet, dice_coefficient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the Improved U-Net.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing OASIS PNG folders.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--output-dir", type=Path, default=Path("./predictions"), help="Where to save outputs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of samples to visualise.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize images before inference.")
    parser.add_argument("--save-overlays", action="store_true", help="Save blended overlay images in addition to panels.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Computation device preference.")
    return parser.parse_args()


def prepare_device(device_preference: str) -> torch.device:
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Instantiate the Improved U-Net and restore trained weights.
    """
    config = UNetConfig(
        in_channels=1,
        num_classes=4,
        base_channels=48,
        depth=4,
        dropout=0.1,
        use_attention=True,
        deep_supervision=True,
    )
    model = build_improved_unet(config).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], np.ndarray]:
    """
    Collect predictions and compute running Dice scores across the dataset.
    """
    images_list: List[torch.Tensor] = []
    preds_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []

    num_classes = model.classifier.out_channels
    sum_intersect = torch.zeros(num_classes, device=device)
    sum_cardinality = torch.zeros(num_classes, device=device)

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        intersect = torch.sum(probs * F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2), dim=(0, 2, 3))
        cardinality = torch.sum(probs + F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2), dim=(0, 2, 3))
        sum_intersect += intersect
        sum_cardinality += cardinality

        images_list.extend(images.cpu())
        preds_list.extend(preds.cpu())
        masks_list.extend(masks.cpu())

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    dice_scores = (2 * sum_intersect / (sum_cardinality + 1e-6)).cpu().numpy()
    return images_list, preds_list, masks_list, dice_scores


def to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert single-channel tensor to a uint8 numpy image for visualisation.
    """
    array = tensor.squeeze().numpy()
    array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    return (array * 255).astype(np.uint8)


def create_overlay(image: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Blend prediction and ground truth overlays onto the grayscale image.
    """
    color_image = np.stack([image] * 3, axis=-1).astype(np.float32) / 255.0

    palette = np.array(
        [
            [0.0, 0.0, 0.0],  # background
            [0.8, 0.3, 0.3],
            [0.3, 0.8, 0.3],
            [0.3, 0.3, 0.8],
        ]
    )

    mask_color = palette[mask]
    pred_color = palette[pred]

    overlay = color_image * 0.6 + mask_color * 0.2 + pred_color * 0.2
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def save_visualisations(
    images: List[torch.Tensor],
    preds: List[torch.Tensor],
    masks: List[torch.Tensor],
    dice_scores: np.ndarray,
    output_dir: Path,
    num_samples: int,
    save_overlays: bool,
) -> None:
    """
    Plot sample predictions and optionally save standalone overlays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = min(num_samples, len(images))

    cols = 3
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for idx in range(num_samples):
        image_np = to_numpy_img(images[idx])
        mask_np = masks[idx].numpy().astype(np.uint8)
        pred_np = preds[idx].numpy().astype(np.uint8)
        overlay_np = create_overlay(image_np, mask_np, pred_np)

        ax = axes[idx]
        ax.imshow(overlay_np)
        ax.axis("off")
        ax.set_title(f"Sample {idx} | Dice per class: {dice_scores.round(3)}")

        if save_overlays:
            overlay_path = output_dir / f"sample_{idx:03d}_overlay.png"
            plt.imsave(overlay_path, overlay_np)

    for extra_ax in axes[num_samples:]:
        extra_ax.axis("off")

    panel_path = output_dir / "prediction_panel.png"
    fig.tight_layout()
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = prepare_device(args.device)

    data_config = DataConfig(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_classes=4,
        normalise=True,
        mean=0.5,
        std=0.5,
        train_augment=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        preload_train=False,
    )

    loaders = create_dataloaders(data_config, shuffle_train=False)
    test_loader = loaders["test"]

    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        images, preds, masks, dice_scores = run_inference(
            model,
            test_loader,
            device,
            max_batches=math.ceil(args.num_samples / args.batch_size) if args.num_samples > 0 else None,
        )

    mean_dice = dice_scores.mean()
    print("Per-class Dice:", dice_scores)
    print(f"Mean Dice: {mean_dice:.4f}")

    save_visualisations(images, preds, masks, dice_scores, args.output_dir, args.num_samples, args.save_overlays)
    print(f"Saved predictions to {args.output_dir}.")


if __name__ == "__main__":
    import math

    main()
