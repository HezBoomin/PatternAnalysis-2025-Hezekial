from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import create_default_transforms
from modules import build_convnext_small


DEFAULT_CHECKPOINT = Path("results/best_model.pt")
DEFAULT_OUTPUT_DIR = Path("results/predictions")
DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/ADNI/AD_NC")
CLASS_NAMES = {0: "NC", 1: "AD"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ConvNeXt-Small inference on ADNI brain slices."
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional image files (JPEG/PNG) to classify. "
            "If omitted, default samples are drawn from the ADNI test set."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint produced by train.py (default: results/best_model.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store annotated prediction figures.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=4,
        help="Number of default images to evaluate when --images is not supplied.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (defaults to CUDA when available).",
    )
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_convnext_small(num_classes=2, in_chans=3)
    state = torch.load(checkpoint, map_location=device)
    if "ema_state" in state:
        state_dict = state["ema_state"]
    elif "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image_path: Path, image_size: int = 224) -> torch.Tensor:
    transform = create_default_transforms(image_size=image_size, augment=False)
    with Image.open(image_path) as img:
        img = img.convert("L")
    tensor = transform(img)
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = model(inputs.to(device))
    return torch.softmax(logits, dim=1).cpu()


def render_prediction(image_path: Path, probs: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    img = Image.open(image_path).convert("L")
    ax.imshow(img, cmap="gray")
    title = "\n".join(f"{label}: {prob:.2%}" for label, prob in probs.items())
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def gather_default_images(root: Path, limit: int) -> List[Path]:
    search_dirs = [
        root / "test" / "AD",
        root / "test" / "NC",
        root / "train" / "AD",
        root / "train" / "NC",
    ]
    collected: List[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in ALLOWED_EXTENSIONS:
                collected.append(path)
                if len(collected) >= limit:
                    return collected
    return collected


def run_inference(
    image_paths: List[Path],
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    model = load_model(checkpoint, device)

    for image_path in image_paths:
        image_tensor = preprocess(image_path)
        probs_tensor = predict(model, image_tensor, device).squeeze()
        probs = {CLASS_NAMES[idx]: float(probs_tensor[idx]) for idx in range(len(CLASS_NAMES))}
        predicted_class = max(probs, key=probs.get)

        print(f"Image: {image_path}")
        for label, prob in probs.items():
            print(f"  {label}: {prob:.4f}")
        print(f"Predicted: {predicted_class}")

        output_path = output_dir / f"{image_path.stem}_prediction.png"
        render_prediction(image_path, probs, output_path)
        print(f"Saved visualisation to: {output_path}\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.images:
        image_paths = args.images
    else:
        image_paths = gather_default_images(DEFAULT_DATA_ROOT, args.sample_limit)
        if not image_paths:
            raise FileNotFoundError(
                f"No images found under {DEFAULT_DATA_ROOT}. "
                "Please supply explicit paths with --images."
            )
        print("No images provided; using default samples:")
        for path in image_paths:
            print(f"  {path}")

    run_inference(image_paths, args.checkpoint, args.output_dir, device)


if __name__ == "__main__":
    main()
