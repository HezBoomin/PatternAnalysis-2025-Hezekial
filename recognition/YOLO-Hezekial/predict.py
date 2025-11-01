from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from dataset import DataModuleConfig, ISICDatasetPaths, create_dataloaders
from modules import DetectionResult, YOLOLesionDetector, YOLOModelConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with the trained lesion detector.")
    parser.add_argument(
        "--weights",
        type=str,
        default="results/lesion_detector_best.pt",
        help="Path to trained weights (default assumes train.py output).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for inference (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image file or directory to run inference on. When omitted, the script uses the ISIC test split.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/groups/comp3710/ISIC2018/",
        help="Root directory containing the ISIC dataset folders (used when --source is not provided).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference",
        help="Directory where annotated images and predictions.json will be written.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence required for a detection to be visualised/logged.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size used when loading ISIC dataset samples.",
    )
    return parser


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------


def load_detector(weights_path: Path, device: str) -> YOLOLesionDetector:
    config = YOLOModelConfig(
        weights_path=str(weights_path),
        device=device,
        class_names=("lesion",),
        num_classes=1,
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    return YOLOLesionDetector(config)


def prepare_source_images(source: Path) -> List[Path]:
    """Return a sorted list of image paths from a file or directory."""
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
    image_paths: List[Path] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(source.glob(extension))
    if not image_paths:
        raise RuntimeError(f"No image files found in '{source}'.")
    return sorted(image_paths)


def image_to_numpy(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def run_inference_on_images(
    detector: YOLOLesionDetector,
    image_paths: Sequence[Path],
    output_dir: Path,
    score_threshold: float,
) -> List[dict]:
    """Run inference on explicit image paths and save annotated copies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_log: List[dict] = []

    for image_path in tqdm(image_paths, desc="Predicting", unit="img"):
        image_np = image_to_numpy(image_path)
        prediction = next(iter(detector.predict(image_np, stream=False, conf=score_threshold)))
        prediction = prediction.cpu()
        filtered = filter_predictions(prediction, score_threshold)
        annotated = draw_detections(image_np, filtered, detector.config.class_names)
        save_path = output_dir / image_path.name
        Image.fromarray(annotated).save(save_path)

        predictions_log.append(
            {
                "image": str(image_path),
                "boxes": filtered.boxes.tolist(),
                "scores": filtered.scores.tolist(),
                "labels": filtered.labels.tolist(),
            }
        )

    return predictions_log


def run_inference_on_test_split(
    detector: YOLOLesionDetector,
    data_root: Path,
    output_dir: Path,
    img_size: int,
    score_threshold: float,
) -> List[dict]:
    """Use the dataset loader to iterate through the test split."""
    paths = ISICDatasetPaths(
        root=str(data_root),
        train_images="ISIC2018_Task1-2_Training_Input_x2",
        train_masks="ISIC2018_Task1_Training_GroundTruth_x2",
        test_images="ISIC2018_Task1-2_Test_Input",
    )
    config = DataModuleConfig(
        paths=paths,
        val_split=0.2,
        seed=13,
        batch_size=4,
        num_workers=2,
        image_size=(img_size, img_size),
        pin_memory=True,
    )
    dataloaders = create_dataloaders(config)
    test_loader = dataloaders["test"]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_log: List[dict] = []

    for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Predicting test", unit="batch")):
        np_images = [
            (img.mul(255.0).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)) for img in images
        ]
        batch_predictions = [pred.cpu() for pred in detector.predict(np_images, conf=score_threshold)]

        for idx, (prediction, target) in enumerate(zip(batch_predictions, targets)):
            filtered = filter_predictions(prediction, score_threshold)
            image_id = str(target.get("orig_id", f"test_{batch_idx}_{idx}"))
            annotated = draw_detections(np_images[idx], filtered, detector.config.class_names)
            save_path = output_dir / f"{image_id}.jpg"
            Image.fromarray(annotated).save(save_path)
            predictions_log.append(
                {
                    "image": image_id,
                    "boxes": filtered.boxes.tolist(),
                    "scores": filtered.scores.tolist(),
                    "labels": filtered.labels.tolist(),
                }
            )

    return predictions_log


def filter_predictions(prediction: DetectionResult, threshold: float) -> DetectionResult:
    """Filter detections by confidence threshold."""
    if prediction.scores.numel() == 0:
        return prediction
    keep = prediction.scores >= threshold
    return DetectionResult(
        boxes=prediction.boxes[keep],
        scores=prediction.scores[keep],
        labels=prediction.labels[keep],
        path=prediction.path,
        original_shape=prediction.original_shape,
    )


def draw_detections(image: np.ndarray, result: DetectionResult, names: Tuple[str, ...]) -> np.ndarray:
    """Draw bounding boxes and labels on an RGB image array."""
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    font = _get_font(size=max(10, image.shape[0] // 35))
    colors = _palette()
    for idx, (box, score, label) in enumerate(zip(result.boxes, result.scores, result.labels)):
        x1, y1, x2, y2 = map(float, box.tolist())
        color = colors[int(label) % len(colors)]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        class_name = _lookup_name(names, int(label))
        caption = f"{class_name} {score:.2f}"
        try:
            text_size = draw.textbbox((x1, y1), caption, font=font)
        except AttributeError:
            w, h = draw.textsize(caption, font=font)
            text_size = (int(x1), int(y1) - h, int(x1 + w), int(y1))
        text_bg = [text_size[0] - 2, text_size[1] - 2, text_size[2] + 2, text_size[3] + 2]
        draw.rectangle([(text_bg[0], text_bg[1]), (text_bg[2], text_bg[3])], fill=color)
        draw.text((x1, y1), caption, fill="white", font=font)
    return np.array(canvas)


def _palette() -> List[Tuple[int, int, int]]:
    return [
        (244, 67, 54),
        (33, 150, 243),
        (76, 175, 80),
        (255, 193, 7),
        (156, 39, 176),
    ]


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _lookup_name(names: Iterable, label: int) -> str:
    if isinstance(names, dict):
        return str(names.get(label, f"class_{label}"))
    if isinstance(names, (list, tuple)):
        if 0 <= label < len(names):
            return str(names[label])
        return f"class_{label}"
    names_list = list(names)
    if 0 <= label < len(names_list):
        return str(names_list[label])
    return f"class_{label}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "predict.log", mode="a", encoding="utf-8"),
        ],
    )

    if "cuda" in args.device and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU.")
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    logging.info("Using device: %s", device)

    detector = load_detector(Path(args.weights), device=str(device))

    if args.source is not None:
        source_paths = prepare_source_images(Path(args.source))
        predictions = run_inference_on_images(
            detector=detector,
            image_paths=source_paths,
            output_dir=output_dir / "annotated",
            score_threshold=args.score_threshold,
        )
    else:
        predictions = run_inference_on_test_split(
            detector=detector,
            data_root=Path(args.data_root),
            output_dir=output_dir / "annotated",
            img_size=args.img_size,
            score_threshold=args.score_threshold,
        )

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)
    logging.info("Saved %d predictions to %s", len(predictions), output_dir / "predictions.json")


if __name__ == "__main__":
    main()
