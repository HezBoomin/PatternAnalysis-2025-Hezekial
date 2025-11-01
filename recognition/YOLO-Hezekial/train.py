import argparse
import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset import (
    DataModuleConfig,
    ISICDatasetPaths,
    ISICDetectionDataset,
    create_dataloaders,
    mask_to_boxes,
    resolve_image_path,
    resolve_mask_path,
)
from modules import (
    DetectionResult,
    MeanAveragePrecision,
    YOLOLesionDetector,
    YOLOModelConfig,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on ISIC lesions.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/groups/comp3710/ISIC2018/",
        help="Path to the root folder containing the ISIC 2018 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory used for experiment artefacts (checkpoints, plots, logs).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction reserved for validation.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for deterministic splits.")
    parser.add_argument("--img-size", type=int, default=640, help="Square image size used during export/training.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="isic-yolo",
        help="Name of the Ultralytics run directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs).")
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained-model/yolo11s.pt",
        help="Initial weights for fine-tuning (relative or absolute path).",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Regenerate YOLO formatted dataset even when cached artefacts exist.",
    )
    return parser


# ---------------------------------------------------------------------------
# Dataset preparation helpers
# ---------------------------------------------------------------------------


def prepare_yolo_dataset(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    paths: ISICDatasetPaths,
    export_dir: Path,
    class_names: Iterable[str],
    force: bool = False,
) -> Path:
    """
    Export PyTorch datasets to a YOLO-compatible folder structure.
    """
    logging.info("Exporting datasets to YOLO format at %s", export_dir)
    yaml_path = export_dir / "isic.yaml"
    if yaml_path.exists() and not force:
        logging.info("Existing YOLO dataset detected at %s; reuse without re-export.", export_dir)
        return export_dir

    image_dirs = {
        "train": Path(paths.root) / paths.train_images,
        "val": Path(paths.root) / paths.train_images,
        "test": Path(paths.root) / paths.test_images,
    }

    label_root = export_dir / "labels"
    image_root = export_dir / "images"
    label_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    split_to_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split, ids in split_to_ids.items():
        list_path = export_dir / f"{split}.txt"
        label_dir = label_root / split
        image_dir = image_root / split
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        with open(list_path, "w", encoding="utf-8") as list_file:
            for image_id in ids:
                source_image_path = resolve_image_path(image_dirs[split], image_id)
                target_image_path = image_dir / source_image_path.name
                if not target_image_path.exists():
                    try:
                        os.symlink(source_image_path, target_image_path)
                    except FileExistsError:
                        pass
                    except OSError:
                        # Symlink may be unsupported; fall back to copying the file once.
                        if not target_image_path.exists():
                            shutil.copy2(source_image_path, target_image_path)

                list_file.write(f"{target_image_path.resolve()}\n")

                label_path = label_dir / f"{image_id}.txt"
                if split == "test":
                    label_path.write_text("", encoding="utf-8")
                    continue

                mask_path = resolve_mask_path(Path(paths.root) / paths.train_masks, image_id)
                mask = Image.open(mask_path).convert("L")
                mask_np = (np.array(mask) > 0).astype(np.uint8)
                boxes = mask_to_boxes(mask_np)
                labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)

                write_yolo_labels(
                    label_path=label_path,
                    boxes=boxes,
                    labels=labels,
                    width=mask_np.shape[1],
                    height=mask_np.shape[0],
                )

    create_dataset_yaml(export_dir=export_dir, class_names=class_names)
    return export_dir


def write_yolo_labels(
    label_path: Path,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    width: int,
    height: int,
) -> None:
    """Write YOLO-format labels (class x_center y_center width height) to disk."""
    with open(label_path, "w", encoding="utf-8") as handle:
        if boxes.numel() == 0:
            return
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            handle.write(f"{int(label)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def create_dataset_yaml(export_dir: Path, class_names: Iterable[str]) -> Path:
    """Create a Ultralytics dataset YAML file pointing to the exported folders."""
    yaml_path = export_dir / "isic.yaml"
    names = list(class_names)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(f"path: {export_dir.resolve()}\n")
        handle.write("train: train.txt\n")
        handle.write("val: val.txt\n")
        handle.write("test: test.txt\n")
        handle.write("names:\n")
        for idx, name in enumerate(names):
            handle.write(f"  {idx}: {name}\n")
    logging.info("Dataset YAML written to %s", yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# Training, evaluation, plotting
# ---------------------------------------------------------------------------


def train_model(
    detector: YOLOLesionDetector,
    data_yaml: Path,
    output_dir: Path,
    experiment_name: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    patience: int,
    num_workers: int,
) -> Dict[str, Path]:
    """Launch Ultralytics training and return paths to useful artefacts."""
    logging.info("Starting YOLO training for %d epochs", epochs)
    safe_workers = max(0, min(num_workers, os.cpu_count() or 0))
    if safe_workers != num_workers:
        logging.info("Adjusting worker count from %d to %d", num_workers, safe_workers)

    detector.model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        project=str(output_dir / "runs"),
        name=experiment_name,
        exist_ok=True,
        device=detector.config.device,
        patience=patience,
        workers=safe_workers,
    )

    trainer = detector.model.trainer
    if trainer is None:
        raise RuntimeError("Ultralytics trainer did not initialise as expected.")

    best_path = Path(trainer.best)
    save_dir = Path(trainer.save_dir)
    logging.info("Training completed. Best weights at %s", best_path)
    return {
        "best": best_path,
        "last": Path(trainer.last),
        "results_csv": save_dir / "results.csv",
        "metrics_csv": save_dir / "metrics.csv",
    }


def evaluate_model(
    detector: YOLOLesionDetector,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """Run evaluation on a dataloader and return mAP metrics."""
    detector.model.to(device)
    detector.model.eval()
    metric = MeanAveragePrecision(iou_thresholds=(0.5, 0.75, 0.9, 0.95))

    for images, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
        np_images = [
            (img.mul(255.0).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)) for img in images
        ]
        predictions = [pred.cpu() for pred in detector.predict(np_images)]

        target_results: List[DetectionResult] = []
        for prediction, target in zip(predictions, targets):
            boxes = target["boxes"].to(torch.float32)
            labels = target["labels"].to(torch.int64)
            scores = torch.ones_like(labels, dtype=torch.float32)
            target_results.append(
                DetectionResult(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    path=target.get("orig_id"),
                    original_shape=prediction.original_shape,
                )
            )
        metric.update(predictions, target_results)

    return metric.compute()


def generate_training_plots(
    results_csv: Path,
    metrics_csv: Path,
    output_path: Path,
) -> None:
    """Plot training/validation curves from Ultralytics CSV outputs."""
    if not results_csv.exists():
        logging.warning("results.csv not found at %s; skipping plot generation.", results_csv)
        return

    epochs: List[int] = []
    train_box: List[float] = []
    val_box: List[float] = []
    train_obj: List[float] = []
    val_obj: List[float] = []
    map50: List[float] = []

    with open(results_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = int(float(row.get("epoch", len(epochs))))
            epochs.append(epoch)
            train_box.append(float(row.get("train/box_loss", 0.0)))
            val_box.append(float(row.get("val/box_loss", 0.0)))
            train_obj.append(float(row.get("train/obj_loss", 0.0)))
            val_obj.append(float(row.get("val/obj_loss", 0.0)))

    if metrics_csv.exists():
        with open(metrics_csv, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                map50.append(float(row.get("metrics/mAP50(B)", 0.0)))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_box, label="Train box loss")
    plt.plot(epochs, val_box, label="Val box loss")
    plt.plot(epochs, train_obj, label="Train obj loss")
    plt.plot(epochs, val_obj, label="Val obj loss")
    if map50:
        plt.plot(epochs[: len(map50)], map50, label="mAP@0.50", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / mAP")
    plt.title("Training progress")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info("Training curves saved to %s", output_path)


def run_test_inference(
    detector: YOLOLesionDetector,
    dataloader,
    output_path: Path,
) -> None:
    """Run inference on the test dataloader and store predictions as JSON."""
    predictions_log: List[Dict[str, object]] = []
    for images, targets in tqdm(dataloader, desc="Testing", unit="batch"):
        np_images = [
            (img.mul(255.0).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)) for img in images
        ]
        batch_predictions = [pred.cpu() for pred in detector.predict(np_images)]
        for prediction, target in zip(batch_predictions, targets):
            entry = {
                "image": target.get("orig_id", prediction.path),
                "boxes": prediction.boxes.tolist(),
                "scores": prediction.scores.tolist(),
                "labels": prediction.labels.tolist(),
            }
            predictions_log.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(predictions_log, handle, indent=2)
    logging.info("Test predictions saved to %s", output_path)


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
            logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8"),
        ],
    )

    logging.info("Training configuration: %s", vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------ setup
    class_names = ("lesion",)
    paths = ISICDatasetPaths(
        root=args.data_root,
        train_images="ISIC2018_Task1-2_Training_Input_x2",
        train_masks="ISIC2018_Task1_Training_GroundTruth_x2",
        test_images="ISIC2018_Task1-2_Test_Input",
    )
    data_config = DataModuleConfig(
        paths=paths,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=max(0, min(args.num_workers, os.cpu_count() or 0)),
        image_size=(args.img_size, args.img_size),
    )
    dataloaders = create_dataloaders(data_config)
    train_dataset = dataloaders["train"].dataset
    val_dataset = dataloaders["val"].dataset
    test_dataset = dataloaders["test"].dataset

    # Export dataset for Ultralytics training
    prepared_dir = output_dir / "yolo_dataset"
    dataset_dir = prepare_yolo_dataset(
        train_dataset.ids,
        val_dataset.ids,
        test_dataset.ids,
        paths,
        prepared_dir,
        class_names=class_names,
        force=args.force_export,
    )
    data_yaml = dataset_dir / "isic.yaml"

    # Model configuration
    model_config = YOLOModelConfig(
        weights_path=args.weights,
        device=args.device,
        class_names=class_names,
        num_classes=1,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        max_det=100,
        half_precision=False,
    )
    detector = YOLOLesionDetector(model_config)

    # Train and evaluate
    artefacts = train_model(
        detector=detector,
        data_yaml=data_yaml,
        output_dir=output_dir,
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.img_size,
        patience=args.patience,
        num_workers=args.num_workers,
    )

    detector.load_checkpoint(artefacts["best"])
    val_metrics = evaluate_model(detector, dataloaders["val"], device=torch.device(args.device))
    metrics_path = output_dir / "val_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(val_metrics, handle, indent=2)
    logging.info("Validation metrics: %s", val_metrics)

    generate_training_plots(
        artefacts["results_csv"],
        artefacts["metrics_csv"],
        output_path=output_dir / "training_curves.png",
    )

    # Run inference on test set
    run_test_inference(
        detector=detector,
        dataloader=dataloaders["test"],
        output_path=output_dir / "test_predictions.json",
    )

    # Save final checkpoint copy under results directory
    final_ckpt = output_dir / "lesion_detector_best.pt"
    exported_ckpt = detector.save_checkpoint(final_ckpt)
    logging.info("Checkpoint exported to %s", exported_ckpt)


if __name__ == "__main__":
    main()
