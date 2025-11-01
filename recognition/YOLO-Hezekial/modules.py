from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------


@dataclass
class YOLOModelConfig:
    """
    Declarative configuration for the lesion detector.
    """

    weights_path: str = "pretrained-model/yolo11s.pt"
    task: str = "detect"
    num_classes: int = 1
    class_names: Tuple[str, ...] = ("lesion",)
    device: str = "cuda"
    half_precision: bool = False
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 100
    freeze_backbone: bool = False
    tile_shape: Optional[Tuple[int, int]] = None
    tile_overlap: float = 0.2
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def as_predict_kwargs(self) -> Dict[str, Any]:
        return {
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_det,
            "half": self.half_precision,
            **self.extra_args,
        }


@dataclass
class DetectionResult:
    """
    Lightweight wrapper for a single-image detection result.

    The tensors are shaped as ``(num_detections, 4)`` for boxes and
    ``(num_detections,)`` for scores/labels.
    """

    boxes: Tensor
    scores: Tensor
    labels: Tensor
    path: Optional[str] = None
    original_shape: Optional[Tuple[int, int]] = None

    def cpu(self) -> "DetectionResult":
        """Return a copy of the result on CPU memory."""
        return DetectionResult(
            boxes=self.boxes.cpu(),
            scores=self.scores.cpu(),
            labels=self.labels.cpu(),
            path=self.path,
            original_shape=self.original_shape,
        )

    def to(self, device: torch.device | str) -> "DetectionResult":
        """Return a copy of the result on the requested device."""
        return DetectionResult(
            boxes=self.boxes.to(device),
            scores=self.scores.to(device),
            labels=self.labels.to(device),
            path=self.path,
            original_shape=self.original_shape,
        )


# ---------------------------------------------------------------------------
# Core detector wrapper
# ---------------------------------------------------------------------------


class YOLOLesionDetector:
    """
    Wrapper around YOLO specialised for ISIC lesion detection.

    The wrapper keeps the rest of the codebase agnostic to the underlying YOLO
    version while still exposing familiar hooks for fine-tuning.
    """

    def __init__(self, config: YOLOModelConfig):
        _raise_if_ultralytics_missing()
        self.config = config
        self.model = _create_ultralytics_model(config.weights_path, task=config.task)
        self._configure_model()
        self._maybe_move_to_device()

    # ------------------------------------------------------------------ setup

    def _configure_model(self) -> None:
        """Apply dataset-specific tweaks to the YOLO model instance."""
        model_impl = getattr(self.model, "model", None)

        if model_impl is not None:
            if hasattr(model_impl, "nc"):
                model_impl.nc = self.config.num_classes

            if hasattr(model_impl, "names"):
                model_impl.names = list(self.config.class_names)

            if self.config.freeze_backbone:
                _freeze_module_parameters(getattr(model_impl, "backbone", None))

        # Update Ultralytics metadata map so exports/reporting show correct names.
        if self.config.class_names:
            names_map = {idx: name for idx, name in enumerate(self.config.class_names)}
            try:
                self.model.names = names_map
            except AttributeError:
                if hasattr(self.model, "model") and hasattr(self.model.model, "names"):
                    self.model.model.names = list(self.config.class_names)
                elif hasattr(self.model, "overrides"):
                    self.model.overrides["names"] = list(self.config.class_names)

    def _maybe_move_to_device(self) -> None:
        """Move the model to the configured device if supported."""
        if self.config.device == "auto":
            return

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Ensure a CUDA-capable GPU is visible."
            )
        try:
            self.model.to(self.config.device)
        except AttributeError:
            if hasattr(self.model, "model"):
                self.model.model.to(self.config.device)

    # ------------------------------------------------------------------ API

    def forward(self, images: Tensor, **kwargs: Any) -> List[DetectionResult]:
        """
        Run a forward pass using already prepared tensors.
        """
        predictions = self.model(images, **kwargs)
        return [_to_detection_result(pred) for pred in predictions]

    def predict(self, source: Any, stream: bool = False, **kwargs: Any) -> Iterable[DetectionResult]:
        """
        Convenience wrapper around ``YOLO.predict`` returning ``DetectionResult`` objects.
        """
        predict_kwargs = {**self.config.as_predict_kwargs(), **kwargs}
        results = self.model.predict(source=source, stream=stream, **predict_kwargs)
        if stream:
            return (_to_detection_result(res) for res in results)
        return [_to_detection_result(res) for res in results]

    def save_checkpoint(self, path: str | Path) -> Path:
        """
        Save the current model weights to ``path`` using Ultralytics export logic.
        """
        output_path = Path(path).with_suffix(".pt")
        self.model.save(filename=str(output_path))
        return output_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load weights from a checkpoint path."""
        checkpoint_path = Path(path)
        self.model = _create_ultralytics_model(str(checkpoint_path), task=self.config.task)
        self._configure_model()
        self._maybe_move_to_device()


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """
    Compute pairwise intersection-over-union between two sets of boxes.
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), device=boxes_a.device)

    tl = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    hw = (br - tl).clamp(min=0)
    intersection = hw[..., 0] * hw[..., 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)

    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


class MeanAveragePrecision:
    """
    Simple stateful mAP calculator based on IoU thresholds.
    """

    def __init__(self, iou_thresholds: Iterable[float] = (0.5, 0.75, 0.9)):
        self.iou_thresholds = tuple(iou_thresholds)
        self.reset()

    def reset(self) -> None:
        """Clear internal buffers."""
        self._predictions: List[DetectionResult] = []
        self._targets: List[DetectionResult] = []

    def update(self, preds: Iterable[DetectionResult], targets: Iterable[DetectionResult]) -> None:
        """Add a batch of predictions/targets to the metric buffers."""
        self._predictions.extend(preds)
        self._targets.extend(targets)

    def compute(self) -> Dict[str, float]:
        """Return the averaged precision per IoU threshold."""
        if not self._predictions:
            return {f"mAP@{thr:.2f}": 0.0 for thr in self.iou_thresholds}

        aps: Dict[float, List[float]] = {thr: [] for thr in self.iou_thresholds}

        for pred, target in zip(self._predictions, self._targets):
            ious = box_iou(pred.boxes, target.boxes)
            for thr in self.iou_thresholds:
                matches = (ious >= thr).any(dim=1)
                tp = matches.sum().item()
                fp = len(matches) - tp
                fn = max(target.boxes.shape[0] - tp, 0)
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                aps[thr].append((precision + recall) / 2.0)

        return {f"mAP@{thr:.2f}": float(torch.tensor(values).mean().item()) for thr, values in aps.items()}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _raise_if_ultralytics_missing() -> None:
    """Ensure Ultralytics is available before constructing the detector."""
    try:
        importlib.import_module("ultralytics")
    except ImportError as exc:  # pragma: no cover - defensive path for missing dep
        raise ImportError(
            "Ultralytics is not installed. Install it with `pip install ultralytics` "
            "before using the YOLOLesionDetector."
        ) from exc


def _freeze_module_parameters(module: Any) -> None:
    """Freeze parameters in the given module if present."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


def _to_detection_result(ultralytics_result: Any) -> DetectionResult:
    """
    Convert an Ultralytics result object into ``DetectionResult``.
    """
    boxes = ultralytics_result.boxes
    if boxes is None:
        return DetectionResult(
            boxes=torch.empty((0, 4)),
            scores=torch.empty((0,)),
            labels=torch.empty((0,), dtype=torch.long),
            path=getattr(ultralytics_result, "path", None),
            original_shape=getattr(ultralytics_result, "orig_shape", None),
        )

    return DetectionResult(
        boxes=boxes.xyxy.to(torch.float32),
        scores=boxes.conf.to(torch.float32),
        labels=boxes.cls.to(torch.long),
        path=getattr(ultralytics_result, "path", None),
        original_shape=getattr(ultralytics_result, "orig_shape", None),
    )


def _create_ultralytics_model(weights_path: str, task: str) -> Any:
    """
    Lazily create a YOLO model instance using Ultralytics without direct imports.
    """
    module = importlib.import_module("ultralytics")
    yolo_cls = getattr(module, "YOLO", None)
    if yolo_cls is None:
        engine = importlib.import_module("ultralytics.yolo.engine.model")
        yolo_cls = getattr(engine, "YOLO")
    return yolo_cls(weights_path, task=task)


__all__ = [
    "YOLOModelConfig",
    "YOLOLesionDetector",
    "DetectionResult",
    "MeanAveragePrecision",
    "box_iou",
    "_create_ultralytics_model",
]
