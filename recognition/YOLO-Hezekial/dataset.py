from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ISICDatasetPaths:
    """
    Collection of folder names used by the ISIC 2018 Task 1-2 dataset release.
    """

    root: str = "/home/groups/comp3710/ISIC2018/"
    train_images: str = "ISIC2018_Task1-2_Training_Input_x2"
    train_masks: str = "ISIC2018_Task1_Training_GroundTruth_x2"
    test_images: str = "ISIC2018_Task1-2_Test_Input"

    def resolve(self) -> Tuple[Path, Path, Path]:
        """Return resolved Paths for the configured folders."""
        root = Path(self.root).expanduser()
        return (
            root / self.train_images,
            root / self.train_masks,
            root / self.test_images,
        )


@dataclass
class DataModuleConfig:
    """
    Parameters controlling dataset creation and dataloader construction.
    """

    paths: ISICDatasetPaths = field(default_factory=ISICDatasetPaths)
    val_split: float = 0.2
    seed: int = 13
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    image_size: Optional[Tuple[int, int]] = (640, 640)


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------


class ISICDetectionDataset(Dataset):
    """
    Lesion detection dataset derived from the ISIC 2018 Task 1-2 release.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_ids: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Tensor, Dict[str, Tensor]], Tuple[Tensor, Dict[str, Tensor]]]] = None,
        image_size: Optional[Tuple[int, int]] = (640, 640),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.image_size = image_size

        paths = ISICDatasetPaths(root=str(root))
        train_images, train_masks, test_images = paths.resolve()

        if split in {"train", "val"}:
            self.image_dir = train_images
            self.mask_dir = train_masks
        elif split == "test":
            self.image_dir = test_images
            self.mask_dir = None
        else:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train', 'val', or 'test'.")

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory '{self.image_dir}' does not exist.")
        if self.mask_dir is not None and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory '{self.mask_dir}' does not exist.")

        if image_ids is not None:
            self.ids = list(image_ids)
        else:
            self.ids = discover_image_ids(self.image_dir)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_id = self.ids[index]
        image_path = resolve_image_path(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        if self.image_size is not None:
            image = image.resize(self.image_size[::-1], Image.BILINEAR)

        image_tensor = _pil_to_tensor(image)
        target = self._build_target(image_id, image_tensor.shape[1:])

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

    def _build_target(self, image_id: str, spatial_shape: Tuple[int, int]) -> Dict[str, Tensor]:
        height, width = spatial_shape
        target: Dict[str, Tensor] = {
            "image_id": torch.tensor([_hash_id(image_id)], dtype=torch.int64),
            "orig_id": image_id,
        }

        if self.mask_dir is None:
            # No annotations available (e.g. test split)
            empty = torch.zeros((0, 4), dtype=torch.float32)
            target.update(
                {
                    "boxes": empty,
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "areas": torch.zeros((0,), dtype=torch.float32),
                    "iscrowd": torch.zeros((0,), dtype=torch.int64),
                    "masks": torch.zeros((0, height, width), dtype=torch.uint8),
                }
            )
            return target

        mask_path = resolve_mask_path(self.mask_dir, image_id)
        mask = Image.open(mask_path).convert("L")

        if self.image_size is not None:
            mask = mask.resize(self.image_size[::-1], Image.NEAREST)

        mask_np = (np.array(mask) > 0).astype(np.uint8)
        masks = torch.from_numpy(mask_np[None, ...])  # (1, H, W) – lesion assumed single instance
        boxes = mask_to_boxes(mask_np)

        labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target.update(
            {
                "boxes": boxes,
                "labels": labels,
                "areas": areas,
                "iscrowd": iscrowd,
                "masks": masks,
            }
        )
        return target


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def discover_image_ids(image_dir: Path) -> List[str]:
    """Return sorted image identifiers without file extensions."""
    ids: List[str] = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        ids.extend([path.stem for path in image_dir.glob(extension)])
    if not ids:
        raise RuntimeError(f"No image files found in '{image_dir}'.")
    ids.sort()
    return ids


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    """Resolve the best matching image path for a given identifier."""
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate an image file for '{image_id}' in '{image_dir}'.")


def resolve_mask_path(mask_dir: Path, image_id: str) -> Path:
    """Retrieve the segmentation mask path for a given image id."""
    for extension in (".png", ".jpg", ".bmp"):
        candidate = mask_dir / f"{image_id}_segmentation{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate a segmentation mask for '{image_id}' in '{mask_dir}'.")


def _pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL image to a float tensor in CxHxW format."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=2)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def mask_to_boxes(mask: np.ndarray) -> Tensor:
    """
    Convert a binary mask into a tensor of bounding boxes in ``xyxy`` format.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    # Single component fallback (common case).
    if mask.max() == 1:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([[x_min, y_min, x_max + 1, y_max + 1]], dtype=np.float32)
        return torch.from_numpy(box)

    # Multiple components (mask encoded with unique ids per lesion).
    boxes: List[List[float]] = []
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue
        instance = mask == instance_id
        y_coords, x_coords = np.where(instance)
        if len(x_coords) == 0:
            continue
        boxes.append([x_coords.min(), y_coords.min(), x_coords.max() + 1, y_coords.max() + 1])

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def _hash_id(identifier: str) -> int:
    """Deterministically convert a string identifier into a 32-bit integer."""
    return hash(identifier) & 0xFFFFFFFF


def split_image_ids(
    ids: Sequence[str],
    val_split: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split a list of image identifiers into train/validation subsets.
    """
    ids = list(ids)
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be within (0, 1).")

    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = max(1, int(len(ids) * val_split))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])
    return train_ids, val_ids


def detection_collate_fn(batch: Iterable[Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    """
    Collate function suitable for detection batches with varying annotations.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def create_dataloaders(
    config: DataModuleConfig,
    train_transforms: Optional[
        Callable[[Tensor, Dict[str, Tensor]], Tuple[Tensor, Dict[str, Tensor]]]
    ] = None,
    eval_transforms: Optional[
        Callable[[Tensor, Dict[str, Tensor]], Tuple[Tensor, Dict[str, Tensor]]]
    ] = None,
) -> Dict[str, DataLoader]:
    """
    Factory returning dataloaders for train/val/test splits.
    """
    train_dir, mask_dir, test_dir = config.paths.resolve()
    all_ids = discover_image_ids(train_dir)
    train_ids, val_ids = split_image_ids(all_ids, config.val_split, config.seed)

    train_dataset = ISICDetectionDataset(
        root=config.paths.root,
        split="train",
        image_ids=train_ids,
        transforms=train_transforms,
        image_size=config.image_size,
    )
    val_dataset = ISICDetectionDataset(
        root=config.paths.root,
        split="val",
        image_ids=val_ids,
        transforms=eval_transforms,
        image_size=config.image_size,
    )
    test_dataset = ISICDetectionDataset(
        root=config.paths.root,
        split="test",
        transforms=eval_transforms,
        image_size=config.image_size,
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            collate_fn=detection_collate_fn,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=detection_collate_fn,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=detection_collate_fn,
        ),
    }
    return loaders


__all__ = [
    "ISICDatasetPaths",
    "ISICDetectionDataset",
    "DataModuleConfig",
    "create_dataloaders",
    "split_image_ids",
    "detection_collate_fn",
    "discover_image_ids",
    "resolve_image_path",
    "resolve_mask_path",
    "mask_to_boxes",
]
