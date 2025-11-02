from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


__all__ = [
    "OASISBrainDataset",
    "DataConfig",
    "create_dataloaders",
    "mask_name_from_image_name",
]


SPLIT_TO_DIRS = {
    "train": ("keras_png_slices_train", "keras_png_slices_seg_train"),
    "val": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test": ("keras_png_slices_test", "keras_png_slices_seg_test"),
}


LABEL_MAP = {
    0: 0,
    85: 1,
    170: 2,
    255: 3,
}


def mask_name_from_image_name(image_name: str) -> str:
    """
    Convert an image filename into the expected mask filename.
    """
    basename = Path(image_name).name
    if basename.startswith("case_"):
        return "seg_" + basename[len("case_") :]
    if basename.startswith("img_"):
        return basename.replace("img_", "seg_", 1)
    if basename.startswith("image_"):
        return basename.replace("image_", "seg_", 1)
    return basename


def _map_mask_to_indices(mask_array: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Map mask grayscale values to contiguous class ids.
    """
    mapped = np.zeros_like(mask_array, dtype=np.uint8)
    for value, idx in LABEL_MAP.items():
        mapped[mask_array == value] = idx

    if mapped.max() >= num_classes:
        mapped = np.clip(mapped, 0, num_classes - 1)

    return torch.from_numpy(mapped.astype(np.int64))


def _random_affine_params(max_rotate: float = 10.0, max_translate: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """
    Draw random rotation (deg) and translation offsets used during augmentation.
    """
    angle = random.uniform(-max_rotate, max_rotate)
    translate = (random.uniform(-max_translate, max_translate), random.uniform(-max_translate, max_translate))
    return angle, translate


class OASISBrainDataset(Dataset):
    """
    Torch dataset that loads slice-level OASIS images and masks with optional augmentation.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        image_size: int = 256,
        num_classes: int = 4,
        augment: bool = False,
        normalise: bool = True,
        mean: float = 0.5,
        std: float = 0.5,
        preload: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if split not in SPLIT_TO_DIRS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(SPLIT_TO_DIRS)}.")
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment
        self.normalise = normalise
        self.mean = mean
        self.std = std
        self.preload = preload

        image_dir_name, mask_dir_name = SPLIT_TO_DIRS[split]
        self.image_dir = self.root / image_dir_name
        self.mask_dir = self.root / mask_dir_name

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self._pairs = self._build_index()

        if preload:
            self._cache = [self._load_pair(paths) for paths in self._pairs]
        else:
            self._cache = None

    def _build_index(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        missing: List[Path] = []
        for image_path in sorted(self.image_dir.glob("*.png")):
            mask_name = mask_name_from_image_name(image_path.name)
            mask_path = self.mask_dir / mask_name
            if mask_path.exists():
                pairs.append((image_path, mask_path))
            else:
                missing.append(mask_path)

        if not pairs:
            raise RuntimeError(
                f"No image/mask pairs found for split '{self.split}'. "
                f"Looked in {self.image_dir} and {self.mask_dir}."
            )
        if missing:
            print(f"[OASISBrainDataset] Warning: {len(missing)} masks missing for split '{self.split}'.")
        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def _load_pair(self, paths: Tuple[Path, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = paths

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            angle, translate = _random_affine_params()
            # Coupled augmentations ensure pixel-alignment with the mask.
            image = TF.affine(
                image,
                angle=angle,
                translate=(
                    translate[0] * image.width,
                    translate[1] * image.height,
                ),
                scale=1.0,
                shear=0.0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=(
                    translate[0] * mask.width,
                    translate[1] * mask.height,
                ),
                scale=1.0,
                shear=0.0,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.25:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.resize(image, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)  # (1,H,W)
        if self.normalise:
            image_tensor = (image_tensor - self.mean) / self.std

        mask_array = np.array(mask, dtype=np.uint8)
        mask_tensor = _map_mask_to_indices(mask_array, self.num_classes)

        if self.augment:
            # Mild intensity jitter helps the network generalise across scanners.
            if random.random() < 0.3:
                noise = torch.randn_like(image_tensor) * 0.05
                image_tensor = torch.clamp(image_tensor + noise, -3.0, 3.0)
            if random.random() < 0.3:
                scale = random.uniform(0.9, 1.1)
                shift = random.uniform(-0.1, 0.1)
                image_tensor = image_tensor * scale + shift

        return image_tensor.float(), mask_tensor.long()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            return self._cache[idx]
        return self._load_pair(self._pairs[idx])


@dataclass
class DataConfig:
    """
    Configuration bundle used by `create_dataloaders`.
    """

    root: Path | str
    batch_size: int = 8
    image_size: int = 256
    num_classes: int = 4
    normalise: bool = True
    mean: float = 0.5
    std: float = 0.5
    train_augment: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    preload_train: bool = False


def create_dataloaders(
    config: DataConfig,
    *,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build PyTorch dataloaders for train/validation/test splits.
    """

    def _dataset(split: str, augment: bool, preload: bool) -> OASISBrainDataset:
        return OASISBrainDataset(
            root=config.root,
            split=split,
            image_size=config.image_size,
            num_classes=config.num_classes,
            augment=augment,
            normalise=config.normalise,
            mean=config.mean,
            std=config.std,
            preload=preload,
        )

    train_ds = _dataset("train", config.train_augment, config.preload_train)
    val_ds = _dataset("val", False, False)
    test_ds = _dataset("test", False, False)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=shuffle_train,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
    }
    return loaders
