from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

__all__ = [
    "ADNIDataset",
    "ADNISample",
    "create_default_transforms",
    "create_dataloaders",
    "load_metadata",
]


DEFAULT_IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAME_TO_LABEL: Dict[str, int] = {
    "AD": 1,
    "NC": 0,
}

METADATA_LABEL_TO_BINARY: Dict[Any, int] = {
    0: 0,
    "0": 0,
    "NC": 0,
    "CN": 0,
    2: 1,
    "2": 1,
    "AD": 1,
}

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

ADNI_ID_PATTERN = re.compile(r"_I(\d+)")


@dataclass(frozen=True)
class ADNISample:
    """Container describing a single training example."""

    path: Path
    label: int
    instance_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return serialisable metadata describing this sample."""
        base: Dict[str, Any] = {
            "path": str(self.path),
            "label": self.label,
            "instance_id": self.instance_id,
        }
        if self.metadata:
            base.update(self.metadata)
        return base


def load_metadata(metadata_path: Optional[Path | str]) -> Dict[str, Dict[str, Any]]:
    """
    Load the metadata file and augment each entry with a binary
    label suitable for AD vs NC classification.
    """
    if metadata_path is None:
        return {}

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        raw_metadata: Dict[str, Dict[str, Any]] = json.load(handle)

    processed: Dict[str, Dict[str, Any]] = {}
    for instance_id, entry in raw_metadata.items():
        entry_copy = dict(entry)
        binary_label = _metadata_label_to_binary(entry_copy.get("label"))
        if binary_label is not None:
            entry_copy["binary_label"] = binary_label
        processed[instance_id] = entry_copy

    return processed


def _metadata_label_to_binary(value: Any) -> Optional[int]:
    """Convert metadata labels (0, 1, 2, NC, AD, etc.) into 0/1 targets."""
    if value is None:
        return None
    if value in METADATA_LABEL_TO_BINARY:
        return METADATA_LABEL_TO_BINARY[value]
    # Ignore labels that are not part of the AD/NC task.
    return None


def create_default_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    augment: bool = False,
) -> transforms.Compose:
    """
    Build torchvision transforms that output ConvNeXt-compatible tensors.

    """
    if augment:
        transform_ops: List[Callable[[Image.Image], Image.Image | torch.Tensor]] = [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    else:
        transform_ops = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(
                int(image_size * 1.1),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    return transforms.Compose(transform_ops)


def _default_loader(path: Path) -> Image.Image:
    """Load an image as a PIL Image in grayscale mode."""
    with Image.open(path) as handle:
        return handle.convert("L")


def _extract_instance_id(path: Path) -> Optional[str]:
    """Extract the ADNI instance identifier (digits following '_I')."""
    match = ADNI_ID_PATTERN.search(path.name)
    if match:
        return match.group(1)
    return None


class ADNIDataset(Dataset):
    """
    PyTorch dataset wrapping ADNI image slices with optional metadata.
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        *,
        metadata_path: Optional[Path | str] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        samples: Optional[Sequence[ADNISample]] = None,
        subset_indices: Optional[Sequence[int]] = None,
        include_metadata: bool = False,
        loader: Callable[[Path], Image.Image] = _default_loader,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.loader = loader
        self.include_metadata = include_metadata

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = load_metadata(metadata_path)

        self.transform = transform or create_default_transforms(
            image_size=image_size,
            augment=False,
        )

        if samples is not None:
            sample_list = list(samples)
        else:
            sample_list = self._build_sample_index()

        if subset_indices is not None:
            sample_list = [sample_list[idx] for idx in subset_indices]

        self.samples: List[ADNISample] = sample_list

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}'. Checked: {self.root_dir / split}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self.loader(sample.path)
        image_tensor = self.transform(image)

        if self.include_metadata:
            return image_tensor, sample.label, sample.to_metadata_dict()
        return image_tensor, sample.label

    def _build_sample_index(self) -> List[ADNISample]:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples: List[ADNISample] = []

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name.upper()
            base_label = CLASS_NAME_TO_LABEL.get(class_name)
            if base_label is None:
                continue

            image_paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                instance_id = _extract_instance_id(image_path)
                metadata_entry = (
                    self.metadata.get(instance_id) if instance_id else None
                )
                label = base_label
                if metadata_entry and "binary_label" in metadata_entry:
                    label = metadata_entry["binary_label"]

                samples.append(
                    ADNISample(
                        path=image_path,
                        label=label,
                        instance_id=instance_id,
                        metadata=metadata_entry,
                    )
                )

        if not samples:
            raise RuntimeError(
                f"No image files with extensions {ALLOWED_IMAGE_EXTENSIONS} "
                f"found under {split_dir}"
            )

        return samples

    def subset(
        self,
        indices: Sequence[int],
        *,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        include_metadata: Optional[bool] = None,
    ) -> "ADNIDataset":
        """
        Create a lightweight subset of the dataset sharing the same metadata cache.
        """
        return ADNIDataset(
            root_dir=self.root_dir,
            split=self.split,
            metadata=self.metadata,
            transform=transform or self.transform,
            samples=[self.samples[idx] for idx in indices],
            include_metadata=self.include_metadata if include_metadata is None else include_metadata,
            loader=self.loader,
        )


def create_dataloaders(
    data_root: Path | str,
    *,
    metadata_path: Optional[Path | str] = None,
    batch_size: int = 16,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 4,
    valid_split: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Convenience function returning ``(train_loader, val_loader, test_loader)``.
    """
    metadata = load_metadata(metadata_path)

    train_transform = create_default_transforms(
        image_size=image_size,
        augment=True,
    )
    eval_transform = create_default_transforms(
        image_size=image_size,
        augment=False,
    )

    train_dataset_full = ADNIDataset(
        data_root,
        split="train",
        metadata=metadata,
        transform=None,
        image_size=image_size,
    )

    if valid_split and 0.0 < valid_split < 1.0:
        generator = torch.Generator().manual_seed(seed)
        val_size = max(1, int(len(train_dataset_full) * valid_split))
        permuted_indices = torch.randperm(len(train_dataset_full), generator=generator)
        val_indices = permuted_indices[:val_size].tolist()
        train_indices = permuted_indices[val_size:].tolist()

        train_subset = [train_dataset_full.samples[idx] for idx in train_indices]
        val_subset = [train_dataset_full.samples[idx] for idx in val_indices]

        train_dataset = ADNIDataset(
            data_root,
            split="train",
            metadata=metadata,
            transform=train_transform,
            samples=train_subset,
        )
        val_dataset = ADNIDataset(
            data_root,
            split="train",
            metadata=metadata,
            transform=eval_transform,
            samples=val_subset,
        )
        val_loader: Optional[DataLoader] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_dataset = ADNIDataset(
            data_root,
            split="train",
            metadata=metadata,
            transform=train_transform,
        )
        val_dataset = None
        val_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    test_dataset = ADNIDataset(
        data_root,
        split="test",
        metadata=metadata,
        transform=eval_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
