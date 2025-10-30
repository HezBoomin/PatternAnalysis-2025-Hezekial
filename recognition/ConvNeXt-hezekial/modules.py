from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


__all__ = [
    "LayerNorm2d",
    "DropPath",
    "ConvNeXtBlock",
    "ConvNeXtStage",
    "ConvNeXtTiny",
    "build_convnext_tiny",
    "freeze_backbone",
    "load_checkpoint",
]


class LayerNorm2d(nn.Module):
    """
    LayerNorm variant that normalises across the channel dimension for 2D feature maps.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """
    Stochastic depth per sample (channel-wise).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    """
    Core ConvNeXt block: depthwise 7x7 convolution, channel-wise LayerNorm,
    MLP with GELU activation, and optional layer scaling plus stochastic depth.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        x = self.drop_path(x)
        return shortcut + x


class ConvNeXtStage(nn.Sequential):
    """
    Sequential container for a stack of ConvNeXt blocks.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: Iterable[float],
        layer_scale_init_value: float,
    ) -> None:
        blocks = [
            ConvNeXtBlock(
                dim=dim,
                drop_path=drop_rate,
                layer_scale_init_value=layer_scale_init_value,
            )
            for drop_rate in drop_path_rates
        ]
        super().__init__(*blocks)


@dataclass(frozen=True)
class ConvNeXtConfig:
    """Configuration dataclass for convenience."""

    depths: Tuple[int, int, int, int]
    dims: Tuple[int, int, int, int]
    layer_scale_init_value: float = 1e-6
    drop_path_rate: float = 0.1
    num_classes: int = 2
    in_chans: int = 3


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny backbone + classification head tailored for AD vs NC tasks.
    """

    def __init__(self, config: Optional[ConvNeXtConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ConvNeXtConfig(
                depths=(3, 3, 9, 3),
                dims=(96, 192, 384, 768),
            )
        self.config = config

        depths = config.depths
        dims = config.dims
        layer_scale = config.layer_scale_init_value

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(config.in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        total_blocks = sum(depths)
        drop_path_rates = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()

        self.stages = nn.ModuleList()
        block_idx = 0
        for stage_idx, depth in enumerate(depths):
            stage_drop_rates = drop_path_rates[block_idx : block_idx + depth]
            block_idx += depth
            stage = ConvNeXtStage(
                dim=dims[stage_idx],
                depth=depth,
                drop_path_rates=stage_drop_rates,
                layer_scale_init_value=layer_scale,
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], config.num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)


def build_convnext_tiny(
    *,
    num_classes: int = 2,
    in_chans: int = 3,
    drop_path_rate: float = 0.1,
    layer_scale_init_value: float = 1e-6,
) -> ConvNeXtTiny:
    """
    Factory function to create a ConvNeXt-Tiny model with custom heads.
    """
    config = ConvNeXtConfig(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        layer_scale_init_value=layer_scale_init_value,
        drop_path_rate=drop_path_rate,
        num_classes=num_classes,
        in_chans=in_chans,
    )
    return ConvNeXtTiny(config=config)


def freeze_backbone(model: ConvNeXtTiny, train_head_only: bool = True) -> None:
    """
    Optionally freeze backbone parameters. Useful for transfer learning when
    fine-tuning on limited medical imaging data.
    """
    for name, param in model.named_parameters():
        if train_head_only and name.startswith("head"):
            param.requires_grad = True
        else:
            param.requires_grad = not train_head_only


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Load weights from a local checkpoint file.

    Returns the state dictionary.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    model.load_state_dict(state_dict, strict=strict)
    return state_dict
