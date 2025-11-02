from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ImprovedUNet",
    "DiceLoss",
    "dice_coefficient",
    "build_improved_unet",
]


def _make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Create a GroupNorm layer with the highest possible number of groups
    that evenly divides the channel dimension.
    """
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class SqueezeExciteBlock(nn.Module):
    """
    Lightweight squeeze-excite module used to highlight informative channels.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualConvBlock(nn.Module):
    """
    Residual double-convolution block with squeeze-excite gating.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_group_norm(mid)
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_group_norm(out_channels)

        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SqueezeExciteBlock(out_channels) if use_se else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)

        out = self.se(out)
        out = out + residual
        return self.act(out)


class GatedSkipConnection(nn.Module):
    """
    Attention gate that modulates encoder skip features based on decoder context.

    This allows the decoder to suppress irrelevant activations (e.g. background)
    without removing the spatial detail provided by the skip connection.
    """

    def __init__(self, skip_channels: int, decoder_channels: int, reduction: int = 2) -> None:
        super().__init__()
        inter_channels = max(skip_channels // reduction, 1)
        self.transform = nn.Sequential(
            nn.Conv2d(skip_channels + decoder_channels, inter_channels, kernel_size=1, bias=False),
            _make_group_norm(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, skip_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.transform(torch.cat([skip, decoder], dim=1))
        return skip * gate


class DownBlock(nn.Module):
    """
    Encoder block: residual conv block followed by strided downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.block = ResidualConvBlock(in_channels, out_channels, dropout=dropout, use_se=use_se)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
        downsampled = self.down(x)
        return x, downsampled


class UpBlock(nn.Module):
    """
    Decoder block: attention-gated skip connection merged with upsampled context.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = (
            GatedSkipConnection(skip_channels, out_channels) if use_attention else nn.Identity()
        )
        self.block = ResidualConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.attention(skip, x) if isinstance(self.attention, GatedSkipConnection) else skip
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ImprovedUNet(nn.Module):
    """
    Attention-augmented residual U-Net for 2D multi-class segmentation.

    Args:
        in_channels: number of channels in the input image (e.g. 1 for MRI).
        num_classes: number of output segmentation classes.
        base_channels: number of feature channels in the first stage.
        depth: number of downsampling operations (>= 3 recommended).
        dropout: dropout probability applied inside residual blocks.
        use_attention: enable attention gates on skip connections.
        deep_supervision: add auxiliary heads for intermediate decoder outputs.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("depth must be >= 3 for a well-formed U-Net.")

        self.deep_supervision = deep_supervision

        # Encoder hierarchy
        enc_blocks: List[DownBlock] = []
        in_ch = in_channels
        channels: List[int] = []
        for level in range(depth):
            out_ch = base_channels * (2**level)
            block = DownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                dropout=dropout if level > 0 else 0.0,
                use_se=True,
            )
            enc_blocks.append(block)
            channels.append(out_ch)
            in_ch = out_ch
        self.encoder = nn.ModuleList(enc_blocks)

        # Bottleneck keeps the receptive field large without further downsampling.
        bottleneck_channels = base_channels * (2**depth)
        self.bottleneck = ResidualConvBlock(
            in_channels=channels[-1],
            out_channels=bottleneck_channels,
            dropout=dropout,
            use_se=True,
        )

        # Decoder mirrors the encoder while incorporating attention-gated skips.
        dec_blocks: List[UpBlock] = []
        dec_heads: List[nn.Conv2d] = []
        decoder_in = bottleneck_channels
        for level in reversed(range(depth)):
            skip_ch = channels[level]
            out_ch = skip_ch
            block = UpBlock(
                in_channels=decoder_in,
                skip_channels=skip_ch,
                out_channels=out_ch,
                dropout=dropout,
                use_attention=use_attention,
            )
            dec_blocks.append(block)
            decoder_in = out_ch

            if deep_supervision:
                head = nn.Conv2d(out_ch, num_classes, kernel_size=1)
                dec_heads.append(head)

        self.decoder = nn.ModuleList(dec_blocks)
        self.classifier = nn.Conv2d(decoder_in, num_classes, kernel_size=1)
        self.auxiliary_heads = nn.ModuleList(dec_heads) if deep_supervision else nn.ModuleList()

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        encoder_features: List[torch.Tensor] = []
        out = x
        for block in self.encoder:
            skip, out = block(out)
            encoder_features.append(skip)

        out = self.bottleneck(out)

        aux_outputs: List[torch.Tensor] = []
        for idx, block in enumerate(self.decoder):
            skip = encoder_features[-(idx + 1)]
            out = block(out, skip)
            if self.deep_supervision:
                head = self.auxiliary_heads[idx]
                aux_outputs.append(head(out))

        logits = self.classifier(out)

        if self.deep_supervision:
            # Upsample auxiliary predictions to the main output size for joint supervision.
            aux_outputs_upsampled = [
                F.interpolate(aux, size=logits.shape[-2:], mode="bilinear", align_corners=False)
                for aux in aux_outputs
            ]
            return logits, aux_outputs_upsampled
        return logits


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    apply_softmax: bool = True,
) -> torch.Tensor:
    """
    Compute the multi-class Dice coefficient between logits and targets.
    """
    if apply_softmax:
        probs = torch.softmax(logits, dim=1)
    else:
        probs = logits

    num_classes = probs.shape[1]
    if targets.ndim == probs.ndim:
        target_one_hot = targets
    else:
        target_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersect = torch.sum(probs * target_one_hot, dim=dims)
    union = torch.sum(probs + target_one_hot, dim=dims)

    dice = (2 * intersect + epsilon) / (union + epsilon)
    return dice


class DiceLoss(nn.Module):
    """
    Soft Dice loss with optional class weighting and log-cosh stabilisation.
    """

    def __init__(
        self,
        weight: Optional[Sequence[float]] = None,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if weight is not None:
            w = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", w)
        else:
            self.weight = None
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = dice_coefficient(logits, targets, epsilon=self.epsilon, apply_softmax=True)
        loss = 1.0 - dice
        if self.weight is not None:
            # Broadcast to (num_classes,) before reduction.
            loss = loss * self.weight
        loss = loss.mean()
        # log-cosh keeps the gradient smooth near convergence.
        return torch.log(torch.cosh(loss + self.epsilon))


def _init_weights(module: nn.Module) -> None:
    """
    Kaiming initialisation tailored for residual conv blocks.
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=5**0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass
class UNetConfig:
    """
    Lightweight configuration container to simplify model creation from YAML / argparse.
    """

    in_channels: int = 1
    num_classes: int = 4
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.1
    use_attention: bool = True
    deep_supervision: bool = True


def build_improved_unet(config: Union[UNetConfig, dict, None] = None) -> ImprovedUNet:
    """
    Factory helper used by training / inference scripts.
    """
    if config is None:
        cfg = UNetConfig()
    elif isinstance(config, dict):
        cfg = UNetConfig(**config)
    elif isinstance(config, UNetConfig):
        cfg = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    model = ImprovedUNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
        use_attention=cfg.use_attention,
        deep_supervision=cfg.deep_supervision,
    )
    return model

