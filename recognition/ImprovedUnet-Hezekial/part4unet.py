#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from glob import glob
from contextlib import nullcontext
import numpy as np
from PIL import Image
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# In[ ]:


DATA_ROOT   = "/home/groups/comp3710/OASIS"
RUNS_ROOT   = "./unet_runs"

IMG_SIZE    = 256       # divisible by 16
NUM_CLASSES = 4
BASE_CH     = 64
BATCH_SIZE  = 16
EPOCHS      = 5        # train longer for >0.9 DSC
LR          = 1e-3
WEIGHT_CE   = 1.0
WEIGHT_DICE = 1.0

if torch.cuda.is_available():
    device = torch.device("cuda")
    amp_enabled = True
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    def autocast_ctx():
        return torch.amp.autocast(device_type="cuda", enabled=True)
else:
    device = torch.device("cpu")
    amp_enabled = False
    scaler = None
    def autocast_ctx():
        return nullcontext()

def make_sequential_dir(root=RUNS_ROOT, prefix="run_", digits=3):
    os.makedirs(root, exist_ok=True)
    i = 1
    while True:
        out = os.path.join(root, f"{prefix}{i:0{digits}d}")
        if not os.path.exists(out):
            os.makedirs(out)
            return out
        i += 1

OUT_DIR = make_sequential_dir(RUNS_ROOT, "run_", 3)


# In[ ]:


TRAIN_IMAGES = os.path.join(DATA_ROOT, "keras_png_slices_train")   # e.g. .../images/*.png
TRAIN_MASKS  = os.path.join(DATA_ROOT, "keras_png_slices_seg_train",)    # e.g. .../masks/*.png
VAL_IMAGES   = os.path.join(DATA_ROOT, "keras_png_slices_validate")
VAL_MASKS    = os.path.join(DATA_ROOT, "keras_png_slices_seg_validate",)
TEST_IMAGES  = os.path.join(DATA_ROOT, "keras_png_slices_test")
TEST_MASKS   = os.path.join(DATA_ROOT, "keras_png_slices_seg_test")


def mask_name_from_image_name(img_name: str) -> str:
    """
    Map image file name to its mask file name for OASIS:
    case_###_slice_##.nii.png  -> seg_###_slice_##.nii.png
    """
    base = os.path.basename(img_name)
    if base.startswith("case_"):
        return "seg_" + base[len("case_"):]
    # fallbacks if there are other prefixes in the dataset
    base = re.sub(r"^(image_|img_)", "seg_", base)
    return base

class OasisSeg(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256, augment=False):
        self.size = img_size
        self.augment = augment
        self.pairs = []

        img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
        missing = []
        for f in img_files:
            img_path  = os.path.join(images_dir, f)
            mask_file = mask_name_from_image_name(f)
            mask_path = os.path.join(masks_dir, mask_file)
            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
            else:
                missing.append(mask_path)

        if not self.pairs:
            raise RuntimeError(
                f"No (image, mask) pairs found. "
                f"Checked {images_dir} vs {masks_dir}. "
                f"E.g. expected {os.path.join(masks_dir, mask_name_from_image_name(img_files[0])) if img_files else '(no images)'}"
            )
        if missing:
            print(f"[OasisSeg] Warning: {len(missing)} masks not found (showing up to 5):")
            for m in missing[:5]:
                print("  missing:", m)

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]

        # --- image ---
        x = Image.open(img_path).convert("L")
        x = TF.resize(x, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        x = TF.to_tensor(x)  # [1,H,W] float 0..1

        # --- mask ---
        y = Image.open(mask_path).convert("L")
        y = TF.resize(y, (self.size, self.size), interpolation=InterpolationMode.NEAREST)
        y = np.array(y, dtype=np.uint8)

        # Map grayscale {0,85,170,255} -> class ids {0,1,2,3}
        if y.max() > 3:
            y = y // 85  # 0->0, 85->1, 170->2, 255->3

        y = torch.from_numpy(y).long().clamp_(0, NUM_CLASSES - 1)  # [H,W] int64

        return x, y

train_ds = OasisSeg(TRAIN_IMAGES, TRAIN_MASKS, IMG_SIZE, augment=True)
val_ds   = OasisSeg(VAL_IMAGES,   VAL_MASKS,   IMG_SIZE, augment=False)
test_ds  = OasisSeg(TEST_IMAGES,  TEST_MASKS,  IMG_SIZE, augment=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=(device.type=="cuda"))
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))


# In[ ]:


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    """Upsample deep tensor to out_ch, concat with skip (skip_ch), then conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)   # <-- key: out_ch + skip_ch

    def forward(self, x, skip):
        x = self.up(x)                     # (B, out_ch, H*, W*)
        # If shapes differ by 1px due to rounding, you can pad/crop here
        x = torch.cat([x, skip], dim=1)    # (B, out_ch + skip_ch, H*, W*)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        # Encoder: 32, 64, 128, 256, 512 channels
        self.c1 = DoubleConv(in_ch,     base)        # -> 32
        self.d1 = Down(base,            base*2)      # -> 64
        self.d2 = Down(base*2,          base*4)      # -> 128
        self.d3 = Down(base*4,          base*8)      # -> 256
        self.d4 = Down(base*8,          base*16)     # -> 512 (bottom)

        # Decoder (mirror): Up(in_ch, skip_ch, out_ch)
        self.u1 = Up(base*16, base*8,  base*8)       # 512 -> 256, concat 256 -> conv(512->256)
        self.u2 = Up(base*8,  base*4,  base*4)       # 256 -> 128, concat 128 -> conv(256->128)
        self.u3 = Up(base*4,  base*2,  base*2)       # 128 -> 64,  concat 64  -> conv(128->64)
        self.u4 = Up(base*2,  base,    base)         # 64  -> 32,  concat 32  -> conv(64->32)

        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.c1(x)           # 32
        x2 = self.d1(x1)          # 64
        x3 = self.d2(x2)          # 128
        x4 = self.d3(x3)          # 256
        x5 = self.d4(x4)          # 512

        x  = self.u1(x5, x4)      # expects out_ch+skip_ch = 256+256 = 512 in the DoubleConv
        x  = self.u2(x,  x3)
        x  = self.u3(x,  x2)
        x  = self.u4(x,  x1)
        return self.out(x)


# In[28]:


def one_hot(labels, num_classes):
    # labels: [N,H,W] long -> [N,C,H,W] float
    return F.one_hot(labels, num_classes).permute(0,3,1,2).float()

def soft_dice_per_class(probs, target_1h, eps=1e-6):
    # probs/target: [N,C,H,W]; returns [C] dice
    dims = (0,2,3)
    intersect = (probs * target_1h).sum(dims)
    denom = probs.sum(dims) + target_1h.sum(dims) + eps
    return 2*intersect / denom

def dice_ce_loss(logits, target, class_weights=None, ignore_bg_in_dice=True):
    # CE expects class indices [N,H,W]
    ce = F.cross_entropy(logits, target, weight=class_weights)

    probs = F.softmax(logits, dim=1)
    tgt1h = one_hot(target, logits.size(1))

    dice_c = soft_dice_per_class(probs, tgt1h)  # [C]
    if ignore_bg_in_dice:
        dice_mean = dice_c[1:].mean()
    else:
        dice_mean = dice_c.mean()
    dice_loss = 1.0 - dice_mean
    return ce + dice_loss, dice_c


# In[ ]:


def colorize(mask):
    # mask: [H,W] with {0,1,2,3}
    palette = torch.tensor([
        [0,0,0],        # bg
        [64, 224, 208], # CSF (teal)
        [255,165,0],    # GM (orange)
        [135,206,250],  # WM (light blue)
    ], dtype=torch.uint8)
    return palette[mask.clamp(min=0, max=palette.size(0)-1)]

model = UNet(in_ch=1, num_classes=NUM_CLASSES, base=32).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

def step_epoch(loader, train=True):
    if train: model.train()
    else:     model.eval()
    loss_sum, count = 0.0, 0
    dsc_sum = torch.zeros(NUM_CLASSES, device=device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(x)
            loss, dice_c = dice_ce_loss(logits, y, class_weights=None, ignore_bg_in_dice=True)

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

        loss_sum += loss.item() * x.size(0)
        dsc_sum  += dice_c.detach() * x.size(0)
        count    += x.size(0)

    return loss_sum / count, (dsc_sum / count).tolist()  # avg per class

# ----- Training loop -----
best_val = 1e9
for epoch in range(1, EPOCHS):
    tr_loss, tr_dsc = step_epoch(train_loader, train=True)
    va_loss, va_dsc = step_epoch(val_loader,   train=False)
    print(f"Epoch {epoch:03d} | "
          f"train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | "
          f"val_DSC per class: {['%.3f'%d for d in va_dsc]}")

    if va_loss < best_val:
        best_val = va_loss
        torch.save(model.state_dict(), "unet_best.pt")

# ----- Testing (report per-class DSC) -----
model.load_state_dict(torch.load("unet_best.pt", map_location=device))
test_loss, test_dsc = step_epoch(test_loader, train=False)
print("TEST per-class DSC:", ["%.3f"%d for d in test_dsc], " | mean(excl bg) = %.3f" % (np.mean(test_dsc[1:])))

# ----- Save a few qualitative overlays -----
os.makedirs("pred_viz", exist_ok=True)
model.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        logits = model(x)
        pred = logits.softmax(1).argmax(1).cpu()   # [N,H,W] class ids

        for b in range(min(x.size(0), 4)):
            img  = (x[b].cpu().squeeze(0).numpy()*255).astype(np.uint8)
            pm   = colorize(pred[b])    # [H,W,3]
            gt   = colorize(y[b])       # [H,W,3]
            Image.fromarray(img).save(f"pred_viz/{i:03d}_{b}_img.png")
            Image.fromarray(pm.numpy()).save(f"pred_viz/{i:03d}_{b}_pred.png")
            Image.fromarray(gt.numpy()).save(f"pred_viz/{i:03d}_{b}_gt.png")
        if i >= 4: break

