# Improved U-Net for OASIS Brain Segmentation
Project: Segment the 2D OASIS brain data set with an Improved UNet [Easy]

Name: Hezekial Octora Yudha Tampubolon

Student ID: 49058466

## Problem Description

Brain structure segmentation from MRI scans is a fundamental task in neuroimaging research and clinical diagnosis. Accurate delineation of brain tissues is essential for studying neurological disorders, tracking disease progression, and planning treatment interventions. This project tackles the automated multi-class segmentation of brain MRI slices from the OASIS dataset, which contains structural brain scans from individuals across different age groups and cognitive states. Manual segmentation is extremely time-consuming and very variative between indivduals, which makes automated approaches incredibly valuable for large-scale neuroimaging studies. 

## Algorithm Overview

This implementation uses an Improved U-Net architecture, which takes the classic U-Net design and enhances it with several modern techniques to boost segmentation performance. Think of the core U-Net as having two main pathways: an encoder that compresses the image while extracting important features, and a decoder that reconstructs the full-resolution segmentation. What makes this version "improved" are the additions of residual connections (allowing gradients to flow more easily during training), squeeze-and-excitation blocks for channel attention (helping the network focus on important features), and gated attention mechanisms on skip connections (selectively emphasizing relevant information while filtering out noise). The encoder progressively downsamples through four stages, learning increasingly abstract representations, while the decoder mirrors this process to produce the final segmentation.

The training approach starts with grayscale brain MRI slices at 256×256 resolution. I preprocess the data with intensity normalization and apply augmentation techniques like random rotations, flips, and intensity adjustments—all while keeping the image and mask perfectly aligned. Each residual block uses grouped normalization and includes squeeze-and-excitation modules that let the network dynamically adjust the importance of different feature channels. I also use deep supervision by adding extra segmentation outputs at intermediate decoder stages, which helps guide learning throughout the network rather than just at the end.

## Model Architecture

<p align="center">
  <img src="assets/improved_unet.png" alt="Training Curves" width="80%">
</p>

*Figure: Improved U-Net architecture showing the encoder pathway (left, blue blocks) progressively downsampling through four stages, a bottleneck with residual connections, and the decoder pathway (right, orange blocks) with attention-gated skip connections (dashed arrows) that merge encoder features with upsampled decoder features. Each block incorporates residual connections, group normalization, and channel attention mechanisms. Deep supervision outputs are generated at multiple decoder scales (green arrows).*

### Architecture Details

The Improved U-Net architecture consists of several sophisticated components working together to achieve precise brain tissue segmentation:

**1. Input Processing:** The network accepts 256×256 single-channel grayscale MRI slices. Images are normalized with zero mean and unit standard deviation to stabilize training and improve convergence.

**2. Encoder Pathway:** Four downsampling stages with progressively increasing channels [32, 64, 128, 256]:
   - Each stage uses a `ResidualConvBlock` with double 3×3 convolutions, group normalization, and a residual shortcut
   - Squeeze-and-excitation blocks recalibrate channel responses based on global context
   - Strided 3×3 convolutions (stride=2) perform spatial downsampling between stages
   - Spatial resolutions: 256×256 → 128×128 → 64×64 → 32×32 → 16×16

**3. Bottleneck:** A residual block at 512 channels maintains a large receptive field without further downsampling, capturing high-level semantic information about brain anatomy.

**4. Decoder Pathway:** Four upsampling stages mirroring the encoder [256, 128, 64, 32]:
   - Transposed convolutions (stride=2) upsample feature maps to higher resolutions
   - `GatedSkipConnection` (attention gates) modulate encoder features before concatenation
   - Each decoder block fuses upsampled features with gated skip connections
   - Bilinear interpolation handles any spatial misalignment

**5. Attention Gates:** Learn to suppress irrelevant regions and highlight tissue boundaries:
   - Combine encoder skip features with decoder context through 1×1 convolutions
   - Generate spatial attention maps via sigmoid activation
   - Element-wise multiplication applies the learned gates to skip connections

**6. Deep Supervision:** Auxiliary segmentation heads at each decoder stage:
   - 1×1 convolutions produce class predictions at multiple scales
   - All outputs upsampled to input resolution for joint supervision
   - Encourages intermediate layers to learn meaningful representations

**7. Final Classifier:** A 1×1 convolution projects the final decoder features to 4 class channels (background, gray matter, white matter, CSF), followed by softmax for probability maps.

This architecture enables the network to learn hierarchical representations, from low-level edges and intensity patterns in early stages to high-level anatomical structures in deeper stages, while attention mechanisms ensure focus on diagnostically relevant tissue boundaries.

## Project Structure
```
.
├── dataset.py          # OASIS dataset loader with augmentation
├── modules.py          # Improved U-Net architecture implementation
├── train.py            # Training script with validation and testing
├── predict.py          # Inference script for visualization
├── assets/             # Directory containing visualizations
└── README.md           # This file
```

## Dependencies

I trained and evaluated the model with these packages:

- Python 3.10.13  
- PyTorch 2.2.2
- NumPy 1.26, SciPy 1.11, Pillow 10.2  
- Matplotlib 3.8 for plotting curves  
- tqdm 4.66 for progress bars

## Data Preparation & Splits

The dataset is the 2D OASIS brain data. It pre-split into:

- `keras_png_slices_train/` and `keras_png_slices_seg_train/`  
- `keras_png_slices_validate/` and `keras_png_slices_seg_validate/`  
- `keras_png_slices_test/` and `keras_png_slices_seg_test/`

Each folder contains 256×256 PNG slices for both images and masks, with subject-level separation already enforced. In numbers, the split is roughly 70 % / 15 % / 15 % of the slices, which provides enough data to train the  model while holding out sufficient validation and test data for unbiased evaluation.

### Pre-processing & Augmentation

Inside `dataset.py` each slice is loaded in grayscale, converted to tensors, and normalised using a simple z-score (`(x - 0.5)/0.5`) which keeps inputs in a narrow range and stabilises GroupNorm layers. Masks are mapped from grayscale values `{0,85,170,255}` to class indices `{0,1,2,3}`. Augmentation follows common recommendations for 2D medical segmentation [2], and includes coupled random rotations (±10°), translations (up to 5 %), horizontal/vertical flips, and mild intensity jitter. These augmentations preserve anatomical plausibility while helping the model generalise to small spatial variations between subjects.

## Training Configuration

I trained with the hyper-parameters inside `train.py`:

- Optimiser: AdamW (lr = 2e-4, weight decay = 1e-4)  
- Scheduler: cosine annealing with warm restarts (period ≈ 8 epochs)  
- Loss: 1.0 × CrossEntropy + 1.0 × Dice with deep supervision heads weighted at 0.4  
- Batch size: 8 slices on the A100 GPU  
- Epochs: 80 (early stopping triggers once all validation Dice ≥ 0.9)

## Results & Example Outputs

The plots below summarise the learning dynamics:

<p align="center">
  <img src="assets/training_curves.png" alt="Training Curves" width="80%">
</p>

Validation loss keeps decreasing while mean Dice stabilises around 0.94, indicating good generalisation without overfitting.

 Using the trained best model and the evaluation results from above, `predict.py` reports the following metrics on the held-out test split:

```
Per-class Dice (BG / GM / WM / CSF): [0.9967, 0.9230, 0.9108, 0.9516]
Mean Dice: 0.9455
```

Every individual class exceeds the required 0.90 Dice threshold on the test data:
- Background 0.9967 → comfortably above 0.90  
- Grey matter 0.9230 → 2.3 pp above the requirement  
- White matter 0.9108 → 1.1 pp above the requirement  
- CSF 0.9516 → strong margin

Because `predict.py` directly loads the best training checkpoint and runs inference on the test loader (no training steps), these numbers reflect pure generalisation performance. The script also saves class probabilities and segmentation masks, so the outputs are reproducible simply by rerunning the same command on the same checkpoint.

Below is the example results:

<p align="center">
  <img src="assets/prediction_panel.png" alt="Prediction Panel" width="55%">
</p>

Each panel overlays the predicted segmentation (cool colours) and the ground-truth mask (warm colours) on top of the MRI slice. The segmentation captures the cortical boundaries and ventricles pretty accurately, with only minor uncertainty in regions where tissues blend together. These samples were drawn from the test set during the `predict.py` run, which further demonstrates that the ≥ 0.9 Dice criterion is satisfied visually as well as numerically.

## References

[1] Xu, Y., Hou, S., Wang, X., Li, D., & Lu, L. (2023). A Medical Image Segmentation Method Based on Improved UNet 3+ Network. *Diagnostics*, *13*(3), 576. https://doi.org/10.3390/diagnostics13030576

[2] Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211. https://doi.org/10.1038/s41592-020-01008-z
