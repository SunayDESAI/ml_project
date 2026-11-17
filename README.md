# Single Photon Image Reconstruction

ES 335 course project: reconstruct clean RGB images from sparse single-photon measurements.

- **Website:** [sunaydesai.github.io/ml_project](https://sunaydesai.github.io/ml_project/)
- **Repository:** [github.com/SunayDESAI/ml_project](https://github.com/SunayDESAI/ml_project)

## Research question

Can learned convolutional models reconstruct an 800 x 800 RGB image from a sparse photon cube, and how do a plain CNN, a U-Net, and an Attention-Residual U-Net compare with a statistical frame-averaging baseline?

One data row is one scene folder. The selected `.npy` photon cube is decoded into 128 RGB frames, concatenated into a 384-channel model input, and paired with one clean `.png` RGB target.

## Data

The source is the [Single Photon Challenge dataset](https://singlephotonchallenge.com/). The training scripts expect a directory containing scene folders:

```text
train/
  <scene_name>/
    <photon_file>.npy
    <clean_file>.png
```

The photon arrays used by the scripts have shape `(1024, 800, 100, 3)`. The 100-width axis stores eight packed binary detections per value; unpacking produces `(1024, 800, 800, 3)`. Clean targets are 800 x 800 RGB PNG images.

The scripts use the first 50 scene folders after sorting. Within each scene, they use the first matching `.npy` and `.png` file. The first 3 sorted scenes are held out and the remaining 47 are used for training. This is a deterministic scene-level split.

## Exact preprocessing

All three learned-model scripts use the same preprocessing:

1. Keep the last 128 temporal frames: `raw[-128:]`.
2. Decode the packed width with `np.unpackbits(arr, axis=2)`.
3. Convert every RGB frame to `float32` and divide by `255.0`.
4. Permute each frame from HWC to CHW.
5. Concatenate 128 frames into `(1, 384, 800, 800)`.
6. Convert the clean PNG target to CHW float values in `[0, 1]`.

The fixed 128-frame window makes the input size consistent, unpacking preserves the binary detections encoded in the photon cube, and division by 255 places inputs and targets on a common scale.

The statistical baseline intentionally differs: it unpacks and averages the last 256 frames, then applies the sensor-response and sRGB conversions described below. Therefore, its comparison is a reference point, not a controlled architecture-only ablation.

## Models and objective

### Statistical baseline

`naive_baseline.py` uses `batch_size = 256`, averages the last 256 unpacked frames, calls `spc_avg_to_rgb(..., factor=0.5)`, and applies `linearrgb_to_srgb`. It can optionally enable or disable response inversion and tonemapping with the two boolean settings, saves each reconstruction to `naivesum_reconstructed/<scene>_reconstructed.png`, and prints the final mean PSNR and SSIM.

### CNN

`cnn_training.py` uses sequential convolutions:

```text
384 -> 256 -> 256 -> 128 -> 128 -> 64 -> 32 -> 3
```

It has no pooling or skip connections.

### U-Net

`unet_training.py` uses two encoder pooling stages, a bottleneck, transposed-convolution upsampling, and concatenated skip connections. Its main channels are 384 -> 256 -> 128 -> 64 in the encoder and 64 -> 128 -> 256 in the decoder.

### Attention-Residual U-Net

`attn_res_unet_training.py` uses:

- residual blocks with two convolutions, ReLU, and an identity or 1 x 1 projection shortcut;
- encoder channels 128 -> 256 -> 512;
- a 1024-channel bottleneck;
- attention gates on all three encoder skip paths;
- decoder channels 512 -> 256 -> 128;
- a final 1 x 1 convolution and sigmoid output.

There is no normalization layer, global residual shortcut, learning-rate scheduler, mixed-precision training, or checkpoint restoration in the supplied scripts.

All learned models use the same objective:

```text
loss = 0.7 * L1 + 0.2 * (1 - SSIM) + 0.1 * VGG perceptual loss
```

The optimizer is Adam with learning rate `1e-4`. Training runs for 50 epochs, processes one scene at a time, and uses frozen ImageNet VGG-16 features through `features[:16]`. The Attention-Residual U-Net clips gradient norm to `1.0`; the CNN and U-Net scripts do not.

## Evaluation and results

After training, each script evaluates the final in-memory model on the 3 held-out scenes. It reports:

- PSNR in dB using `skimage.metrics.peak_signal_noise_ratio`;
- plain single-scale SSIM using `skimage.metrics.structural_similarity`, `channel_axis=2`, and `data_range=1.0`.

The website shows the supplied three-scene averages:

| Model | PSNR (dB) | SSIM |
|---|---:|---:|
| Naive Sum baseline | 20.955 | 0.2924 |
| CNN | 26.517 | 0.7607 |
| U-Net | 27.493 | 0.7863 |
| Attention-Residual U-Net | 29.673 | 0.8407 |

Per-scene values stored in `js/main.js` are:

| Scene | Naive PSNR / SSIM | CNN PSNR / SSIM | U-Net PSNR / SSIM | Attention-Residual U-Net PSNR / SSIM |
|---|---|---|---|---|
| attic | 21.769 / 0.3077 | 28.410 / 0.7470 | 29.820 / 0.7940 | 31.040 / 0.8370 |
| bachelors-quarters | 22.152 / 0.2611 | 27.480 / 0.8110 | 28.670 / 0.8310 | 31.800 / 0.8700 |
| barbershop | 18.943 / 0.3085 | 23.660 / 0.7240 | 23.990 / 0.7340 | 26.180 / 0.8150 |

The scene images and interactive comparison are in `assets/results/`. The underlying chart values and per-scene values are in `js/main.js`. The Attention-Residual U-Net is strongest on the supplied three-scene averages, while barbershop is the weakest displayed scene for every learned model. The three-scene sample is too small for broad generalization claims.

## Setup and reproduction

The learned scripts require Python 3.10 or a compatible Python version and:

```bash
pip install torch torchvision numpy pillow matplotlib scikit-image torchmetrics
```

VGG-16 ImageNet weights are loaded by TorchVision and may be downloaded the first time a script runs.

Edit the `DATASET` constant near the top of the selected script to point to the local `train` directory, then run:

```bash
python cnn_training.py
python unet_training.py
python attn_res_unet_training.py
```

Run the baseline separately. It additionally requires the `visionsim` package used by the challenge preprocessing utilities:

```bash
python naive_baseline.py
```

The scripts print the split sizes, epoch losses, per-scene PSNR/SSIM, and evaluation images. They do not save checkpoints or provide a separate inference command; the final evaluation occurs at the end of each run.

## Repository contents

```text
index.html                    Submission website
css/style.css                 Website styling
js/main.js                    Charts, metrics, tabs, and image comparison
naive_baseline.py             256-frame statistical baseline
cnn_training.py               CNN training and evaluation
unet_training.py              U-Net training and evaluation
attn_res_unet_training.py    Attention-Residual U-Net training and evaluation
assets/results/               Ground truth and supplied reconstruction images
```

## Limitations

- Only 50 sorted scenes are used, with 3 held out for evaluation.
- The evaluation set is small and scene-level metric variation is substantial.
- The baseline uses 256 frames while learned models use 128.
- Each scene is processed as one 800 x 800 sample, which requires substantial memory.
- The scripts select the first matching files rather than validating a multi-file scene manifest.
- The supplied code does not perform controlled ablations or report uncertainty intervals.

## AI-use disclosure

AI assistance was used to help draft and revise code comments, website structure, documentation wording, and consistency checks. The model descriptions, preprocessing, split, metrics, and reported results were checked against the repository scripts and supplied reconstruction outputs. No data, metric values, or experimental claims were generated by the AI.

## References

1. Single Photon Challenge dataset and task: [singlephotonchallenge.com](https://singlephotonchallenge.com/).
2. PyTorch documentation: [pytorch.org/docs](https://pytorch.org/docs/stable/).
3. TorchVision VGG-16 model documentation: [torchvision.models.vgg16](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html).
4. TorchMetrics structural similarity documentation: [Structural Similarity Index](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html).
5. scikit-image metric documentation: [skimage.metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html).
6. Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE TIP, 2004: [doi:10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861).
7. Simonyan and Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” 2014: [arXiv:1409.1556](https://arxiv.org/abs/1409.1556).

## GitHub Pages deployment

Enable GitHub Pages from repository Settings → Pages, selecting the `main` branch and repository root. The current site is a static HTML/CSS/JavaScript page and does not load the Python training scripts.
