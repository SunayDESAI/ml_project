# Single Photon Image Reconstruction

**ES 335 Course Project**

A deep learning project for reconstructing high-quality RGB images from sparse single-photon measurements, comparing statistical frame averaging, CNN, U-Net, and Attention-Residual U-Net architectures.

## 🌐 Live Site

> Deploy on GitHub Pages and update this link.

## 📁 Structure

```
ml_project/
├── index.html          # Main website (single-page)
├── css/
│   └── style.css       # Dark tech theme styles
├── js/
│   └── main.js         # Charts, tabs, interactive compare, particles
├── unet_training.py     # U-Net training and evaluation with L1 + SSIM + VGG loss
├── cnn_training.py      # CNN training and evaluation with L1 + SSIM + VGG loss
├── attn_res_unet_training.py # Attention-Residual U-Net training and evaluation
├── assets/
│   ├── placeholder_gen.svg
│   └── results/        # Place your reconstruction PNGs here
└── .gitignore
```

## 🧠 Training the U-Net

`unet_training.py`, `cnn_training.py`, and `attn_res_unet_training.py` contain complete training and evaluation workflows. Their composite objective matches the methodology shown on the site:

```
loss = 0.7 * L1 + 0.2 * (1 - SSIM) + 0.1 * VGG perceptual loss
```

Run it in an environment with PyTorch, TorchVision, TorchMetrics, NumPy, Pillow, Matplotlib, and scikit-image:

```bash
python unet_training.py
# or
python cnn_training.py
# or
python attn_res_unet_training.py
```

The script expects the Kaggle dataset at the `DATASET` path defined near the top of the file. The training code is repository-only and is not loaded by the deployed website.

## 🚀 Deploy to GitHub Pages

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Then: GitHub → Repo Settings → Pages → Source: `main` branch → `/root`

## 📌 Customizing with Your Results

1. **Images**: Drop architecture reconstruction outputs into `assets/results/<arch>.png`
2. **Metrics**: Edit `archMetrics` in `js/main.js` with real PSNR/SSIM values
3. **Chart data**: Edit `data.psnr.values` and `data.ssim.values` arrays in `js/main.js`
4. **Tables**: Fill in the `—` cells in the Results table in `index.html`
