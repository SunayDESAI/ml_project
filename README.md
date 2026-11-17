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
├── assets/
│   ├── placeholder_gen.svg
│   └── results/        # Place your reconstruction PNGs here
└── .gitignore
```

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
