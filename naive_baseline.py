import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from visionsim.emulate.spc import spc_avg_to_rgb
from visionsim.utils.color import linearrgb_to_srgb


DATASET = "/kaggle/input/datasets/sunaydesai/single-photon-challenge/train"
batch_size = 256
invert_response = True
tonemap = True
output_dir = Path("naivesum_reconstructed")


def main():
    path = Path(DATASET)
    scenes = sorted(os.listdir(DATASET))[:50]
    test_scenes = scenes[:3]
    train_scenes = scenes[3:]
    psnr_values = []
    ssim_values = []

    print("Total scenes:", len(scenes))
    print("Train:", len(train_scenes))
    print("Test:", len(test_scenes))
    output_dir.mkdir(exist_ok=True)

    for scene in test_scenes:
        print("\nProcessing:", scene)
        scene_path = path / scene
        npy_file = next(scene_path.glob("*.npy"))
        png_file = next(scene_path.glob("*.png"))

        photon = np.load(npy_file)
        photon = np.unpackbits(photon[-batch_size:], axis=2)
        image = photon.sum(axis=0) / batch_size

        if invert_response:
            image = spc_avg_to_rgb(image, factor=0.5)
        if tonemap:
            image = linearrgb_to_srgb(image)

        image = np.clip(image, 0, 1)
        reconstructed = (image * 255).astype(np.uint8)

        clean = np.array(Image.open(png_file).convert("RGB"))
        clean_norm = clean.astype(np.float32) / 255.0
        pred_norm = reconstructed.astype(np.float32) / 255.0

        psnr = peak_signal_noise_ratio(clean_norm, pred_norm, data_range=1.0)
        ssim = structural_similarity(
            clean_norm,
            pred_norm,
            channel_axis=2,
            data_range=1.0,
        )
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        print("PSNR:", psnr)
        print("SSIM:", ssim)

        save_path = output_dir / f"{scene}_reconstructed.png"
        Image.fromarray(reconstructed).save(save_path)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(clean)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed)
        plt.title(f"Naive Sum\nPSNR: {psnr:.3f}, SSIM: {ssim:.4f}")
        plt.axis("off")

        plt.suptitle(scene)
        plt.tight_layout()
        plt.show()

    print("\n========== FINAL RESULTS ==========")
    print("Average PSNR:", np.mean(psnr_values))
    print("Average SSIM:", np.mean(ssim_values))


if __name__ == "__main__":
    main()
