import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.models import VGG16_Weights, vgg16


DATASET = "/kaggle/input/datasets/sunaydesai/single-photon-challenge/train"
EPOCHS = 50
LEARNING_RATE = 1e-4


def unpack(arr):
    """Keep the last 128 frames and unpack the packed width dimension."""
    arr = arr[-128:]
    return np.unpackbits(arr, axis=2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + identity)


class AttentionBlock(nn.Module):
    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()
        self.Wg = nn.Conv2d(gate_channels, intermediate_channels, 1)
        self.Wx = nn.Conv2d(skip_channels, intermediate_channels, 1)
        self.psi = nn.Conv2d(intermediate_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate, skip):
        gate_features = self.Wg(gate)
        skip_features = self.Wx(skip)
        attention = self.relu(gate_features + skip_features)
        attention = self.sigmoid(self.psi(attention))
        return skip * attention


class AttentionResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ResidualBlock(384, 128)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(256, 512)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(512, 1024)

        self.att3 = AttentionBlock(512, 512, 256)
        self.att2 = AttentionBlock(256, 256, 128)
        self.att1 = AttentionBlock(128, 128, 64)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = ResidualBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = ResidualBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualBlock(256, 128)

        self.out = nn.Conv2d(128, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.bottleneck(self.pool3(e3))

        up1 = self.up1(bottleneck)
        dec1 = self.dec1(torch.cat([up1, self.att3(up1, e3)], dim=1))

        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, self.att2(up2, e2)], dim=1))

        up3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat([up3, self.att1(up3, e1)], dim=1))

        return self.sigmoid(self.out(dec3))


class VGGPerceptualLoss(nn.Module):
    """VGG-16 feature loss using frozen ImageNet features through layer 16."""

    def __init__(self):
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, prediction, target):
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.features(prediction), self.features(target))


def load_scene(scene_path):
    files = os.listdir(scene_path)
    npy_file = next(file for file in files if file.endswith(".npy"))
    png_file = next(file for file in files if file.endswith(".png"))

    photon = unpack(np.load(os.path.join(scene_path, npy_file)))
    frames = []
    for index in range(128):
        image = photon[index].astype(np.float32)
        image = image / (255.0)
        frames.append(torch.from_numpy(image).permute(2, 0, 1))
    noisy = torch.cat(frames, dim=0).unsqueeze(0)

    clean = np.array(Image.open(os.path.join(scene_path, png_file)))
    clean = torch.from_numpy(clean.astype(np.float32) / 255.0)
    clean = clean.permute(2, 0, 1).unsqueeze(0)
    return noisy, clean


def evaluate(model, test_scenes, device):
    model.eval()

    for scene in test_scenes:
        noisy, clean = load_scene(os.path.join(DATASET, scene))
        noisy = noisy.to(device)
        clean = clean.to(device)

        with torch.no_grad():
            prediction = model(noisy)

        prediction_image = prediction.cpu()[0].permute(1, 2, 0).numpy()
        prediction_image = np.clip(prediction_image, 0, 1)
        clean_image = clean.cpu()[0].permute(1, 2, 0).numpy()

        psnr = peak_signal_noise_ratio(clean_image, prediction_image, data_range=1.0)
        ssim = structural_similarity(
            clean_image,
            prediction_image,
            channel_axis=2,
            data_range=1.0,
        )

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth")
        plt.imshow(clean_image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(prediction_image)
        plt.axis("off")

        plt.suptitle(f"PSNR: {psnr:.2f}  SSIM: {ssim:.3f}")
        plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    path = Path(DATASET)
    scenes_npy = sorted(path.rglob("*.npy"))
    scenes_png = sorted(path.rglob("*.png"))
    print("Total photon files:", len(scenes_npy))
    print("Total clean images:", len(scenes_png))

    scenes = sorted(os.listdir(DATASET))[:50]
    test_scenes = scenes[:3]
    train_scenes = scenes[3:]
    print("Train:", len(train_scenes))
    print("Test:", len(test_scenes))

    sample_photon = np.load(scenes_npy[0])
    print("Photon original shape:", sample_photon.shape)
    print("Photon after unpack:", unpack(sample_photon).shape)
    sample_clean = np.array(Image.open(scenes_png[0]))
    print("Clean image shape:", sample_clean.shape)

    model = AttentionResUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)

    train_losses = []
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for scene in train_scenes:
            noisy, clean = load_scene(os.path.join(DATASET, scene))
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)
            l1 = F.l1_loss(output, clean)

            ssim_metric.reset()
            ssim_value = ssim_metric(output, clean)
            ssim_metric.reset()
            ssim_loss = 1.0 - ssim_value

            vgg_loss = perceptual_loss(output, clean)
            loss = 0.7 * l1 + 0.2 * ssim_loss + 0.1 * vgg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_scenes)
        train_losses.append(average_loss)
        print("Epoch", epoch + 1, "Loss", average_loss)

    evaluate(model, test_scenes, device)


if __name__ == "__main__":
    main()
