MRI k-space Reconstruction
==========================
Simulates 2D MRI k-space data acquisition and reconstructs the image
using the Fast Fourier Transform (FFT).

Demonstrates:
- 2D FFT and inverse FFT
- k-space sampling patterns
- SNR measurement
- Effect of noise on reconstruction quality

Author: [Lewis Jenkins]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def create_phantom(size: int = 256) -> np.ndarray:
    """
    Create a simple Shepp-Logan style MRI phantom.
    Returns a 2D array representing tissue contrast.
    """
    phantom = np.zeros((size, size))
    cx, cy = size // 2, size // 2

    Y, X = np.ogrid[:size, :size]
    outer = ((X - cx) / (0.92 * cx)) ** 2 + ((Y - cy) / (0.69 * cy)) ** 2
    phantom[outer <= 1] = 1.0

    inner = ((X - cx) / (0.74 * cx)) ** 2 + ((Y - cy) / (0.61 * cy)) ** 2
    phantom[inner <= 1] = 0.4

    for x_off, y_off, radius, value in [
        (-0.22, 0.0, 0.11, 0.8),
        (0.22, 0.0, 0.11, 0.8),
    ]:
        feature = (X - (cx + x_off * cx)) ** 2 + (Y - (cy + y_off * cy)) ** 2
        phantom[feature <= (radius * cx) ** 2] = value

    return phantom


def simulate_kspace(image: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Simulate MRI k-space data from a ground truth image.
    Applies 2D FFT and adds complex Gaussian noise to simulate scanner noise.
    """
    kspace = np.fft.fftshift(np.fft.fft2(image))
    noise = noise_std * (
        np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape)
    )
    return kspace + noise


def reconstruct_from_kspace(kspace: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from k-space via inverse 2D FFT.
    Returns magnitude image.
    """
    image = np.fft.ifft2(np.fft.ifftshift(kspace))
    return np.abs(image)


def measure_snr(image: np.ndarray, signal_mask: np.ndarray, noise_mask: np.ndarray) -> float:
    """
    Measure image SNR using signal and noise ROI masks.
    SNR = mean(signal ROI) / std(noise ROI)
    """
    signal = np.mean(image[signal_mask])
    noise = np.std(image[noise_mask])
    return signal / noise if noise > 0 else np.inf


def undersample_kspace(kspace: np.ndarray, acceleration: int = 2) -> np.ndarray:
    """
    Simulate k-space undersampling.
    Zeros out every `acceleration`-th line in the phase encode direction.
    """
    undersampled = kspace.copy()
    undersampled[::acceleration, :] = 0
    return undersampled


def plot_results(phantom, kspace, reconstructed, undersampled_recon):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("MRI k-space Reconstruction Pipeline", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(phantom, cmap="gray")
    axes[0, 0].set_title("Ground Truth Phantom")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.log1p(np.abs(kspace)), cmap="inferno")
    axes[0, 1].set_title("k-space (log magnitude)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(reconstructed, cmap="gray")
    axes[0, 2].set_title("Reconstructed Image (full sampling)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(reconstructed - phantom, cmap="bwr", vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title("Reconstruction Error")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(undersampled_recon, cmap="gray")
    axes[1, 1].set_title("Undersampled Reconstruction (R=2)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(undersampled_recon - phantom, cmap="bwr", vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title("Undersampling Artefacts")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved to reconstruction_results.png")


def main():
    np.random.seed(42)
    print("MRI k-space Reconstruction Demo")
    print("=" * 40)

    print("Creating phantom...")
    phantom = create_phantom(size=256)

    print("Simulating k-space acquisition (noise_std=0.01)...")
    kspace = simulate_kspace(phantom, noise_std=0.01)

    print("Reconstructing image...")
    reconstructed = reconstruct_from_kspace(kspace)

    size = phantom.shape[0]
    signal_mask = np.zeros(phantom.shape, dtype=bool)
    signal_mask[size//4:3*size//4, size//4:3*size//4] = True
    noise_mask = np.zeros(phantom.shape, dtype=bool)
    noise_mask[:20, :20] = True

    snr = measure_snr(reconstructed, signal_mask, noise_mask)
    print(f"Reconstruction SNR: {snr:.1f}")

    print("Simulating R=2 undersampling...")
    kspace_us = undersample_kspace(kspace, acceleration=2)
    undersampled_recon = reconstruct_from_kspace(kspace_us)
    snr_us = measure_snr(undersampled_recon, signal_mask, noise_mask)
    print(f"Undersampled SNR:   {snr_us:.1f}")
    print(f"SNR penalty from undersampling: {100*(snr - snr_us)/snr:.1f}%")

    plot_results(phantom, kspace, reconstructed, undersampled_recon)


if __name__ == "__main__":
    main()