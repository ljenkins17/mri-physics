import numpy as np
import matplotlib.pyplot as plt


def create_phantom(size: int = 256) -> np.ndarray:
    # Create an empty image array of given size
    phantom = np.zeros((size, size))
    
    # Find the centre coordinates of the image
    cx, cy = size // 2, size // 2

    # Create coordinate grids for the whole image
    Y, X = np.ogrid[:size, :size]
    
    # Define outer ellipse (represents body/skull boundary)
    # Points inside the ellipse equation <= 1 are set to 1.0 (bright tissue)
    outer = ((X - cx) / (0.92 * cx)) ** 2 + ((Y - cy) / (0.69 * cy)) ** 2
    phantom[outer <= 1] = 1.0

    # Define inner ellipse (represents brain tissue, lower signal intensity)
    inner = ((X - cx) / (0.74 * cx)) ** 2 + ((Y - cy) / (0.61 * cy)) ** 2
    phantom[inner <= 1] = 0.4

    # Add two circular features representing ventricles or lesions
    for x_off, y_off, radius, value in [
        (-0.22, 0.0, 0.11, 0.8),  # left feature
        (0.22, 0.0, 0.11, 0.8),   # right feature
    ]:
        # Calculate distance of each pixel from the feature centre
        feature = (X - (cx + x_off * cx)) ** 2 + (Y - (cy + y_off * cy)) ** 2
        # Set pixels within the radius to the given intensity value
        phantom[feature <= (radius * cx) ** 2] = value

    return phantom


def simulate_kspace(image: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    # Apply 2D FFT to convert image from spatial domain to k-space (frequency domain)
    # fftshift reorders the output so that the DC component (zero frequency) is at the centre
    kspace = np.fft.fftshift(np.fft.fft2(image))
    
    # Generate complex Gaussian noise to simulate real scanner noise
    # MRI noise is complex because the signal has both real and imaginary components
    noise = noise_std * (
        np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape)
    )
    
    # Add noise to the k-space data
    return kspace + noise


def reconstruct_from_kspace(kspace: np.ndarray) -> np.ndarray:
    # Reverse the fftshift before applying the inverse FFT
    # Then apply 2D inverse FFT to convert back from k-space to image domain
    image = np.fft.ifft2(np.fft.ifftshift(kspace))
    
    # Take the magnitude to discard the phase component
    # In MRI we typically only display the magnitude image
    return np.abs(image)


def measure_snr(image: np.ndarray, signal_mask: np.ndarray, noise_mask: np.ndarray) -> float:
    # Calculate mean signal intensity within the signal region of interest
    signal = np.mean(image[signal_mask])
    
    # Calculate standard deviation of pixel values in the noise region
    # A background region with no tissue gives a pure noise measurement
    noise = np.std(image[noise_mask])
    
    # SNR = mean signal / noise std — return infinity if noise is zero
    return signal / noise if noise > 0 else np.inf


def undersample_kspace(kspace: np.ndarray, acceleration: int = 2) -> np.ndarray:
    # Copy k-space so we don't modify the original
    undersampled = kspace.copy()
    
    # Zero out every nth line in the phase encode direction
    # This simulates accelerated acquisition (e.g. parallel imaging)
    # Acceleration factor of 2 means half the lines are acquired
    undersampled[::acceleration, :] = 0
    
    return undersampled


def plot_results(phantom, kspace, reconstructed, undersampled_recon):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("MRI k-space Reconstruction Pipeline", fontsize=14, fontweight="bold")

    # Top row: the forward pipeline (ground truth → k-space → reconstruction)
    axes[0, 0].imshow(phantom, cmap="gray")
    axes[0, 0].set_title("Ground Truth Phantom")
    axes[0, 0].axis("off")

    # Display log of k-space magnitude — log scale needed because
    # the DC component at the centre is many orders of magnitude larger than edges
    axes[0, 1].imshow(np.log1p(np.abs(kspace)), cmap="inferno")
    axes[0, 1].set_title("k-space (log magnitude)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(reconstructed, cmap="gray")
    axes[0, 2].set_title("Reconstructed Image (full sampling)")
    axes[0, 2].axis("off")

    # Bottom row: error analysis and undersampling effects
    # bwr (blue-white-red) colormap shows positive and negative errors clearly
    axes[1, 0].imshow(reconstructed - phantom, cmap="bwr", vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title("Reconstruction Error")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(undersampled_recon, cmap="gray")
    axes[1, 1].set_title("Undersampled Reconstruction (R=2)")
    axes[1, 1].axis("off")

    # Wider colour scale for undersampling artefacts — they are much larger than noise errors
    axes[1, 2].imshow(undersampled_recon - phantom, cmap="bwr", vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title("Undersampling Artefacts")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved to reconstruction_results.png")


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    print("MRI k-space Reconstruction Demo")
    print("=" * 40)

    print("Creating phantom...")
    phantom = create_phantom(size=256)

    print("Simulating k-space acquisition (noise_std=0.01)...")
    kspace = simulate_kspace(phantom, noise_std=0.01)

    print("Reconstructing image...")
    reconstructed = reconstruct_from_kspace(kspace)

    # Define signal ROI in the centre of the image where tissue is present
    size = phantom.shape[0]
    signal_mask = np.zeros(phantom.shape, dtype=bool)
    signal_mask[size//4:3*size//4, size//4:3*size//4] = True
    
    # Define noise ROI in the corner of the image where there is no tissue
    noise_mask = np.zeros(phantom.shape, dtype=bool)
    noise_mask[:20, :20] = True

    snr = measure_snr(reconstructed, signal_mask, noise_mask)
    print(f"Reconstruction SNR: {snr:.1f}")

    # Simulate undersampling and measure the SNR penalty
    print("Simulating R=2 undersampling...")
    kspace_us = undersample_kspace(kspace, acceleration=2)
    undersampled_recon = reconstruct_from_kspace(kspace_us)
    snr_us = measure_snr(undersampled_recon, signal_mask, noise_mask)
    print(f"Undersampled SNR:   {snr_us:.1f}")
    print(f"SNR penalty from undersampling: {100*(snr - snr_us)/snr:.1f}%")

    plot_results(phantom, kspace, reconstructed, undersampled_recon)


if __name__ == "__main__":
    main()