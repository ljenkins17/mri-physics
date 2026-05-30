# MRI Physics & Medical Imaging

**MSc Medical Physics · NHS Clinical Scientist**

`Python` · `C++` · `CUDA` · `MATLAB`

---

Clinical MRI physicist with experience in scanner QA, image reconstruction,
signal processing, and quantitative imaging. This repository contains
computational work from my NHS clinical scientist training.

---

## Projects

### [`mri_reconstruction/`](./mri_reconstruction/)
**k-space Acquisition and FFT Image Reconstruction**

Simulation of 2D MRI k-space data acquisition, FFT-based image reconstruction,
SNR measurement, and the effects of k-space undersampling on image quality.
Directly relevant to parallel imaging, compressed sensing, and accelerated
acquisition techniques.

```bash
cd mri_reconstruction
pip install -r requirements.txt
python3 kspace_reconstruction.py
```

---

## Coming Soon

- `signal_processing/` — MR spectroscopy peak fitting, noise power spectrum analysis
- `image_analysis/` — SNR/CNR measurement, geometric distortion, automated QA pipelines
- `cpp_reconstruction/` — High-performance C++ and CUDA reconstruction pipeline

---

## Literature

### MRI Physics & Image Reconstruction
- **Bernstein, King & Zhou** — *Handbook of MRI Pulse Sequences* (2004). Comprehensive
  reference for MRI acquisition — covers k-space sampling strategies that motivate
  the undersampling simulation in `mri_reconstruction/`.
- **Liang & Lauterbur** — *Principles of Magnetic Resonance Imaging* (2000). Rigorous
  treatment of the signal equation and Fourier relationship between k-space and image
  space — the theoretical foundation for the FFT reconstruction implemented here.
- **Lustig, Donoho & Pauly** — *Sparse MRI: The application of compressed sensing for
  rapid MR imaging* (2007), Magnetic Resonance in Medicine. Landmark paper on
  undersampled reconstruction — directly motivates the undersampling artefact analysis
  in `mri_reconstruction/`.

### Signal Processing
- **Oppenheim & Schafer** — *Discrete-Time Signal Processing* (2009). Standard
  reference for digital signal processing — covers the FFT algorithm, sampling theory,
  and filtering that underpin MRI reconstruction.
- **Proakis & Manolakis** — *Digital Signal Processing* (2006). Practical companion
  to Oppenheim — good on implementation of FFT-based algorithms.

### Image Quality & QA
- **IPEM Report 112** — *Quality Control and Artefacts in Magnetic Resonance Imaging*
  (2017). UK clinical standard for MRI QA — directly relevant to the SNR/CNR
  measurements and geometric distortion analysis planned in `image_analysis/`.
- **NEMA MS 1-2008** — *Determination of Signal-to-Noise Ratio in Diagnostic Magnetic
  Resonance Imaging*. The standard measurement protocol implemented in
  `image_analysis/`.

---

## Contact

[LinkedIn](https://www.linkedin.com/in/lewis-jenkins-565605190/) · [Email](lewis.jenkins96@gmail.com)

*Also see: [quant-finance](https://github.com/ljenkins17/quant-finance) ·
[ml-engineering](https://github.com/ljenkins17/ml-engineering)*