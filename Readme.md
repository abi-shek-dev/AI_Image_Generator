# Generative AI Lab  
## Domain Translation & Image Synthesis

**Project Status:** ✅ Completed  
**Frameworks:** PyTorch, Torchvision  
**Architectures:** CycleGAN (ResNet-based), Diffusion Model (U-Net based)

---

## 📖 Project Overview

This project explores two fundamental paradigms of modern **Generative AI**:

- Unpaired Image-to-Image Translation  
- Denoising Diffusion Probabilistic Models (DDPM)

The goal is to demonstrate how neural networks can modify existing reality through translation and generate structure from random noise through probabilistic modeling.

The project consists of two independent generative engines:
- **The Translator (CycleGAN)**
- **The Denoiser (Diffusion Model)**

---

## 📂 Project Structure

```bash

Generative_AI_Lab/
│
├── checkpoints/             # Saved model weights  
│   ├── cyclegan/            # Horse → Zebra model  
│   └── diffusion/           # Diffusion model  
│
├── data/                    # Datasets  
│   ├── horse2zebra/         # Unpaired training images  
│   └── denoise_data/        # CIFAR-10 (binary format)  
│
├── models/                  # Model architectures  
│   ├── cycle_gan.py         # ResNet Generator & PatchGAN Discriminator  
│   └── unet.py              # U-Net for diffusion  
│
├── utils/  
│   └── diffusion_utils.py   # Noise scheduling & diffusion math  
│
├── train_translator.py      # CycleGAN training  
├── test_translator.py       # Translation visualization  
│
├── train_denoiser.py        # Diffusion training  
├── test_denoiser.py         # Noise → Image restoration  
│
└── README.md

```

---

## ⚙️ Requirements

```bash
pip install torch torchvision matplotlib tqdm requests pillow
```

---

## 📥 Data Setup

Run this script first to download and organize the required datasets (Horse2Zebra and CIFAR-10):

```bash
python download_data.py
```

---

## 🖥️ Hardware Used

- **GPU:** NVIDIA RTX 3050 (4GB VRAM)  
- **CUDA:** 11.8 / 12.x  

---

## 🦓 Engine 1: The Translator (CycleGAN)

Implements **Cycle-Consistent Adversarial Networks**  
(Zhu et al., ICCV 2017).

### Concept

Learns bidirectional mappings:
- **G: X → Y** (Horse → Zebra)  
- **F: Y → X**  

With cycle consistency:
F(G(X)) ≈ X

### Architecture

- Generator: ResNet-based (9 residual blocks)  
- Discriminator: PatchGAN (70×70)  
- Loss:
  - Adversarial Loss  
  - Cycle Consistency Loss  

### How to Run

Train:
```bash
python train_translator.py
```

Test:
```bash
python test_translator.py
```

---

## 🌫️ Engine 2: The Denoiser (Diffusion Model)

Implements a simplified **DDPM**  
(Ho et al., NeurIPS 2020).

### Concept

- Forward process: Adds Gaussian noise  
- Reverse process: U-Net predicts and removes noise  

### Architecture

- Backbone: U-Net  
- Objective: Minimize MSE between actual and predicted noise  

### How to Run

Train:
```bash
python train_denoiser.py
```

Test:
```bash
python test_denoiser.py
```

---

## 📊 Results & Observations

### CycleGAN
- Early epochs: Blurry structure  
- Later epochs: Clear zebra patterns  

### Diffusion
- Recovers semantic structure from heavy noise  
- Demonstrates learned image distribution  

---

## 📚 References

- Zhu et al., ICCV 2017  
- Ho et al., NeurIPS 2020  
- Krizhevsky et al., CIFAR-10  

---

## 🧪 Summary

This project demonstrates adversarial learning and probabilistic generative modeling for image translation and synthesis.
