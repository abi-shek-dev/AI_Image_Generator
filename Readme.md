# Generative AI Lab - By Xe54z (Abishek / abi-shek-dev)

## Domain Translation, Image Synthesis & Restoration

**Project Status:** ✅ Completed  

**Frameworks:** PyTorch, Torchvision  
**Architectures:** CycleGAN (ResNet), Diffusion Model (U-Net), VQ-VAE + Transformer  

---

## 📖 Project Overview

This project explores three fundamental paradigms of modern Generative AI:

- Unpaired Image-to-Image Translation (Style Transfer)  
- Denoising Diffusion Probabilistic Models (DDPM)  
- Transformer-Based Image Generation (Vector Quantization)  

The goal is to demonstrate how neural networks can modify reality (Translation), restore structure (Denoising), and hallucinate new images from scratch (Generation).

The project consists of three independent generative engines:

- The Translator (CycleGAN)  
- The Denoiser (Diffusion Model)  
- The Generator (VQ-VAE + Transformer)  

---

## 📂 Project Structure

```bash
Generative_AI_Lab/
│
├── batch_scripts/           # Bulk generation tools
│   ├── batch_translator.py  # Generates Horse → Zebra dataset (600 img)
│   ├── batch_denoiser.py    # Generates Clean/Noisy/Restored dataset
│   └── batch_transformer.py # Generates Input/Output pairs
│
├── checkpoints/             # Saved model weights
│   ├── cyclegan/            # Horse → Zebra model
│   ├── diffusion/           # Diffusion model
│   ├── vqvae/               # VQ-VAE Autoencoder
│   └── transformer/         # PixelTransformer
│
├── data/                    # Datasets
│   ├── horse2zebra/         # Unpaired training images
│   ├── denoise_data/        # CIFAR-10 (binary format)
│
├── models/                  # Model architectures
│   ├── cycle_gan.py         # ResNet Generator & PatchGAN Discriminator
│   ├── unet.py              # U-Net for diffusion
│   ├── vqvae.py             # Vector Quantized Variational Autoencoder
│   └── transformer.py       # Autoregressive PixelTransformer
│
├── utils/
│   ├── diffusion_utils.py   # Noise scheduling & diffusion math
│   └── download_data.py     # Download data for the model
│
├── train_translator.py      # CycleGAN training
├── test_translator.py       # Translation visualization
│
├── train_denoiser.py        # Diffusion training
├── test_denoiser.py         # Noise → Image restoration
│
├── train_vqvae.py           # Stage 1: Train Discrete Codebook
├── train_transformer.py     # Stage 2: Train Autoregressive Prior
├── test_transformer.py      # Generate images from scratch
│
├── main_app.py              # Main app for interactive testing
├── monitor_dashboard.py     # Monitoring dashboard
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install torch torchvision matplotlib tqdm requests pillow
```

---

## 📥 Data Setup

Run this script first to download and organize the required datasets (CIFAR-10):

```bash
python download_data.py
```

To download horse2zebra dataset:

```bash
https://www.kaggle.com/datasets/balnyaupane/horse2zebra
```

---

## 🖥️ Hardware Used

- **GPU:** NVIDIA RTX 3050 (4GB VRAM)  
- **CUDA:** 11.8 / 12.x  

---

## 🦓 Engine 1: The Translator (CycleGAN)

Implements **Cycle-Consistent Adversarial Networks**  
(Zhu et al., ICCV 2017).

### CycleGAN Concept

Learns bidirectional mappings:

- **G:** X → Y (Horse → Zebra)  
- **F:** Y → X  

With cycle consistency:

```
F(G(X)) ≈ X
```

### CycleGAN Architecture

- Generator: ResNet-based (9 residual blocks)  
- Discriminator: PatchGAN (70×70)  

### Loss

- Adversarial Loss  
- Cycle Consistency Loss  

### How to Run CycleGAN

**Train:**

```bash
python train_translator.py
```

**Test:**

```bash
python test_translator.py
```

Example output for 0 epochs:
<img width="998" height="503" alt="image" src="https://github.com/user-attachments/assets/8b2e802d-51a6-4e1f-ab6e-3f64c68e054d" />

Example output for 20 epochs:
<img width="998" height="498" alt="image" src="https://github.com/user-attachments/assets/1e4bac95-65c2-4979-8185-a7871028058e" />

Example output for 59 epochs:
<img width="996" height="501" alt="image" src="https://github.com/user-attachments/assets/e902d014-ac11-420a-ad28-c947e355169b" />

---

## 🌫️ Engine 2: The Denoiser (Diffusion Model)

Implements a simplified **DDPM**  
(Ho et al., NeurIPS 2020).

### Diffusion Concept

- Forward process: Adds Gaussian noise  
- Reverse process: U-Net predicts and removes noise  

### Diffusion Architecture

- Backbone: U-Net  

### Objective

Minimize MSE between actual and predicted noise

### How to Run Diffusion

**Train:**

```bash
python train_denoiser.py
```

**Test:**

```bash
python test_denoiser.py
```

Example output for 35 epochs with 50 noise steps:
<img width="1203" height="498" alt="image" src="https://github.com/user-attachments/assets/8f5c1073-bfce-4d5d-b035-c4bb5fe17c44" />

Example output for 35 epochs with 100 noise steps:
<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/c7f882e5-5601-475a-b443-d4c9f59a61e0" />

Example output for 35 epochs with 200 noise steps:
<img width="1195" height="498" alt="image" src="https://github.com/user-attachments/assets/eda8b755-f371-4076-85e0-17dc9038f988" />

---

## 🧩 Engine 3: The Generator (Transformer-Based)

Implements **Vector Quantized Variational Autoencoder (VQ-VAE)**  
combined with an **Autoregressive Transformer**  
(similar to DALL-E 1 / VQGAN).

### Transformer Concept

**Stage 1 (Compression):**  
The VQ-VAE learns to compress images into a discrete *Codebook* of tokens.

**Stage 2 (Generation):**  
A Transformer learns the sequence of these tokens to generate new images from imagination.

### Transformer Architecture

- **VQ-VAE:** Encoder, Vector Quantizer (Codebook), Decoder  
- **Transformer:** Causal Convolutional Transformer (PixelCNN style)

### How to Run Transformer

**Train the VQ-VAE:**

```bash
python train_vqvae.py
```

**Train the Transformer:**

```bash
python train_transformer.py
```

**Generate new images:**

```bash
python test_transformer.py
```

---

## 📦 Batch Data Generation

Tools included to generate large-scale datasets (600+ images).

```text
batch_translator.py  → Converts Zebras to Horses
batch_denoiser.py    → Adds & Removes Noise
batch_transformer.py → Generates Input/Output Pairs
```

**Run example:**

```bash
python batch_scripts/batch_translator.py
```

---

## 📊 Results & Observations

**CycleGAN**

- Early epochs: Blurry structure  
- Later epochs: Clear zebra patterns  

**Diffusion**

- Recovers semantic structure from heavy noise  
- Demonstrates learned image distribution  

**Transformer**

- Successfully learns discrete representations (Tokens)  
- Generates novel image compositions  
  (Impressionist style on CIFAR-10)

---

## 📚 References

- Zhu et al., ICCV 2017 (CycleGAN)  
- Ho et al., NeurIPS 2020 (DDPM)  
- Van den Oord et al., NeurIPS 2017 (VQ-VAE)  
- Krizhevsky et al., CIFAR-10  
