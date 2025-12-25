import os
import sys

# --- 🔧 PATH FIXER ---
current_script_path = os.path.abspath(__file__)
batch_folder = os.path.dirname(current_script_path)
project_root = os.path.dirname(batch_folder)
sys.path.append(project_root)
# ---------------------

import torch
from torchvision import transforms, datasets
from models.unet import SimpleUNet
from utils.diffusion_utils import DiffusionEngine
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = os.path.join(project_root, "checkpoints/diffusion/unet_epoch34.pth") # Checks for epoch 34
DATA_PATH = os.path.join(project_root, "data/denoise_data")

# Output Folders
BASE_DIR = os.path.join(batch_folder, "batch_denoiser_results")
DIR_ORIGINAL = os.path.join(BASE_DIR, "1_original")
DIR_NOISY = os.path.join(BASE_DIR, "2_noisy_input")
DIR_RESTORED = os.path.join(BASE_DIR, "3_restored_output")

NUM_IMAGES = 600
NOISE_STEPS = 50 # Keeping it at 50 for clean results

def setup_folders():
    for d in [DIR_ORIGINAL, DIR_NOISY, DIR_RESTORED]:
        if not os.path.exists(d):
            os.makedirs(d)

def smart_denoise_loop(model, diffusion, noisy_image, steps):
    """Refines the image step-by-step for better quality"""
    model.eval()
    x = noisy_image.clone()
    
    for i in reversed(range(1, steps)):
        t = (torch.ones(x.shape[0]) * i).long().to(DEVICE)
        
        with torch.no_grad():
            predicted_noise = model(x) # Fix: Model takes only x
            
        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
        
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
    return x

def save_tensor_as_png(tensor, path):
    """Helper to save a tensor as an image file"""
    # Un-normalize from [-1, 1] to [0, 1]
    t = tensor.squeeze().cpu().detach()
    t = (t + 1) / 2
    t = t.clamp(0, 1)
    
    # Save using Matplotlib (easiest way to handle formats)
    plt.imsave(path, t.permute(1, 2, 0).numpy())

def run_batch_denoising():
    print(f"🚀 Starting Batch Denoising on {DEVICE}...")
    
    # 1. Setup
    setup_folders()
    
    # 2. Load Model
    # Auto-find latest checkpoint if specific one is missing
    model_path = CHECKPOINT
    if not os.path.exists(model_path):
        print(f"⚠️ Checkpoint {model_path} not found. Searching folder...")
        ckpt_dir = os.path.dirname(model_path)
        if os.path.exists(ckpt_dir):
            files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
            if files:
                model_path = os.path.join(ckpt_dir, files[-1])
                print(f"✅ Found alternative: {model_path}")
            else:
                print("❌ No checkpoints found!")
                return
        else:
            print("❌ Checkpoint folder missing.")
            return

    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    diffusion = DiffusionEngine(device=DEVICE)

    # 3. Load Data (CIFAR-10 Test Set)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.CIFAR10(root=DATA_PATH, train=False, transform=transform, download=True)

    print(f"🎨 Processing {NUM_IMAGES} images...")

    # 4. Processing Loop
    for i in tqdm(range(NUM_IMAGES), desc="Denoising"):
        # Get Image
        img, _ = dataset[i]
        img = img.unsqueeze(0).to(DEVICE)
        
        # File Name (e.g., 001.png)
        file_name = f"{i+1:03d}.png"
        
        # A. Save Original
        save_tensor_as_png(img, os.path.join(DIR_ORIGINAL, file_name))
        
        # B. Create & Save Noisy Input
        t = torch.tensor([NOISE_STEPS]).to(DEVICE)
        noisy_img, _ = diffusion.add_noise(img, t)
        save_tensor_as_png(noisy_img, os.path.join(DIR_NOISY, file_name))
        
        # C. Restore & Save Output
        restored_img = smart_denoise_loop(model, diffusion, noisy_img, NOISE_STEPS)
        save_tensor_as_png(restored_img, os.path.join(DIR_RESTORED, file_name))

    print(f"\n✅ Done! Files saved in: {BASE_DIR}")

if __name__ == "__main__":
    run_batch_denoising()