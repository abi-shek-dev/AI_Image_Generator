import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.unet import SimpleUNet
from utils.diffusion_utils import DiffusionEngine
import os
from tqdm import tqdm
import time

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
BATCH_SIZE = 64
EPOCHS = 60          # Goal: 60 Epochs
DATA_PATH = "data/denoise_data" 

# --- RESUME CONFIG (FIXED) ---
# We point directly to where your file is in the screenshot:
RESUME_PATH = "checkpoints/diffusion/unet_epoch19.pth" 
RESUME_EPOCH_NUM = 19

def train():
    print(f"Initializing Denoising Training on {DEVICE}...")
    
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    
    dataset = datasets.CIFAR10(root=DATA_PATH, train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    
    # 2. Load Model & Engine
    model = SimpleUNet().to(DEVICE)
    diffusion = DiffusionEngine(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss() 

    # --- RESUME LOGIC ---
    start_epoch = 0
    if os.path.exists(RESUME_PATH):
        print(f"🔄 Found checkpoint: {RESUME_PATH}")
        print("   Loading weights...")
        # Load weights but ignore potential size mismatches (just in case)
        state_dict = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        start_epoch = RESUME_EPOCH_NUM + 1
        print(f"✅ Resuming training from Epoch {start_epoch} to {EPOCHS}...")
    else:
        print(f"⚠️ Checkpoint not found at: {RESUME_PATH}")
        print("   Starting training from SCRATCH (Epoch 0).")

    # --- START TIMER ---
    overall_start_time = time.time()

    # 3. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        loop = tqdm(dataloader, leave=True)
        
        for i, (images, _) in enumerate(loop):
            images = images.to(DEVICE)
            
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.add_noise(images, t)
            
            # Pass 't' for correct training
            try:
                predicted_noise = model(x_t, t)
            except TypeError:
                predicted_noise = model(x_t)
            
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())
        
        # Save every 5 epochs OR the final epoch
        os.makedirs("checkpoints/diffusion", exist_ok=True)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            torch.save(model.state_dict(), f"checkpoints/diffusion/unet_epoch{epoch}.pth")
            print(f"✅ Saved Checkpoint for Epoch {epoch}")

    # --- END TIMER ---
    overall_end_time = time.time()
    total_seconds = overall_end_time - overall_start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"\n🎉 DENOISER TRAINING FINISHED!")
    print(f"⏱️ Total Time Taken: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    train()