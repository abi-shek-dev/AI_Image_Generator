import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.unet import SimpleUNet
from utils.diffusion_utils import DiffusionEngine
import os
from tqdm import tqdm  # <--- NEW: Import the progress bar tool

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
BATCH_SIZE = 64
EPOCHS = 20
DATA_PATH = "data/denoise_data" 

def train():
    print(f"Initializing Denoising Training on {DEVICE}...")
    
    # 1. Prepare Data (CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    
    dataset = datasets.CIFAR10(root=DATA_PATH, train=True, transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    
    # 2. Load Model & Engine
    model = SimpleUNet().to(DEVICE)
    diffusion = DiffusionEngine(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss() 

    # 3. Training Loop
    for epoch in range(EPOCHS):
        # Wrap the loader with tqdm for the progress bar
        loop = tqdm(dataloader, leave=True)
        
        for i, (images, _) in enumerate(loop):
            images = images.to(DEVICE)
            
            # A. Sample random timestamps
            t = diffusion.sample_timesteps(images.shape[0])
            
            # B. Add noise
            x_t, noise = diffusion.add_noise(images, t)
            
            # C. Predict noise (SimpleUNet only takes the image)
            predicted_noise = model(x_t)
            
            # D. Loss & Optimize
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # UPDATE PROGRESS BAR
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())
        
        # Save Checkpoint
        os.makedirs("checkpoints/diffusion", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/diffusion/unet_epoch{epoch}.pth")

if __name__ == "__main__":
    train()