import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.unet import SimpleUNet
from utils.diffusion_utils import DiffusionEngine
import os

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
BATCH_SIZE = 32
EPOCHS = 100 # PhD projects require long training!
DATA_PATH = "data/denoise_data" # Put some high-quality JPGs here

def train():
    print(f"Initializing Denoising Training on {DEVICE}...")
    
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Keep small for testing
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Normalize to [-1, 1]
    ])
    
    # Use ImageFolder - expects structure: data/denoise_data/class_name/image.jpg
    dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Load Model & Engine
    model = SimpleUNet().to(DEVICE)
    diffusion = DiffusionEngine(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss() # We compare predicted noise vs actual noise

    # 3. Training Loop
    for epoch in range(EPOCHS):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            
            # A. Sample random timestamps (t)
            t = diffusion.sample_timesteps(images.shape[0])
            
            # B. Add noise to the images (Forward Process)
            x_t, noise = diffusion.add_noise(images, t)
            
            # C. Model predicts the noise
            predicted_noise = model(x_t)
            
            # D. Calculate Loss & Optimize
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
        
        # Save Checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/diffusion/unet_epoch{epoch}.pth")

if __name__ == "__main__":
    # Create checkpoint folder if it doesn't exist
    os.makedirs("checkpoints/diffusion", exist_ok=True)
    train()