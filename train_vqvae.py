import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.vqvae import VQVAE
import os
import glob
from tqdm import tqdm

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 60               # ⬆️ INCREASED TO 60
DATA_PATH = "data/denoise_data"
CHECKPOINT_DIR = "checkpoints/vqvae"

def train():
    print(f"Initializing VQ-VAE Training on {DEVICE}...")
    
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root=DATA_PATH, train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    # 2. Load Model
    model = VQVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- RESUME LOGIC ---
    start_epoch = 0
    if os.path.exists(CHECKPOINT_DIR):
        # Find all .pth files
        files = glob.glob(f"{CHECKPOINT_DIR}/vqvae_epoch*.pth")
        if files:
            # Sort by epoch number (e.g., epoch5, epoch19, epoch20)
            # We assume filename format is fixed: "vqvae_epochX.pth"
            latest_file = max(files, key=os.path.getctime) # Pick the most recently created file
            print(f"🔄 Found checkpoint: {latest_file}")
            
            try:
                model.load_state_dict(torch.load(latest_file, map_location=DEVICE))
                # Extract epoch number from filename roughly or just trust the user
                # Let's try to parse it: "checkpoints/vqvae/vqvae_epoch19.pth"
                base = os.path.basename(latest_file) # vqvae_epoch19.pth
                num = base.replace("vqvae_epoch", "").replace(".pth", "")
                start_epoch = int(num) + 1
                print(f"✅ Resuming from Epoch {start_epoch} to {EPOCHS}...")
            except:
                print("⚠️ Warning: Could not parse epoch number. Continuing with weights loaded.")
    
    # 3. Training Loop
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        loop = tqdm(dataloader, leave=True)
        total_loss = 0
        
        for i, (images, _) in enumerate(loop):
            images = images.to(DEVICE)
            
            # Forward pass
            vq_loss, data_recon, _ = model(images)
            
            # Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(data_recon, images)
            
            # Total Loss
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # Save Checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        # Save every 5 epochs OR the final one
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/vqvae_epoch{epoch}.pth")
            print(f"✅ Saved VQ-VAE Epoch {epoch}")

    print("\n🎉 VQ-VAE TRAINING FINISHED!")

if __name__ == "__main__":
    train()