import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.vqvae import VQVAE
from models.transformer import PixelTransformer
import os
import glob
from tqdm import tqdm

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# We look for the FINAL VQVAE model (or the latest one)
VQVAE_DIR = "checkpoints/vqvae"
TRANSFORMER_DIR = "checkpoints/transformer"
EPOCHS = 40           # ⬆️ TARGET: 40 Epochs

def get_latest_vqvae():
    if not os.path.exists(VQVAE_DIR): return None
    files = glob.glob(f"{VQVAE_DIR}/vqvae_epoch*.pth")
    if not files: return None
    # Sort by epoch number (simple alphanumeric sort usually works for standard naming)
    return max(files, key=os.path.getctime)

def train_transformer():
    print(f"Initializing Transformer Training on {DEVICE}...")

    # 1. Load Pre-trained VQ-VAE (Frozen)
    vqvae_path = get_latest_vqvae()
    if not vqvae_path:
        print("❌ Error: No VQ-VAE checkpoints found. Please run train_vqvae.py first!")
        return
        
    print(f"🔒 Loading Frozen VQ-VAE from: {vqvae_path}")
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=DEVICE))
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False # Freeze VQ-VAE
    
    # 2. Setup Data
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="data/denoise_data", train=True, transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. Initialize Transformer
    transformer = PixelTransformer().to(DEVICE)
    optimizer = optim.Adam(transformer.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --- RESUME LOGIC ---
    start_epoch = 0
    if os.path.exists(TRANSFORMER_DIR):
        files = glob.glob(f"{TRANSFORMER_DIR}/trans_epoch*.pth")
        if files:
            latest = max(files, key=os.path.getctime)
            print(f"🔄 Resuming Transformer from: {latest}")
            transformer.load_state_dict(torch.load(latest, map_location=DEVICE))
            try:
                base = os.path.basename(latest)
                num = base.replace("trans_epoch", "").replace(".pth", "")
                start_epoch = int(num) + 1
            except:
                pass

    # 4. Training Loop
    transformer.train()
    for epoch in range(start_epoch, EPOCHS):
        loop = tqdm(dataloader, leave=True)
        for images, _ in loop:
            images = images.to(DEVICE)
            
            # Step A: Encode images to Tokens using VQ-VAE
            with torch.no_grad():
                z = vqvae._encoder(images)
                z = vqvae._pre_vq_conv(z)
                _, _, indices = vqvae._vq_vae(z)
                
                # Reshape indices to sequence (Batch, 256)
                indices = indices.view(images.shape[0], -1)

            # Step B: Train Transformer
            # Predict next token based on previous ones
            inputs = indices[:, :-1]   # 0 to N-1
            targets = indices[:, 1:]   # 1 to N
            
            outputs = transformer(inputs)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # Save
        os.makedirs(TRANSFORMER_DIR, exist_ok=True)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            torch.save(transformer.state_dict(), f"{TRANSFORMER_DIR}/trans_epoch{epoch}.pth")

    print("🎉 Transformer Training Complete!")

if __name__ == "__main__":
    train_transformer()