import torch
from torchvision import transforms, datasets
from models.vqvae import VQVAE
import matplotlib.pyplot as plt
import os
import random

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Default target file
DEFAULT_CHECKPOINT = "checkpoints/vqvae/vqvae_epoch29.pth" 

def test_vqvae():
    print("Testing VQ-VAE Reconstruction...")
    
    # 1. Determine which checkpoint to load
    model_path = DEFAULT_CHECKPOINT
    
    if not os.path.exists(model_path):
        print(f"⚠️ Checkpoint not found at: {model_path}")
        # Try to find ANY checkpoint in the folder
        if os.path.exists("checkpoints/vqvae"):
            files = sorted(os.listdir("checkpoints/vqvae"))
            # Filter only .pth files
            files = [f for f in files if f.endswith(".pth")]
            
            if files:
                model_path = os.path.join("checkpoints/vqvae", files[-1])
                print(f"✅ Found alternative: {model_path}")
            else:
                print("❌ No checkpoint files found in checkpoints/vqvae/")
                return
        else:
            print("❌ Checkpoint folder does not exist.")
            return

    print(f"Loading model from {model_path}...")
    model = VQVAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Get a Random Image (CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root="data/denoise_data", train=False, transform=transform, download=False)
    
    if len(dataset) == 0:
        print("❌ Dataset not found or empty.")
        return

    # Pick random images to test
    indices = random.sample(range(len(dataset)), 5) # Test 5 images at once
    
    # 3. Visualization Loop
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]
        img = img.unsqueeze(0).to(DEVICE) # Add batch dim
        
        # --- THE MAGIC PASS ---
        with torch.no_grad():
            # The model returns: loss, reconstructed_image, perplexity
            # We only care about the reconstruction (2nd return value)
            _, reconstructed, _ = model(img)

        # Helper to clean up image for plotting
        def process(t):
            t = t.squeeze().cpu().detach()
            t = t * 0.5 + 0.5 # Un-normalize
            t = t.clamp(0, 1)
            return t.permute(1, 2, 0)

        # Plot Original
        axes[0, i].imshow(process(img))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        
        # Plot Reconstruction
        axes[1, i].imshow(process(reconstructed))
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    print("✅ Test Complete! Check the popup window.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vqvae()