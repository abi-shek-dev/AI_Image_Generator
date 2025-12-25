import torch
from torchvision import transforms, datasets
from models.unet import SimpleUNet
from utils.diffusion_utils import DiffusionEngine
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm # Progress bar

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/diffusion/unet_epoch34.pth" # Ensure this matches your latest file

def smart_denoise_loop(model, diffusion, noisy_image, start_t):
    """
    Restores the image STEP-BY-STEP (Iterative Sampling).
    This creates much sharper results than a single guess.
    """
    model.eval()
    x = noisy_image.clone()
    
    # We loop backwards from start_t down to 1 (e.g., 300 -> 299 -> ... -> 0)
    print(f"   Using Smart Sampling to clean image ({start_t} steps)...")
    for i in tqdm(reversed(range(1, start_t)), total=start_t-1):
        t = (torch.ones(x.shape[0]) * i).long().to(DEVICE)
        
        # 1. Get model prediction (What is the noise?)
        with torch.no_grad():
            predicted_noise = model(x)
            
        # 2. Get the math values for this specific step
        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
        
        # 3. Remove a tiny layer of noise
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x) # No noise in the final step
            
        # The Magic Formula (Langevin Dynamics)
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
    return x

def test_denoising():
    print("Testing Denoising AI (Advanced Sampling)...")
    
    if not os.path.exists(CHECKPOINT):
        print(f"❌ Checkpoint not found: {CHECKPOINT}")
        return

    # 1. Load Model
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    
    diffusion = DiffusionEngine(device=DEVICE)

    # 2. Get Random Image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.CIFAR10(root="data/denoise_data", train=False, transform=transform, download=False)
    
    idx = random.randint(0, len(dataset) - 1)
    x0, label = dataset[idx]
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Object: {classes[label]}")

    x0 = x0.unsqueeze(0).to(DEVICE)
    
    # 3. Add Noise (Simulation)
    # Let's try 200 steps. It's enough to corrupt the image but easier to recover.
    STEPS = 200 
    t = torch.tensor([STEPS]).to(DEVICE)
    noisy_image, _ = diffusion.add_noise(x0, t)
    
    # 4. AI Restoration (The Loop)
    reconstructed = smart_denoise_loop(model, diffusion, noisy_image, STEPS)

    # 5. Visualize
    def show_tensor(t, title, ax):
        t = t.squeeze().cpu().detach()
        t = (t + 1) / 2
        t = t.clamp(0, 1)
        ax.imshow(t.permute(1, 2, 0))
        ax.set_title(title)
        ax.axis("off")

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    show_tensor(x0, f"Original ({classes[label]})", ax[0])
    show_tensor(noisy_image, "Input (Noisy)", ax[1])
    show_tensor(reconstructed, "AI Output (Smart Sampled)", ax[2])
    
    plt.show()

if __name__ == "__main__":
    test_denoising()