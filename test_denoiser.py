import torch
from torchvision import transforms, datasets
from models.unet import SimpleUNet
from utils.diffusion_utils import DiffusionEngine
import matplotlib.pyplot as plt
import os
import random  # Import random to pick different images

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# This file is created after Epoch 0 of training
CHECKPOINT = "checkpoints/diffusion/unet_epoch0.pth" 

def test_denoising():
    print("Testing Denoising AI...")
    
    # 1. Load Model
    if not os.path.exists(CHECKPOINT):
        print(f"❌ Checkpoint not found: {CHECKPOINT}")
        print("   Please wait for train_denoiser.py to finish Epoch 0.")
        return

    print(f"Loading model from {CHECKPOINT}...")
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    diffusion = DiffusionEngine(device=DEVICE)

    # 2. Get a Random Image from CIFAR-10
    # We use the test set (images the model hasn't seen during training)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Normalize [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(root="data/denoise_data", train=False, transform=transform, download=False)
    
    # --- RANDOM SELECTION ---
    image_index = random.randint(0, len(dataset) - 1)
    x0, label = dataset[image_index]
    
    # Map label index to class name so we know what it is
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = classes[label]
    
    print(f"Testing on Image Index: {image_index}")
    print(f"Object Type: {class_name}")

    x0 = x0.unsqueeze(0).to(DEVICE) # Add batch dimension
    
    # 3. Add Noise (Simulate a corrupted image)
    # Timestep 300 adds about 30-40% noise. Increase to 500 for harder difficulty.
    t = torch.tensor([300]).to(DEVICE)
    noisy_image, noise_added = diffusion.add_noise(x0, t)
    
    # 4. AI Restoration (Predict & Remove Noise)
    with torch.no_grad():
        # The model guesses what the noise is
        predicted_noise = model(noisy_image)
    
    # We subtract the guessed noise to attempt restoration (Single Step Estimation)
    reconstructed = noisy_image - predicted_noise 

    # 5. Visualize Results
    def show_tensor(t, title, ax):
        t = t.squeeze().cpu().detach()
        t = (t + 1) / 2 # Un-normalize to [0, 1] for display
        t = t.clamp(0, 1)
        ax.imshow(t.permute(1, 2, 0))
        ax.set_title(title)
        ax.axis("off")

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    
    show_tensor(x0, f"Original ({class_name})", ax[0])
    show_tensor(noisy_image, "Input (Noisy)", ax[1])
    show_tensor(reconstructed, "AI Output (Restored)", ax[2])
    
    plt.show()
    print("✅ Test Complete! Check the popup window.")

if __name__ == "__main__":
    test_denoising()