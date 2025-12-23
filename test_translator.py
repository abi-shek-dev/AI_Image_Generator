import torch
from torchvision import transforms
from PIL import Image
from models.cycle_gan import Generator
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# This will be the file created after Epoch 0 finishes
CHECKPOINT_FILE = "checkpoints/cyclegan/gen_horse2zebra_0.pth" 
TEST_IMAGE_PATH = "data/horse2zebra/testA/n02381460_120.jpg" # We pick a random horse to test

def test():
    print(f"Loading generator from {CHECKPOINT_FILE}...")
    
    # 1. Check if the training has actually saved a file yet
    if not os.path.exists(CHECKPOINT_FILE):
        print("❌ Wait! The training hasn't finished Epoch 0 yet.")
        print("   Let the training run until you see a file in 'checkpoints/cyclegan/'")
        return

    # 2. Load the Model
    gen = Generator().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    gen.load_state_dict(checkpoint)
    gen.eval() # Switch to "Test Mode" (turns off randomness)

    # 3. Prepare the Image
    print(f"Loading image: {TEST_IMAGE_PATH}")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Check if test image exists (just in case)
    if not os.path.exists(TEST_IMAGE_PATH):
        # Fallback: Find ANY jpg in the test folder
        test_dir = "data/horse2zebra/testA"
        files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if len(files) > 0:
            full_path = os.path.join(test_dir, files[0])
            print(f"⚠️ specific image not found, using: {full_path}")
            image = Image.open(full_path).convert("RGB")
        else:
            print("❌ No test images found!")
            return
    else:
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 4. Run the Magic
    with torch.no_grad():
        fake_zebra = gen(input_tensor)

    # 5. Show the Result
    # We need to un-normalize the colors to show them correctly
    fake_zebra = fake_zebra.squeeze().cpu().detach()
    fake_zebra = fake_zebra * 0.5 + 0.5 # Reverse the normalization
    
    # Convert input too
    real_horse = transform(image).squeeze().cpu()
    real_horse = real_horse * 0.5 + 0.5

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(real_horse.permute(1, 2, 0))
    ax[0].set_title("Real Horse (Input)")
    ax[0].axis("off")

    ax[1].imshow(fake_zebra.permute(1, 2, 0))
    ax[1].set_title("AI Generated Zebra (Output)")
    ax[1].axis("off")

    plt.show()
    print("✅ Test Complete! Check the popup window.")

if __name__ == "__main__":
    test()