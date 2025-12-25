import os
import sys

# --- 🔧 PATH FIXER (Must be at the top) ---
current_script_path = os.path.abspath(__file__)
batch_folder = os.path.dirname(current_script_path)
project_root = os.path.dirname(batch_folder)
sys.path.append(project_root)
# ------------------------------------------

import torch
from torchvision import transforms
from PIL import Image
from models.cycle_gan import Generator
import shutil
import random
from tqdm import tqdm

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = os.path.join(project_root, "checkpoints/cyclegan/gen_horse2zebra_59.pth")

# Folders
BASE_DIR = os.path.join(batch_folder, "batch_translation_results")
INPUT_DIR = os.path.join(BASE_DIR, "input_horses")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_zebras")

# Source Data
SOURCE_DATASET = os.path.join(project_root, "data/horse2zebra/trainA")
NUM_IMAGES = 600

def setup_folders():
    """Creates folders and copies images with sequential numbering (001.jpg - 600.jpg)"""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Check if input folder is empty
    existing_files = os.listdir(INPUT_DIR)
    
    if len(existing_files) < NUM_IMAGES:
        print(f"📂 Populating {INPUT_DIR} with {NUM_IMAGES} images (Renamed 001-600)...")
        
        if not os.path.exists(SOURCE_DATASET):
            print(f"❌ Error: Source dataset not found at {SOURCE_DATASET}")
            return False

        # Get all horse images
        all_horses = [f for f in os.listdir(SOURCE_DATASET) if f.endswith('.jpg')]
        
        if len(all_horses) < NUM_IMAGES:
            print(f"⚠️ Warning: Only found {len(all_horses)} images. Using all of them.")
            selected_horses = all_horses
        else:
            selected_horses = random.sample(all_horses, NUM_IMAGES)

        # --- COPY & RENAME LOOP ---
        for i, file_name in enumerate(tqdm(selected_horses, desc="Renaming & Copying")):
            src = os.path.join(SOURCE_DATASET, file_name)
            
            # Create new name: 001.jpg, 002.jpg, ... 600.jpg
            new_name = f"{i+1:03d}.jpg" 
            dst = os.path.join(INPUT_DIR, new_name)
            
            shutil.copy(src, dst)
    else:
        print(f"✅ Input folder already populated. Skipping copy.")
    
    return True

def run_batch_translation():
    print(f"🚀 Starting Batch Translation (Sequential Names) on {DEVICE}...")

    # 1. Setup Data
    if not setup_folders():
        return

    # 2. Load Model
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"❌ Checkpoint not found: {CHECKPOINT_FILE}")
        return

    print("LOADING MODEL...")
    gen = Generator().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    gen.load_state_dict(checkpoint)
    gen.eval()

    # 3. Processing Loop
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Get files and sort them so 001 processes before 002
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")])
    
    print(f"🎨 Translating {len(image_files)} images...")
    
    for filename in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(INPUT_DIR, filename)
        save_path = os.path.join(OUTPUT_DIR, filename) # Saves as 001.jpg, etc.

        if os.path.exists(save_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                fake_zebra = gen(input_tensor)

            fake_zebra = fake_zebra.squeeze().cpu().detach()
            fake_zebra = fake_zebra * 0.5 + 0.5
            fake_zebra = fake_zebra.clamp(0, 1)
            
            output_image = transforms.ToPILImage()(fake_zebra)
            output_image.save(save_path)

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    print(f"\n✅ Done! Check the folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_batch_translation()