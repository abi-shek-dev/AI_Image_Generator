import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.cycle_gan import Generator
from models.discriminator import Discriminator
import os
from tqdm import tqdm
import glob
import time

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
BATCH_SIZE = 1 
EPOCHS = 60           # Target Goal
RESUME_EPOCH = 35     # We resume from the last safe file (35)
DATA_DIR = "data/horse2zebra"
CHECKPOINT_DIR = "checkpoints/cyclegan"
LOAD_MODEL = True     

def train():
    print(f"Initializing CycleGAN Training on {DEVICE}...")

    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Initialize Models
    gen_AB = Generator().to(DEVICE)
    gen_BA = Generator().to(DEVICE)
    disc_A = Discriminator().to(DEVICE)
    disc_B = Discriminator().to(DEVICE)

    # 3. Optimizers
    opt_gen = optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()), 
        lr=LR, betas=(0.5, 0.999)
    )
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()), 
        lr=LR, betas=(0.5, 0.999)
    )

    # 4. Resume Logic
    start_epoch = 0
    if LOAD_MODEL:
        file_AB = f"{CHECKPOINT_DIR}/gen_horse2zebra_{RESUME_EPOCH}.pth"
        file_BA = f"{CHECKPOINT_DIR}/gen_zebra2horse_{RESUME_EPOCH}.pth"

        if os.path.exists(file_AB) and os.path.exists(file_BA):
            print(f"🔄 Found checkpoint for Epoch {RESUME_EPOCH}! Loading...")
            gen_AB.load_state_dict(torch.load(file_AB, map_location=DEVICE))
            gen_BA.load_state_dict(torch.load(file_BA, map_location=DEVICE))
            
            # Start from the NEXT epoch
            start_epoch = RESUME_EPOCH + 1
            print(f"✅ Resuming training from Epoch {start_epoch} to {EPOCHS}...")
        else:
            print(f"❌ Checkpoint {file_AB} not found. Starting from scratch.")

    # 5. Losses
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # --- CUSTOM DATA LOADER ---
    from torch.utils.data import Dataset
    from PIL import Image

    class HorseZebraDataset(Dataset):
        def __init__(self, root_A, root_B, transform=None):
            self.files_A = sorted(glob.glob(os.path.join(root_A, "*.jpg")))
            self.files_B = sorted(glob.glob(os.path.join(root_B, "*.jpg")))
            self.transform = transform

        def __len__(self):
            return min(len(self.files_A), len(self.files_B))

        def __getitem__(self, index):
            img_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))
            img_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert("RGB"))
            return img_A, img_B

    train_dataset = HorseZebraDataset(
        root_A=f"{DATA_DIR}/trainA", 
        root_B=f"{DATA_DIR}/trainB", 
        transform=transform
    )
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- START TIMER ---
    overall_start_time = time.time()

    # 6. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (real_A, real_B) in enumerate(loop):
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            # --- TRAIN DISCRIMINATORS ---
            fake_A = gen_BA(real_B)
            fake_B = gen_AB(real_A)

            # D_A & D_B
            D_A_real = disc_A(real_A)
            D_A_fake = disc_A(fake_A.detach())
            loss_D_A = (mse(D_A_real, torch.ones_like(D_A_real)) + mse(D_A_fake, torch.zeros_like(D_A_fake))) / 2

            D_B_real = disc_B(real_B)
            D_B_fake = disc_B(fake_B.detach())
            loss_D_B = (mse(D_B_real, torch.ones_like(D_B_real)) + mse(D_B_fake, torch.zeros_like(D_B_fake))) / 2

            loss_D = (loss_D_A + loss_D_B) / 2

            opt_disc.zero_grad()
            loss_D.backward()
            opt_disc.step()

            # --- TRAIN GENERATORS ---
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            cycle_A = gen_BA(fake_B)
            cycle_B = gen_AB(fake_A)
            loss_cycle_A = L1(real_A, cycle_A)
            loss_cycle_B = L1(real_B, cycle_B)

            id_A = gen_BA(real_A)
            id_B = gen_AB(real_B)
            loss_id_A = L1(real_A, id_A)
            loss_id_B = L1(real_B, id_B)

            loss_G = (loss_G_A + loss_G_B) + (loss_cycle_A + loss_cycle_B) * 10 + (loss_id_A + loss_id_B) * 5

            opt_gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            if idx % 200 == 0:
                loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
                loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

        # --- UPDATED SAVING LOGIC ---
        # Save if it's a multiple of 5 OR if it's the very last epoch (to fix the previous bug)
        if (epoch % 5 == 0) or (epoch == EPOCHS - 1):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(gen_AB.state_dict(), f"{CHECKPOINT_DIR}/gen_horse2zebra_{epoch}.pth")
            torch.save(gen_BA.state_dict(), f"{CHECKPOINT_DIR}/gen_zebra2horse_{epoch}.pth")
            print(f"✅ Saved Checkpoint for Epoch {epoch}")

    # --- END TIMER ---
    overall_end_time = time.time()
    total_seconds = overall_end_time - overall_start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"\n🎉 TRAINING FINISHED!")
    print(f"⏱️ Total Time Taken: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    train()