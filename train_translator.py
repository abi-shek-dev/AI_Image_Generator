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

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
BATCH_SIZE = 1 # CycleGAN works best with batch size 1
EPOCHS = 20
DATA_DIR = "data/horse2zebra"

def train():
    print(f"Initializing CycleGAN Training on {DEVICE}...")

    # 1. Prepare Data
    # CycleGAN expects folders: trainA (Horses) and trainB (Zebras)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # We use a custom trick to load unrelated folders as a pair
    # For simplicity in this script, we iterate manually or use a custom Dataset
    # Here we load them separately and zip them in the loop
    dataset_A = ImageFolder(root=DATA_DIR, transform=transform) # This expects subfolders inside trainA
    # NOTE: ImageFolder requires a subfolder structure. 
    # If horse2zebra/trainA contains images directly, we need a custom loader.
    # Let's use a simpler custom loader class below.

    # 2. Initialize Models
    # G_AB: Horse -> Zebra | G_BA: Zebra -> Horse
    gen_AB = Generator().to(DEVICE)
    gen_BA = Generator().to(DEVICE)
    
    # D_A: Checks Real Horse | D_B: Checks Real Zebra
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

    # 4. Losses
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # --- CUSTOM DATA LOADER FOR UNPAIRED IMAGES ---
    from torch.utils.data import Dataset
    from PIL import Image
    import glob

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

    # Setup the loader
    train_dataset = HorseZebraDataset(
        root_A=f"{DATA_DIR}/trainA", 
        root_B=f"{DATA_DIR}/trainB", 
        transform=transform
    )
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (real_A, real_B) in enumerate(loop):
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            # --- TRAIN DISCRIMINATORS ---
            # Generate Fakes
            fake_A = gen_BA(real_B) # Zebra -> Fake Horse
            fake_B = gen_AB(real_A) # Horse -> Fake Zebra

            # D_A Checks (Real Horse vs Fake Horse)
            D_A_real = disc_A(real_A)
            D_A_fake = disc_A(fake_A.detach())
            loss_D_A = (mse(D_A_real, torch.ones_like(D_A_real)) + mse(D_A_fake, torch.zeros_like(D_A_fake))) / 2

            # D_B Checks (Real Zebra vs Fake Zebra)
            D_B_real = disc_B(real_B)
            D_B_fake = disc_B(fake_B.detach())
            loss_D_B = (mse(D_B_real, torch.ones_like(D_B_real)) + mse(D_B_fake, torch.zeros_like(D_B_fake))) / 2

            loss_D = (loss_D_A + loss_D_B) / 2

            opt_disc.zero_grad()
            loss_D.backward()
            opt_disc.step()

            # --- TRAIN GENERATORS ---
            # Adversarial Loss (Trick the Discriminators)
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            # Cycle Consistency Loss (Horse -> Zebra -> Horse should look like original)
            cycle_A = gen_BA(fake_B)
            cycle_B = gen_AB(fake_A)
            loss_cycle_A = L1(real_A, cycle_A)
            loss_cycle_B = L1(real_B, cycle_B)

            loss_G = (loss_G_A + loss_G_B) + (loss_cycle_A + loss_cycle_B) * 10

            opt_gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            if idx % 200 == 0:
                loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
                loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

        # Save Checkpoint
        if epoch % 1 == 0:
            os.makedirs("checkpoints/cyclegan", exist_ok=True)
            torch.save(gen_AB.state_dict(), f"checkpoints/cyclegan/gen_horse2zebra_{epoch}.pth")
            torch.save(gen_BA.state_dict(), f"checkpoints/cyclegan/gen_zebra2horse_{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch}")

if __name__ == "__main__":
    train()