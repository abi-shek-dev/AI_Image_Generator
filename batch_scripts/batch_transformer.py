import os
import sys

# --- 🔧 PATH FIXER ---
current_script_path = os.path.abspath(__file__)
batch_folder = os.path.dirname(current_script_path)
project_root = os.path.dirname(batch_folder)
sys.path.append(project_root)
# ---------------------

import torch
import torch.nn.functional as F
from models.vqvae import VQVAE
from models.transformer import PixelTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VQVAE_DIR = os.path.join(project_root, "checkpoints/vqvae")
TRANSFORMER_DIR = os.path.join(project_root, "checkpoints/transformer")

# Output Structure
BASE_DIR = os.path.join(batch_folder, "batch_transformer_results")
INPUT_DIR = os.path.join(BASE_DIR, "input_seeds")       # The Random Start
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_output") # The Final Image

NUM_IMAGES = 600
BATCH_SIZE = 20  
TEMPERATURE = 1.0 # Controls diversity

def setup_folders():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_latest_checkpoint(folder, prefix):
    if not os.path.exists(folder): return None
    files = glob.glob(os.path.join(folder, f"{prefix}*.pth"))
    if not files: return None
    return max(files, key=os.path.getctime)

def save_batch(tensors, start_index, folder):
    """Saves a batch of tensors as PNG files"""
    tensors = tensors.cpu().detach()
    tensors = tensors * 0.5 + 0.5 # Un-normalize
    tensors = tensors.clamp(0, 1)
    
    for i in range(tensors.shape[0]):
        file_name = f"{start_index + i + 1:03d}.png"
        path = os.path.join(folder, file_name)
        plt.imsave(path, tensors[i].permute(1, 2, 0).numpy())

def run_transformer_batch():
    print(f"🚀 Starting Transformer Generation on {DEVICE}...")
    setup_folders()

    # 1. Load Models
    vqvae_path = get_latest_checkpoint(VQVAE_DIR, "vqvae_epoch")
    trans_path = get_latest_checkpoint(TRANSFORMER_DIR, "trans_epoch")

    if not vqvae_path or not trans_path:
        print("❌ Error: Checkpoints not found. Train VQ-VAE & Transformer first.")
        return

    print(f"   Loading VQ-VAE: {os.path.basename(vqvae_path)}")
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=DEVICE))
    vqvae.eval()

    print(f"   Loading Transformer: {os.path.basename(trans_path)}")
    transformer = PixelTransformer().to(DEVICE)
    transformer.load_state_dict(torch.load(trans_path, map_location=DEVICE))
    transformer.eval()

    # 2. Generation Loop
    print(f"🎨 Generating {NUM_IMAGES} Input/Output pairs...")
    
    num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
    current_img_count = 0
    seq_len = 16 * 16 # 256 tokens

    for b in tqdm(range(num_batches), desc="Processing Batches"):
        current_batch_size = min(BATCH_SIZE, NUM_IMAGES - current_img_count)
        
        # --- STEP A: Create Input (Random Start) ---
        indices = torch.zeros(current_batch_size, seq_len, dtype=torch.long).to(DEVICE)
        
        # Randomize the first token to get unique starts
        indices[:, 0] = torch.randint(0, 512, (current_batch_size,)).to(DEVICE)
        
        # Visualize the "Input" (The mostly empty/random canvas)
        with torch.no_grad():
            z_q_input = vqvae._vq_vae.embeddings(indices) 
            z_q_input = z_q_input.permute(0, 2, 1).view(current_batch_size, 64, 16, 16)
            input_images = vqvae._decoder(z_q_input)
            
        # Save INPUTS
        save_batch(input_images, current_img_count, INPUT_DIR)

        # --- STEP B: Generate Output (The Dream) ---
        with torch.no_grad():
            # Autoregressive loop (Predict all tokens)
            for i in range(1, seq_len):
                logits = transformer(indices) 
                current_logits = logits[:, :, i]
                probs = F.softmax(current_logits / TEMPERATURE, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                indices[:, i] = next_token

            # Decode final result
            z_q_output = vqvae._vq_vae.embeddings(indices) 
            z_q_output = z_q_output.permute(0, 2, 1).view(current_batch_size, 64, 16, 16)
            output_images = vqvae._decoder(z_q_output)

        # Save OUTPUTS
        save_batch(output_images, current_img_count, OUTPUT_DIR)
        
        current_img_count += current_batch_size

    print(f"\n✅ Done! Check folder: {BASE_DIR}")

if __name__ == "__main__":
    run_transformer_batch()