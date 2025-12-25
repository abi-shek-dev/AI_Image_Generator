import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from models.vqvae import VQVAE
from models.transformer import PixelTransformer
import os
import glob

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VQVAE_DIR = "checkpoints/vqvae"
TRANSFORMER_DIR = "checkpoints/transformer"
NUM_SAMPLES = 8  # How many images to generate

def get_latest_checkpoint(folder, prefix):
    if not os.path.exists(folder): return None
    files = glob.glob(f"{folder}/{prefix}*.pth")
    if not files: return None
    return max(files, key=os.path.getctime)

def generate_images():
    print(f"✨ Initializing Generative AI on {DEVICE}...")

    # 1. Load VQ-VAE (The Painter)
    vqvae_path = get_latest_checkpoint(VQVAE_DIR, "vqvae_epoch")
    if not vqvae_path:
        print("❌ Error: VQ-VAE checkpoint not found.")
        return
    print(f"🎨 Loading VQ-VAE from {vqvae_path}...")
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=DEVICE))
    vqvae.eval()

    # 2. Load Transformer (The Dreamer)
    trans_path = get_latest_checkpoint(TRANSFORMER_DIR, "trans_epoch")
    if not trans_path:
        print("❌ Error: Transformer checkpoint not found.")
        return
    print(f"🧠 Loading Transformer from {trans_path}...")
    transformer = PixelTransformer().to(DEVICE)
    transformer.load_state_dict(torch.load(trans_path, map_location=DEVICE))
    transformer.eval()

    # 3. The Dreaming Loop (Autoregressive Generation)
    # The latent map size is 16x16 = 256 tokens
    seq_len = 16 * 16 
    
    # Start with empty tokens (zeros)
    indices = torch.zeros(NUM_SAMPLES, seq_len, dtype=torch.long).to(DEVICE)
    
    print("⏳ Generating tokens (this takes a moment)...")
    with torch.no_grad():
        for i in range(seq_len):
            # Ask Transformer to predict the next token based on what we have so far
            logits = transformer(indices) # [Batch, Vocab, Seq_Len]
            
            # Focus only on the current step 'i'
            current_logits = logits[:, :, i]
            
            # Apply Temperature (Higher = More creative/random, Lower = More stable)
            temperature = 0.7 
            probs = F.softmax(current_logits / temperature, dim=-1)
            
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Store the prediction
            indices[:, i] = next_token
            
            # Simple progress indicator
            if i % 50 == 0:
                print(f"   ...generated {i}/{seq_len} tokens")

    print("🎨 Decoding tokens into images...")
    
    # 4. Decode (Turn tokens back into pixels)
    with torch.no_grad():
        # Look up the vectors for these token indices
        z_q = vqvae._vq_vae.embeddings(indices) # [Batch, Seq, Dim]
        
        # Reshape back to 2D Image Map: [Batch, Dim, 16, 16]
        z_q = z_q.permute(0, 2, 1).view(NUM_SAMPLES, 64, 16, 16)
        
        # Run through VQ-VAE Decoder
        generated_images = vqvae._decoder(z_q)

    # 5. Visualize
    def process(t):
        t = t.squeeze().cpu().detach()
        t = t * 0.5 + 0.5 # Un-normalize
        t = t.clamp(0, 1)
        return t.permute(1, 2, 0)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("AI Generated Images (Transformer)", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < NUM_SAMPLES:
            ax.imshow(process(generated_images[i]))
            ax.axis("off")
            
    plt.tight_layout()
    plt.show()
    print("✅ Generation Complete!")

if __name__ == "__main__":
    generate_images()