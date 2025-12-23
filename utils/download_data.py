import os
import zipfile
import requests
from torchvision import datasets
import shutil

# --- CONFIGURATION ---
PROJECT_ROOT = "." 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def download_file(url, save_path):
    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for 404 errors
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

def setup_cyclegan_data():
    """
    Downloads the official Horse2Zebra dataset.
    Updated URL to efrosgans.eecs.berkeley.edu
    """
    dataset_name = "horse2zebra"
    # NEW WORKING URL
    url = f"https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset_name}.zip"
    
    zip_path = os.path.join(DATA_DIR, f"{dataset_name}.zip")
    extract_path = os.path.join(DATA_DIR)
    
    # 1. Check if already extracted
    if os.path.exists(os.path.join(DATA_DIR, dataset_name)):
        print(f"✅ {dataset_name} already exists. Skipping.")
        return

    # 2. Download
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(zip_path):
        download_file(url, zip_path)

    # 3. Unzip
    print("Unzipping...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ Setup complete: {extract_path}/{dataset_name}")
        
        # Clean up zip file to save space
        os.remove(zip_path)
        
    except zipfile.BadZipFile:
        print("❌ Error: The file downloaded is not a valid zip.")
        print("Try deleting the .zip file manually and checking the URL.")

def setup_denoising_data():
    """
    Downloads CIFAR-10 for denoising.
    """
    print("Setting up Denoising Data (CIFAR-10)...")
    denoise_path = os.path.join(DATA_DIR, "denoise_data")
    datasets.CIFAR10(root=denoise_path, train=True, download=True)
    print("✅ Denoising data ready.")

if __name__ == "__main__":
    setup_cyclegan_data()
    setup_denoising_data()