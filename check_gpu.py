import torch

def check_gpu():
    print("Checking system hardware...")
    
    if torch.cuda.is_available():
        print("✅ SUCCESS: CUDA (NVIDIA GPU) is available!")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test a small calculation
        x = torch.rand(5, 5).to("cuda")
        print("   Test Tensor successfully created on GPU.")
    else:
        print("❌ WARNING: No GPU detected. Training will be VERY slow.")
        print("   Did you install the CUDA version of PyTorch?")
        print("   Try running: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu()