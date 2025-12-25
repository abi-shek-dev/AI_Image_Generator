import os
import sys
import time
from utils.download_data import setup_cyclegan_data, setup_denoising_data

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("=" * 60)
    print("       🎨  GENERATIVE AI LAB: MISSION CONTROL  🎨")
    print("=" * 60)
    print("Select a task to run:")
    print("-" * 60)

def main():
    while True:
        print_header()
        
        # --- MENU OPTIONS ---
        print(" [1] 📥 Download/Setup Data (Run this first!)")
        print(" [2] 🖥️  Check GPU Status")
        print(" [3] 📊 System Monitor (Dashboard)")
        print("-" * 30)
        print(" [4] 🌫️  Train Denoiser (Diffusion)")
        print(" [5] ✨ Test Denoiser")
        print("-" * 30)
        print(" [6] 🦓 Train Translator (CycleGAN - Horse2Zebra)")
        print(" [7] 🐴 Test Translator")
        print("-" * 30)
        print(" [Q] 🚪 Quit")
        print("=" * 60)
        
        choice = input("Enter your choice: ").upper().strip()

        # --- EXECUTION LOGIC ---
        if choice == "1":
            print("\n🚀 Starting Data Download...")
            # We import the functions directly to run them
            try:
                setup_cyclegan_data()
                setup_denoising_data()
                input("\n✅ Data Setup Complete. Press Enter to continue...")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

        elif choice == "2":
            os.system("python check_gpu.py")
            input("\nPress Enter to return to menu...")

        elif choice == "3":
            # This runs the dashboard. User uses Ctrl+C to exit it.
            os.system("python monitor_dashboard.py")

        elif choice == "4":
            print("\n🧠 Starting Denoising Training...")
            os.system("python train_denoiser.py")
            input("\nTraining stopped. Press Enter to continue...")

        elif choice == "5":
            print("\n✨ Running Denoiser Test...")
            os.system("python test_denoiser.py")
            input("\nTest complete. Press Enter to continue...")

        elif choice == "6":
            print("\n🦓 Starting CycleGAN Training...")
            os.system("python train_translator.py")
            input("\nTraining stopped. Press Enter to continue...")

        elif choice == "7":
            print("\n🐴 Running Translator Test...")
            os.system("python test_translator.py")
            input("\nTest complete. Press Enter to continue...")

        elif choice == "Q":
            print("\nExiting Mission Control. Goodbye!")
            sys.exit()

        else:
            print("\n❌ Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit()