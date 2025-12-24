import time
import os
import psutil
import subprocess
import sys

def clear_screen():
    # Clears the terminal screen for a clean "dashboard" look
    os.system('cls' if os.name == 'nt' else 'clear')

def get_gpu_info():
    try:
        # We use the nvidia-smi command which comes with your drivers
        # It asks for: temperature, utilization, and memory used
        cmd = "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        
        # Parse the output (e.g., "72, 98, 3500, 4096")
        temp, util, mem_used, mem_total = output.split(', ')
        return int(temp), int(util), int(mem_used), int(mem_total)
    except:
        return None, 0, 0, 0

def get_progress_bar(percent, width=20):
    # Creates a visual bar: [████████░░]
    filled_len = int(width * percent // 100)
    bar = '█' * filled_len + '░' * (width - filled_len)
    return f"[{bar}]"

def main():
    print("Initializing System Monitor... (Press Ctrl+C to Stop)")
    time.sleep(1)

    try:
        while True:
            # 1. Get Stats
            cpu_usage = psutil.cpu_percent(interval=0.5)
            ram = psutil.virtual_memory()
            gpu_temp, gpu_util, gpu_mem_used, gpu_mem_total = get_gpu_info()

            # 2. Clear Screen
            clear_screen()

            # 3. Print Header
            print("="*50)
            print(f"    🚀 AI TRAINING MISSION CONTROL    ")
            print("="*50)

            # 4. CPU Section
            print(f"\n🧠  CPU Usage:    {cpu_usage}%")
            print(f"    {get_progress_bar(cpu_usage)}")
            print(f"    RAM Used:     {ram.percent}% ({ram.used // (1024**3)} GB / {ram.total // (1024**3)} GB)")

            # 5. GPU Section (The Important Part!)
            if gpu_temp is not None:
                # Color code the temperature warning
                temp_status = "✅ OK"
                if gpu_temp > 80: temp_status = "⚠️ HOT"
                if gpu_temp > 87: temp_status = "🔥 DANGER"

                print("-" * 50)
                print(f"🎮  GPU (RTX 3050):")
                print(f"    Temperature:  {gpu_temp}°C  {temp_status}")
                print(f"    Utilization:  {gpu_util}%")
                print(f"    {get_progress_bar(gpu_util)}")
                print(f"    VRAM Usage:   {gpu_mem_used} MB / {gpu_mem_total} MB")
            else:
                print("\n⚠️  GPU Not Found (Are NVIDIA drivers installed?)")

            print("-" * 50)
            print("\n(Press Ctrl+C to Exit Monitor)")
            
            # Refresh every 2 seconds
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping Monitor. Goodbye!")

if __name__ == "__main__":
    main()