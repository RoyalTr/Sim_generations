import platform
import psutil
import multiprocessing
import subprocess

def get_cpu_model_windows():
    try:
        output = subprocess.check_output(
            "wmic cpu get Name", shell=True
        ).decode(errors="ignore").split("\n")
        # Filter empty lines and take first valid line after header
        names = [line.strip() for line in output if line.strip()]
        return names[1] if len(names) > 1 else "Unknown"
    except Exception:
        return "Unknown"

def get_cpu_vendor_windows():
    try:
        output = subprocess.check_output(
            "wmic cpu get Manufacturer", shell=True
        ).decode(errors="ignore").split("\n")
        vendors = [line.strip() for line in output if line.strip()]
        return vendors[1] if len(vendors) > 1 else "Unknown"
    except Exception:
        return "Unknown"

def cpu_info():
    print("=== CPU Information ===\n")

    # Basic system info
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"CPU Vendor: {get_cpu_vendor_windows()}")
    print(f"Exact CPU Model: {get_cpu_model_windows()}\n")

    # Core and processor counts
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    print(f"Physical cores: {physical_cores}")
    print(f"Logical processors: {logical_cores}")
    print(f"multiprocessing.cpu_count(): {multiprocessing.cpu_count()}")

    # Hyper-Threading / SMT Detection
    if logical_cores and physical_cores:
        if logical_cores > physical_cores:
            print("Hyper-Threading / SMT: ENABLED")
        else:
            print("Hyper-Threading / SMT: DISABLED or NOT SUPPORTED")
    else:
        print("Unable to determine Hyper-Threading status")
    print()

    # CPU frequency
    freq = psutil.cpu_freq()
    if freq:
        print(f"Max Frequency: {freq.max:.2f} MHz")
        print(f"Min Frequency: {freq.min:.2f} MHz")
        print(f"Current Frequency: {freq.current:.2f} MHz\n")

    # Per-core usage
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"  Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%\n")

    # CPU times
    print("CPU Times:")
    for k, v in psutil.cpu_times()._asdict().items():
        print(f"  {k}: {v:.2f} seconds")

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("The 'psutil' package is required. Install it with:\n  pip install psutil")
    else:
        cpu_info()
