#!/usr/bin/env python3
"""
cpu_info.py
AUTHOR: Dr. Royal Truman
VERSION: 0.9
"""
import platform
import multiprocessing
import subprocess
import sys

try:
    import psutil
    import cpuinfo
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  'psutil' or 'py-cpuinfo' not installed. Install with: pip install psutil py-cpuinfo\n")

# ===========================
# PLATFORM-SPECIFIC HELPERS
# ===========================

def get_cpu_model_windows():
# If necessary write an analog get_cpu_model_windows()
    try:
        output = subprocess.check_output("wmic cpu get Name", shell=True, stderr=subprocess.DEVNULL).decode(errors="ignore").splitlines()
        names = [line.strip() for line in output if line.strip()]
        return names[1] if len(names) > 1 else "Unknown"
    except Exception:
        return "Unknown"

def get_cpu_vendor_windows():
    try:
        output = subprocess.check_output("wmic cpu get Manufacturer", shell=True, stderr=subprocess.DEVNULL).decode(errors="ignore").splitlines()
        vendors = [line.strip() for line in output if line.strip()]
        return vendors[1] if len(vendors) > 1 else "Unknown"
    except Exception:
        return "Unknown"

# ===========================
# UTILITY FUNCTIONS
# ===========================

def bytes_to_mb(b):
    """Convert bytes to MB (integer division)."""
    return b // (1024 * 1024) if b else None

def safe_getattr(obj, attr, default="N/A"):
    """Safely get attribute or dict key with fallback."""
    try:
        return getattr(obj, attr, default) if hasattr(obj, attr) else obj.get(attr, default)
    except Exception:
        return default

# ===========================
# CACHE INFO DISPLAY
# ===========================

def display_cache_info():
    """Display CPU cache sizes and definitions."""
    try:
        info = cpuinfo.get_cpu_info()
    except Exception as e:
        print(f"❌ Could not retrieve CPU info: {e}")
        return

    definitions = {
        'l1_data_cache_size': "Size of L1 data cache per core (in MB) — holds active variables like p_A_t, N.",
        'l1_instruction_cache_size': "Size of L1 instruction cache per core (in MB) — holds currently executing code.",
        'l2_cache_size': "Total L2 cache size per core or cluster (in MB) — stores frequently accessed data close to CPU for speed.",
        'l3_cache_size': "Total shared L3 cache size across all cores (in MB) — acts as a last-level buffer before slower main memory (DRAM).",
        'l2_cache_line_size': "Size (in bytes) of each cache line in L2 — smallest unit of data transferred between cache levels or memory.",
        'l2_cache_associativity': "Number of ways L2 cache is set-associative — affects how memory addresses map to cache slots (higher = more flexible, less conflict)."
    }

    preferred_order = [
        'l1_data_cache_size',
        'l1_instruction_cache_size',
        'l2_cache_size',
        'l3_cache_size',
        'l2_cache_line_size',
        'l2_cache_associativity'
    ]

    available_cache_keys = [k for k in info.keys() if 'cache' in k.lower()]

    print("=== Detected Cache Info ===")
    # Show preferred keys first
    for key in preferred_order:
        if key in info and info[key] is not None:
            val = bytes_to_mb(info[key]) if 'size' in key and key != 'l2_cache_line_size' else info[key]
            print(f"{key}: {val}")

    # Show any extra cache keys
    for key in available_cache_keys:
        if key not in preferred_order and info[key] is not None:
            val = bytes_to_mb(info[key]) if 'size' in key and key != 'l2_cache_line_size' else info[key]
            print(f"{key}: {val}")

    print("\n=== Definitions ===")
    # Show definitions only for keys that exist
    for key in preferred_order:
        if key in info and info[key] is not None and key in definitions:
            print(f"{key}: {definitions[key]}")

    for key in available_cache_keys:
        if key not in preferred_order and key in definitions:
            print(f"{key}: {definitions[key]}")

# ===========================
# CORE & PROCESSOR INFO
# ===========================

def display_core_info():
    """Display physical/logical core counts and HT/SMT status."""
    logical = multiprocessing.cpu_count()
    physical = psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else None

    print(f"Physical cores: {physical or 'N/A'}")
    print(f"Logical processors: {logical}")
    print(f"multiprocessing.cpu_count(): {logical}")

    if physical and logical:
        ht_status = "ENABLED" if logical > physical else "DISABLED or NOT SUPPORTED"
        print(f"Hyper-Threading / SMT: {ht_status}")
    else:
        print("Unable to determine Hyper-Threading status")
    print()

# ===========================
# CPU FREQUENCY INFO
# ===========================

def display_frequency_info():
    """Display CPU frequency stats if available."""
    if not PSUTIL_AVAILABLE:
        return
    freq = psutil.cpu_freq()
    if freq:
        print(f"Max Frequency: {freq.max:.2f} MHz")
        print(f"Min Frequency: {freq.min:.2f} MHz")
        print(f"Current Frequency: {freq.current:.2f} MHz")
    print()

# ===========================
# CPU USAGE & TIMES
# ===========================

def display_usage_and_times():
    """Display per-core usage and CPU times."""
    if not PSUTIL_AVAILABLE:
        return

    print("CPU Usage Per Core (sampled over 1 sec):")
    percents = psutil.cpu_percent(percpu=True, interval=1)
    for i, pct in enumerate(percents):
        print(f"  Core {i}: {pct}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

    print("\nCPU Times (cumulative since boot):")
    cpu_times = psutil.cpu_times()
    for k, v in cpu_times._asdict().items():
        print(f"  {k}: {v:.2f} seconds")
    print()

# ===========================
# MAIN DISPLAY FUNCTION
# ===========================

def display_system_info():
    """Display basic system and CPU identification."""
    print("=== CPU Information ===\n")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"CPU Vendor: {get_cpu_vendor_windows()}")
    print(f"Exact CPU Model: {get_cpu_model_windows()}\n")

# ===========================
# MAIN ENTRY POINT
# ===========================

def main():
    display_system_info()

    if PSUTIL_AVAILABLE:
        display_core_info()
        display_frequency_info()
        display_usage_and_times()
        display_cache_info()
    else:
        print("⚠️  Install 'psutil' and 'py-cpuinfo' for full CPU diagnostics:")
        print("   pip install psutil py-cpuinfo\n")
        print("=== Basic Info ===")
        print(f"CPU cores (multiprocessing): {multiprocessing.cpu_count()}")
        print(f"CPU Vendor: {get_cpu_vendor_windows()}")
        print(f"CPU Model: {get_cpu_model_windows()}")

if __name__ == "__main__":
    main()