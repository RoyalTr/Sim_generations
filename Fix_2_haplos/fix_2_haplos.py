#!/usr/bin/env python3
"""
fix_2_haplos.py
AUTHOR: Dr. Royal Truman
VERSION: 0.9
"""
import numpy as np
import os
import sys
import time
import multiprocessing
import shutil  # for efficient file copying
import warnings

start_time = time.time()

warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)

MASTER_SEED = 42  # Hardcoded master seed for reproducible results across runs

# === Load YAML config file ===
# CONFIG_FILE = "config_v1.yaml"
CONFIG_FILE = "config_2_haplos_v1.yaml"

try:
    import yaml
except ImportError:
    yaml = None
    print("❌ Required package 'pyyaml' not found.")
    print("👉 Install it with: pip install pyyaml")
    sys.exit(1)

# Check if config file exists
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Configuration file '{CONFIG_FILE}' not found in your directory.")
    print(f" Copy it over, and modify the parameters if wished, then rerun.")
    yaml = None  # Avoid unnecessary warning from PyCharm
    sys.exit(1)

# Load config
try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"❌ Error reading or parsing '{CONFIG_FILE}': {e}")
    print("Please check that the file is valid YAML (correct indentation, colons, etc.).")
    sys.exit(1)

# Extract runtime parameters
try:
    Repetitions = int(config['Repetitions'])
    max_generations = int(config['max_generations'])
    document_results_every_generation = bool(config['document_results_every_generation'])

    # Validate required ranges
    if Repetitions < 1:
        raise ValueError("Repetitions must be >= 1")
    if max_generations < 1:
        raise ValueError("max_generations must be >= 1")
    if document_results_every_generation not in [True, False]:
        raise ValueError("document_results_every_generation must be true or false")

except KeyError as e:
    print(f"❌ Missing required setting in config: {e}")
    print(f"🔧 Please make sure '{CONFIG_FILE}' includes all required keys.")
    sys.exit(1)
except ValueError as e:
    print(f"❌ Invalid value in config: {e}")
    sys.exit(1)


# Prevent system sleep on Windows
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def column_headings():
    """
    Column headings for in_2_haplos.txt
    """
    return "Ni;r;K;s_A;attempts;h_A;p_A_i;s_B;h_B;p_B_i"

def results_headings():
    """
    Column headings for out_2_haplos.txt — updated column order
    """
    return ("Sim Nr;Rep;Ni;r;K;"
            "s A;s B;h A;h B;p A i;p B i;attempts;"
            "Prob A fix;Prob a fix;"
            "N A fix;N a fix;N hetero Aa lost;"
            "Gen A fix;Gen a fix;Gen hetero Aa lost;"
            "Prob B fix;Prob b fix;"
            "N B fix;N b fix;N hetero Bb lost;"
            "Gen B fix;Gen b fix;Gen hetero Bb lost;"
            "Gen pan-homoz;N pan-homoz;Gen pan-hetero lost;N pan-hetero lost")

def results_headings_avg():
    """
    Column headings for out_avg_2_haplos.txt — updated column order
    """
    return ("Sim Nr;Reps;Ni;r;K;"
            "s A;s B;h A;h B;p A i;p B i;attempts;"
            "Prob A fix;Prob a fix;"
            "N A fix;N a fix;N hetero Aa lost;"
            "Gen A fix;Gen a fix;Gen hetero Aa lost;"
            "Prob B fix;Prob b fix;"
            "N B fix;N b fix;N hetero Bb lost;"
            "Gen B fix;Gen b fix;Gen hetero Bb lost;"
            "Gen pan-homoz;N pan-homoz;Gen pan-hetero lost;N pan-hetero lost")

def per_generation_headings():
    return ("SimNr;Rep;attempt;Ni;r;K;s_A;h_A;p_A_i;s_B;h_B;p_B_i;attempts;generation;N;"
            "freq_A;freq_Aa;freq_a;freq_B;freq_Bb;freq_b;pan_heteroz;pan_homoz")

# Filenames
results_filename_avg = "out_avg_2_haplos.txt"

example_rows = [
    "100;0.01;1000;0.001;1000;0.5;0.05;0.002;0.7;0.1",
    "1000;0.05;20000;0.005;20000;0.2;0.1;0.001;0.3;0.2"
]

input_filename = "in_2_haplos.txt"
headings = column_headings()

# --- Input File Handling ---
if not os.path.exists(input_filename):
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter parameters in in_2_haplos.txt, then rerun.")
    sys.exit(0)

with open(input_filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

if not lines:
    with open(input_filename, "a") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("in_2_haplos.txt was empty. Example data added. Please edit and rerun.")
    sys.exit(0)

if lines[0] != headings:
    print(f"Error: Expected heading: '{headings}', got '{lines[0]}'")
    sys.exit(0)

if len(lines) == 1:
    with open(input_filename, "a") as f:
        for row in example_rows:
            f.write(row + "\n")
    print("Please add your parameters to in_2_haplos.txt and rerun.")
    sys.exit(0)

valid_data = []
error_found = False

for line_num, line in enumerate(lines[1:], start=2):
    parts = line.split(";")
    if len(parts) != 10:
        print(f"Error: Line {line_num} must have 10 parameters.")
        error_found = True
        continue

    try:
        Ni = int(parts[0])
        if not (1 <= Ni <= 1_000_000_000):
            print(f"Error: Ni in line {line_num} must be between 1 and 1,000,000,000.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: Ni in line {line_num} must be integer.")
        error_found = True
        continue

    try:
        r = float(parts[1])
        if r <= -1:
            print(f"Error: r in line {line_num} must be > -1.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: r in line {line_num} must be number.")
        error_found = True
        continue

    try:
        K = int(parts[2])
        if K < Ni:
            print(f"Error: K >= Ni required in line {line_num}.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: K in line {line_num} must be integer.")
        error_found = True
        continue

    try:
        s_A = float(parts[3])
        if not (-2 <= s_A <= 2):
            print(f"Error: s_A in line {line_num} must be in [-2, 2].")
            error_found = True
            continue
    except ValueError:
        print(f"Error: s_A in line {line_num} must be number.")
        error_found = True
        continue

    try:
        attempts = int(parts[4])
        if not (1 <= attempts <= 10_000_000_000):
            print(f"Error: attempts in line {line_num} must be 1–1e10.")
            error_found = True
            continue
    except ValueError:
        print(f"Error: attempts in line {line_num} must be integer.")
        error_found = True
        continue

    try:
        h_A = float(parts[5])
        if not (-1 <= h_A <= 1):
            print(f"Error: h_A in line {line_num} must be in [-1, 1].")
            error_found = True
            continue
    except ValueError:
        print(f"Error: h_A in line {line_num} must be number.")
        error_found = True
        continue

    try:
        p_A_i = float(parts[6])
        if not (0.0 <= p_A_i <= 1.0):
            print(f"Error: p_A_i in line {line_num} must be in [0,1].")
            error_found = True
            continue
    except ValueError:
        print(f"Error: p_A_i in line {line_num} must be number.")
        error_found = True
        continue

    try:
        s_B = float(parts[7])
        if not (-2 <= s_B <= 2):
            print(f"Error: s_B in line {line_num} must be in [-2, 2].")
            error_found = True
            continue
    except ValueError:
        print(f"Error: s_B in line {line_num} must be number.")
        error_found = True
        continue

    try:
        h_B = float(parts[8])
        if not (-1 <= h_B <= 1):
            print(f"Error: h_B in line {line_num} must be in [-1, 1].")
            error_found = True
            continue
    except ValueError:
        print(f"Error: h_B in line {line_num} must be number.")
        error_found = True
        continue

    try:
        p_B_i = float(parts[9])
        if not (0.0 <= p_B_i <= 1.0):
            print(f"Error: p_B_i in line {line_num} must be in [0,1].")
            error_found = True
            continue
    except ValueError:
        print(f"Error: p_B_i in line {line_num} must be number.")
        error_found = True
        continue

    valid_data.append((Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i))

if error_found:
    print("Please correct errors in in_2_haplos.txt and rerun.")
    sys.exit(0)


# === Simulation Function (modified to write per-gen data to file) ===
def simulate_population(Ni, r, K, s_A, p_A_i, s_B, p_B_i, total_generations, attempts, h_A, h_B, sim_nr, rep, rng, temp_filename=None):
    # We no longer accumulate per_gen_data in memory
    # If temp_filename is provided, we write directly to it

    # Aggregators
    A_count = a_count = B_count = b_count = 0
    sum_N_A = sum_N_a = sum_N_B = sum_N_b = 0.0
    sum_gen_A = sum_gen_a = sum_gen_B = sum_gen_b = 0.0
    sum_total_gens = sum_total_N = 0.0
    num_both_fixed = 0

    sum_gen_hetero_lost = 0.0
    sum_N_hetero_lost = 0.0
    count_hetero_lost = 0  # number of attempts where loss was detected

    # Fitness values
    w_AA = 1.0 + s_A
    w_Aa = 1.0 + h_A * s_A
    w_aa = 1.0

    w_BB = 1.0 + s_B
    w_Bb = 1.0 + h_B * s_B
    w_bb = 1.0

    # Beverton-Holt constants
    r1 = 1.0 + r
    rK = r / K if K > 0 else 0.0

    # If writing per-gen data, open file once for efficiency
    per_gen_file = None
    if temp_filename:
        per_gen_file = open(temp_filename, "w")
        # Write header for temp file? No — we'll write header once in main file.
        # We assume main process already wrote global header.

    for attempt in range(attempts):
        N = Ni
        p_A = p_A_i
        p_B = p_B_i

        # Fixation tracking
        A_fixed = a_fixed = False
        B_fixed = b_fixed = False
        gen_A = gen_a = gen_B = gen_b = np.nan
        N_A = N_a = N_B = N_b = np.nan

        pan_hetero_lost = False
        gen_hetero_lost = np.nan
        N_hetero_lost = np.nan

        for gen in range(total_generations):
            # --- Record per-generation data ---
            if temp_filename and per_gen_file:
                freq_A = p_A
                freq_Aa = 2.0 * p_A * (1.0 - p_A)
                freq_a = 1.0 - p_A
                freq_B = p_B
                freq_Bb = 2.0 * p_B * (1.0 - p_B)
                freq_b = 1.0 - p_B
                pan_heteroz = freq_Aa * freq_Bb
                pan_homoz = (freq_A**2 + freq_a**2) * (freq_B**2 + freq_b**2)
                entry = (
                    sim_nr, rep, attempt + 1, Ni, r, K,
                    s_A, h_A, p_A_i, s_B, h_B, p_B_i, attempts,
                    gen, N,
                    freq_A, freq_Aa, freq_a, freq_B, freq_Bb, freq_b,
                    pan_heteroz, pan_homoz
                )

                # Format the line
                formatted_values = []
                for i, val in enumerate(entry):
                    if i in [0, 1, 2, 3, 5, 12, 13, 14]:  # integers
                        s_val = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
                        formatted_values.append(s_val)
                    elif i == 4:  # r
                        formatted_values.append(f"{val:.4f}")
                    elif i in [6, 8, 9, 11]:  # s_A, p_A_i, s_B, p_B_i
                        formatted_values.append(f"{val:.10f}")
                    elif i in [7, 10]:  # h_A, h_B
                        formatted_values.append(f"{val:.3f}")
                    else:  # frequencies
                        formatted_values.append(f"{val:.8f}")
                line = ";".join(formatted_values) + "\n"
                per_gen_file.write(line)

            # --- Check fixation at start of generation ---
            if not (A_fixed or a_fixed):
                if p_A >= 1.0:
                    A_fixed = True
                    gen_A = gen
                    N_A = N
                elif p_A <= 0.0:
                    a_fixed = True
                    gen_a = gen
                    N_a = N

            if not (B_fixed or b_fixed):
                if p_B >= 1.0:
                    B_fixed = True
                    gen_B = gen
                    N_B = N
                elif p_B <= 0.0:
                    b_fixed = True
                    gen_b = gen
                    N_b = N

            if not pan_hetero_lost:
                if (A_fixed or a_fixed or B_fixed or b_fixed):
                    pan_hetero_lost = True
                    gen_hetero_lost = gen
                    N_hetero_lost = N
                    sum_gen_hetero_lost += gen_hetero_lost
                    sum_N_hetero_lost += N_hetero_lost
                    count_hetero_lost += 1

            # End if both genes have fixed (pan-homozygous state)
            if (A_fixed or a_fixed) and (B_fixed or b_fixed):
                sum_total_gens += gen
                sum_total_N += N
                num_both_fixed += 1  # Number of attempts where both loci reached fixation

                if A_fixed:
                    A_count += 1
                    sum_gen_A += gen_A
                    sum_N_A += N_A
                elif a_fixed:
                    a_count += 1
                    sum_gen_a += gen_a
                    sum_N_a += N_a

                if B_fixed:
                    B_count += 1
                    sum_gen_B += gen_B
                    sum_N_B += N_B
                elif b_fixed:
                    b_count += 1
                    sum_gen_b += gen_b
                    sum_N_b += N_b

                break

            # --- Calculate population growth/decrease each generation if r <> 0 ---
            if r != 0:
                # Use Beverton-Holt formulation for non-negative growth (stable discrete density dependence)
                # For r < 0 BH can't be used, use continuous exponential decay for negative growth 
                if r >= 0.0:
                    N_float = (N * r1) / (1.0 + rK * N)  # Beverton-Holt
                else:
                    # Use np.exp(r) which is safe for r near zero and gives pure decay if r < 0
                    N_float = N * np.exp(r)
                # Protect against possible tiny negative rounding artefacts
                if N_float <= 0.0:
                    N = 0
                else:
                    # Use stochastic rounding to round up or down to convert N to integer numbers
                    frac = N_float - int(N_float)
                    if rng.random() < frac:
                        N = int(N_float) + 1
                    else:
                        N = int(N_float)
            
            # If population extinct, break out immediately AND handle fixation for un-fixed alleles
            if N == 0:
                if not (A_fixed or a_fixed):
                    a_fixed = True
                    gen_a = gen
                    N_a = N
                if not (B_fixed or b_fixed):
                    b_fixed = True
                    gen_b = gen
                    N_b = N
                break

            # --- Update allele frequencies ---
            if not (A_fixed or a_fixed):
                mean_w_A = (p_A**2) * w_AA + (2 * p_A * (1-p_A)) * w_Aa + ((1-p_A)**2) * w_aa
                if mean_w_A > 0:
                    p_A = (p_A * (p_A * w_AA + (1-p_A) * w_Aa)) / mean_w_A
                n_A = rng.binomial(2 * N, p_A)
                p_A = n_A / (2 * N)

            if not (B_fixed or b_fixed):
                mean_w_B = (p_B**2) * w_BB + (2 * p_B * (1-p_B)) * w_Bb + ((1-p_B)**2) * w_bb
                if mean_w_B > 0:
                    p_B = (p_B * (p_B * w_BB + (1-p_B) * w_Bb)) / mean_w_B
                n_B = rng.binomial(2 * N, p_B)
                p_B = n_B / (2 * N)

    # Close per-gen file if open
    if per_gen_file:
        per_gen_file.close()

    # --- Compute averages ---
    def safe_div(x, n): return x / n if n > 0 else np.nan
    return (
        safe_div(sum_N_A, A_count),
        safe_div(sum_N_a, a_count),
        safe_div(sum_N_B, B_count),
        safe_div(sum_N_b, b_count),

        safe_div(sum_gen_A, A_count),
        safe_div(sum_gen_a, a_count),
        safe_div(sum_gen_B, B_count),
        safe_div(sum_gen_b, b_count),

        A_count / attempts,
        a_count / attempts,
        B_count / attempts,
        b_count / attempts,

        safe_div(sum_total_gens, num_both_fixed),
        safe_div(sum_total_N, num_both_fixed),

        safe_div(sum_gen_hetero_lost, count_hetero_lost),
        safe_div(sum_N_hetero_lost, count_hetero_lost),

        temp_filename  # Return temp filename instead of per_gen_data list
    )

# Worker & Main Execution
def worker(job):
    sim_nr, rep, Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i = job   

    # Generate unique seed for this specific (SimNr, Rep) combination. Workers will produce independent results
    # Guarantee rerunning the same in_2_haplos.txt file will always generate identical results
    unique_seed = MASTER_SEED + (sim_nr * 1000) + rep

    # Initialize a dedicated random number generator with the unique seed
    rng = np.random.RandomState(unique_seed)

    # Create temp filename for per-gen data if enabled
    temp_filename = None
    if document_results_every_generation:
        temp_filename = f"temp_Sim{sim_nr}_Rep{rep}.txt"

    # Pass the seeded RNG and temp_filename to the simulation function
    results = simulate_population(
        Ni=Ni, r=r, K=K, s_A=s_A, p_A_i=p_A_i,
        s_B=s_B, p_B_i=p_B_i, total_generations=max_generations,
        attempts=attempts, h_A=h_A, h_B=h_B, sim_nr=sim_nr, rep=rep, rng=rng,
        temp_filename=temp_filename
    )

    (avg_N_A, avg_N_a, avg_N_B, avg_N_b,
     avg_A_fix_gen, avg_a_fix_gen, avg_B_fix_gen, avg_b_fix_gen,
     A_fix_prob, a_fix_prob, B_fix_prob, b_fix_prob,
     avg_total_generations, avg_total_N,
     avg_gen_hetero_lost, avg_N_hetero_lost,
     temp_filename_used) = results

    return (
        sim_nr, rep, Ni, r, K,
        s_A, s_B, h_A, h_B, p_A_i, p_B_i, attempts,
        A_fix_prob, a_fix_prob,
        avg_N_A, avg_N_a,
        avg_A_fix_gen, avg_a_fix_gen,
        B_fix_prob, b_fix_prob,
        avg_N_B, avg_N_b,
        avg_B_fix_gen, avg_b_fix_gen,
        avg_gen_hetero_lost, avg_N_hetero_lost,
        avg_total_generations, avg_total_N,
        temp_filename_used  # Now returns temp filename (or None) instead of per_gen_data list
    )

if __name__ == '__main__':
    max_processes = multiprocessing.cpu_count()
    print(f"Hardware can support up to {max_processes} processes.")

    # Initialize output files — OVERWRITE them at start of run
    results_filename = "out_2_haplos.txt"
    per_gen_filename = "out_2_haplos_per_gen.txt"
    results_filename_avg = "out_avg_2_haplos.txt"

    # 🚫 OVERWRITE out_2_haplos.txt — write header fresh
    with open(results_filename, "w") as f:
        f.write(results_headings() + "\n")

    # 🚫 OVERWRITE out_avg_2_haplos.txt — write header fresh
    with open(results_filename_avg, "w") as f:
        f.write(results_headings_avg() + "\n")

    # Handle per-generation file — write header once
    if document_results_every_generation:
        with open(per_gen_filename, "w") as f:
            f.write(per_generation_headings() + "\n")

    # Process each SimNr sequentially
    for sim_nr, params in enumerate(valid_data, start=1):
        Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i = params
        print(f"Processing SimNr {sim_nr}/{len(valid_data)}...")

        # Create jobs for this SimNr (all repetitions)
        jobs = []
        for rep in range(1, Repetitions + 1):
            jobs.append((sim_nr, rep, Ni, r, K, s_A, attempts, h_A, p_A_i, s_B, h_B, p_B_i))

        # Process all repetitions for this SimNr in parallel
        with multiprocessing.Pool(processes=max_processes) as pool:
            sim_results = pool.map(worker, jobs)

        # Sort results by repetition for this SimNr (for consistent summary output)
        sim_results_sorted = sorted(sim_results, key=lambda x: x[1])

        # ✅ NEW: Merge per-generation temp files in Rep order
        if document_results_every_generation:
            print(f"  → Merging per-generation data for SimNr {sim_nr}...")
            with open(per_gen_filename, "a") as master_f:
                for res in sim_results_sorted:
                    temp_file = res[-1]  # last element is temp_filename
                    if temp_file and os.path.exists(temp_file):
                        with open(temp_file, "r") as tf:
                            shutil.copyfileobj(tf, master_f)
                        os.remove(temp_file)  # cleanup — comment out if you want to keep temp files

        # Write detailed results for this SimNr — APPEND during run
        with open(results_filename, "a") as f:
            for res in sim_results_sorted:
                (sim_nr, rep, Ni, r, K, s_A, s_B, h_A, h_B, p_A_i, p_B_i, attempts,
                 A_fix_prob, a_fix_prob, avg_N_A, avg_N_a, avg_A_gen, avg_a_gen,
                 B_fix_prob, b_fix_prob, avg_N_B, avg_N_b, avg_B_gen, avg_b_gen,
                 avg_gen_hetero_lost, avg_N_hetero_lost, avg_total_gens, avg_total_N,
                 temp_filename_used) = res

                # Compute derived values
                N_hetero_Aa_lost = (A_fix_prob * avg_N_A) + (a_fix_prob * avg_N_a)
                Gen_hetero_Aa_lost = (A_fix_prob * avg_A_gen) + (a_fix_prob * avg_a_gen)
                N_hetero_Bb_lost = (B_fix_prob * avg_N_B) + (b_fix_prob * avg_N_b)
                Gen_hetero_Bb_lost = (B_fix_prob * avg_B_gen) + (b_fix_prob * avg_b_gen)

                line_parts = [
                    str(sim_nr), str(rep), str(Ni), str(r), str(K),
                    str(s_A), str(s_B),
                    f"{h_A:.8f}", f"{h_B:.8f}",
                    f"{p_A_i:.8f}", f"{p_B_i:.8f}", str(attempts),
                    f"{A_fix_prob:.8f}", f"{a_fix_prob:.8f}",
                    f"{avg_N_A:.8f}", f"{avg_N_a:.8f}", f"{N_hetero_Aa_lost:.8f}",
                    f"{avg_A_gen:.8f}", f"{avg_a_gen:.8f}", f"{Gen_hetero_Aa_lost:.8f}",
                    f"{B_fix_prob:.8f}", f"{b_fix_prob:.8f}",
                    f"{avg_N_B:.8f}", f"{avg_N_b:.8f}", f"{N_hetero_Bb_lost:.8f}",
                    f"{avg_B_gen:.8f}", f"{avg_b_gen:.8f}", f"{Gen_hetero_Bb_lost:.8f}",
                    f"{avg_total_gens:.8f}", f"{avg_total_N:.8f}",
                    f"{avg_gen_hetero_lost:.8f}", f"{avg_N_hetero_lost:.8f}"
                ]
                f.write(";".join(line_parts) + "\n")

        # Compute and write averaged results for this SimNr — APPEND during run
        A_fix = np.nanmean([r[12] for r in sim_results_sorted])
        a_fix = np.nanmean([r[13] for r in sim_results_sorted])
        B_fix = np.nanmean([r[18] for r in sim_results_sorted])
        b_fix = np.nanmean([r[19] for r in sim_results_sorted])

        A_gen = np.nanmean([r[16] for r in sim_results_sorted])
        a_gen = np.nanmean([r[17] for r in sim_results_sorted])
        B_gen = np.nanmean([r[22] for r in sim_results_sorted])
        b_gen = np.nanmean([r[23] for r in sim_results_sorted])

        avg_N_A = np.nanmean([r[14] for r in sim_results_sorted])
        avg_N_a = np.nanmean([r[15] for r in sim_results_sorted])
        avg_N_B = np.nanmean([r[20] for r in sim_results_sorted])
        avg_N_b = np.nanmean([r[21] for r in sim_results_sorted])

        avg_total_gens = np.nanmean([r[26] for r in sim_results_sorted])
        avg_total_N = np.nanmean([r[27] for r in sim_results_sorted])

        avg_gen_hetero_lost = np.nanmean([r[24] for r in sim_results_sorted])
        avg_N_hetero_lost = np.nanmean([r[25] for r in sim_results_sorted])

        # Perform calculations avoid storing 'nan' that are not justified.
        # For locus A
        if A_fix == 1.0:
            N_hetero_Aa_lost = avg_N_A
            Gen_hetero_Aa_lost = A_gen
        elif a_fix == 1.0:
            N_hetero_Aa_lost = avg_N_a
            Gen_hetero_Aa_lost = a_gen
        else:
            N_hetero_Aa_lost = (A_fix * avg_N_A) + (a_fix * avg_N_a)
            Gen_hetero_Aa_lost = (A_fix * A_gen) + (a_fix * a_gen)

        # For locus B
        if B_fix == 1.0:
            N_hetero_Bb_lost = avg_N_B
            Gen_hetero_Bb_lost = B_gen
        elif b_fix == 1.0:
            N_hetero_Bb_lost = avg_N_b
            Gen_hetero_Bb_lost = b_gen
        else:
            N_hetero_Bb_lost = (B_fix * avg_N_B) + (b_fix * avg_N_b)
            Gen_hetero_Bb_lost = (B_fix * B_gen) + (b_fix * b_gen)

        line_parts = [
            str(sim_nr), str(Repetitions), str(Ni), str(r), str(K),
            str(s_A), str(s_B),
            f"{h_A:.8f}", f"{h_B:.8f}",
            f"{p_A_i:.8f}", f"{p_B_i:.8f}", str(attempts),
            f"{A_fix:.8f}", f"{a_fix:.8f}",
            f"{avg_N_A:.8f}", f"{avg_N_a:.8f}", f"{N_hetero_Aa_lost:.8f}",
            f"{A_gen:.8f}", f"{a_gen:.8f}", f"{Gen_hetero_Aa_lost:.8f}",
            f"{B_fix:.8f}", f"{b_fix:.8f}",
            f"{avg_N_B:.8f}", f"{avg_N_b:.8f}", f"{N_hetero_Bb_lost:.8f}",
            f"{B_gen:.8f}", f"{b_gen:.8f}", f"{Gen_hetero_Bb_lost:.8f}",
            f"{avg_total_gens:.8f}", f"{avg_total_N:.8f}",
            f"{avg_gen_hetero_lost:.8f}", f"{avg_N_hetero_lost:.8f}"
        ]
        with open(results_filename_avg, "a") as f:
            f.write(";".join(line_parts) + "\n")

    print(f"💾 Detailed results saved to {results_filename}")
    if document_results_every_generation:
        print(f"💾 Per-generation results stored in file: {per_gen_filename}")
    print(f"💾 Averaged results saved to {results_filename_avg}")

    current_dir = os.getcwd()
    print(f"📁 Output files stored in: {current_dir}")

    end_time = time.time()
    print(f"⏱️ Execution time: {end_time - start_time:.2f} seconds")