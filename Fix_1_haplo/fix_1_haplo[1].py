#!/usr/bin/env python3
"""
fix_1_gene.py
AUTHOR: Dr. Royal Truman
VERSION: 0.9
"""
import numpy as np
import os
import sys
import time
import multiprocessing
import tempfile
import warnings

start_time = time.time()

warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)

MASTER_SEED = 42  # Hardcoded master seed to generate reproducible results if Input_data.txt is rerun

# Get the folder where this program is located
curr_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(curr_dir, 'config_1_haplo_v1.yaml')
input_filename = os.path.join(curr_dir, 'in_1_haplo.txt')
results_filename = os.path.join(curr_dir, 'out_1_haplo.txt')
results_filename_per_generation = os.path.join(curr_dir, 'out_1_haplo_per_gen.txt')
results_filename_avg = "out_avg_1_haplo.txt"

# input_filename = "in_1_haplo.txt"
# results_filename = "out_1_haplo.txt"
# CONFIG_FILE = "config_1_haplo_v1.yaml"
# results_filename_per_generation = "out_1_haplo_per_gen.txt"

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

# === Prevent system sleep on Windows (when running overnight) ===
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

# === Input data handling ===
def column_headings():
    return "Ni;r;K;s_A;attempts;h_A;p_A_i"

def results_headings():
    return ("SimNr;Rep;Ni;r;K;s_A;h_A;p_A_i;"
            "attempts;Prob A fix;Prob a fix;"
            "N A fix;N a fix;N homozygous and hetero lost;"
            "Gen A fix;Gen a fix;Gen homozygous and hetero lost")

output_headings_avg = ("SimNr;Reps;Ni;r;K;s_A;h_A;p_A_i;"
                       "attempts;Prob A fix;Prob a fix;"
                       "N A fix;N a fix;N homozygous and hetero lost;"
                       "Gen A fix;Gen a fix;Gen homozygous and hetero lost")

headings = column_headings()

example_rows = [
    "100;0.01;1000;0.001;1000;0.5;0.05",
    "1000;0.05;20000;0.005;20000;0.2;0.1"
]

if not os.path.exists(input_filename):
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters to run in file in_1_haplo.txt (see example data), then rerun the program.")
    sys.exit(0)
    
with open(input_filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

if not lines:
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters you want in file in_1_haplo.txt (see example data), then rerun the program.")
    sys.exit(0)

if lines[0] != headings:
    print(f"Error: The heading in in_1_haplo.txt is incorrect. Expected: '{headings}' but got '{lines[0]}'")
    print("Please correct the heading in in_1_haplo.txt to match the expected format.")
    sys.exit(0)

if len(lines) == 1:
    with open(input_filename, "a") as f:
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters you want in file in_1_haplo.txt (see example data), then rerun the program.")
    sys.exit(0)

valid_data = []
error_found = False
for line_num, line in enumerate(lines[1:], start=2):
    parts = line.split(";")
    if len(parts) != 7:
        print(f"Seven parameters must be found in file in_1_haplo.txt in line {line_num}. Please correct and rerun.")
        error_found = True
        continue
    try:
        Ni = int(parts[0])
        if not (1 <= Ni <= 1000000000):
            print(f"The value of Ni (initial population size) in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#1 The data in line {line_num} is wrong (Ni, initial population size). Please correct")
        error_found = True
        continue
    try:
        r = float(parts[1])
        if r <= -1.0:
            print(f"The value of r (growth rate / generation) in line {line_num} must be > -1.0. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"#2 The data in line {line_num} is wrong (r). Please correct")
        error_found = True
        continue
    try:
        K = int(parts[2])
        if not (K >= Ni):
            print(f"The value of K (carrying capacity) in line {line_num} must be greater or equal to initial population size. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"#3 The data in line {line_num} is wrong (K). Please correct")
        error_found = True
        continue
    try:
        s_A = float(parts[3])
        if not (-2 <= s_A <= 2):
            print(f"The value of s_A in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#4 The data in line {line_num} is wrong (s_A). Please correct")
        error_found = True
        continue
    try:
        attempts = int(parts[4])
        if not (1 <= attempts <= 1000000000000):
            print(f"The value of attempts in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#5 The data in line {line_num} is wrong (attempts). Please correct")
        error_found = True
        continue
    try:
        h_A = float(parts[5])
        if not (-1 <= h_A <= 1):
            print(f"#6 The value of h_A in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#7 The data in line {line_num} is wrong (h_A). Please correct")
        error_found = True
        continue
    try:
        p_A_i = float(parts[6])
        if not (0.0 <= p_A_i <= 1.0):
            print(f"#8 The value of p_A_i in line {line_num} must be between 0 and 1. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"#9 The data in line {line_num} is wrong (p_A_i). Please correct")
        error_found = True
        continue

    valid_data.append((Ni, r, K, s_A, attempts, h_A, p_A_i))

if error_found:
    print("Please correct the data and rerun the program")
    sys.exit(0)


# === Simulation Function (corrected to ensure consistent generation recording) ===
def simulate_population(Ni, r, K, s_A, p_A_i, max_generations, attempts, h_A, rng, per_gen_filename=None):
    """
    Simulate population genetics model.
    If per_gen_filename is provided, writes per-generation data directly to that file (one line per generation).
    Returns: summary stats + per_gen_filename (not the data itself)
    
    CRITICAL FIX: Ensures that the generation recorded for fixation events is consistent
    between per-generation file and summary statistics.
    """
    a_count = 0
    A_count = 0
    sum_A_fix_gens = 0.0
    sum_a_fix_gens = 0.0
    sum_N_A_final = 0.0
    sum_N_a_final = 0.0

    r1 = 1.0 + r
    rK = r / K

    # Open file handle if filename provided
    per_gen_file_handle = None
    if per_gen_filename:
        try:
            per_gen_file_handle = open(per_gen_filename, 'w')
        except Exception as e:
            print(f"⚠️  Warning: Could not open temp file {per_gen_filename} for writing: {e}")
            per_gen_file_handle = None

    fitness_AA = 1.0 + s_A
    fitness_Aa = 1.0 + h_A * s_A
    fitness_aa = 1.0

    for attempt_idx in range(1, attempts + 1):
        N = Ni
        p_A_t = p_A_i

        for gen in range(max_generations):
            # Calculate derived frequencies for current state
            freq_a = 1.0 - p_A_t
            freq_Aa = 2.0 * p_A_t * freq_a
            homoz = p_A_t ** 2 + freq_a ** 2
            
            # ALWAYS write per-generation data for current state first
            if per_gen_file_handle is not None:
                per_gen_file_handle.write(f"{attempt_idx};{gen};{N};{p_A_t:.8f};{freq_a:.8f};{freq_Aa:.8f};{homoz:.8f}\n")

            # Check for fixation AFTER writing the data
            if p_A_t == 0.0:
                a_count += 1
                fixation_gen = gen  # Record the same generation that was just written to per-gen file
                sum_a_fix_gens += fixation_gen
                sum_N_a_final += N
                break
            elif p_A_t == 1.0:
                A_count += 1
                fixation_gen = gen  # Record the same generation that was just written to per-gen file
                sum_A_fix_gens += fixation_gen
                sum_N_A_final += N
                break

            # Continue with population genetics calculations only if no fixation
            freq_AA = p_A_t * p_A_t
            freq_Aa = 2.0 * p_A_t * (1.0 - p_A_t)
            freq_aa = (1.0 - p_A_t) * (1.0 - p_A_t)

            mean_fitness = freq_AA * fitness_AA + freq_Aa * fitness_Aa + freq_aa * fitness_aa

            numerator_A = 2.0 * freq_AA * fitness_AA + freq_Aa * fitness_Aa
            fit_A = numerator_A / (2.0 * mean_fitness)

            # Use Beverton-Holt formulation for non-negative growth (stable discrete density dependence)
            # Use continuous exponential decay for negative growth (r < 0)
            if r != 0:
                if r >= 0.0:
                    N_float = N * r1 / (1.0 + rK * N)  # Beverton-Holt
                else:
                    # Exponential decay for negative r (continuous-time step)
                    # Use np.exp(r) which is safe for r near zero and gives pure decay if r < 0
                    N_float = N * np.exp(r)

                # Protect against possible tiny negative rounding artefacts
                if N_float <= 0.0:
                    N = 0
                else:
                    # Use stochastic rounding
                    frac = N_float - int(N_float)
                    if rng.random() < frac:
                        N = int(N_float) + 1
                    else:
                        N = int(N_float)

            # If population extinct, break out immediately
            if N <= 0:
                break           
            
            n_A_alleles = rng.binomial(2 * N, float(fit_A))
            p_A_t = n_A_alleles / (2 * N)

            # <<< FIXED: Defensive check against NaN from floating-point instability
            if np.isnan(p_A_t):
                break

    # Close file if opened
    if per_gen_file_handle:
        per_gen_file_handle.close()

    avg_N_A = sum_N_A_final / A_count if A_count > 0 else np.nan
    avg_N_a = sum_N_a_final / a_count if a_count > 0 else np.nan
    a_fix_prob = a_count / attempts
    A_fix_prob = A_count / attempts
    avg_A_fix_gen = sum_A_fix_gens / A_count if A_count > 0 else np.nan
    avg_a_fix_gen = sum_a_fix_gens / a_count if a_count > 0 else np.nan

    # Return filename instead of data list
    return avg_N_A, avg_N_a, A_fix_prob, avg_A_fix_gen, a_fix_prob, avg_a_fix_gen, per_gen_filename

def worker(job):
    """
    Worker function for multiprocessing.
    Instead of returning large per-generation data lists, writes them to temp file.
    Returns only summary stats + temp filename.
    """
    sim_nr, rep, Ni, r, K, s_A, attempts, h_A, p_A_i = job
    
    # Generate unique seed for this specific (SimNr, Rep) combination. Workers will produce independent results
    # Guarantee rerunning the same in_1_haplo.txt file will always generate identical results
    unique_seed = MASTER_SEED + (sim_nr * 1000) + rep
    
    # Initialize a dedicated random number generator with the unique seed
    rng = np.random.RandomState(unique_seed)
    
    # Generate temp filename for per-generation data if needed
    per_gen_filename = None
    if document_results_every_generation:
        try:
            temp_dir = tempfile.gettempdir()
            # Use unique seed in filename to avoid collisions
            per_gen_filename = os.path.join(temp_dir, f"per_gen_sim{sim_nr}_rep{rep}_seed{unique_seed}.tmp")
        except Exception as e:
            print(f"⚠️  Warning: Could not generate temp filename for Sim {sim_nr} Rep {rep}: {e}")

    # Pass the seeded RNG and temp filename to simulation
    result = simulate_population(Ni, r, K, s_A, p_A_i, max_generations, attempts, h_A, rng, per_gen_filename)
    return sim_nr, rep, Ni, r, K, s_A, h_A, p_A_i, attempts, *result

# === Main Execution ===
if __name__ == '__main__':
    max_processes = multiprocessing.cpu_count()
    print(f"Hardware can support up to {max_processes} processes.")

    # Initialize per-generation results file if needed
    if document_results_every_generation:
        if os.path.exists(results_filename_per_generation):
            os.remove(results_filename_per_generation)
        with open(results_filename_per_generation, "w") as f:
            f.write("SimNr;Rep;attempt;Ni;r;K;s_A;h_A;p_A_i;attempts;generation;N;freq_A;freq_Aa;freq_a;homoz\n")

    # Initialize summary results files
    output_headings = results_headings()
    
    if os.path.exists(results_filename):
        os.remove(results_filename)
    with open(results_filename, "w") as f:
        f.write(output_headings + "\n")

    if Repetitions > 1:
        if os.path.exists(results_filename_avg):
            os.remove(results_filename_avg)
        with open(results_filename_avg, "w") as f:
            f.write(output_headings_avg + "\n")

    # Process each SimNr sequentially
    for sim_nr, (Ni, r, K, s_A, attempts, h_A, p_A_i) in enumerate(valid_data, start=1):
        print(f"Processing SimNr {sim_nr}/{len(valid_data)}...")
        
        # Create jobs for all repetitions of this SimNr
        jobs = []
        for rep_idx in range(1, Repetitions + 1):
            jobs.append((sim_nr, rep_idx, Ni, r, K, s_A, attempts, h_A, p_A_i))
        
        # Run all repetitions in parallel, but maintain order
        with multiprocessing.Pool(processes=max_processes) as pool:
            # pool.map maintains order - Rep 1 result will be at index 0, Rep 2 at index 1, etc.
            rep_results = pool.map(worker, jobs)
        
        # Now rep_results contains results in correct Rep order
        # Write results for this SimNr in correct Rep order
        
        # ✅ CRITICAL: Write per-generation data in strict Rep order (Rep 1, then Rep 2, ...)
        # ✅ Each temp file is fully appended before moving to next — no interleaving
        # ✅ All reps for current SimNr merged before next SimNr starts
        if document_results_every_generation:
            with open(results_filename_per_generation, "a") as f_gen:
                for result in rep_results:  # Already in Rep order: index 0 = Rep 1, index 1 = Rep 2, etc.
                    (sim_nr_res, rep_res, Ni_res, r_res, K_res, s_A_res, h_A_res, p_A_i_res, attempts_res,
                     avg_N_A, avg_N_a, A_fix_prob, avg_A_fix_gen, a_fix_prob, avg_a_fix_gen, per_gen_filename) = result
                    
                    # If temp file exists, read and append all its content
                    if per_gen_filename and os.path.exists(per_gen_filename):
                        try:
                            with open(per_gen_filename, 'r') as tmp_f:
                                for line in tmp_f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    parts = line.split(';')
                                    if len(parts) != 7:
                                        continue
                                    attempt_idx, gen, N, freq_A, freq_a, freq_Aa, homoz = parts
                                    # Write with full context: SimNr;Rep;... preserving global sort order
                                    f_gen.write(f"{sim_nr_res};{rep_res};{attempt_idx};{Ni_res};{r_res:.4f};{K_res};{s_A_res:.10f};{h_A_res:.3f};{p_A_i_res:.10f};{attempts_res};{gen};{N};{freq_A};{freq_Aa};{freq_a};{homoz}\n")
                            # ✅ Safe to remove temp file after successful read
                            os.remove(per_gen_filename)
                        except Exception as e:
                            print(f"⚠️  Warning: Error reading or deleting temp file {per_gen_filename}: {e}")
                            # Do not crash — continue with next repetition
        
        # Write individual repetition results for this SimNr in Rep order
        with open(results_filename, "a") as f:
            for result in rep_results:  # Already in Rep order
                (sim_nr_res, rep_res, Ni_val, r_val, K_val, s_A_val, h_A_val, p_A_i_val, attempts_val,
                 avg_N_A, avg_N_a, A_fix_prob, avg_A_fix_gen, a_fix_prob, avg_a_fix_gen, _) = result

                N_homoz_hetero_lost = A_fix_prob * avg_N_A + a_fix_prob * avg_N_a
                Gen_homoz_hetero_lost = A_fix_prob * avg_A_fix_gen + a_fix_prob * avg_a_fix_gen

                Ni_int = f"{int(Ni_val)}" if not np.isnan(Ni_val) else "nan"
                K_int = f"{int(K_val)}" if not np.isnan(K_val) else "nan"

                line = (f"{sim_nr_res};{rep_res};{Ni_int};{r_val:.8f};{K_int};"
                        f"{s_A_val:.8f};{h_A_val:.8f};{p_A_i_val:.8f};"
                        f"{attempts_val};"
                        f"{A_fix_prob:.7f};{a_fix_prob:.7f};"
                        f"{avg_N_A:.3f};{avg_N_a:.3f};{N_homoz_hetero_lost:.3f};"
                        f"{avg_A_fix_gen:.3f};{avg_a_fix_gen:.3f};{Gen_homoz_hetero_lost:.3f}")
                f.write(line + "\n")
        
        # Calculate and write averaged results for this SimNr if multiple repetitions
        if Repetitions > 1:
            # Need to properly aggregate raw data across all reps to match Excel calculation
            # Collect raw counts and sums instead of averages to avoid nested averaging bias
            
            total_A_count = 0
            total_a_count = 0
            total_A_fix_gens_sum = 0.0
            total_a_fix_gens_sum = 0.0
            total_N_A_final_sum = 0.0
            total_N_a_final_sum = 0.0
            total_attempts = 0
            
            # Extract the raw aggregated values that were calculated in simulate_population
            for result in rep_results:
                # result structure: sim_nr, rep, Ni, r, K, s_A, h_A, p_A_i, attempts, 
                #                   avg_N_A, avg_N_a, A_fix_prob, avg_A_fix_gen, a_fix_prob, avg_a_fix_gen, per_gen_filename
                rep_attempts = result[8]  # attempts
                rep_A_fix_prob = result[11]  # A_fix_prob  
                rep_a_fix_prob = result[13]  # a_fix_prob
                rep_avg_A_fix_gen = result[12]  # avg_A_fix_gen
                rep_avg_a_fix_gen = result[14]  # avg_a_fix_gen  
                rep_avg_N_A = result[9]  # avg_N_A
                rep_avg_N_a = result[10]  # avg_N_a
                
                # Convert back to raw counts and sums
                rep_A_count = int(rep_A_fix_prob * rep_attempts)
                rep_a_count = int(rep_a_fix_prob * rep_attempts)
                
                total_attempts += rep_attempts
                total_A_count += rep_A_count
                total_a_count += rep_a_count
                
                # Reconstruct sums from averages (only if there were fixation events)
                if rep_A_count > 0 and not np.isnan(rep_avg_A_fix_gen):
                    total_A_fix_gens_sum += rep_avg_A_fix_gen * rep_A_count
                if rep_a_count > 0 and not np.isnan(rep_avg_a_fix_gen):
                    total_a_fix_gens_sum += rep_avg_a_fix_gen * rep_a_count
                if rep_A_count > 0 and not np.isnan(rep_avg_N_A):
                    total_N_A_final_sum += rep_avg_N_A * rep_A_count
                if rep_a_count > 0 and not np.isnan(rep_avg_N_a):
                    total_N_a_final_sum += rep_avg_N_a * rep_a_count
            
            # Calculate true overall averages (matching Excel calculation approach)
            avg_A_fix_prob = total_A_count / total_attempts
            avg_a_fix_prob = total_a_count / total_attempts
            avg_A_fix_gens_val = total_A_fix_gens_sum / total_A_count if total_A_count > 0 else np.nan
            avg_a_fix_gens_val = total_a_fix_gens_sum / total_a_count if total_a_count > 0 else np.nan
            avg_N_A = total_N_A_final_sum / total_A_count if total_A_count > 0 else np.nan
            avg_N_a = total_N_a_final_sum / total_a_count if total_a_count > 0 else np.nan

            # Avoid generating unjustified values of 'nan' when one allele has probability = 1.0
            if avg_A_fix_prob == 1.0:
                N_homoz_hetero_lost = avg_N_A
                Gen_homoz_hetero_lost = avg_A_fix_gens_val
            elif avg_a_fix_prob == 1.0:
                N_homoz_hetero_lost = avg_N_a
                Gen_homoz_hetero_lost = avg_a_fix_gens_val
            else:
                N_homoz_hetero_lost = avg_A_fix_prob * avg_N_A + avg_a_fix_prob * avg_N_a
                Gen_homoz_hetero_lost = avg_A_fix_prob * avg_A_fix_gens_val + avg_a_fix_prob * avg_a_fix_gens_val

            Ni_int = f"{int(Ni)}" if not np.isnan(Ni) else "nan"
            K_int = f"{int(K)}" if not np.isnan(K) else "nan"

            avg_line = (f"{sim_nr};{Repetitions};{Ni_int};{r:.8f};{K_int};"
                        f"{s_A:.8f};{h_A:.8f};{p_A_i:.8f};"
                        f"{attempts};"
                        f"{avg_A_fix_prob:.7f};{avg_a_fix_prob:.7f};"
                        f"{avg_N_A:.3f};{avg_N_a:.3f};{N_homoz_hetero_lost:.3f};"
                        f"{avg_A_fix_gens_val:.3f};{avg_a_fix_gens_val:.3f};{Gen_homoz_hetero_lost:.3f}")
            
            with open(results_filename_avg, "a") as f:
                f.write(avg_line + "\n")

    print(f"💾 Results stored in {results_filename}.")
    
    if document_results_every_generation:
        print(f"💾 Detailed per-generation results stored in {results_filename_per_generation}.")
    
    if Repetitions > 1:
        print(f"💾 Average values across repetitions stored in {results_filename_avg}")

    current_dir = os.getcwd()
    print(f"📁 Output files stored in: {current_dir}")

    # end_time = time.time()
    execution_time = time.time() - start_time
    print(f"⏱️ Total runtime: {execution_time:.2f} seconds")