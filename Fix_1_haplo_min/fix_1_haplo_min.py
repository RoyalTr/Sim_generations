"""
AUTHOR: Dr. Royal Truman
VERSION: 0.9 
"""
import numpy as np
import os
import sys
import time
import multiprocessing
import yaml

# Load configuration
with open("config_1_haplo_min_v1.yaml", "r") as f:
    config = yaml.safe_load(f)

Repetitions = config["Repetitions"]  # How many times to repeat every attempt for every simulation
max_generations = config["max_generations"]  # Terminate early or prevent excessive resource usage

# --- CONSTANTS ---
MASTER_SEED = 42
input_filename = "in_1_haplo_min.txt"
results_filename = "out_1_haplo_min.txt"
results_filename_avg = "out_avg_1_haplo_min.txt"
# -----------------------------------------------------------

# Prevent system sleep on Windows (e.g., when running overnight)
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

start_time = time.time()

def column_headings():
    return "Pop size;Select coeff;p0;attempts"

def results_headings():
    return "SimNr;Rep;Pop size;Select coeff;p0;attempts;Prob allele loss;Prob allele fix;St. dev. allele fix;Aver gen to fix;St. dev. gen to fix"

output_headings_avg = "SimNr;Reps;Pop size;Select coeff;p0;attempts;Prob allele loss;Prob of allele fix;St. dev. allele fix;Aver gen to fix;St. dev. gen to fix"

example_rows = [
    "1000;0.001;0.1;5000",
    "10000;0.005;0.2;20000"
]

headings = column_headings()

if not os.path.exists(input_filename):
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters to run in file in_1_haplo_min.txt (see example data), then rerun the program.")
    sys.exit(0)
    
with open(input_filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

if not lines:
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters to run in file in_1_haplo_min.txt (see example data), then rerun the program.")
    sys.exit(0)

if lines[0] != headings:
    os.remove(input_filename)
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters you want in file in_1_haplo_min.txt (see example data), then rerun the program.")
    sys.exit(0)

if len(lines) == 1:
    with open(input_filename, "a") as f:
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters you want in file in_1_haplo_min.txt (see example data), then rerun the program.")
    sys.exit(0)

valid_data = []
error_found = False
for line_num, line in enumerate(lines[1:], start=2):
    parts = line.split(";")
    if len(parts) != 4:  # Changed from 3 to 4 parameters
        print(f"The data in file in_1_haplo_min.txt in line {line_num} is wrong. Please correct and rerun.")
        error_found = True
        continue
    try:
        N_val = int(parts[0])
        if not (1 <= N_val <= 1000000000000):
            print(f"The data in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except:
        print(f"The data in line {line_num} is wrong. Please correct")
        error_found = True
        continue
    try:
        s_val = float(parts[1])
        if not (-2 <= s_val <= 2):
            print(f"The data in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except:
        print(f"The data in line {line_num} is wrong. Please correct")
        error_found = True
        continue
    try:
        p0_val = float(parts[2])
        if not (0.0 <= p0_val <= 1.0):
            print(f"The data in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except:
        print(f"The data in line {line_num} is wrong. Please correct")
        error_found = True
        continue
    try:
        attempts_val = int(parts[3])  # Changed from parts[2] to parts[3]
        if not (1 <= attempts_val <= 1000000000000):
            print(f"The data in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except:
        print(f"The data in line {line_num} is wrong. Please correct")
        error_found = True
        continue

    valid_data.append((N_val, s_val, p0_val, attempts_val))  # Now includes p0_val

if error_found:
    print("Please correct the data and rerun the program")
    sys.exit(0)

def simulate_population(N, s, p0, max_generations, attempts):
    losses = 0
    fixation = 0
    fixation_generations = []  # Store actual generation numbers when fixation occurred
    p_fix  = 1 - (1 / (2 * N))  # Threshold for declaring fixation

    for i in range(attempts):
        p_t = p0
        for gen in range(max_generations):
            if p_t == 0.0:
                losses += 1
                break
            elif p_t > p_fix:
                fixation += 1
                fixation_generations.append(gen)  # Record generation of fixation
                break

            # Genotype frequencies (Hardy–Weinberg)
            f_AA = p_t * p_t
            f_Aa = 2.0 * p_t * (1.0 - p_t)
            f_aa = (1.0 - p_t) * (1.0 - p_t)

            # Fitness values
            W_AA = 1.0 + s
            W_Aa = 1.0 + 0.5 * s
            W_aa = 1.0

            # Mean fitness
            meanW = f_AA * W_AA + f_Aa * W_Aa + f_aa * W_aa

            # Allele frequency after selection
            p_sel = (f_AA * W_AA + 0.5 * f_Aa * W_Aa) / meanW

            # Genetic drift: binomial sampling
            C = np.random.binomial(2 * N, p_sel)
            p_t = C / (2 * N)
    
    return losses, fixation, fixation_generations

def worker(job):
    idx, rep, N, s, p0, max_generations, attempts = job
    # Guarantee rerunning the same in_1_haplo_min.txt file will always generate identical results
    np.random.seed(MASTER_SEED + idx * 1000 + rep)
    losses, fixation, fixation_generations = simulate_population(N, s, p0, max_generations, attempts)
    return idx, rep, N, s, p0, attempts, losses, fixation, fixation_generations  # Now includes p0 in return

if __name__ == '__main__':
    max_processes = multiprocessing.cpu_count()
    print(f"Maximum number of processes supported: {max_processes}")
    
    jobs = []
    for idx, (N, s, p0, attempts) in enumerate(valid_data, start=1):  # Now unpacks p0 from valid_data
        for rep in range(1, Repetitions + 1):
            jobs.append((idx, rep, N, s, p0, max_generations, attempts))  # p0 now comes from input data
            
    with multiprocessing.Pool(processes=max_processes) as pool:
        results = pool.map(worker, jobs)
    
    individual_results_sorted = sorted(results, key=lambda x: (x[0], x[1]))
    
    # Pool all raw outcomes across repetitions, i.e., NOT averages per-repetition
    from collections import defaultdict
    pooled_data = defaultdict(lambda: {
        'total_losses': 0,
        'total_fixations': 0,
        'all_fixation_generations': [],
        'N': None,
        's': None,
        'p0': None,
        'attempts_per_rep': 0,
        'repetitions': 0
    })

    # Aggregate raw data per parameter set (idx)
    for res in individual_results_sorted:
        idx, rep, N, s, p0, attempts, losses, fixation, fixation_gens = res  # Now includes p0
        pooled_data[idx]['total_losses'] += losses
        pooled_data[idx]['total_fixations'] += fixation
        pooled_data[idx]['all_fixation_generations'].extend(fixation_gens)
        pooled_data[idx]['N'] = N
        pooled_data[idx]['s'] = s
        pooled_data[idx]['p0'] = p0  # Store p0
        pooled_data[idx]['attempts_per_rep'] = attempts
        pooled_data[idx]['repetitions'] += 1

    # Compute pooled statistics for each parameter set
    results_by_param = []
    for idx, data in pooled_data.items():
        total_trials = data['repetitions'] * data['attempts_per_rep']
        total_losses = data['total_losses']
        total_fixations = data['total_fixations']
        all_gens = data['all_fixation_generations']
        
        # Pooled probability of loss and fixation
        p_loss = total_losses / total_trials
        p_fix = total_fixations / total_trials
        
        # Standard error of fixation probability (based on total binomial trials)
        std_err_fix = np.sqrt(p_fix * (1.0 - p_fix) / total_trials) if total_trials > 0 else 0.0
        
        # Mean and sample std dev of generations to fixation (only if any fixations occurred)
        if len(all_gens) > 0:
            avg_gen_fix = np.mean(all_gens)
            # Use sample standard deviation (ddof=1) for unbiased estimate
            std_gen_fix = np.std(all_gens, ddof=1) if len(all_gens) > 1 else 0.0
        else:
            avg_gen_fix = np.nan
            std_gen_fix = np.nan

        results_by_param.append((
            idx,
            data['N'],
            data['s'],
            data['p0'],  # Now includes p0
            data['attempts_per_rep'],
            p_loss,
            p_fix,
            std_err_fix,
            avg_gen_fix,
            std_gen_fix
        ))

    # Write individual repetition results to out_1_haplo_min.txt
    output_headings = results_headings()
    lines_to_write = []
    for rec in individual_results_sorted:
        idx, rep, N, s, p0, attempts, losses, fixation, fixation_gens = rec  # Now includes p0
        total_trials_rep = attempts
        p_loss_rep = losses / total_trials_rep
        p_fix_rep = fixation / total_trials_rep
        std_err_rep = np.sqrt(p_fix_rep * (1.0 - p_fix_rep) / total_trials_rep) if total_trials_rep > 0 else 0.0
        
        # For individual rep: compute mean and std of fixation gens (if any)
        if len(fixation_gens) > 0:
            avg_gen_rep = np.mean(fixation_gens)
            std_gen_rep = np.std(fixation_gens, ddof=1) if len(fixation_gens) > 1 else 0.0
        else:
            avg_gen_rep = np.nan
            std_gen_rep = np.nan

        line = f"{idx};{rep};{N};{s};{p0};{attempts};{p_loss_rep:.8f};{p_fix_rep:.8f};{std_err_rep:.10f};{avg_gen_rep:.2f};{std_gen_rep:.2f}"  # Now includes p0
        lines_to_write.append(line)
    
    skip_creation_individual = False
    try:
        if os.path.exists(results_filename):
            os.remove(results_filename)
    except Exception as e:
        skip_creation_individual = True

    if not skip_creation_individual:
        with open(results_filename, "w") as f:
            f.write(output_headings + "\n")
            for line in lines_to_write:
                f.write(line + "\n")
        print(f"Results stored in file {results_filename}.")
    else:
        print(f"Skipped creating results file {results_filename} due to file access issues.")
    
    # Write pooled average results to out_avg_1_haplo_min.txt
    if Repetitions > 1:
        skip_creation = False
        try:
            if os.path.exists(results_filename_avg):
                os.remove(results_filename_avg)
        except Exception as e:
            print(f"Warning: Could not delete {results_filename_avg} because: {e}")
            skip_creation = True
        if not skip_creation:
            with open(results_filename_avg, "w") as f:
                f.write(output_headings_avg + "\n")
        avg_lines = []
        for rec in results_by_param:
            line = f"{rec[0]};{Repetitions};{rec[1]};{rec[2]};{rec[3]};{rec[4]};{rec[5]:.8f};{rec[6]:.8f};{rec[7]:.10f};{rec[8]:.2f};{rec[9]:.2f}"  # Adjusted indices for p0 and column order
            avg_lines.append(line)
        if not skip_creation:
            with open(results_filename_avg, "a") as f:
                for line in avg_lines:
                    f.write(line + "\n")
            print(f"Statistics based on all repetitions and attempts were stored in file: {results_filename_avg}")
        else:
            print(f"Skipped creating average results file {results_filename_avg} due to file access issues.")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")