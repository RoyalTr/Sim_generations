#!/usr/bin/env python3
"""
aver_per_sim_2_haplos.py
AUTHOR: Dr. Royal Truman
VERSION: 0.9
"""
import csv
import os
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

# Record when the script starts
start_time = time.time()

# Prevent system sleep on Windows
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def process_roll_forward_and_average_streaming(data, col_index, target_col, fix_condition, max_gen_col='generation'):
    """
    Yields one row per actual generation g: [actual_g, ave_effective_gen, ave_N, ...]
    - actual_g: actual generation index (for internal use)
    - ave_effective_gen: average of min(g, fix_gen) across all runs → output as "Gen"
    - Y-values: averaged using roll-forward logic (actual row if g <= fix_gen, else final row)
    """
    runs = {}  # key: (Rep, attempt) → { 'rows': [...], 'fix_gen': int, 'final_values': row }

    for row in data:
        key = (row[col_index['Rep']], row[col_index['attempt']])
        gen = row[col_index[max_gen_col]]

        if key not in runs:
            runs[key] = {'rows': [], 'fix_gen': None, 'final_values': None}
        runs[key]['rows'].append(row)

    max_generation = 0
    for key, run in runs.items():
        rows = run['rows']
        fix_gen = None
        for i, row in enumerate(rows):
            val = row[col_index[target_col]]
            if fix_condition(val):
                fix_gen = i
                run['final_values'] = row
                break
        if fix_gen is None:
            fix_gen = len(rows) - 1
            run['final_values'] = rows[-1]
        run['fix_gen'] = fix_gen
        max_generation = max(max_generation, fix_gen)

    n_col = col_index['N']
    freq_cols = ['freq_A', 'freq_a', 'freq_Aa', 'freq_B', 'freq_b', 'freq_Bb', 'pan_heteroz', 'pan_homoz']
    freq_idxs = [col_index.get(c) for c in freq_cols]
    num_runs = len(runs)

    for g in range(max_generation + 1):
        sum_effective_gen = 0.0
        sum_N = 0.0
        sum_freqs = [0.0] * len(freq_idxs)

        for run in runs.values():
            fix_gen = run['fix_gen']
            row = run['rows'][g] if g <= fix_gen else run['final_values']
            effective_gen = g if g <= fix_gen else fix_gen
            sum_effective_gen += effective_gen
            sum_N += row[n_col]
            for i, idx in enumerate(freq_idxs):
                if idx is not None:
                    sum_freqs[i] += row[idx]

        n = num_runs
        ave_effective_gen = sum_effective_gen / n
        ave_N = sum_N / n
        ave_freqs = [s / n for s in sum_freqs]

        if target_col == 'freq_Aa':
            yield [g, ave_effective_gen, ave_N, ave_freqs[0], ave_freqs[1], ave_freqs[2]]
        elif target_col == 'freq_Bb':
            yield [g, ave_effective_gen, ave_N, ave_freqs[3], ave_freqs[4], ave_freqs[5]]
        elif target_col == 'pan_heteroz':
            yield [g, ave_effective_gen, ave_N, ave_freqs[6]]
        elif target_col == 'pan_homoz':
            yield [g, ave_effective_gen, ave_N, ave_freqs[7]]


def format_row(sim_nr, result_row):
    """
    Format output row: [actual_g, ave_effective_gen, ave_N, ...] →
    Output: SimNr, ave_effective_gen (as "Gen"), ave_N, ...
    """
    actual_g, ave_gen, ave_N, *values = result_row
    row = [str(sim_nr)]
    row.append(f"{ave_gen:.3f}")
    row.append(f"{ave_N:.3f}")
    for val in values:
        row.append(f"{val:.6f}")
    return row


def process_one_sim_nr(args):
    """
    Worker: reads input, filters SimNr, computes all metric rows, returns them.
    Does NOT write to disk — returns structured results for main process to write in order.
    Returns: (sim_nr, success, results_dict) where results_dict[target_col] = list of formatted rows
    """
    sim_nr, output_specs, input_file = args

    int_columns = {'SimNr', 'attempt', 'Rep', 'generation'}
    float_columns = {
        'N', 'freq_A', 'freq_a', 'freq_Aa',
        'freq_B', 'freq_b', 'freq_Bb',
        'pan_heteroz', 'pan_homoz'
    }

    data = []
    col_index = None

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            headers = next(reader)
            col_index = {name: i for i, name in enumerate(headers)}

            converters = []
            for header in headers:
                if header in int_columns:
                    converters.append(int)
                elif header in float_columns:
                    converters.append(float)
                else:
                    converters.append(str)

            sim_nr_idx = col_index['SimNr']

            for row in reader:
                if not row:
                    continue
                try:
                    converted_row = [
                        converter(value.strip()) for converter, value in zip(converters, row)
                    ]
                    row_sim_nr = converted_row[sim_nr_idx]

                    if row_sim_nr > sim_nr:
                        break

                    if row_sim_nr == sim_nr:
                        data.append(converted_row)

                except (ValueError, IndexError, KeyError):
                    continue

    except Exception as e:
        return sim_nr, False, {}

    if not data or not col_index:
        return sim_nr, False, {}

    required_cols = ['SimNr', 'attempt', 'Rep', 'generation', 'N',
                     'freq_Aa', 'freq_Bb', 'pan_heteroz', 'pan_homoz']
    missing_cols = [c for c in required_cols if c not in col_index]
    if missing_cols:
        return sim_nr, False, {}

    metrics = [
        ('freq_Aa', lambda x: abs(x) < 1e-9),
        ('freq_Bb', lambda x: abs(x) < 1e-9),
        ('pan_heteroz', lambda x: abs(x) < 1e-9),
        ('pan_homoz', lambda x: abs(x - 1.0) < 1e-9),
    ]

    results_dict = {}
    success = True

    for target_col, fix_condition in metrics:
        if target_col not in output_specs:
            continue

        rows = []
        try:
            for result_row in process_roll_forward_and_average_streaming(
                data, col_index, target_col, fix_condition
            ):
                formatted_row = format_row(sim_nr, result_row)
                rows.append(formatted_row)
            results_dict[target_col] = rows
        except Exception:
            success = False

    return sim_nr, success, results_dict


def main():
    input_file = 'out_2_haplos_per_gen.txt'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Scanning SimNrs and validating sort order...")

    sim_nrs = set()
    col_index = None

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            headers = next(reader)
            col_index = {name: i for i, name in enumerate(headers)}
            sim_nr_idx = col_index['SimNr']

            prev_sim_nr = None
            for row_num, row in enumerate(reader, start=2):
                if not row:
                    continue
                try:
                    sim_nr = int(row[sim_nr_idx].strip())
                    sim_nrs.add(sim_nr)

                    if prev_sim_nr is not None and sim_nr < prev_sim_nr:
                        print(f"❌ ABORT: SimNr order violation at input line {row_num}: "
                              f"previous={prev_sim_nr}, current={sim_nr}. "
                              f"Workers rely on sorted SimNr for early-break optimization. "
                              f"Please sort the input file by SimNr, Rep, attempt, generation and rerun.")
                        sys.exit(1)

                    prev_sim_nr = sim_nr

                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error scanning file: {e}")
        return

    sims = sorted(sim_nrs)
    if not sims:
        print("No valid SimNrs found.")
        return

    # Define output files and headers
    outputs = {
        'A_haplo_AVE_vals_per_gen_2H.txt': "Sim Nr;Gen;N;Freq A;Freq a;Freq Aa",
        'B_haplo_AVE_vals_per_gen_2H.txt': "Sim Nr;Gen;N;Freq B;Freq b;Freq Bb",
        'Pan-heteroz_AVE_vals_per_gen_2H.txt': "Sim Nr;Gen;N;Pan heteroz",
        'Pan-homoz_AVE_vals_per_gen_2H.txt': "Sim Nr;Gen;N;Pan homoz",
    }

    # Prepare output specs
    output_specs = {
        'freq_Aa': {'filename': 'A_haplo_AVE_vals_per_gen_2H.txt'},
        'freq_Bb': {'filename': 'B_haplo_AVE_vals_per_gen_2H.txt'},
        'pan_heteroz': {'filename': 'Pan-heteroz_AVE_vals_per_gen_2H.txt'},
        'pan_homoz': {'filename': 'Pan-homoz_AVE_vals_per_gen_2H.txt'},
    }

    # Initialize files with headers
    for filename, header in outputs.items():
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(header.split(';'))
        except Exception as e:
            print(f"Error initializing {filename}: {e}")
            return

    # Prepare args
    args_list = [(sim_nr, output_specs, input_file) for sim_nr in sims]

    max_workers = os.cpu_count() or 1
    print(f"Processing {len(sims)} SimNrs using {max_workers} workers...")

    completed = 0
    total = len(sims)

    multiprocessing.set_start_method('spawn', force=True)

    # 📦 Collect all results first
    all_results = {}  # sim_nr -> { target_col -> [rows] }

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_sim_nr, args): args[0] for args in args_list}

        for future in as_completed(futures):
            sim_nr = futures[future]
            try:
                sim_nr, success, results_dict = future.result()
                if success:
                    all_results[sim_nr] = results_dict
                    completed += 1
                    print(f"✅ [{completed}/{total}] Processed SimNr {sim_nr}")
                else:
                    print(f"❌ [{completed}/{total}] Failed SimNr {sim_nr}")
            except Exception as e:
                print(f"💥 Exception processing SimNr {sim_nr}: {e}")

    # ✍️ Now write in SimNr order

    # For each metric/file, write all SimNr blocks in order
    for target_col, spec in output_specs.items():
        filename = spec['filename']
        try:
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                for sim_nr in sorted(all_results.keys()):  # ascending SimNr
                    if target_col in all_results[sim_nr]:
                        for row in all_results[sim_nr][target_col]:
                            writer.writerow(row)
        except Exception as e:
            print(f"⚠️ Error writing {target_col} to {filename}: {e}")

    # Final summary
    print(f"💾 Output files created:")
    for fname in outputs.keys():
        print(f"...{fname}")

    print(f"📁 Output files stored in: {os.getcwd()}")
    execution_time = time.time() - start_time
    print(f"⏱️  Total runtime: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()