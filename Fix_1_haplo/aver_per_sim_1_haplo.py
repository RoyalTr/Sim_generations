#!/usr/bin/env python3
"""
aver_per_sim_1_haplo.py
AUTHOR: Dr. Royal Truman
VERSION: 0.9
"""
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import tempfile

# Prevent system sleep on Windows.
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def process_roll_forward_and_average_streaming(data, target_col, fix_condition, result_cols):
    run_metadata = {}
    run_timeseries = {}
    max_generation = -1

    def process_run(key, rows):
        nonlocal max_generation
        fix_gen = None
        timeseries = {}

        for row in rows:
            gen = row['generation']
            val = row[target_col]
            N_val = row['N']
            freqs_tuple = tuple(row[col] for col in result_cols)
            timeseries[gen] = (N_val, freqs_tuple)

            if fix_gen is None and fix_condition(val):
                fix_gen = gen

        if fix_gen is None:
            fix_gen = max(timeseries.keys()) if timeseries else 0

        final_N, final_freqs = timeseries.get(fix_gen, (0, tuple(0.0 for _ in result_cols)))
        run_metadata[key] = (fix_gen, final_N, final_freqs)
        run_timeseries[key] = timeseries
        max_generation = max(max_generation, fix_gen)

    current_key = None
    current_rows = []

    for row in data:
        key = (row['Rep'], row['attempt'])
        if current_key is None:
            current_key = key
        elif key != current_key:
            process_run(current_key, current_rows)
            current_rows = []
            current_key = key
        current_rows.append(row)

    if current_rows:
        process_run(current_key, current_rows)

    if not run_metadata:
        return

    if max_generation < 0:
        return

    for g in range(0, max_generation + 1):
        total_count = 0
        sum_effective_gen = 0.0
        sum_N = 0.0
        sum_freqs = [0.0] * len(result_cols)

        for key in run_metadata:
            fix_gen, final_N, final_freqs = run_metadata[key]
            timeseries = run_timeseries[key]

            effective_gen_for_run = g if g <= fix_gen else fix_gen

            if g <= fix_gen:
                if g in timeseries:
                    N_val, freqs_val = timeseries[g]
                else:
                    N_val, freqs_val = final_N, final_freqs
            else:
                N_val, freqs_val = final_N, final_freqs

            total_count += 1
            sum_effective_gen += effective_gen_for_run
            sum_N += N_val
            for i in range(len(result_cols)):
                sum_freqs[i] += freqs_val[i]

        if total_count == 0:
            continue

        avg_effective_gen = sum_effective_gen / total_count
        avg_N = sum_N / total_count
        avg_freqs = [s / total_count for s in sum_freqs]

        yield (g, avg_effective_gen, avg_N, *avg_freqs)


def process_roll_forward_and_average_streaming_fix_A(data):
    # ! Initialize default row from first gen of first run since the downstream programs expect data for all SimNr
    first_row = None
    for row in data:
        if row['generation'] == 0:
            first_row = row
            break
    if first_row is None:
        first_row = data[0] if data else None  # fallback

    run_metadata = {}
    run_timeseries = {}
    max_generation = -1

    def process_run(key, rows):
        nonlocal max_generation
        fix_gen = None
        timeseries = {}

        for row in rows:
            gen = row['generation']
            val = row['freq_A']
            N_val = row['N']
            freqs_tuple = (row['freq_A'], row['freq_a'], row['freq_Aa'])
            timeseries[gen] = (N_val, freqs_tuple)

        last_gen = max(timeseries.keys()) if timeseries else None
        if last_gen is not None:
            last_freq_A = timeseries[last_gen][1][0]
            if abs(last_freq_A - 1.0) < 1e-10:
                fix_gen = last_gen
                final_N, final_freqs = timeseries[fix_gen]
                run_metadata[key] = (fix_gen, final_N, final_freqs)
                run_timeseries[key] = timeseries
                max_generation = max(max_generation, fix_gen)

    current_key = None
    current_rows = []

    for row in data:
        key = (row['Rep'], row['attempt'])
        if current_key is None:
            current_key = key
        elif key != current_key:
            process_run(current_key, current_rows)
            current_rows = []
            current_key = key
        current_rows.append(row)

    if current_rows:
        process_run(current_key, current_rows)

    if not run_metadata and first_row is None:
        return  # no runs, no default

    if max_generation < 0 and first_row is None:
        return

    # ! Yield default row for this SimNr if any runs exist or even if none (via first_row)
    if first_row is not None:
        yield (0, 0.0, first_row['N'], first_row['freq_A'])

    if not run_metadata:
        return

    for g in range(0, max_generation + 1):
        total_count = 0
        sum_effective_gen = 0.0
        sum_N = 0.0
        sum_freq_A = 0.0

        for key in run_metadata:
            fix_gen, final_N, final_freqs = run_metadata[key]
            timeseries = run_timeseries[key]

            effective_gen_for_run = g if g <= fix_gen else fix_gen

            if g <= fix_gen:
                if g in timeseries:
                    N_val, freqs_val = timeseries[g]
                else:
                    N_val, freqs_val = final_N, final_freqs
            else:
                N_val, freqs_val = final_N, final_freqs

            total_count += 1
            sum_effective_gen += effective_gen_for_run
            sum_N += N_val
            sum_freq_A += freqs_val[0]

        if total_count == 0:
            continue

        avg_effective_gen = sum_effective_gen / total_count
        avg_N = sum_N / total_count
        avg_freq_A = sum_freq_A / total_count

        yield (g, avg_effective_gen, avg_N, avg_freq_A)


def process_roll_forward_and_average_streaming_fix_a(data):
    # ! Initialize default row from first gen of first run
    first_row = None
    for row in data:
        if row['generation'] == 0:
            first_row = row
            break
    if first_row is None:
        first_row = data[0] if data else None  # fallback

    run_metadata = {}
    run_timeseries = {}
    max_generation = -1

    def process_run(key, rows):
        nonlocal max_generation
        fix_gen = None
        timeseries = {}

        for row in rows:
            gen = row['generation']
            val = row['freq_a']
            N_val = row['N']
            freqs_tuple = (row['freq_A'], row['freq_a'], row['freq_Aa'])
            timeseries[gen] = (N_val, freqs_tuple)

        last_gen = max(timeseries.keys()) if timeseries else None
        if last_gen is not None:
            last_freq_a = timeseries[last_gen][1][1]
            if abs(last_freq_a - 1.0) < 1e-10:
                fix_gen = last_gen
                final_N, final_freqs = timeseries[fix_gen]
                run_metadata[key] = (fix_gen, final_N, final_freqs)
                run_timeseries[key] = timeseries
                max_generation = max(max_generation, fix_gen)

    current_key = None
    current_rows = []

    for row in data:
        key = (row['Rep'], row['attempt'])
        if current_key is None:
            current_key = key
        elif key != current_key:
            process_run(current_key, current_rows)
            current_rows = []
            current_key = key
        current_rows.append(row)

    if current_rows:
        process_run(current_key, current_rows)

    if not run_metadata and first_row is None:
        return

    if max_generation < 0 and first_row is None:
        return

    # ! Yield default row for this SimNr
    if first_row is not None:
        yield (0, 0.0, first_row['N'], first_row['freq_a'])

    if not run_metadata:
        return

    for g in range(0, max_generation + 1):
        total_count = 0
        sum_effective_gen = 0.0
        sum_N = 0.0
        sum_freq_a = 0.0

        for key in run_metadata:
            fix_gen, final_N, final_freqs = run_metadata[key]
            timeseries = run_timeseries[key]

            effective_gen_for_run = g if g <= fix_gen else fix_gen

            if g <= fix_gen:
                if g in timeseries:
                    N_val, freqs_val = timeseries[g]
                else:
                    N_val, freqs_val = final_N, final_freqs
            else:
                N_val, freqs_val = final_N, final_freqs

            total_count += 1
            sum_effective_gen += effective_gen_for_run
            sum_N += N_val
            sum_freq_a += freqs_val[1]

        if total_count == 0:
            continue

        avg_effective_gen = sum_effective_gen / total_count
        avg_N = sum_N / total_count
        avg_freq_a = sum_freq_a / total_count

        yield (g, avg_effective_gen, avg_N, avg_freq_a)


def process_one_sim_nr(args):
    sim_nr, input_file, temp_dir = args

    field_types = {
        'SimNr': int,
        'Rep': int,
        'attempt': int,
        'generation': int,
        'N': int,
        'freq_A': float,
        'freq_Aa': float,
        'freq_a': float
    }

    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            found_target = False

            for row in reader:
                if not any(value.strip() for value in row.values()):
                    continue

                try:
                    sim_val_str = row['SimNr'].strip()
                    if not sim_val_str:
                        continue
                    row_sim_nr = int(sim_val_str)
                except (ValueError, KeyError):
                    continue

                if not found_target:
                    if row_sim_nr < sim_nr:
                        continue
                    elif row_sim_nr > sim_nr:
                        break
                    else:
                        found_target = True
                else:
                    if row_sim_nr > sim_nr:
                        break

                try:
                    parsed = {}
                    for field, conv in field_types.items():
                        val = row[field].strip()
                        parsed[field] = conv(val) if val else None
                    data.append(parsed)
                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print(f"❌ SimNr {sim_nr}: Error reading file: {e}")
        return sim_nr, False

    if not data:
        return sim_nr, False

    raw_results = process_roll_forward_and_average_streaming(
        data,
        target_col='freq_Aa',
        fix_condition=lambda x: abs(x) < 1e-10,
        result_cols=['freq_A', 'freq_a', 'freq_Aa']
    )

    temp_file = os.path.join(temp_dir, f"sim_{sim_nr}.tmp")

    try:
        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            for r in raw_results:
                g, avg_effective_gen, ave_N, freq_A, freq_a, freq_Aa = r
                homoz = 1.0 - freq_Aa
                formatted_row = [
                    str(sim_nr),
                    f"{avg_effective_gen:.3f}",
                    f"{ave_N:.3f}",
                    f"{freq_A:.6f}",
                    f"{freq_a:.6f}",
                    f"{freq_Aa:.6f}",
                    f"{homoz:.6f}"
                ]
                writer.writerow(formatted_row)
    except Exception as e:
        print(f"⚠️  SimNr {sim_nr}: Error writing temp file: {e}")
        return sim_nr, False


    # ========================
    # OUTPUT 1: only runs where freq_A == 1.0 at end, but ALWAYS include gen=0 row
    # ========================
    raw_results_1_fix = process_roll_forward_and_average_streaming_fix_A(data)
    temp_file_1_fix = os.path.join(temp_dir, f"sim_{sim_nr}_1_fix.tmp")
    try:
        with open(temp_file_1_fix, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            for r in raw_results_1_fix:
                g, avg_effective_gen, ave_N, freq_A = r
                formatted_row = [
                    str(sim_nr),
                    f"{avg_effective_gen:.3f}",
                    f"{ave_N:.3f}",
                    f"{freq_A:.6f}"
                ]
                writer.writerow(formatted_row)
    except Exception as e:
        print(f"⚠️  SimNr {sim_nr}: Error writing 1_fix temp file: {e}")


    # ========================
    # OUTPUT 2: only runs where freq_a == 1.0 at end, but ALWAYS include gen=0 row
    # ========================
    raw_results_2_fix = process_roll_forward_and_average_streaming_fix_a(data)
    temp_file_2_fix = os.path.join(temp_dir, f"sim_{sim_nr}_2_fix.tmp")
    try:
        with open(temp_file_2_fix, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            for r in raw_results_2_fix:
                g, avg_effective_gen, ave_N, freq_a = r
                formatted_row = [
                    str(sim_nr),
                    f"{avg_effective_gen:.3f}",
                    f"{ave_N:.3f}",
                    f"{freq_a:.6f}"
                ]
                writer.writerow(formatted_row)
    except Exception as e:
        print(f"⚠️  SimNr {sim_nr}: Error writing 2_fix temp file: {e}")

    return sim_nr, True


def get_sim_nrs(filename):
    last_sim_nr = None
    seen = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if not any(value.strip() for value in row.values()):
                    continue

                try:
                    sim_nr = int(row['SimNr'].strip())
                except (ValueError, KeyError):
                    continue

                if last_sim_nr is not None and sim_nr < last_sim_nr:
                    print(f"⚠️  Warning: File not sorted by SimNr. Found {sim_nr} after {last_sim_nr}. "
                          f"Early-exit optimization may miss data.")

                last_sim_nr = sim_nr

                if sim_nr not in seen:
                    seen.add(sim_nr)
                    yield sim_nr

    except Exception as e:
        print(f"Error scanning SimNrs: {e}")
        raise


def main():
    start_time = time.time()
    input_file = 'out_1_haplo_per_gen.txt'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    output_file = 'A_haplo_AVE_vals_per_gen_1H.txt'
    header = "SimNr;EffGen;N;freq A;freq a;hetero;homoz"

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header.split(';'))
    except Exception as e:
        print(f"Error initializing output file: {e}")
        return

    print("Scanning SimNrs (and verifying sort order)...")

    sim_nrs = sorted(get_sim_nrs(input_file))
    if not sim_nrs:
        print("No SimNrs found.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [(sim_nr, input_file, temp_dir) for sim_nr in sim_nrs]

        max_workers = os.cpu_count() or 4
        print(f"Processing {len(sim_nrs)} SimNrs using {max_workers} workers...")

        multiprocessing.set_start_method('spawn', force=True)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one_sim_nr, args): args[0] for args in args_list}

            completed = 0
            total = len(sim_nrs)

            for future in as_completed(futures):
                sim_nr = futures[future]
                try:
                    sim_nr, success = future.result()
                    if success:
                        completed += 1
                        print(f"✅ [{completed}/{total}] Completed SimNr {sim_nr}")
                    else:
                        print(f"❌ [{completed}/{total}] Failed SimNr {sim_nr}")
                except Exception as e:
                    print(f"💥 Exception processing SimNr {sim_nr}: {e}")

        # Concatenate original output
        try:
            with open(output_file, 'a', newline='', encoding='utf-8') as f_out:
                for sim_nr in sim_nrs:
                    temp_file = os.path.join(temp_dir, f"sim_{sim_nr}.tmp")
                    if os.path.exists(temp_file):
                        with open(temp_file, 'r', encoding='utf-8') as f_in:
                            for line in f_in:
                                f_out.write(line)
        except Exception as e:
            print(f"Error writing final output: {e}")
            return

        output_file_1_fix = 'haplo_AVE_A_per_gen_1H.txt'
        
        try:
            with open(output_file_1_fix, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out, delimiter=';')
                writer.writerow(["SimNr", "EffGen", "N", "freq_A"])
            with open(output_file_1_fix, 'a', newline='', encoding='utf-8') as fout:
                for sim_nr in sim_nrs:
                    temp_file = os.path.join(temp_dir, f"sim_{sim_nr}_1_fix.tmp")
                    if os.path.exists(temp_file):
                        with open(temp_file, 'r', encoding='utf-8') as fin:
                            for line in fin:
                                fout.write(line)
        except Exception as e:
            print(f"Error writing 1_fix final output: {e}")

        output_file_2_fix = 'haplo_AVE_a__per_gen_1H.txt'

        try:
            with open(output_file_2_fix, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out, delimiter=';')
                writer.writerow(["SimNr", "EffGen", "N", "freq_a"])
            with open(output_file_2_fix, 'a', newline='', encoding='utf-8') as fout:
                for sim_nr in sim_nrs:
                    temp_file = os.path.join(temp_dir, f"sim_{sim_nr}_2_fix.tmp")
                    if os.path.exists(temp_file):
                        with open(temp_file, 'r', encoding='utf-8') as fin:
                            for line in fin:
                                fout.write(line)
        except Exception as e:
            print(f"Error writing 2_fix final output: {e}")

    print(f"💾 Outputs in '{output_file}', '{output_file_1_fix}', and '{output_file_2_fix}'.")

    execution_time = time.time() - start_time
    print(f"⏱️ Total runtime: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()