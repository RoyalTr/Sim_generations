#!/usr/bin/env python3
"""
Kimura_NumIntegr.py
Alternative to using an exact finite-state solver (WFES)
Numerical solver for Kimura fixation probability using genotype fitness model:
    W_AA = 1 + s
    W_Aa = 1 + h*s
    W_aa = 1

The model assumes populations with discrete, non-overlapping generations (like the standard Wright–Fisher model)

Reads input file (semicolon-separated) with columns:
    Ne;s;h;p0
and writes output file with added columns:
    Pfix; method

Author: Dr. Royal Truman
"""
from __future__ import annotations
import argparse
import math
import sys
from typing import Tuple
import numpy as np
import time


# Define input/output files and grid size as constants
INPUT_FILE = "in_Kimura.txt"
OUTPUT_FILE = "out_Kimura_NumIntegr.txt"
GRID_SIZE = 10001  # High-accuracy grid size for numeric integration


def kimura_fixation_probability_numeric(Ne: float, s: float, h: float, p0: float) -> Tuple[float, str]:
    """
    Compute fixation probability pi(p0) using the diffusion result for the genotype fitness model.
    Discussed in the technical document Kimura_NumIntegr.docx
    Returns (pi, method) where method is one of:
      - 'erf-closed'  : analytic reduction to erf differences (fast & stable)
      - 'h=0.5-closed' : semidominant closed form using expm1 (stable)
      - 'numeric'      : numeric integration with exponent shift + trapezoid

    Numerical details:
      Phi(y) = gamma*(h*y + 0.5*(1-2h)*y^2)  with gamma = 4 * Ne * s
      Integrand psi(y) = exp(-Phi(y)).

    The calculation returns a float between 0 and 1 (or np.nan on numerical failure).
    """
    # Boundaries
    if p0 <= 0.0:
        return 0.0, "boundary"
    if p0 >= 1.0:
        return 1.0, "boundary"

    # Convert to floats
    Ne = float(Ne)
    s = float(s)
    h = float(h)
    p0 = float(p0)

    # gamma is the common factor
    gamma = 4.0 * Ne * s

    # If gamma is essentially zero => neutral-ish -> return p0
    if abs(gamma) < 1e-14:
        return p0, "near-neutral"

    # Write Phi(y) = A*y + B*y^2
    A = gamma * h
    B = gamma * 0.5 * (1.0 - 2.0 * h)  # could be positive, zero, or negative

    # Case B == 0  -> Phi(y) = A*y -> closed form (semidominant when h=0.5)
    if abs(B) < 1e-30:
        # pi = (1 - exp(-A*p0)) / (1 - exp(-A)), where A = gamma*h = gamma*0.5 when h=0.5
        # but note: when h=0.5 this reduces to the standard semidominant closed form with gamma=4Ne s
        # we use the expm1 function for numerical stability
        num = -math.expm1(-A * p0)
        den = -math.expm1(-A)
        # NOTE: The 'h=0-limit' case is theoretically possible but practically unreachable
        # due to the earlier near-neutral check (|gamma| < 1e-14). It is retained only
        # as a defensive safeguard and should never be observed in normal execution.
        if abs(den) < 1e-16:
            return p0, "h=0-limit (this should never occur!)"
        return float(num / den), "h=0.5-closed"

    # If B > 0 we can express integrals in terms of erf() differences and cancel large prefactors:
    # Integral_0^p0 exp(-B y^2 - A y) dy = prefactor * [erf(tp) - erf(t0)]
    # where tp = sqrt(B)*p0 + t0, t0 = A/(2 sqrt(B)), and prefactor cancels in ratio.
    if B > 0.0:
        sqrtB = math.sqrt(B)
        t0 = A / (2.0 * sqrtB)
        tp = sqrtB * p0 + t0
        t1 = sqrtB * 1.0 + t0
        # Use math.erf (available in stdlib)
        erf = math.erf
        num = erf(tp) - erf(t0)
        den = erf(t1) - erf(t0)
        # If denominator is tiny, fall back to numeric integration
        if not math.isfinite(den) or abs(den) < 1e-18:
            # fall back to numeric safe integration
            Pfix = _kimura_numeric_integral_shift(Ne, s, h, p0)  # discussed in the technical writeup
            # return Pfix, "numeric-fallback-erf-small-den"
            return Pfix, "numeric-erf-small-den"
        return float(num / den), "erf-closed"

    # If B < 0 we cannot use real erf safely (it would require erfi with imaginary arguments).
    # Fall back to robust numeric integration with exponent shift trick to avoid overflow.
    Pfix = _kimura_numeric_integral_shift(Ne, s, h, p0)
    return Pfix, "numeric"


def _kimura_numeric_integral_shift(Ne: float, s: float, h: float, p0: float) -> float:
    """
    Robust numeric integration of psi(y) = exp(-Phi(y)) with shift:
      - compute phi(y) on grid [0,1]
      - shift by phi_min to avoid overflow: psi_shift = exp(-(phi - phi_min))
      - compute cumulative integral via trapezoid
      - numerator = interp(cum, p0), denominator=cum[-1], return numerator/denominator
    """
    # Build grid
    grid_size = GRID_SIZE
    if grid_size < 3:
        grid_size = 3
    z = np.linspace(0.0, 1.0, grid_size)

    gamma = 4.0 * float(Ne) * float(s)
    phi = gamma * (h * z + 0.5 * (1.0 - 2.0 * h) * (z * z))

    # Shift for numerical stability (largest exponent becomes zero)
    phi_min = float(np.min(phi))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        psi_shift = np.exp(-(phi - phi_min))

    # Cumulative trapezoid: cum[i] = integral_0^{z[i]} psi_shift(t) dt
    dz = np.diff(z)
    mid = 0.5 * (psi_shift[:-1] + psi_shift[1:])
    areas = mid * dz
    cum = np.concatenate(([0.0], np.cumsum(areas)))

    den = float(cum[-1])
    if not np.isfinite(den) or den <= 0.0:
        return float('nan')

    num = float(np.interp(p0, z, cum))
    # Ratio is same as (∫0^p0 psi) / (∫0^1 psi) because shift cancels
    return float(num / den)


# ----------------------------
# Command-line / file I/O
# ----------------------------
def _read_input(path: str):
    import pandas as pd
    df = pd.read_csv(path, sep=';')
    required = {'Ne', 's', 'h', 'p0'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Input file must contain columns: {sorted(required)}")
    return df


def _write_output(df, path: str):
    df.to_csv(path, sep=';', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kimura numerical fixation probability solver")
    parser.add_argument("--input", "-i", default=INPUT_FILE, help="Input file (semicolon-separated) with columns Ne;s;h;p0")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="Output file (semicolon-separated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose progress")
    args = parser.parse_args(argv)

    try:
        df = _read_input(args.input)
    except FileNotFoundError:
        print(f"Input file '{args.input}' not found. Create '{args.input}' with header Ne;s;h;p0 and rows.", file=sys.stderr)
        return 2
    except Exception as e:
        print("Error reading input:", e, file=sys.stderr)
        return 3

    out_rows = []
    for idx, row in df.iterrows():
        try:
            Ne = float(row["Ne"])
            s = float(row["s"])
            h = float(row["h"])
            p0 = float(row["p0"])
        except Exception as e:
            print(f"Skipping row {idx} due to conversion error: {e}", file=sys.stderr)
            out_rows.append([row.get("Ne"), row.get("s"), row.get("h"), row.get("p0"), float("nan"), "bad-input"])
            continue

        pi_val, method = kimura_fixation_probability_numeric(Ne, s, h, p0)
        out_rows.append([Ne, s, h, p0, pi_val, method])
        if args.verbose:
            print(f"row {idx}: Ne={Ne}, s={s}, h={h}, p0={p0} -> pi={pi_val:.12g} ({method})")

    import pandas as pd
    out_df = pd.DataFrame(out_rows, columns=["Ne", "s", "h", "p0", "Pfix", "method"])
    out_df['Ne'] = out_df['Ne'].astype('Int64')  # Convert Ne to integer
    _write_output(out_df, args.output)
    if args.verbose:
        print(f"Wrote {len(out_df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()  # Run the program logic and get its exit status to document program execution status
    print(f"Results stored in file: {OUTPUT_FILE}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    raise SystemExit(exit_code)
