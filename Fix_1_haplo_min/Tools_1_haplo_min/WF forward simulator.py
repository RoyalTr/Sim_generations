'''
WF forward simulator.py	(Wright-Fisher forward simulator)

This is a minimal, vectorized diploid Wright–Fisher forward simulator with selection and dominance, matching the
assumptions of Kimura’s diffusion mathematics.
The standard genotype fitness convention was used:
WAA = 1 + s; WAa = 1 + hs; Waa = 1
Based on a standard algorithm that appears in many places:
- Ewens (2004), “Mathematical Population Genetics” — formal definition.
- Hartl & Clark (2007), “Principles of Population Genetics” — Wright–Fisher model with selection.
- Teaching code: many universities post small Python/R/Matlab versions online, but they all follow the same basic pseudocode.
The vectorized style keeps all replicates in a NumPy array and updates them in parallel to maximize speed.
'''

import numpy as np
import pandas as pd
import time

# ------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------
INPUT_FILE = "in_WF forward simulator.txt"
OUTPUT_FILE = "out_WF forward simulator.txt"
MAX_GENS = 1_000_000_000  # Fixed maximum generations (1 billion)


# ------------------------------------------------------------
# Vectorized Wright–Fisher simulator
# ------------------------------------------------------------
start_time = time.time()


def wright_fisher_sim(Ne, s, h, p0, attempts):
    """
    Wright–Fisher forward simulations with selection (Model II).

    Parameters
    ----------
    Ne : int
        Diploid effective population size
    s : float
        Selection coefficient (W_AA=1+s, W_Aa=1+hs, W_aa=1)
    h : float
        Dominance coefficient
    p0 : float
        Initial allele frequency (0..1)
    attempts : int
        Number of replicate populations (must be provided)

    Returns
    -------
    fix_prob : float
        Fraction of replicates that fixed
    loss_prob : float
        Fraction that lost
    mean_fix_time : float
        Mean generations to fixation (NaN if no fixation)
    """
    n = 2 * Ne  # gene copies
    counts = np.full(attempts, int(round(p0 * n)))
    gens = np.zeros(attempts, dtype=int)
    fixed = np.zeros(attempts, dtype=bool)
    lost = np.zeros(attempts, dtype=bool)

    for g in range(1, MAX_GENS + 1):
        active = ~(fixed | lost)
        if not np.any(active):
            break

        x = counts[active] / n

        # Fitness model
        wAA, wAa, waa = 1 + s, 1 + h * s, 1.0
        mean_w = x**2 * wAA + 2 * x * (1 - x) * wAa + (1 - x) ** 2 * waa
        p_prime = (x**2 * wAA + x * (1 - x) * wAa) / mean_w

        counts[active] = np.random.binomial(n, p_prime)
        gens[active] = g

        fixed[active] = counts[active] == n
        lost[active] = counts[active] == 0

    fix_prob = fixed.mean()
    loss_prob = lost.mean()
    mean_fix_time = gens[fixed].mean() if fixed.any() else np.nan
    return fix_prob, loss_prob, mean_fix_time


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE, sep=";")

    # Ensure required columns are present
    required_columns = {"Ne", "s", "h", "p0", "attempts"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    results = []
    for _, row in df.iterrows():
        Ne = int(row["Ne"])
        s = float(row["s"])
        h = float(row["h"])
        p0 = float(row["p0"])
        attempts = int(row["attempts"])  # No default — must be in input

        fix_prob, loss_prob, mean_fix_time = wright_fisher_sim(
            Ne, s, h, p0, attempts=attempts
        )

        results.append([Ne, s, h, p0, attempts,
                        fix_prob, loss_prob, mean_fix_time])

    out_df = pd.DataFrame(
        results,
        columns=[
            "Ne", "s", "h", "p0", "attempts",
            "fix_prob", "loss_prob", "mean_fix_time"
        ]
    )
    out_df.to_csv(OUTPUT_FILE, sep=";", index=False)
    print(f"Results written to {OUTPUT_FILE}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")