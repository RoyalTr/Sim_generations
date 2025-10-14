import pandas as pd
import numpy as np
from decimal import Decimal, getcontext

# Set maximum precision for Decimal calculations
getcontext().prec = 50  # 50 decimal places of precision

# Define input and output file names as constants
INPUT_FILE = 'in_Kimura.txt'
OUTPUT_FILE = 'out_Kimura_approx.txt'

def kimura_fixation_probability(Ne, s, h, p0):
    """
    Kimura's diffusion theory involves a partial differential equation that, in its full generality,
    requires numerical integration to solve exactly. Students are taught a well-known analytical
    approximation that works well under certain assumptions (like weak selection, large population sizes, etc.).
    This program uses this approximation with 50 decimal point calculations.
    
    Parameters:
    Ne: Effective population size
    s: Selection coefficient
    h: Dominance coefficient
    p0: Initial frequency of allele A
    
    Returns:
    Probability of fixation (high precision)
    """
    # Convert to Decimal for high precision calculations
    Ne_d = Decimal(str(Ne))
    s_d = Decimal(str(s))
    h_d = Decimal(str(h))
    p0_d = Decimal(str(p0))
    
    # Handle the case where s = 0 (neutral)
    if s_d == 0:
        return float(p0_d)
    
    # Calculate the exponents with high precision
    numerator_exp = -4 * Ne_d * h_d * s_d * p0_d
    denominator_exp = -4 * Ne_d * h_d * s_d
    
    # Calculate the probability using Kimura's formula with high precision
    # Use exp() method of Decimal for high precision exponential
    numerator = 1 - numerator_exp.exp()  # ​1 − exp(−4Nehsp0)
    denominator = 1 - denominator_exp.exp()
    
    # Handle edge cases
    if denominator == 0:
        return float(p0_d)  # Neutral case
    
    # Return as float for compatibility with pandas
    return float(numerator / denominator)

def main():
    try:
        # Read the input file
        df = pd.read_csv(INPUT_FILE, delimiter=';')
        
        # Check if required columns exist
        required_columns = ['Ne', 's', 'h', 'p0']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input file must contain columns: {required_columns}")
            return
        
        # Add a column for fixation probability
        df['Prob of allele fix'] = df.apply(
            lambda row: kimura_fixation_probability(row['Ne'], row['s'], row['h'], row['p0']), 
            axis=1
        )
        
        # Save results to output file with high precision
        df.to_csv(OUTPUT_FILE, sep=';', index=False, float_format='%.12f')
        print(f"Results saved to: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'")
        print("Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()