import csv
from decimal import Decimal, getcontext

# Set precision to 60 to safely maintain 50 decimal places during computation
getcontext().prec = 60

def main():
    MAX_GENS = 500
    s = Decimal('0.05')
    h = Decimal('0.5')
    x = Decimal('0.005')  # initial frequency at generation 0

    # Open CSV file for writing
    with open('Bootstrap.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['g', 'x', 'mean_w', 'p_prime'])

        # Generation 0
        writer.writerow([0, x, '', ''])  # mean_w and p_prime not defined at g=0

        for g in range(1, MAX_GENS + 1):
            wAA = Decimal('1') + s
            wAa = Decimal('1') + h * s
            waa = Decimal('1')

            x2 = x * x
            x1_minus_x = x * (1 - x)

            mean_w = x2 * wAA + 2 * x1_minus_x * wAa + (1 - x) ** 2 * waa
            p_prime = (x2 * wAA + x1_minus_x * wAa) / mean_w

            # Write row: g, x (from previous gen), mean_w, p_prime
            writer.writerow([g, x, mean_w, p_prime])

            # Update x for next generation
            x = p_prime

    print("Bootstrap.csv' has been created.")

if __name__ == "__main__":
    main()