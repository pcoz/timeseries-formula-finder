"""Test hybrid decomposition on sunspot data"""
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppf import HybridDecomposer, print_hybrid_result, FormType


def load_sunspot_data(filepath, start_year=None):
    """Load sunspot data from SILSO CSV format"""
    years = []
    sunspots = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                year = int(parts[0])
                if start_year is None or year >= start_year:
                    decimal_year = float(parts[2])
                    sunspot_num = float(parts[3])
                    years.append(decimal_year)
                    sunspots.append(sunspot_num)

    return np.array(years), np.array(sunspots)


def main():
    print("=" * 70)
    print("HYBRID DECOMPOSITION: SUNSPOT DATA")
    print("=" * 70)
    print()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data", "sunspot_monthly.csv"
    )

    # Use last 50 years for faster analysis
    years, sunspots = load_sunspot_data(data_path, start_year=1974)

    print(f"Data points: {len(sunspots)}")
    print(f"Year range: {years[0]:.1f} to {years[-1]:.1f}")
    print(f"Sunspot range: {sunspots.min():.1f} to {sunspots.max():.1f}")
    print()

    # Test with EMD
    print("Testing with EMD...")
    decomposer = HybridDecomposer(method="emd", noise_threshold=0.3, min_r_squared=0.2)
    result = decomposer.analyze(sunspots)

    print_hybrid_result(result)

    # Interpretation
    print()
    print("=" * 70)
    print("SCIENTIFIC INTERPRETATION")
    print("=" * 70)

    for comp in result.get_signal_components():
        if comp.form_type == FormType.SINE:
            # Sunspot data is monthly
            period_months = comp.period
            period_years = period_months / 12
            if 8 < period_years < 15:
                print(f"Detected ~11-year solar cycle: {period_years:.1f} years")
            elif 70 < period_years < 130:
                print(f"Possible Gleissberg cycle: {period_years:.1f} years")
            else:
                print(f"Cycle detected: {period_years:.1f} years")

    print(f"\nTotal variance explained by signal components: {result.variance_explained_ratio:.1%}")

    # Compare with SSA
    print()
    print("=" * 70)
    print("TESTING WITH SSA (window=132 months ~ 11 years)")
    print("=" * 70)

    decomposer_ssa = HybridDecomposer(
        method="ssa",
        window_size=132,  # ~11 years
        n_components=6,
        noise_threshold=0.3
    )
    result_ssa = decomposer_ssa.analyze(sunspots)

    print_hybrid_result(result_ssa)


if __name__ == "__main__":
    main()
