"""Test hybrid decomposition on Mars orbital data"""
import csv
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppf import HybridDecomposer, print_hybrid_result


def load_mars_data(filepath):
    distances = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            distances.append(float(row['distance_au']))
    return np.array(distances)


def main():
    print("=" * 70)
    print("HYBRID DECOMPOSITION: MARS ORBITAL DATA")
    print("=" * 70)
    print()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data", "mars_radial_distance.csv"
    )
    distances = load_mars_data(data_path)
    print(f"Data points: {len(distances)}")
    print(f"Range: {distances.min():.4f} to {distances.max():.4f} AU")
    print()

    # Test with basic EMD (no multiprocessing issues)
    print("Testing with EMD...")
    decomposer = HybridDecomposer(method="emd", noise_threshold=0.4)
    result = decomposer.analyze(distances)

    print_hybrid_result(result)

    # Interpretation
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    for comp in result.get_signal_components():
        if comp.form_type and comp.form_type.value == "sine":
            # Mars data is 5-day intervals
            period_days = comp.period * 5
            print(f"Detected orbital period: {period_days:.0f} days (actual: ~687 days)")

    print(f"\nVariance explained by signal: {result.variance_explained_ratio:.1%}")

    # Also test SSA
    print()
    print("=" * 70)
    print("TESTING WITH SSA")
    print("=" * 70)

    decomposer_ssa = HybridDecomposer(method="ssa", window_size=70, n_components=5)
    result_ssa = decomposer_ssa.analyze(distances)

    print_hybrid_result(result_ssa)


if __name__ == "__main__":
    main()
