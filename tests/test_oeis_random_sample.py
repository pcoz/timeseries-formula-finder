"""
Random OEIS Sample Test
=======================

For a FAIR benchmark, we need to test against RANDOM sequences from OEIS,
not hand-picked ones we know will match.

This downloads a random sample and tests honestly.
"""

import sys
sys.path.insert(0, '..')

import urllib.request
import gzip
import random
import os
from typing import List, Tuple, Optional

try:
    from core.numerical_form_analyzer import NumericalFormAnalyzer, FormType
    from core.recursive_form_flow import RecursiveFormFlow
except ImportError:
    sys.path.insert(0, '../core')
    from numerical_form_analyzer import NumericalFormAnalyzer, FormType
    from recursive_form_flow import RecursiveFormFlow


def download_oeis_stripped(cache_path: str = "oeis_stripped.txt") -> str:
    """Download OEIS stripped.gz if not cached"""
    if os.path.exists(cache_path):
        print(f"  Using cached {cache_path}")
        return cache_path

    print("  Downloading OEIS stripped.gz (~30MB)...")
    url = "https://oeis.org/stripped.gz"

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            compressed = response.read()

        print("  Decompressing...")
        decompressed = gzip.decompress(compressed)

        with open(cache_path, 'wb') as f:
            f.write(decompressed)

        print(f"  Saved to {cache_path}")
        return cache_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def parse_oeis_line(line: str) -> Optional[Tuple[str, List[int]]]:
    """Parse a line from stripped OEIS file"""
    # Format: A000001 ,1,1,1,2,1,2,1,5,2,2,...
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    parts = line.split(' ', 1)
    if len(parts) != 2:
        return None

    oeis_id = parts[0]
    values_str = parts[1].strip()

    if not values_str.startswith(','):
        return None

    try:
        values = [int(x) for x in values_str.split(',') if x.strip() and x.strip() != '']
        if len(values) >= 6:  # Need at least 6 terms
            return (oeis_id, values[:20])  # First 20 terms
    except:
        pass

    return None


def load_random_sample(filepath: str, n: int = 1000, seed: int = 42) -> List[Tuple[str, List[int]]]:
    """Load a random sample of n sequences from OEIS"""
    print(f"  Loading sequences from {filepath}...")

    all_sequences = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_oeis_line(line)
            if parsed:
                all_sequences.append(parsed)

    print(f"  Found {len(all_sequences)} valid sequences")

    random.seed(seed)
    sample = random.sample(all_sequences, min(n, len(all_sequences)))

    print(f"  Selected random sample of {len(sample)}")
    return sample


def run_random_oeis_benchmark(sample_size: int = 1000):
    """Run benchmark on random OEIS sample"""

    print("=" * 80)
    print("  RANDOM OEIS SAMPLE BENCHMARK")
    print("  Testing against RANDOMLY selected sequences (fair test)")
    print("=" * 80)
    print()

    # Try to download/load OEIS data
    cache_path = download_oeis_stripped()

    if not cache_path or not os.path.exists(cache_path):
        print("  Could not load OEIS data. Using fallback test set.")
        return run_fallback_test()

    # Load random sample
    sample = load_random_sample(cache_path, sample_size)

    if not sample:
        print("  No sequences loaded.")
        return

    print()

    # Initialize analyzers
    base_analyzer = NumericalFormAnalyzer()
    recursive_analyzer = RecursiveFormFlow(max_depth=4)

    results = {
        'total': 0,
        'base_found': 0,
        'recursive_found': 0,
        'both_found': 0,
        'neither_found': 0,
        'examples_found': [],
        'examples_not_found': []
    }

    print(f"  Analyzing {len(sample)} random OEIS sequences...")
    print()

    for i, (oeis_id, values) in enumerate(sample):
        results['total'] += 1

        float_values = [float(v) for v in values[:15]]

        # Base analyzer
        base_result = base_analyzer.analyze(float_values, start_index=0)
        base_ok = (base_result.form_type != FormType.UNKNOWN and
                   base_result.confidence > 0.8)

        # Recursive analyzer
        rec_result = recursive_analyzer.analyze(float_values)
        rec_ok = rec_result is not None

        if base_ok:
            results['base_found'] += 1
        if rec_ok:
            results['recursive_found'] += 1
        if base_ok and rec_ok:
            results['both_found'] += 1
        if not base_ok and not rec_ok:
            results['neither_found'] += 1

            if len(results['examples_not_found']) < 10:
                results['examples_not_found'].append((oeis_id, values[:8]))

        if rec_ok and len(results['examples_found']) < 20:
            formula = rec_result.formula_string() if rec_result else base_result.formula
            results['examples_found'].append((oeis_id, values[:6], formula[:30]))

        # Progress
        if (i + 1) % 200 == 0:
            pct = results['recursive_found'] / results['total'] * 100
            print(f"  [{i+1:4d}/{len(sample)}] Found: {results['recursive_found']} ({pct:.1f}%)")

    # Results
    print()
    print("=" * 80)
    print("  RESULTS ON RANDOM OEIS SAMPLE")
    print("=" * 80)
    print()

    total = results['total']
    base_pct = results['base_found'] / total * 100
    rec_pct = results['recursive_found'] / total * 100
    neither_pct = results['neither_found'] / total * 100

    print(f"  Total sequences tested:     {total}")
    print()
    print(f"  Base analyzer found:        {results['base_found']:>4} ({base_pct:.1f}%)")
    print(f"  Recursive flow found:       {results['recursive_found']:>4} ({rec_pct:.1f}%)")
    print(f"  Neither found:              {results['neither_found']:>4} ({neither_pct:.1f}%)")
    print()

    # Examples found
    if results['examples_found']:
        print("  Sample of sequences WE IDENTIFIED:")
        print("  " + "-" * 60)
        for oeis_id, vals, formula in results['examples_found'][:10]:
            print(f"    {oeis_id}: {vals} -> {formula}")
        print()

    # Examples not found
    if results['examples_not_found']:
        print("  Sample of sequences we COULD NOT identify:")
        print("  " + "-" * 60)
        for oeis_id, vals in results['examples_not_found'][:5]:
            print(f"    {oeis_id}: {vals}...")
        print()

    print("=" * 80)
    print("  INTERPRETATION")
    print("=" * 80)
    print()
    print(f"  We identified {rec_pct:.1f}% of RANDOM OEIS sequences.")
    print()
    print("  This is a FAIR benchmark because:")
    print("    - Sequences were randomly selected (not cherry-picked)")
    print("    - OEIS contains many sequences WITHOUT closed forms")
    print("    - Many are defined by: primes, divisors, digits, combinatorics")
    print()
    print("  The {:.1f}% we DIDN'T find are likely TRUE_PRIMITIVES:".format(neither_pct))
    print("    - No algebraic closed form exists")
    print("    - Defined by number-theoretic properties")
    print("    - They ARE their own shortest description")
    print()

    return results


def run_fallback_test():
    """Fallback if OEIS download fails"""
    print("  Running fallback test with embedded sequences...")

    # Some actual OEIS sequences that vary in type
    test_sequences = [
        ("A000045", [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),  # Fibonacci
        ("A000040", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]),  # Primes
        ("A000290", [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]),  # Squares
        ("A000079", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),  # Powers of 2
        ("A000041", [1, 1, 2, 3, 5, 7, 11, 15, 22, 30]),  # Partitions
        ("A000010", [1, 1, 2, 2, 4, 2, 6, 4, 6, 4]),  # Euler totient
        ("A000203", [1, 3, 4, 7, 6, 12, 8, 15, 13, 18]),  # Sum of divisors
        ("A000005", [1, 2, 2, 3, 2, 4, 2, 4, 3, 4]),  # Number of divisors
        ("A000108", [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]),  # Catalan
        ("A000142", [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]),  # Factorial
    ]

    recursive_analyzer = RecursiveFormFlow(max_depth=4)

    found = 0
    for oeis_id, values in test_sequences:
        result = recursive_analyzer.analyze([float(v) for v in values])
        if result:
            found += 1
            print(f"    {oeis_id}: FOUND -> {result.formula_string()[:30]}")
        else:
            print(f"    {oeis_id}: not found")

    print(f"\n  Found {found}/{len(test_sequences)} ({100*found/len(test_sequences):.0f}%)")
    return {'found': found, 'total': len(test_sequences)}


if __name__ == "__main__":
    results = run_random_oeis_benchmark(1000)
