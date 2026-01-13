"""
Analyze OEIS detection rates and verify claims about closed-form prevalence.
"""

import sys
sys.path.insert(0, '..')

import random
from typing import List, Tuple, Optional
from collections import defaultdict

try:
    from core.numerical_form_analyzer import NumericalFormAnalyzer, FormType
    from core.recursive_form_flow import RecursiveFormFlow
except ImportError:
    sys.path.insert(0, '../core')
    from numerical_form_analyzer import NumericalFormAnalyzer, FormType
    from recursive_form_flow import RecursiveFormFlow


def parse_oeis_line(line: str) -> Optional[Tuple[str, List[int]]]:
    """Parse a line from stripped OEIS file"""
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
        if len(values) >= 6:
            return (oeis_id, values[:20])
    except:
        pass

    return None


def analyze_detection(sample_size: int = 500):
    """Detailed analysis of what we detect vs miss"""

    print("=" * 80)
    print("  DETAILED OEIS DETECTION ANALYSIS")
    print("=" * 80)
    print()

    # Load sequences
    all_sequences = []
    with open("oeis_stripped.txt", 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_oeis_line(line)
            if parsed:
                all_sequences.append(parsed)

    print(f"  Total OEIS sequences: {len(all_sequences)}")

    # Random sample
    random.seed(42)
    sample = random.sample(all_sequences, min(sample_size, len(all_sequences)))

    # Initialize analyzer
    recursive_analyzer = RecursiveFormFlow(max_depth=4)

    found = []
    not_found = []

    print(f"  Analyzing {len(sample)} sequences...")
    print()

    for i, (oeis_id, values) in enumerate(sample):
        float_values = [float(v) for v in values[:15]]
        result = recursive_analyzer.analyze(float_values)

        if result:
            found.append((oeis_id, values[:10], result.formula_string()))
        else:
            not_found.append((oeis_id, values[:10]))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(sample)}] Found so far: {len(found)} ({100*len(found)/(i+1):.1f}%)")

    print()
    print("=" * 80)
    print("  RESULTS")
    print("=" * 80)
    print()
    print(f"  Found:     {len(found):>4} ({100*len(found)/len(sample):.1f}%)")
    print(f"  Not found: {len(not_found):>4} ({100*len(not_found)/len(sample):.1f}%)")
    print()

    # Analyze what we FOUND
    print("=" * 80)
    print("  SEQUENCES WE IDENTIFIED (sample of 30)")
    print("=" * 80)
    for oeis_id, vals, formula in found[:30]:
        print(f"  {oeis_id}: {vals[:6]}... -> {formula[:40]}")

    print()
    print("=" * 80)
    print("  SEQUENCES WE DID NOT IDENTIFY (sample of 30)")
    print("=" * 80)
    for oeis_id, vals in not_found[:30]:
        print(f"  {oeis_id}: {vals[:8]}...")

    # Categorize the "not found" by looking at patterns
    print()
    print("=" * 80)
    print("  ANALYSIS OF 'NOT FOUND' SEQUENCES")
    print("=" * 80)
    print()

    # Check for common patterns in not-found
    categories = defaultdict(list)

    for oeis_id, vals in not_found[:200]:
        # Check if it looks like primes (increasing, gaps vary)
        if all(v > 1 for v in vals) and len(set(vals)) == len(vals):
            diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            if len(set(diffs)) > len(diffs) * 0.6:  # High variance in differences
                categories['irregular_increasing'].append(oeis_id)
                continue

        # Check if it has repeating small values (divisor-like)
        if max(vals) < 100 and len(set(vals)) < len(vals) * 0.7:
            categories['small_repeating'].append(oeis_id)
            continue

        # Check for digit-like sequences
        if all(0 <= v <= 9 for v in vals):
            categories['digit_sequences'].append(oeis_id)
            continue

        # Check for partition-like (monotonic increasing, specific growth)
        if vals == sorted(vals) and vals[0] >= 1:
            ratios = [vals[i+1]/vals[i] if vals[i] > 0 else 0 for i in range(min(5, len(vals)-1))]
            if all(1 < r < 3 for r in ratios if r > 0):
                categories['moderate_growth'].append(oeis_id)
                continue

        categories['other'].append(oeis_id)

    print("  Rough categorization of unidentified sequences:")
    print()
    for cat, ids in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:25s}: {len(ids):>3} ({100*len(ids)/len(not_found[:200]):.0f}%)")
        print(f"      Examples: {ids[:5]}")
        print()

    # Look up some specific famous sequences
    print("=" * 80)
    print("  CHECKING SPECIFIC FAMOUS SEQUENCES")
    print("=" * 80)
    print()

    famous = {
        'A000040': 'Primes',
        'A000045': 'Fibonacci',
        'A000079': 'Powers of 2',
        'A000290': 'Squares',
        'A000217': 'Triangular',
        'A000041': 'Partitions',
        'A000010': 'Euler totient',
        'A000005': 'Divisor count',
        'A000203': 'Divisor sum',
        'A000108': 'Catalan',
        'A000142': 'Factorial',
        'A000012': 'All ones',
        'A000027': 'Natural numbers',
        'A000032': 'Lucas',
        'A001477': 'Non-negative integers',
        'A000244': 'Powers of 3',
        'A001045': 'Jacobsthal',
        'A000225': '2^n - 1 (Mersenne)',
        'A000051': '2^n + 1',
    }

    # Find these in our data
    seq_lookup = {s[0]: s[1] for s in all_sequences}

    found_famous = []
    missed_famous = []

    for oeis_id, name in famous.items():
        if oeis_id in seq_lookup:
            vals = seq_lookup[oeis_id]
            float_vals = [float(v) for v in vals[:15]]
            result = recursive_analyzer.analyze(float_vals)

            if result:
                found_famous.append((oeis_id, name, result.formula_string()[:30]))
            else:
                missed_famous.append((oeis_id, name, vals[:8]))

    print("  Famous sequences WE FOUND:")
    for oeis_id, name, formula in found_famous:
        print(f"    {oeis_id} ({name}): {formula}")

    print()
    print("  Famous sequences we MISSED:")
    for oeis_id, name, vals in missed_famous:
        print(f"    {oeis_id} ({name}): {vals}...")

    print()
    print("=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    print()

    # Count how many famous "has closed form" vs "no closed form"
    has_form = ['A000045', 'A000079', 'A000290', 'A000217', 'A000108', 'A000142',
                'A000012', 'A000027', 'A000032', 'A001477', 'A000244', 'A001045',
                'A000225', 'A000051']
    no_form = ['A000040', 'A000041', 'A000010', 'A000005', 'A000203']

    found_with_form = len([f for f in found_famous if f[0] in has_form])
    found_without_form = len([f for f in found_famous if f[0] in no_form])

    print(f"  Of famous sequences WITH known closed forms: found {found_with_form}/{len(has_form)}")
    print(f"  Of famous sequences WITHOUT closed forms: found {found_without_form}/{len(no_form)} (should be 0)")
    print()

    if found_without_form == 0:
        print("  VERIFIED: 0% false positive rate on famous non-closed-form sequences")
    else:
        print(f"  WARNING: {found_without_form} false positives detected!")

    return {
        'found': len(found),
        'not_found': len(not_found),
        'found_famous': found_famous,
        'missed_famous': missed_famous
    }


if __name__ == "__main__":
    analyze_detection(500)
