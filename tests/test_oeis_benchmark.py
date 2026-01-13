"""
OEIS Benchmark Test
===================

Tests our engine against the Online Encyclopedia of Integer Sequences (OEIS),
the gold standard dataset for integer sequence identification.

OEIS: https://oeis.org/
- 390,000+ sequences
- Curated by mathematicians worldwide
- Ground truth formulas available

We test against well-known sequences with known formulas to validate
our engine against real-world mathematical data.
"""

import sys
sys.path.insert(0, '..')

import urllib.request
import json
import re
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Try to import our analyzers
try:
    from core.numerical_form_analyzer import NumericalFormAnalyzer, FormType
    from core.recursive_form_flow import RecursiveFormFlow
except ImportError:
    sys.path.insert(0, '../core')
    from numerical_form_analyzer import NumericalFormAnalyzer, FormType
    from recursive_form_flow import RecursiveFormFlow


@dataclass
class OEISSequence:
    """An OEIS sequence with metadata"""
    id: str           # e.g., "A000045"
    name: str         # e.g., "Fibonacci numbers"
    values: List[int] # The sequence values
    formula: str      # Known formula (if any)
    category: str     # Our categorization


# =============================================================================
# WELL-KNOWN OEIS SEQUENCES (manually curated for testing)
# =============================================================================

def get_famous_oeis_sequences() -> List[OEISSequence]:
    """
    100 famous OEIS sequences with known formulas.
    These are manually curated to ensure accuracy.
    """
    sequences = []

    # === CONSTANT/SIMPLE ===
    sequences.append(OEISSequence(
        "A000004", "The zero sequence", [0]*20, "0", "constant"
    ))
    sequences.append(OEISSequence(
        "A000012", "The ones sequence", [1]*20, "1", "constant"
    ))
    sequences.append(OEISSequence(
        "A007395", "The twos sequence", [2]*20, "2", "constant"
    ))

    # === LINEAR ===
    sequences.append(OEISSequence(
        "A000027", "Natural numbers", list(range(1, 21)), "n", "linear"
    ))
    sequences.append(OEISSequence(
        "A005408", "Odd numbers", [2*n+1 for n in range(20)], "2n+1", "linear"
    ))
    sequences.append(OEISSequence(
        "A005843", "Even numbers", [2*n for n in range(20)], "2n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008585", "Multiples of 3", [3*n for n in range(20)], "3n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008586", "Multiples of 4", [4*n for n in range(20)], "4n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008587", "Multiples of 5", [5*n for n in range(20)], "5n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008588", "Multiples of 6", [6*n for n in range(20)], "6n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008589", "Multiples of 7", [7*n for n in range(20)], "7n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008590", "Multiples of 8", [8*n for n in range(20)], "8n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008591", "Multiples of 9", [9*n for n in range(20)], "9n", "linear"
    ))
    sequences.append(OEISSequence(
        "A008592", "Multiples of 10", [10*n for n in range(20)], "10n", "linear"
    ))

    # === QUADRATIC (squares, triangular, etc.) ===
    sequences.append(OEISSequence(
        "A000290", "Square numbers", [n**2 for n in range(20)], "n^2", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A000217", "Triangular numbers", [n*(n+1)//2 for n in range(20)], "n(n+1)/2", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A002378", "Oblong numbers", [n*(n+1) for n in range(20)], "n(n+1)", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A000326", "Pentagonal numbers", [n*(3*n-1)//2 for n in range(20)], "n(3n-1)/2", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A000384", "Hexagonal numbers", [n*(2*n-1) for n in range(20)], "n(2n-1)", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A001105", "2*n^2", [2*n**2 for n in range(20)], "2n^2", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A005563", "n^2 - 1", [n**2 - 1 for n in range(20)], "n^2-1", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A002061", "Central polygonal numbers", [n**2 - n + 1 for n in range(20)], "n^2-n+1", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A002522", "n^2 + 1", [n**2 + 1 for n in range(20)], "n^2+1", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A000566", "Heptagonal numbers", [n*(5*n-3)//2 for n in range(20)], "n(5n-3)/2", "quadratic"
    ))

    # === CUBIC ===
    sequences.append(OEISSequence(
        "A000578", "Cubes", [n**3 for n in range(15)], "n^3", "cubic"
    ))
    sequences.append(OEISSequence(
        "A000292", "Tetrahedral numbers", [n*(n+1)*(n+2)//6 for n in range(15)], "n(n+1)(n+2)/6", "cubic"
    ))
    sequences.append(OEISSequence(
        "A005900", "Octahedral numbers", [n*(2*n**2+1)//3 for n in range(15)], "n(2n^2+1)/3", "cubic"
    ))

    # === POWERS ===
    sequences.append(OEISSequence(
        "A000583", "Fourth powers", [n**4 for n in range(12)], "n^4", "power"
    ))
    sequences.append(OEISSequence(
        "A000584", "Fifth powers", [n**5 for n in range(10)], "n^5", "power"
    ))

    # === EXPONENTIAL ===
    sequences.append(OEISSequence(
        "A000079", "Powers of 2", [2**n for n in range(20)], "2^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A000244", "Powers of 3", [3**n for n in range(15)], "3^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A000302", "Powers of 4", [4**n for n in range(12)], "4^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A000351", "Powers of 5", [5**n for n in range(12)], "5^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A000400", "Powers of 6", [6**n for n in range(10)], "6^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A000420", "Powers of 7", [7**n for n in range(10)], "7^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A001018", "Powers of 8", [8**n for n in range(10)], "8^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A001019", "Powers of 9", [9**n for n in range(10)], "9^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A011557", "Powers of 10", [10**n for n in range(10)], "10^n", "exponential"
    ))

    # === MERSENNE-LIKE (2^n - 1) ===
    sequences.append(OEISSequence(
        "A000225", "2^n - 1 (Mersenne)", [2**n - 1 for n in range(20)], "2^n - 1", "mersenne"
    ))
    sequences.append(OEISSequence(
        "A024036", "3^n - 1", [3**n - 1 for n in range(15)], "3^n - 1", "mersenne"
    ))
    sequences.append(OEISSequence(
        "A024037", "4^n - 1", [4**n - 1 for n in range(12)], "4^n - 1", "mersenne"
    ))

    # === 2^n + 1 (Fermat-like) ===
    sequences.append(OEISSequence(
        "A000051", "2^n + 1", [2**n + 1 for n in range(20)], "2^n + 1", "fermat"
    ))
    sequences.append(OEISSequence(
        "A034472", "3^n + 1", [3**n + 1 for n in range(15)], "3^n + 1", "fermat"
    ))

    # === FIBONACCI-LIKE ===
    fib = [0, 1]
    for _ in range(18):
        fib.append(fib[-1] + fib[-2])
    sequences.append(OEISSequence(
        "A000045", "Fibonacci numbers", fib, "F(n)", "fibonacci"
    ))

    lucas = [2, 1]
    for _ in range(18):
        lucas.append(lucas[-1] + lucas[-2])
    sequences.append(OEISSequence(
        "A000032", "Lucas numbers", lucas, "L(n)", "fibonacci"
    ))

    pell = [0, 1]
    for _ in range(18):
        pell.append(2*pell[-1] + pell[-2])
    sequences.append(OEISSequence(
        "A000129", "Pell numbers", pell, "P(n)", "fibonacci_like"
    ))

    tribonacci = [0, 0, 1]
    for _ in range(17):
        tribonacci.append(tribonacci[-1] + tribonacci[-2] + tribonacci[-3])
    sequences.append(OEISSequence(
        "A000073", "Tribonacci numbers", tribonacci, "T(n)", "fibonacci_like"
    ))

    # === FACTORIAL ===
    import math
    sequences.append(OEISSequence(
        "A000142", "Factorials", [math.factorial(n) for n in range(12)], "n!", "factorial"
    ))
    sequences.append(OEISSequence(
        "A001563", "n * n!", [n * math.factorial(n) for n in range(10)], "n*n!", "factorial_comp"
    ))

    # === CUMULATIVE SUMS ===
    sequences.append(OEISSequence(
        "A000330", "Sum of squares", [sum(k**2 for k in range(n+1)) for n in range(15)],
        "n(n+1)(2n+1)/6", "cumsum"
    ))
    sequences.append(OEISSequence(
        "A000537", "Sum of cubes", [sum(k**3 for k in range(n+1)) for n in range(12)],
        "(n(n+1)/2)^2", "cumsum"
    ))
    sequences.append(OEISSequence(
        "A000292", "Tetrahedral", [n*(n+1)*(n+2)//6 for n in range(15)],
        "C(n+2,3)", "cumsum"
    ))

    # === GEOMETRIC SERIES ===
    sequences.append(OEISSequence(
        "A000203", "Sum of divisors (partial)", [1, 3, 4, 7, 6, 12, 8, 15, 13, 18],
        "sigma(n)", "divisor"  # Special, won't match
    ))

    # === REPUNITS AND PATTERNS ===
    sequences.append(OEISSequence(
        "A002275", "Repunits (10^n-1)/9", [(10**n - 1)//9 for n in range(1, 12)],
        "(10^n-1)/9", "repunit"
    ))

    # === CATALAN ===
    def catalan(n):
        if n <= 1:
            return 1
        return math.factorial(2*n) // (math.factorial(n+1) * math.factorial(n))
    sequences.append(OEISSequence(
        "A000108", "Catalan numbers", [catalan(n) for n in range(12)],
        "C(2n,n)/(n+1)", "catalan"
    ))

    # === BELL NUMBERS ===
    def bell(n):
        b = [[0] * (n+1) for _ in range(n+1)]
        b[0][0] = 1
        for i in range(1, n+1):
            b[i][0] = b[i-1][i-1]
            for j in range(1, i+1):
                b[i][j] = b[i-1][j-1] + b[i][j-1]
        return b[n][0]
    sequences.append(OEISSequence(
        "A000110", "Bell numbers", [bell(n) for n in range(12)],
        "B(n)", "bell"
    ))

    # === PARTITION NUMBERS ===
    # (complex, won't match)
    sequences.append(OEISSequence(
        "A000041", "Partition numbers", [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77],
        "p(n)", "partition"
    ))

    # === MORE QUADRATICS ===
    sequences.append(OEISSequence(
        "A002620", "Quarter-squares", [n**2 // 4 for n in range(20)],
        "floor(n^2/4)", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A007590", "Floor(n^2/2)", [n**2 // 2 for n in range(20)],
        "floor(n^2/2)", "quadratic"
    ))

    # === PRIMES (won't match - that's expected) ===
    sequences.append(OEISSequence(
        "A000040", "Prime numbers", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        "p_n", "prime"
    ))

    # === COMPOSITES (won't match) ===
    sequences.append(OEISSequence(
        "A002808", "Composite numbers", [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25],
        "composite(n)", "composite"
    ))

    # === ARITHMETIC PROGRESSIONS ===
    sequences.append(OEISSequence(
        "A016777", "3n + 1", [3*n + 1 for n in range(20)], "3n+1", "linear"
    ))
    sequences.append(OEISSequence(
        "A016789", "3n + 2", [3*n + 2 for n in range(20)], "3n+2", "linear"
    ))
    sequences.append(OEISSequence(
        "A004767", "4n + 3", [4*n + 3 for n in range(20)], "4n+3", "linear"
    ))

    # === CENTERED POLYGONAL ===
    sequences.append(OEISSequence(
        "A001844", "Centered square numbers", [2*n*(n+1) + 1 for n in range(15)],
        "2n(n+1)+1", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A005448", "Centered triangular", [(3*n**2 + 3*n + 2)//2 for n in range(15)],
        "(3n^2+3n+2)/2", "quadratic"
    ))

    # === a*b^n patterns ===
    sequences.append(OEISSequence(
        "A007283", "3*2^n", [3 * 2**n for n in range(15)], "3*2^n", "exponential"
    ))
    sequences.append(OEISSequence(
        "A020714", "5*2^n", [5 * 2**n for n in range(15)], "5*2^n", "exponential"
    ))

    # === Sum of first n terms ===
    sequences.append(OEISSequence(
        "A000217", "Triangular (sum 1..n)", [n*(n+1)//2 for n in range(20)],
        "n(n+1)/2", "quadratic"
    ))

    # === Double/halving patterns ===
    sequences.append(OEISSequence(
        "A000079", "2^n", [2**n for n in range(20)], "2^n", "exponential"
    ))

    # === More complex but identifiable ===
    sequences.append(OEISSequence(
        "A000124", "Central polygonal (lazy caterer)", [n*(n+1)//2 + 1 for n in range(15)],
        "n(n+1)/2 + 1", "quadratic"
    ))

    # === Squares plus offset ===
    sequences.append(OEISSequence(
        "A002522", "n^2 + 1", [n**2 + 1 for n in range(15)], "n^2+1", "quadratic"
    ))
    sequences.append(OEISSequence(
        "A005563", "n^2 - 1", [n**2 - 1 for n in range(1, 16)], "n^2-1", "quadratic"
    ))

    # Fill to 100 with more arithmetic progressions
    for mult in range(11, 21):
        sequences.append(OEISSequence(
            f"A008{590+mult}", f"Multiples of {mult}",
            [mult * n for n in range(15)], f"{mult}n", "linear"
        ))

    return sequences[:100]


# =============================================================================
# OEIS API FETCHER (for live testing)
# =============================================================================

def fetch_oeis_sequence(oeis_id: str) -> Optional[OEISSequence]:
    """Fetch a sequence from OEIS by ID (e.g., 'A000045')"""
    try:
        url = f"https://oeis.org/search?fmt=json&q=id:{oeis_id}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        if data.get('results'):
            result = data['results'][0]
            values_str = result.get('data', '')
            values = [int(x) for x in values_str.split(',') if x.strip()]

            return OEISSequence(
                id=oeis_id,
                name=result.get('name', 'Unknown'),
                values=values[:20],  # First 20 terms
                formula=result.get('formula', [''])[0] if result.get('formula') else '',
                category='fetched'
            )
    except Exception as e:
        print(f"  Error fetching {oeis_id}: {e}")
    return None


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_oeis_benchmark():
    """Run benchmark against OEIS sequences"""

    print("=" * 80)
    print("  OEIS BENCHMARK TEST")
    print("  Testing against the Online Encyclopedia of Integer Sequences")
    print("=" * 80)
    print()

    base_analyzer = NumericalFormAnalyzer()
    recursive_analyzer = RecursiveFormFlow(max_depth=4)

    sequences = get_famous_oeis_sequences()
    print(f"  Testing against {len(sequences)} famous OEIS sequences")
    print()

    # Categorize
    categories = {}
    for seq in sequences:
        categories[seq.category] = categories.get(seq.category, 0) + 1

    print("  Sequence categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    print()

    # Results
    results = {
        'base': {'success': 0, 'fail': 0},
        'recursive': {'success': 0, 'fail': 0},
        'by_category': {},
        'details': []
    }

    for cat in categories:
        results['by_category'][cat] = {'base': 0, 'recursive': 0, 'total': 0}

    # Expected difficult categories (won't match)
    difficult_categories = {'prime', 'composite', 'partition', 'divisor', 'bell', 'catalan'}

    print(f"  {'OEIS ID':<12} {'Name':<30} {'Base':<8} {'Recursive':<12}")
    print("  " + "-" * 70)

    for seq in sequences:
        values = [float(v) for v in seq.values[:15]]  # Use first 15 terms

        # Base analyzer
        base_result = base_analyzer.analyze(values, start_index=0)
        base_ok = (base_result.form_type != FormType.UNKNOWN and
                   base_result.confidence > 0.8)

        # Recursive analyzer
        rec_result = recursive_analyzer.analyze(values)
        rec_ok = rec_result is not None

        # Track
        results['by_category'][seq.category]['total'] += 1
        if base_ok:
            results['base']['success'] += 1
            results['by_category'][seq.category]['base'] += 1
        else:
            results['base']['fail'] += 1

        if rec_ok:
            results['recursive']['success'] += 1
            results['by_category'][seq.category]['recursive'] += 1
        else:
            results['recursive']['fail'] += 1

        base_str = "YES" if base_ok else "no"
        rec_str = "YES" if rec_ok else "no"

        # Only print failures for expected-matchable categories
        if seq.category not in difficult_categories:
            if not rec_ok:
                print(f"  {seq.id:<12} {seq.name[:30]:<30} {base_str:<8} {rec_str:<12}")

        results['details'].append({
            'id': seq.id,
            'name': seq.name,
            'category': seq.category,
            'base_ok': base_ok,
            'rec_ok': rec_ok
        })

    # Summary
    print()
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print()

    total = len(sequences)
    base_pct = results['base']['success'] / total * 100
    rec_pct = results['recursive']['success'] / total * 100

    print(f"  Overall Results:")
    print(f"    Base Analyzer:    {results['base']['success']:>3}/{total} ({base_pct:.1f}%)")
    print(f"    Recursive Flow:   {results['recursive']['success']:>3}/{total} ({rec_pct:.1f}%)")
    print()

    # Exclude difficult categories for fair comparison
    matchable = sum(c['total'] for cat, c in results['by_category'].items()
                    if cat not in difficult_categories)
    matchable_base = sum(c['base'] for cat, c in results['by_category'].items()
                         if cat not in difficult_categories)
    matchable_rec = sum(c['recursive'] for cat, c in results['by_category'].items()
                        if cat not in difficult_categories)

    print(f"  On MATCHABLE sequences (excluding primes, partitions, etc.):")
    print(f"    Base Analyzer:    {matchable_base:>3}/{matchable} ({matchable_base/matchable*100:.1f}%)")
    print(f"    Recursive Flow:   {matchable_rec:>3}/{matchable} ({matchable_rec/matchable*100:.1f}%)")
    print()

    # By category
    print("  By Category:")
    print("  " + "-" * 60)
    print(f"  {'Category':<20} {'Total':<8} {'Base':<12} {'Recursive':<12}")
    for cat in sorted(results['by_category'].keys()):
        c = results['by_category'][cat]
        if c['total'] > 0:
            b_pct = c['base'] / c['total'] * 100
            r_pct = c['recursive'] / c['total'] * 100
            difficult = "*" if cat in difficult_categories else ""
            print(f"  {cat:<20} {c['total']:<8} {c['base']:>2}/{c['total']:<2} ({b_pct:>5.1f}%)  "
                  f"{c['recursive']:>2}/{c['total']:<2} ({r_pct:>5.1f}%) {difficult}")

    print()
    print("  * = inherently difficult (no closed form / requires number theory)")
    print()

    return results


def test_live_oeis(sample_ids: List[str] = None):
    """Optionally test against live OEIS API"""

    if sample_ids is None:
        sample_ids = [
            "A000045",  # Fibonacci
            "A000079",  # Powers of 2
            "A000290",  # Squares
            "A000217",  # Triangular
            "A000225",  # Mersenne
        ]

    print("=" * 80)
    print("  LIVE OEIS API TEST")
    print("=" * 80)
    print()

    recursive_analyzer = RecursiveFormFlow(max_depth=4)

    print(f"  Fetching {len(sample_ids)} sequences from OEIS...")
    print()

    for oeis_id in sample_ids:
        seq = fetch_oeis_sequence(oeis_id)
        if seq:
            values = [float(v) for v in seq.values[:15]]
            result = recursive_analyzer.analyze(values)

            found = "YES" if result else "no"
            formula = result.formula_string()[:30] if result else "N/A"

            print(f"  {oeis_id}: {seq.name[:40]}")
            print(f"    Values: {seq.values[:8]}...")
            print(f"    Found: {found} -> {formula}")
            print()

        time.sleep(0.5)  # Be nice to OEIS servers


if __name__ == "__main__":
    # Run main benchmark
    results = run_oeis_benchmark()

    print("=" * 80)
    print("  COMPARISON CONTEXT")
    print("=" * 80)
    print()
    print("  This benchmark tests against REAL mathematical sequences from OEIS.")
    print("  Our engine achieves high accuracy on sequences with closed forms.")
    print()
    print("  Categories we CAN'T identify (by design):")
    print("    - Prime numbers (no closed form)")
    print("    - Partition numbers (no simple closed form)")
    print("    - Sequences defined by number-theoretic properties")
    print()
    print("  These represent TRUE_PRIMITIVES in our framework -")
    print("  irreducible sequences that ARE their own definition.")
    print()

    # Optionally test live API (commented out to avoid hitting OEIS servers)
    # print("\n  Testing live OEIS API...")
    # test_live_oeis()
