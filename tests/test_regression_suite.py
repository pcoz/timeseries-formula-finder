"""
Regression Test Suite
=====================

Captures current behavior before making changes.
Run before and after modifications to ensure nothing breaks.

Usage:
    python tests/test_regression_suite.py > baseline.txt
    # Make changes
    python tests/test_regression_suite.py > after.txt
    # Compare results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.super_dimensional_completeness import (
    DerivativeTower,
    GeneratingFunction,
    SequenceSignature,
)
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import json


@dataclass
class RegressionTestCase:
    """A single regression test case"""
    name: str
    sequence: List[float]
    category: str


@dataclass
class RegressionResult:
    """Result of running a regression test"""
    name: str
    sequence: List[float]
    category: str
    path_found: bool
    path: Optional[List[str]]
    is_constant: bool
    is_linear: bool
    is_quadratic: bool
    is_exponential: bool
    is_fibonacci: bool
    polynomial_degree: Optional[int]
    exponential_base: Optional[float]


def create_regression_test_cases() -> List[RegressionTestCase]:
    """Create comprehensive test cases covering all current functionality"""

    cases = []

    # ===================
    # CONSTANTS
    # ===================
    for c in [0, 1, -1, 5, 3.14, 1e6, 1e-6]:
        cases.append(RegressionTestCase(
            name=f"const_{c}",
            sequence=[float(c)] * 10,
            category="constant"
        ))

    # ===================
    # LINEAR: an + b
    # ===================
    linear_params = [
        (1, 0), (2, 0), (-1, 0), (0.5, 0),
        (1, 1), (2, 3), (-2, 5), (1, -1),
        (1000, 0), (0.001, 0),
    ]
    for a, b in linear_params:
        cases.append(RegressionTestCase(
            name=f"linear_{a}n+{b}",
            sequence=[float(a*n + b) for n in range(10)],
            category="linear"
        ))

    # ===================
    # QUADRATIC: an^2 + bn + c
    # ===================
    quad_params = [
        (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1),
        (2, 0, 0), (2, 3, 1), (-1, 0, 0), (0.5, 0, 0),
    ]
    for a, b, c in quad_params:
        cases.append(RegressionTestCase(
            name=f"quad_{a}n2+{b}n+{c}",
            sequence=[float(a*n**2 + b*n + c) for n in range(10)],
            category="quadratic"
        ))

    # ===================
    # CUBIC: an^3 + bn^2 + cn + d
    # ===================
    cubic_params = [
        (1, 0, 0, 0), (1, 1, 1, 1), (2, -1, 3, 0),
    ]
    for a, b, c, d in cubic_params:
        cases.append(RegressionTestCase(
            name=f"cubic_{a}n3+{b}n2+{c}n+{d}",
            sequence=[float(a*n**3 + b*n**2 + c*n + d) for n in range(10)],
            category="cubic"
        ))

    # ===================
    # HIGHER POLYNOMIALS
    # ===================
    cases.append(RegressionTestCase(
        name="poly_n4",
        sequence=[float(n**4) for n in range(10)],
        category="polynomial"
    ))
    cases.append(RegressionTestCase(
        name="poly_n5",
        sequence=[float(n**5) for n in range(10)],
        category="polynomial"
    ))

    # ===================
    # EXPONENTIAL: a * r^n
    # ===================
    exp_params = [
        (1, 2), (1, 3), (1, 0.5), (1, 1.5),
        (2, 2), (5, 2), (1, 10), (1, 1.01), (1, 0.99),
        (1, -2), (1, -1.5),  # Negative bases
    ]
    for a, r in exp_params:
        cases.append(RegressionTestCase(
            name=f"exp_{a}*{r}^n",
            sequence=[float(a * r**n) for n in range(10)],
            category="exponential"
        ))

    # ===================
    # OFFSET EXPONENTIAL: a * r^n + c
    # ===================
    offset_exp_params = [
        (1, 2, -1), (1, 2, 1), (1, 2, 5),
        (2, 3, -2), (1, 0.5, 10),
    ]
    for a, r, c in offset_exp_params:
        cases.append(RegressionTestCase(
            name=f"offset_exp_{a}*{r}^n+{c}",
            sequence=[float(a * r**n + c) for n in range(10)],
            category="offset_exponential"
        ))

    # ===================
    # SUM OF POLYNOMIAL AND EXPONENTIAL
    # ===================
    cases.append(RegressionTestCase(
        name="sum_2^n+n",
        sequence=[float(2**n + n) for n in range(10)],
        category="sum_poly_exp"
    ))
    cases.append(RegressionTestCase(
        name="sum_2^n+n^2",
        sequence=[float(2**n + n**2) for n in range(10)],
        category="sum_poly_exp"
    ))

    # ===================
    # FIBONACCI-LIKE
    # ===================
    fib = [1, 1]
    for _ in range(8):
        fib.append(fib[-1] + fib[-2])
    cases.append(RegressionTestCase(
        name="fibonacci_standard",
        sequence=[float(x) for x in fib],
        category="fibonacci"
    ))

    lucas = [2, 1]
    for _ in range(8):
        lucas.append(lucas[-1] + lucas[-2])
    cases.append(RegressionTestCase(
        name="fibonacci_lucas",
        sequence=[float(x) for x in lucas],
        category="fibonacci"
    ))

    # ===================
    # DEGENERATE CASES
    # ===================
    cases.append(RegressionTestCase(
        name="degenerate_quad_a0",
        sequence=[float(0*n**2 + 3*n + 2) for n in range(10)],
        category="degenerate"
    ))
    cases.append(RegressionTestCase(
        name="degenerate_linear_a0",
        sequence=[float(0*n + 5) for n in range(10)],
        category="degenerate"
    ))

    # ===================
    # PRIMITIVES (should NOT find paths)
    # ===================
    cases.append(RegressionTestCase(
        name="primitive_primes",
        sequence=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        category="primitive"
    ))
    cases.append(RegressionTestCase(
        name="primitive_pi_digits",
        sequence=[3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
        category="primitive"
    ))
    cases.append(RegressionTestCase(
        name="primitive_random",
        sequence=[7, 2, 9, 4, 1, 8, 3, 6, 5, 0],
        category="primitive"
    ))

    # ===================
    # EXCLUDED FORMS (should NOT find paths)
    # ===================
    cases.append(RegressionTestCase(
        name="excluded_n*2^n",
        sequence=[float(n * 2**n) for n in range(10)],
        category="excluded"
    ))
    cases.append(RegressionTestCase(
        name="excluded_n^2*2^n",
        sequence=[float(n**2 * 2**n) for n in range(10)],
        category="excluded"
    ))

    # ===================
    # FORMS THAT WOULD BENEFIT FROM LOG (currently excluded)
    # ===================
    # These should currently fail but might work after adding LOG
    cases.append(RegressionTestCase(
        name="log_candidate_n_log_n",
        sequence=[float(n * math.log(n+1)) for n in range(10)],  # n+1 to avoid log(0)
        category="log_candidate"
    ))
    cases.append(RegressionTestCase(
        name="log_candidate_log_n",
        sequence=[float(math.log(n+1)) for n in range(10)],
        category="log_candidate"
    ))
    cases.append(RegressionTestCase(
        name="log_candidate_n^2_log_n",
        sequence=[float(n**2 * math.log(n+1)) for n in range(10)],
        category="log_candidate"
    ))

    # ===================
    # EDGE CASES
    # ===================
    cases.append(RegressionTestCase(
        name="edge_all_zeros",
        sequence=[0.0] * 10,
        category="edge"
    ))
    cases.append(RegressionTestCase(
        name="edge_alternating",
        sequence=[float((-1)**n) for n in range(10)],
        category="edge"
    ))
    cases.append(RegressionTestCase(
        name="edge_large_values",
        sequence=[float(1e10 * n) for n in range(10)],
        category="edge"
    ))
    cases.append(RegressionTestCase(
        name="edge_small_values",
        sequence=[float(1e-10 * n) for n in range(10)],
        category="edge"
    ))

    return cases


def run_regression_test(case: RegressionTestCase) -> RegressionResult:
    """Run a single regression test and capture all relevant outputs"""

    sig = SequenceSignature.from_sequence(case.sequence, max_depth=6)

    return RegressionResult(
        name=case.name,
        sequence=case.sequence[:6],  # First 6 for display
        category=case.category,
        path_found=sig.transform_path is not None,
        path=sig.transform_path,
        is_constant=sig.is_constant,
        is_linear=sig.is_linear,
        is_quadratic=sig.is_quadratic,
        is_exponential=sig.is_exponential,
        is_fibonacci=sig.is_fibonacci,
        polynomial_degree=sig.polynomial_degree,
        exponential_base=sig.exponential_base,
    )


def run_all_regression_tests() -> Tuple[List[RegressionResult], dict]:
    """Run all regression tests and return results with summary"""

    cases = create_regression_test_cases()
    results = [run_regression_test(case) for case in cases]

    # Compute summary by category
    summary = {}
    for r in results:
        if r.category not in summary:
            summary[r.category] = {'total': 0, 'paths_found': 0}
        summary[r.category]['total'] += 1
        if r.path_found:
            summary[r.category]['paths_found'] += 1

    return results, summary


def print_results(results: List[RegressionResult], summary: dict):
    """Print results in a format suitable for comparison"""

    print("=" * 80)
    print("REGRESSION TEST RESULTS")
    print("=" * 80)
    print()

    # Group by category
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)

    for category, cat_results in sorted(by_category.items()):
        print(f"\n## {category.upper()}")
        print("-" * 60)

        for r in cat_results:
            path_str = ' -> '.join(r.path) if r.path else 'None'
            print(f"  {r.name}")
            print(f"    path_found: {r.path_found}")
            print(f"    path: {path_str}")
            print(f"    is_constant: {r.is_constant}")
            print(f"    is_linear: {r.is_linear}")
            print(f"    is_quadratic: {r.is_quadratic}")
            print(f"    is_exponential: {r.is_exponential}")
            print(f"    is_fibonacci: {r.is_fibonacci}")
            print(f"    polynomial_degree: {r.polynomial_degree}")
            print(f"    exponential_base: {r.exponential_base}")
            print()

    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)

    for category, stats in sorted(summary.items()):
        pct = stats['paths_found'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {category}: {stats['paths_found']}/{stats['total']} paths found ({pct:.1f}%)")

    total = sum(s['total'] for s in summary.values())
    total_found = sum(s['paths_found'] for s in summary.values())
    print(f"\n  TOTAL: {total_found}/{total} paths found ({total_found/total*100:.1f}%)")
    print()


def export_results_json(results: List[RegressionResult], filename: str):
    """Export results to JSON for programmatic comparison"""

    data = []
    for r in results:
        data.append({
            'name': r.name,
            'category': r.category,
            'path_found': r.path_found,
            'path': r.path,
            'is_constant': r.is_constant,
            'is_linear': r.is_linear,
            'is_quadratic': r.is_quadratic,
            'is_exponential': r.is_exponential,
            'is_fibonacci': r.is_fibonacci,
            'polynomial_degree': r.polynomial_degree,
            'exponential_base': r.exponential_base,
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results exported to {filename}")


def compare_results(before_file: str, after_file: str):
    """Compare two JSON result files and report differences"""

    with open(before_file) as f:
        before = {r['name']: r for r in json.load(f)}

    with open(after_file) as f:
        after = {r['name']: r for r in json.load(f)}

    print("=" * 80)
    print("REGRESSION COMPARISON")
    print("=" * 80)
    print()

    changes = []

    for name in before:
        if name not in after:
            changes.append(f"REMOVED: {name}")
            continue

        b, a = before[name], after[name]

        # Check for changes in key fields
        if b['path_found'] != a['path_found']:
            changes.append(f"CHANGED path_found: {name}: {b['path_found']} -> {a['path_found']}")
        elif b['path'] != a['path']:
            changes.append(f"CHANGED path: {name}: {b['path']} -> {a['path']}")

        for field in ['is_constant', 'is_linear', 'is_quadratic', 'is_exponential', 'is_fibonacci']:
            if b[field] != a[field]:
                changes.append(f"CHANGED {field}: {name}: {b[field]} -> {a[field]}")

    for name in after:
        if name not in before:
            changes.append(f"ADDED: {name}")

    if changes:
        print("DIFFERENCES FOUND:")
        for c in changes:
            print(f"  {c}")
    else:
        print("NO DIFFERENCES - regression test passed")

    print()
    return len(changes) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Regression test suite')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'),
                        help='Compare two JSON result files')
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        results, summary = run_all_regression_tests()
        print_results(results, summary)

        if args.export:
            export_results_json(results, args.export)
