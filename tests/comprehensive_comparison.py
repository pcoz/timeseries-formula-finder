"""
Comprehensive comparison of sequence detection methods.

Compares:
1. Our Framework (validated) - with validation layer
2. Our Framework (not validated) - raw detection
3. Sympy pattern matching
4. Numpy polynomial fitting
5. Simple difference/ratio detection
"""

import sys
sys.path.insert(0, '..')

import time
import numpy as np
from typing import List, Tuple
import sympy
from sympy import symbols, simplify, Poly, factorial, fibonacci
from sympy.polys.polytools import degree

try:
    from core.recursive_form_flow import RecursiveFormFlow
    from core.form_validator import FormValidator, ValidationResult
except ImportError:
    sys.path.insert(0, '../core')
    from recursive_form_flow import RecursiveFormFlow
    from form_validator import FormValidator, ValidationResult


# Test sequences with KNOWN ground truth
TEST_SEQUENCES = {
    'has_form': [
        ('linear_1', [1,2,3,4,5,6,7,8,9,10,11,12]),
        ('linear_2', [0,2,4,6,8,10,12,14,16,18,20,22]),
        ('linear_3', [5,8,11,14,17,20,23,26,29,32,35,38]),
        ('squares', [0,1,4,9,16,25,36,49,64,81,100,121]),
        ('cubes', [0,1,8,27,64,125,216,343,512,729,1000,1331]),
        ('triangular', [0,1,3,6,10,15,21,28,36,45,55,66]),
        ('powers_2', [1,2,4,8,16,32,64,128,256,512,1024,2048]),
        ('powers_3', [1,3,9,27,81,243,729,2187,6561,19683,59049,177147]),
        ('fibonacci', [0,1,1,2,3,5,8,13,21,34,55,89]),
        ('factorial', [1,1,2,6,24,120,720,5040,40320,362880]),
    ],
    'no_form': [
        ('primes', [2,3,5,7,11,13,17,19,23,29,31,37]),
        ('divisor_count', [1,2,2,3,2,4,2,4,3,4,2,6]),
        ('totient', [1,1,2,2,4,2,6,4,6,4,10,4]),
        ('partition', [1,1,2,3,5,7,11,15,22,30,42,56]),
        ('pi_digits', [3,1,4,1,5,9,2,6,5,3,5,8]),
        ('divisor_sum', [1,3,4,7,6,12,8,15,13,18,12,28]),
        ('random_1', [47,23,91,15,82,36,58,74,29,63,11,88]),
        ('random_2', [3,14,15,92,65,35,89,79,32,38,46,26]),
    ]
}


def method_our_validated(values: List[float]) -> Tuple[bool, str, float]:
    """Our framework with validation"""
    start = time.perf_counter()
    validator = FormValidator(strict_mode=True)
    result = validator.validate(values)
    elapsed = time.perf_counter() - start

    if result.validation == ValidationResult.TRUE_FORM:
        formula = result.form.formula_string() if result.form else 'unknown'
        return (True, formula[:30], elapsed)
    return (False, 'rejected', elapsed)


def method_our_raw(values: List[float]) -> Tuple[bool, str, float]:
    """Our framework without validation (raw detection)"""
    start = time.perf_counter()
    analyzer = RecursiveFormFlow(max_depth=4)
    result = analyzer.analyze(values)
    elapsed = time.perf_counter() - start

    if result:
        return (True, result.formula_string()[:30], elapsed)
    return (False, 'no_form', elapsed)


def method_sympy_pattern(values: List[float]) -> Tuple[bool, str, float]:
    """Sympy pattern matching - try to find polynomial or recognize sequence"""
    start = time.perf_counter()

    n = symbols('n')
    ints = [int(v) for v in values]

    # Try polynomial interpolation
    try:
        from sympy import interpolate
        points = [(i, ints[i]) for i in range(len(ints))]
        poly = interpolate(points, n)

        # Check if it's a "simple" polynomial (degree <= 4)
        poly_simplified = simplify(poly)
        if poly_simplified.is_polynomial(n):
            deg = degree(Poly(poly_simplified, n))
            if deg <= 4:
                elapsed = time.perf_counter() - start
                return (True, f'poly_deg_{deg}', elapsed)
    except:
        pass

    # Check for known sequences
    # Fibonacci
    try:
        is_fib = True
        for i in range(2, min(10, len(ints))):
            if ints[i] != ints[i-1] + ints[i-2]:
                is_fib = False
                break
        if is_fib and len(ints) >= 3:
            elapsed = time.perf_counter() - start
            return (True, 'fibonacci', elapsed)
    except:
        pass

    # Factorial
    try:
        is_fact = True
        expected = 1
        for i, v in enumerate(ints):
            if i > 0:
                expected *= i
            if v != expected:
                is_fact = False
                break
        if is_fact and len(ints) >= 3:
            elapsed = time.perf_counter() - start
            return (True, 'factorial', elapsed)
    except:
        pass

    elapsed = time.perf_counter() - start
    return (False, 'no_pattern', elapsed)


def method_numpy_polyfit(values: List[float], max_degree: int = 5) -> Tuple[bool, str, float]:
    """Numpy polynomial fitting"""
    start = time.perf_counter()

    x = np.arange(len(values))
    y = np.array(values)

    for degree in range(1, max_degree + 1):
        try:
            coeffs = np.polyfit(x, y, degree)
            predicted = np.polyval(coeffs, x)
            error = np.max(np.abs(predicted - y))

            if error < 1e-6:
                elapsed = time.perf_counter() - start
                return (True, f'poly_deg_{degree}', elapsed)
        except:
            continue

    elapsed = time.perf_counter() - start
    return (False, 'no_fit', elapsed)


def method_simple_detection(values: List[float]) -> Tuple[bool, str, float]:
    """Simple difference/ratio detection"""
    start = time.perf_counter()

    # Constant
    if len(set(values)) == 1:
        return (True, 'constant', time.perf_counter() - start)

    # Linear
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    if len(set(round(d, 6) for d in diffs)) == 1:
        return (True, 'linear', time.perf_counter() - start)

    # Quadratic
    if len(diffs) >= 2:
        d2 = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
        if len(set(round(d, 6) for d in d2)) == 1:
            return (True, 'quadratic', time.perf_counter() - start)

    # Exponential
    if all(v > 0 for v in values):
        ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
        if len(set(round(r, 6) for r in ratios)) == 1:
            return (True, 'exponential', time.perf_counter() - start)

    # Fibonacci
    is_fib = True
    for i in range(2, len(values)):
        if abs(values[i] - (values[i-1] + values[i-2])) > 1e-6:
            is_fib = False
            break
    if is_fib:
        return (True, 'fibonacci', time.perf_counter() - start)

    return (False, 'unknown', time.perf_counter() - start)


def run_comparison():
    """Run comprehensive comparison"""

    print("=" * 90)
    print("  COMPREHENSIVE METHOD COMPARISON")
    print("=" * 90)
    print()

    methods = {
        'Our Framework (validated)': method_our_validated,
        'Our Framework (raw)': method_our_raw,
        'Sympy Pattern': method_sympy_pattern,
        'Numpy Polyfit': method_numpy_polyfit,
        'Simple Detection': method_simple_detection,
    }

    results = {name: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'time': 0.0}
               for name in methods}

    all_tests = []
    for name, values in TEST_SEQUENCES['has_form']:
        all_tests.append((name, values, True))
    for name, values in TEST_SEQUENCES['no_form']:
        all_tests.append((name, values, False))

    # Per-sequence results
    print(f"  {'Sequence':<15} {'Has Form':<10}", end='')
    for name in methods:
        print(f" {name[:10]:<10}", end='')
    print()
    print("  " + "-" * 80)

    for seq_name, values, has_form in all_tests:
        float_values = [float(v) for v in values]

        print(f"  {seq_name:<15} {'YES' if has_form else 'NO':<10}", end='')

        for method_name, method_func in methods.items():
            found, form, elapsed = method_func(float_values)
            results[method_name]['time'] += elapsed

            if has_form and found:
                results[method_name]['tp'] += 1
                status = 'TP'
            elif has_form and not found:
                results[method_name]['fn'] += 1
                status = 'FN'
            elif not has_form and found:
                results[method_name]['fp'] += 1
                status = 'FP!'
            else:
                results[method_name]['tn'] += 1
                status = 'TN'

            print(f" {status:<10}", end='')
        print()

    print("  " + "-" * 80)
    print()

    # Summary table
    print("=" * 90)
    print("  SUMMARY STATISTICS")
    print("=" * 90)
    print()
    print(f"  {'Method':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'FP Rate':>10} {'Time':>10}")
    print("  " + "-" * 80)

    for method_name, r in results.items():
        total = r['tp'] + r['tn'] + r['fp'] + r['fn']
        accuracy = (r['tp'] + r['tn']) / total * 100 if total > 0 else 0
        precision = r['tp'] / (r['tp'] + r['fp']) * 100 if (r['tp'] + r['fp']) > 0 else 100
        recall = r['tp'] / (r['tp'] + r['fn']) * 100 if (r['tp'] + r['fn']) > 0 else 0
        fp_rate = r['fp'] / (r['fp'] + r['tn']) * 100 if (r['fp'] + r['tn']) > 0 else 0

        print(f"  {method_name:<30} {accuracy:>9.1f}% {precision:>9.1f}% {recall:>9.1f}% {fp_rate:>9.1f}% {r['time']*1000:>8.1f}ms")

    print()

    # Analysis
    print("=" * 90)
    print("  ANALYSIS")
    print("=" * 90)
    print()
    print("  KEY METRICS:")
    print()

    # Find best by each metric
    best_accuracy = max(results.items(), key=lambda x: (x[1]['tp'] + x[1]['tn']) / (x[1]['tp'] + x[1]['tn'] + x[1]['fp'] + x[1]['fn']))
    best_precision = min(results.items(), key=lambda x: x[1]['fp'])  # Lowest FP
    best_speed = min(results.items(), key=lambda x: x[1]['time'])

    print(f"  Best Accuracy:    {best_accuracy[0]}")
    print(f"  Best Precision:   {best_precision[0]} (lowest false positives)")
    print(f"  Fastest:          {best_speed[0]}")
    print()

    print("  CRITICAL INSIGHT:")
    print("  -----------------")
    print("  FALSE POSITIVE RATE is the most important metric for real applications.")
    print("  A system that claims patterns exist when they don't is dangerous.")
    print()

    fp_rates = {name: r['fp'] / (r['fp'] + r['tn']) * 100 if (r['fp'] + r['tn']) > 0 else 0
                for name, r in results.items()}

    print("  False Positive Rates:")
    for name, rate in sorted(fp_rates.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if rate == 0 else ""
        print(f"    {name:<30}: {rate:>5.1f}%{marker}")

    print()
    return results


def speed_benchmark(n_iterations: int = 50):
    """Speed benchmark on a standard sequence"""

    print()
    print("=" * 90)
    print("  SPEED BENCHMARK")
    print("=" * 90)
    print()

    test_seq = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0]

    methods = {
        'Our Framework (validated)': lambda: method_our_validated(test_seq),
        'Our Framework (raw)': lambda: method_our_raw(test_seq),
        'Sympy Pattern': lambda: method_sympy_pattern(test_seq),
        'Numpy Polyfit': lambda: method_numpy_polyfit(test_seq),
        'Simple Detection': lambda: method_simple_detection(test_seq),
    }

    print(f"  Running {n_iterations} iterations on squares sequence [0,1,4,9,16,...]")
    print()

    for name, func in methods.items():
        start = time.perf_counter()
        for _ in range(n_iterations):
            func()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / n_iterations * 1000
        ops_per_sec = n_iterations / elapsed

        print(f"  {name:<30}: {avg_ms:>8.3f} ms/call  ({ops_per_sec:>8.0f} ops/sec)")

    print()


if __name__ == "__main__":
    run_comparison()
    speed_benchmark()
