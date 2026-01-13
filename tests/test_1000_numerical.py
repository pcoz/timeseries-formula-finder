"""
1000 Numerical Form Tests
=========================

Test the engine's ability to identify minimal forms from pure number sequences.

This tests the core thesis: given any set of numbers, can we find the
essential form (צורה) of which they are merely a representation?
"""

import sys
sys.path.insert(0, '..')

import random
import math
from typing import List, Tuple, Optional
from core.numerical_form_analyzer import (
    NumericalFormAnalyzer, NumericalForm, FormType
)


# =============================================================================
# TEST GENERATORS
# =============================================================================

def generate_constant_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate constant sequence tests"""
    tests = []
    for _ in range(count):
        c = random.randint(-100, 100)
        length = random.randint(5, 15)
        seq = [c] * length
        tests.append((seq, f"{c}", FormType.CONSTANT))
    return tests


def generate_arithmetic_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate arithmetic sequence tests: a + n*d"""
    tests = []
    for _ in range(count):
        a = random.randint(-50, 50)
        d = random.randint(-20, 20)
        if d == 0:
            d = random.choice([-1, 1]) * random.randint(1, 10)
        length = random.randint(5, 12)
        seq = [a + i * d for i in range(length)]
        tests.append((seq, f"arithmetic(a={a}, d={d})", FormType.ARITHMETIC_SEQ))
    return tests


def generate_geometric_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate geometric sequence tests: a * r^n"""
    tests = []
    for _ in range(count):
        a = random.randint(1, 10)
        r = random.choice([2, 3, 0.5, 4, 5, 1.5, 2.5])
        length = random.randint(5, 10)
        seq = [a * (r ** i) for i in range(length)]
        tests.append((seq, f"geometric(a={a}, r={r})", FormType.GEOMETRIC_SEQ))
    return tests


def generate_linear_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate linear tests: an + b"""
    tests = []
    for _ in range(count):
        a = random.randint(1, 20)
        b = random.randint(-30, 30)
        length = random.randint(5, 12)
        start = random.randint(0, 3)
        seq = [a * (i + start) + b for i in range(length)]
        tests.append((seq, f"linear({a}*n + {b})", FormType.LINEAR))
    return tests


def generate_quadratic_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate quadratic tests: an² + bn + c"""
    tests = []
    for _ in range(count):
        a = random.randint(1, 5)
        b = random.randint(-10, 10)
        c = random.randint(-20, 20)
        length = random.randint(6, 10)
        start = random.randint(0, 2)
        seq = [a * (i+start)**2 + b * (i+start) + c for i in range(length)]
        tests.append((seq, f"quadratic({a}n² + {b}n + {c})", FormType.QUADRATIC))
    return tests


def generate_power_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate n^k tests (squares, cubes, etc.)"""
    tests = []
    for _ in range(count):
        k = random.choice([2, 2, 2, 3, 3, 4])  # Weight toward squares
        length = random.randint(5, 10)
        seq = [(i + 1) ** k for i in range(length)]
        tests.append((seq, f"n^{k}", FormType.POWER))
    return tests


def generate_exponential_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate exponential tests: a * b^n"""
    tests = []
    for _ in range(count):
        a = random.randint(1, 5)
        b = random.choice([2, 3, 4, 5, 10])
        length = random.randint(5, 8)
        seq = [a * (b ** i) for i in range(length)]
        tests.append((seq, f"{a} * {b}^n", FormType.EXPONENTIAL))
    return tests


def generate_triangular_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate triangular number tests: n(n+1)/2"""
    tests = []
    for _ in range(count):
        length = random.randint(6, 12)
        seq = [(i + 1) * (i + 2) // 2 for i in range(length)]
        tests.append((seq, "n(n+1)/2", FormType.TRIANGULAR))
    return tests


def generate_fibonacci_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate Fibonacci-like tests: f(n) = f(n-1) + f(n-2)"""
    tests = []
    for _ in range(count):
        f0 = random.randint(1, 5)
        f1 = random.randint(1, 5)
        length = random.randint(8, 15)
        seq = [f0, f1]
        for i in range(2, length):
            seq.append(seq[-1] + seq[-2])
        tests.append((seq, f"fib(f0={f0}, f1={f1})", FormType.FIBONACCI))
    return tests


def generate_factorial_tests(count: int) -> List[Tuple[List[float], str, FormType]]:
    """Generate factorial tests: n!"""
    tests = []
    for _ in range(count):
        start = random.randint(0, 2)
        length = random.randint(5, 8)
        seq = [math.factorial(i + start) for i in range(length)]
        tests.append((seq, f"(n+{start})!" if start > 0 else "n!", FormType.FACTORIAL))
    return tests


def generate_all_tests(total: int = 1000) -> List[Tuple[List[float], str, FormType]]:
    """Generate all 1000 tests across categories"""
    tests = []

    # Distribution of test types (roughly balanced)
    tests.extend(generate_constant_tests(80))
    tests.extend(generate_arithmetic_tests(150))
    tests.extend(generate_geometric_tests(100))
    tests.extend(generate_linear_tests(150))
    tests.extend(generate_quadratic_tests(150))
    tests.extend(generate_power_tests(100))
    tests.extend(generate_exponential_tests(100))
    tests.extend(generate_triangular_tests(60))
    tests.extend(generate_fibonacci_tests(60))
    tests.extend(generate_factorial_tests(50))

    # Shuffle to mix categories
    random.shuffle(tests)

    return tests[:total]


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_1000_tests():
    """Run 1000 numerical form identification tests"""
    print("=" * 80)
    print("  1000 NUMERICAL FORM IDENTIFICATION TESTS")
    print("=" * 80)
    print()
    print("  Testing: Can the engine identify the minimal form (essential structure)")
    print("  from pure number sequences?")
    print()

    random.seed(42)  # Reproducible results
    analyzer = NumericalFormAnalyzer()

    # Generate tests
    tests = generate_all_tests(1000)

    print(f"  Generated {len(tests)} test sequences")
    print()

    # Results tracking
    results = {
        'total': 0,
        'identified': 0,
        'verified': 0,
        'by_type': {},
        'complexity_savings': [],
        'failures': []
    }

    for form_type in FormType:
        results['by_type'][form_type.value] = {'total': 0, 'identified': 0, 'verified': 0}

    # Run tests
    for i, (sequence, expected_desc, expected_type) in enumerate(tests):
        results['total'] += 1
        results['by_type'][expected_type.value]['total'] += 1

        # Analyze
        form = analyzer.analyze(sequence, start_index=0)

        # Check identification (form type matches or is acceptable equivalent)
        type_match = (
            form.form_type == expected_type or
            # Mathematically valid equivalences:
            (expected_type == FormType.ARITHMETIC_SEQ and form.form_type == FormType.LINEAR) or
            (expected_type == FormType.GEOMETRIC_SEQ and form.form_type == FormType.EXPONENTIAL) or
            (expected_type == FormType.LINEAR and form.form_type == FormType.ARITHMETIC_SEQ) or
            (expected_type == FormType.EXPONENTIAL and form.form_type == FormType.GEOMETRIC_SEQ) or
            # n^2 IS quadratic (special case with b=0, c=0)
            (expected_type == FormType.POWER and form.form_type == FormType.QUADRATIC) or
            # triangular n(n+1)/2 IS quadratic (0.5n^2 + 0.5n)
            (expected_type == FormType.TRIANGULAR and form.form_type == FormType.QUADRATIC)
        )

        if type_match and form.confidence > 0.8:
            results['identified'] += 1
            results['by_type'][expected_type.value]['identified'] += 1

            # Verify
            if analyzer.verify(sequence, form):
                results['verified'] += 1
                results['by_type'][expected_type.value]['verified'] += 1

                # Track complexity savings
                savings = len(sequence) - form.complexity
                results['complexity_savings'].append(savings)
            else:
                results['failures'].append({
                    'sequence': sequence[:5],
                    'expected': expected_desc,
                    'got': str(form),
                    'reason': 'verification failed'
                })
        else:
            results['failures'].append({
                'sequence': sequence[:5],
                'expected': expected_desc,
                'expected_type': expected_type.value,
                'got': str(form),
                'got_type': form.form_type.value,
                'confidence': form.confidence,
                'reason': 'type mismatch' if not type_match else 'low confidence'
            })

        # Progress
        if (i + 1) % 200 == 0:
            pct = results['identified'] / results['total'] * 100
            print(f"  [{i+1:4d}/1000] {results['identified']} identified ({pct:.1f}%)")

    # Final summary
    print()
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print()

    id_pct = results['identified'] / results['total'] * 100
    ver_pct = results['verified'] / results['total'] * 100

    print(f"  Form Identification: {results['identified']}/{results['total']} ({id_pct:.1f}%)")
    print(f"  Verified Correct:    {results['verified']}/{results['total']} ({ver_pct:.1f}%)")
    print()

    print("  By Form Type:")
    print("  " + "-" * 60)
    for form_type in FormType:
        stats = results['by_type'].get(form_type.value, {'total': 0, 'identified': 0, 'verified': 0})
        if stats['total'] > 0:
            id_pct = stats['identified'] / stats['total'] * 100
            ver_pct = stats['verified'] / stats['total'] * 100
            bar = "#" * int(ver_pct / 5)
            print(f"    {form_type.value:<15}: {stats['verified']:>3}/{stats['total']:<3} "
                  f"({ver_pct:5.1f}%) {bar}")

    # Complexity analysis
    if results['complexity_savings']:
        avg_savings = sum(results['complexity_savings']) / len(results['complexity_savings'])
        max_savings = max(results['complexity_savings'])
        total_savings = sum(results['complexity_savings'])

        print()
        print("  Complexity Reduction:")
        print("  " + "-" * 60)
        print(f"    Average savings per sequence: {avg_savings:.1f} elements")
        print(f"    Maximum savings: {max_savings} elements")
        print(f"    Total savings across all tests: {total_savings} elements")

        # Calculate compression ratio
        total_elements = sum(len(t[0]) for t in tests[:results['verified']])
        total_form_complexity = total_elements - total_savings
        compression = (1 - total_form_complexity / total_elements) * 100
        print(f"    Compression ratio: {compression:.1f}%")

    # Show some failures
    if results['failures']:
        print()
        print("  Sample Failures (first 10):")
        print("  " + "-" * 60)
        for f in results['failures'][:10]:
            print(f"    {f['sequence']}... expected {f.get('expected_type', f['expected'])}")
            print(f"         got {f.get('got_type', f['got'])} ({f.get('reason', '')})")

    print()
    print("=" * 80)
    if results['verified'] >= 900:
        print(f"  SUCCESS: {results['verified']}/1000 forms correctly identified and verified!")
    else:
        print(f"  RESULT: {results['verified']}/1000 forms verified ({ver_pct:.1f}%)")
    print("=" * 80)

    return results


def run_complexity_analysis():
    """Analyze how much complexity reduction the form identification achieves"""
    print()
    print("=" * 80)
    print("  COMPLEXITY ANALYSIS: FORM VS RAW REPRESENTATION")
    print("=" * 80)
    print()

    analyzer = NumericalFormAnalyzer()

    # Specific examples
    examples = [
        ([i * 2 for i in range(100)], "100 even numbers"),
        ([i ** 2 for i in range(1, 51)], "50 squares"),
        ([2 ** i for i in range(20)], "20 powers of 2"),
        ([math.factorial(i) for i in range(10)], "10 factorials"),
        ([1, 1] + [0] * 18, "Fibonacci seed"),  # Will compute rest
    ]

    # Fix fibonacci
    fib = [1, 1]
    for _ in range(18):
        fib.append(fib[-1] + fib[-2])
    examples[4] = (fib, "20 Fibonacci numbers")

    print(f"  {'Sequence':<25} {'Raw':<8} {'Form':<8} {'Reduction':<12} {'Formula'}")
    print("  " + "-" * 80)

    total_raw = 0
    total_form = 0

    for seq, desc in examples:
        form = analyzer.analyze(seq, start_index=0)
        raw_complexity = len(seq)
        form_complexity = form.complexity

        total_raw += raw_complexity
        total_form += form_complexity

        reduction = (1 - form_complexity / raw_complexity) * 100
        print(f"  {desc:<25} {raw_complexity:<8} {form_complexity:<8} {reduction:>5.1f}%       {form.formula[:30]}")

    print("  " + "-" * 80)
    overall_reduction = (1 - total_form / total_raw) * 100
    print(f"  {'TOTAL':<25} {total_raw:<8} {total_form:<8} {overall_reduction:>5.1f}%")
    print()
    print(f"  The essential form requires {overall_reduction:.1f}% less information than raw storage.")
    print(f"  This is the difference between storing {total_raw} numbers vs {total_form} parameters.")
    print()

    return {'total_raw': total_raw, 'total_form': total_form, 'reduction': overall_reduction}


def demonstrate_form_equivalence():
    """Show that different number sequences can have the same underlying form"""
    print()
    print("=" * 80)
    print("  FORM EQUIVALENCE: SAME ESSENCE, DIFFERENT MANIFESTATION")
    print("=" * 80)
    print()

    analyzer = NumericalFormAnalyzer()

    # Different arithmetic sequences with same form structure
    groups = [
        {
            'name': "Arithmetic sequences (different parameters, same form)",
            'sequences': [
                [2, 4, 6, 8, 10],
                [5, 10, 15, 20, 25],
                [100, 103, 106, 109, 112],
                [-10, -7, -4, -1, 2],
            ]
        },
        {
            'name': "Quadratic sequences (different parameters, same form)",
            'sequences': [
                [1, 4, 9, 16, 25],       # n²
                [2, 8, 18, 32, 50],      # 2n²
                [0, 3, 8, 15, 24],       # n² - 1
                [1, 5, 13, 25, 41],      # 2n² - n
            ]
        },
        {
            'name': "Exponential sequences (different bases, same form)",
            'sequences': [
                [1, 2, 4, 8, 16],        # 2^n
                [1, 3, 9, 27, 81],       # 3^n
                [2, 6, 18, 54, 162],     # 2 * 3^n
            ]
        }
    ]

    for group in groups:
        print(f"  {group['name']}:")
        print()
        for seq in group['sequences']:
            form = analyzer.analyze(seq, start_index=0)
            print(f"    {str(seq):<30} -> {form.form_type.value}: {form.formula}")
        print()

    print("  Key insight: The FORM TYPE is the essential structure.")
    print("  The parameters are accidental properties that vary.")
    print()


if __name__ == "__main__":
    # Run main 1000 tests
    results = run_1000_tests()

    # Show complexity analysis
    complexity = run_complexity_analysis()

    # Demonstrate form equivalence
    demonstrate_form_equivalence()

    print("=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    print()
    print("  The numerical form analyzer demonstrates that:")
    print()
    print("  1. FORM IDENTIFICATION: Given raw numbers, we can identify the")
    print("     minimal generating form (the essential structure).")
    print()
    print("  2. COMPLEXITY REDUCTION: The form representation requires")
    print(f"     ~{complexity['reduction']:.0f}% less information than raw storage.")
    print()
    print("  3. FORM EQUIVALENCE: Different number sequences can share the")
    print("     same essential form with different parameters.")
    print()
    print("  This validates the thesis: numbers are representations of forms,")
    print("  and we can algorithmically navigate to the essential form.")
    print()
