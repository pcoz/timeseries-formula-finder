"""
Validated Benchmark Test
========================

Tests the form validator's ability to DISCRIMINATE between:
- TRUE closed forms (should validate)
- Coincidental fits (should reject)
"""

import sys
sys.path.insert(0, '..')

import random
from typing import List, Tuple

try:
    from core.form_validator import FormValidator, ValidationResult
except ImportError:
    sys.path.insert(0, '../core')
    from form_validator import FormValidator, ValidationResult


def get_test_sequences():
    """Get sequences with known ground truth"""
    sequences = []

    # === TRUE FORMS (should validate) ===

    # Linear - definitely has closed form
    sequences.append(("linear_1", [1,2,3,4,5,6,7,8,9,10,11,12], True, "n"))
    sequences.append(("linear_2", [0,2,4,6,8,10,12,14,16,18,20,22], True, "2n"))
    sequences.append(("linear_3", [5,8,11,14,17,20,23,26,29,32,35,38], True, "3n+5"))

    # Quadratic - definitely has closed form
    sequences.append(("squares", [0,1,4,9,16,25,36,49,64,81,100,121], True, "n^2"))
    sequences.append(("triangular", [0,1,3,6,10,15,21,28,36,45,55,66], True, "n(n+1)/2"))

    # Exponential - definitely has closed form
    sequences.append(("pow2", [1,2,4,8,16,32,64,128,256,512,1024,2048], True, "2^n"))
    sequences.append(("pow3", [1,3,9,27,81,243,729,2187,6561,19683], True, "3^n"))

    # Fibonacci - has closed form (Binet's formula)
    sequences.append(("fib", [0,1,1,2,3,5,8,13,21,34,55,89], True, "fibonacci"))

    # Factorial - has closed form (product)
    sequences.append(("fact", [1,1,2,6,24,120,720,5040,40320,362880], True, "n!"))

    # === NO CLOSED FORM (should reject) ===

    # Primes - NO algebraic closed form exists
    sequences.append(("primes", [2,3,5,7,11,13,17,19,23,29,31,37], False, "primes"))

    # Digit sum - depends on base representation, not algebraic
    sequences.append(("digit_sum", [0,1,2,3,4,5,6,7,8,9,1,2], False, "digit sum"))

    # Number of divisors - no closed form
    sequences.append(("divisors", [1,2,2,3,2,4,2,4,3,4,2,6], False, "divisor count"))

    # Euler totient - no simple closed form
    sequences.append(("totient", [1,1,2,2,4,2,6,4,6,4,10,4], False, "euler phi"))

    # Digits of pi - transcendental, no form
    sequences.append(("pi_digits", [3,1,4,1,5,9,2,6,5,3,5,8], False, "pi digits"))

    # Random - no form by construction
    random.seed(123)
    sequences.append(("random_1", [random.randint(1,50) for _ in range(12)], False, "random"))
    random.seed(456)
    sequences.append(("random_2", [random.randint(1,100) for _ in range(12)], False, "random"))

    # Prime indicator (0/1) - no closed form
    sequences.append(("prime_char", [0,1,1,0,1,0,1,0,0,0,1,0], False, "prime indicator"))

    # Composite numbers - no closed form
    sequences.append(("composites", [4,6,8,9,10,12,14,15,16,18,20,21], False, "composites"))

    # Sum of proper divisors - no closed form
    sequences.append(("aliquot", [0,1,1,3,1,6,1,7,4,8,1,16], False, "aliquot sum"))

    return sequences


def run_validated_benchmark():
    """Run benchmark with validation"""

    print("=" * 80)
    print("  VALIDATED BENCHMARK: Testing Discrimination Ability")
    print("=" * 80)
    print()
    print("  This tests whether we can DISTINGUISH:")
    print("    - TRUE closed forms (should validate)")
    print("    - Coincidental/impossible forms (should reject)")
    print()

    sequences = get_test_sequences()
    validator = FormValidator(strict_mode=True)

    results = {
        'true_positives': 0,   # Has form, we validate
        'true_negatives': 0,   # No form, we reject
        'false_positives': 0,  # No form, but we validate (BAD)
        'false_negatives': 0,  # Has form, but we reject
    }

    print("  Testing sequences...")
    print("  " + "-" * 70)

    for name, values, has_form, description in sequences:
        result = validator.validate([float(v) for v in values])

        is_valid = result.validation == ValidationResult.TRUE_FORM
        correct = (is_valid == has_form)

        if has_form and is_valid:
            results['true_positives'] += 1
            status = "TP"
        elif not has_form and not is_valid:
            results['true_negatives'] += 1
            status = "TN"
        elif not has_form and is_valid:
            results['false_positives'] += 1
            status = "FP"
        else:
            results['false_negatives'] += 1
            status = "FN"

        mark = "OK" if correct else "XX"
        print(f"    [{mark}] {status} {name:<15} | gen={result.generalization_score:.0%} "
              f"stab={result.stability_score:.0%} | {result.validation.value}")

    # Summary
    print()
    print("=" * 80)
    print("  RESULTS")
    print("=" * 80)
    print()

    total = sum(results.values())
    tp, tn = results['true_positives'], results['true_negatives']
    fp, fn = results['false_positives'], results['false_negatives']

    accuracy = (tp + tn) / total * 100

    # Precision: of all we said "TRUE_FORM", how many actually had forms?
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

    # Recall: of all that had forms, how many did we find?
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    # False positive rate: of those without forms, how many did we wrongly validate?
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

    print(f"  True Positives (has form, validated):    {tp:>2}")
    print(f"  True Negatives (no form, rejected):      {tn:>2}")
    print(f"  False Positives (no form, validated):    {fp:>2}")
    print(f"  False Negatives (has form, rejected):    {fn:>2}")
    print()
    print(f"  Accuracy:            {accuracy:>5.1f}%")
    print(f"  Precision:           {precision:>5.1f}%")
    print(f"  Recall:              {recall:>5.1f}%")
    print(f"  False Positive Rate: {fpr:>5.1f}%")
    print()

    print("=" * 80)
    print("  INTERPRETATION")
    print("=" * 80)
    print()

    if fpr < 20:
        print(f"  EXCELLENT: False positive rate of {fpr:.0f}% means we correctly")
        print(f"  discriminate between true forms and coincidental patterns.")
    elif fpr < 40:
        print(f"  GOOD: False positive rate of {fpr:.0f}% shows reasonable")
        print(f"  discrimination ability with some room for improvement.")
    else:
        print(f"  NEEDS WORK: False positive rate of {fpr:.0f}% indicates")
        print(f"  difficulty distinguishing true forms from coincidence.")

    print()
    print(f"  The validation layer transforms our engine from a")
    print(f"  'pattern finder' into a 'pattern verifier'.")
    print()

    return results


if __name__ == "__main__":
    run_validated_benchmark()
