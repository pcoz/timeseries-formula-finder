"""
100 Test Cases for Stuck Classifier + Four Cuts
================================================
"""

import sys
sys.path.insert(0, '..')

from core.stuck_classifier import classify_stuck, StuckType
from core.four_cuts import (
    RegMachine, RegInstr, RegOp,
    test_all_cuts, solve_with_best_cut,
    reg_to_alg, alg_flow, alg_to_expr_str
)

def generate_tests():
    """Generate 100 diverse test cases."""
    tests = []

    # === CATEGORY 1: TRUE PRIMITIVES (25 tests) ===
    # Binary operations on distinct variables
    tests.append(("x+y", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("x-y", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("x*y", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("a+b", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("p*q", "primitive", "TRUE_PRIMITIVE"))

    # Three-variable combinations
    tests.append(("x+y+z", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("x*y+z", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("x+y*z", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("a+b+c", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("a*b*c", "primitive", "TRUE_PRIMITIVE"))

    # Four-variable combinations
    tests.append(("(a+b)+(c+d)", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("(a*b)+(c*d)", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("(a+b)*(c+d)", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("(a*b)*(c*d)", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("a+b+c+d", "primitive", "TRUE_PRIMITIVE"))

    # Five-variable combinations
    tests.append(("a+b+c+d+e", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("(a+b)+(c+d)+e", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("a*b+c*d+e", "primitive", "TRUE_PRIMITIVE"))

    # Minimal powers (already optimal)
    tests.append(("x*x", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("x*x*x", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("y*y", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("y*y*y", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("z*z", "primitive", "TRUE_PRIMITIVE"))

    # Mixed minimal
    tests.append(("x*x+y", "primitive", "TRUE_PRIMITIVE"))
    tests.append(("x+y*y", "primitive", "TRUE_PRIMITIVE"))

    # === CATEGORY 2: ISOMORPHISMS (25 tests) ===
    # Difference of squares pattern
    tests.append(("(x+y)*(x-y)", "isomorphism", "ISOMORPHISM"))
    tests.append(("(a+b)*(a-b)", "isomorphism", "ISOMORPHISM"))
    tests.append(("(p+q)*(p-q)", "isomorphism", "ISOMORPHISM"))
    tests.append(("(m+n)*(m-n)", "isomorphism", "ISOMORPHISM"))

    # Repeated addition (can be multiplication)
    tests.append(("x+x", "isomorphism", "ISOMORPHISM"))
    tests.append(("x+x+x", "isomorphism", "ISOMORPHISM"))
    tests.append(("x+x+x+x", "isomorphism", "ISOMORPHISM"))
    tests.append(("x+x+x+x+x", "isomorphism", "ISOMORPHISM"))
    tests.append(("y+y", "isomorphism", "ISOMORPHISM"))
    tests.append(("y+y+y", "isomorphism", "ISOMORPHISM"))
    tests.append(("y+y+y+y", "isomorphism", "ISOMORPHISM"))
    tests.append(("z+z+z", "isomorphism", "ISOMORPHISM"))
    tests.append(("a+a+a+a+a", "isomorphism", "ISOMORPHISM"))
    tests.append(("b+b+b+b+b+b", "isomorphism", "ISOMORPHISM"))

    # More difference of squares variants
    tests.append(("(x-y)*(x+y)", "isomorphism", "ISOMORPHISM"))
    tests.append(("(a-b)*(a+b)", "isomorphism", "ISOMORPHISM"))

    # Large repeated additions
    tests.append(("x+x+x+x+x+x", "isomorphism", "ISOMORPHISM"))
    tests.append(("x+x+x+x+x+x+x", "isomorphism", "ISOMORPHISM"))
    tests.append(("y+y+y+y+y+y+y+y", "isomorphism", "ISOMORPHISM"))

    # Mixed repeated
    tests.append(("a+a", "isomorphism", "ISOMORPHISM"))
    tests.append(("b+b", "isomorphism", "ISOMORPHISM"))
    tests.append(("c+c+c", "isomorphism", "ISOMORPHISM"))
    tests.append(("d+d+d+d", "isomorphism", "ISOMORPHISM"))
    tests.append(("e+e+e+e+e", "isomorphism", "ISOMORPHISM"))
    tests.append(("f+f+f+f+f+f", "isomorphism", "ISOMORPHISM"))

    # === CATEGORY 3: OPTIMIZATIONS - POWERS (25 tests) ===
    # x^4 and above can be optimized
    tests.append(("x*x*x*x", "power_opt", "OPTIMIZATION"))
    tests.append(("x*x*x*x*x", "power_opt", "OPTIMIZATION"))
    tests.append(("x*x*x*x*x*x", "power_opt", "OPTIMIZATION"))
    tests.append(("x*x*x*x*x*x*x", "power_opt", "OPTIMIZATION"))
    tests.append(("x*x*x*x*x*x*x*x", "power_opt", "OPTIMIZATION"))

    tests.append(("y*y*y*y", "power_opt", "OPTIMIZATION"))
    tests.append(("y*y*y*y*y", "power_opt", "OPTIMIZATION"))
    tests.append(("y*y*y*y*y*y", "power_opt", "OPTIMIZATION"))
    tests.append(("y*y*y*y*y*y*y*y", "power_opt", "OPTIMIZATION"))

    tests.append(("z*z*z*z", "power_opt", "OPTIMIZATION"))
    tests.append(("z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z", "power_opt", "OPTIMIZATION"))  # z^16

    tests.append(("a*a*a*a", "power_opt", "OPTIMIZATION"))
    tests.append(("a*a*a*a*a*a*a*a", "power_opt", "OPTIMIZATION"))

    tests.append(("b*b*b*b*b*b", "power_opt", "OPTIMIZATION"))
    tests.append(("c*c*c*c*c*c*c", "power_opt", "OPTIMIZATION"))

    # More powers
    tests.append(("p*p*p*p", "power_opt", "OPTIMIZATION"))
    tests.append(("q*q*q*q*q", "power_opt", "OPTIMIZATION"))
    tests.append(("r*r*r*r*r*r", "power_opt", "OPTIMIZATION"))
    tests.append(("s*s*s*s*s*s*s", "power_opt", "OPTIMIZATION"))
    tests.append(("t*t*t*t*t*t*t*t", "power_opt", "OPTIMIZATION"))

    # Large powers
    tests.append(("x*x*x*x*x*x*x*x*x", "power_opt", "OPTIMIZATION"))  # x^9
    tests.append(("x*x*x*x*x*x*x*x*x*x", "power_opt", "OPTIMIZATION"))  # x^10
    tests.append(("x*x*x*x*x*x*x*x*x*x*x*x", "power_opt", "OPTIMIZATION"))  # x^12
    tests.append(("y*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y", "power_opt", "OPTIMIZATION"))  # y^16
    tests.append(("z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z*z", "power_opt", "OPTIMIZATION"))  # z^32

    # === CATEGORY 4: OPTIMIZATIONS - CSE (25 tests) ===
    # Repeated subexpressions
    tests.append(("(x+y)+(x+y)", "cse", "OPTIMIZATION"))
    tests.append(("(x+y)*(x+y)", "cse", "OPTIMIZATION"))
    tests.append(("(a+b)+(a+b)", "cse", "OPTIMIZATION"))
    tests.append(("(a*b)+(a*b)", "cse", "OPTIMIZATION"))
    tests.append(("(a*b)*(a*b)", "cse", "OPTIMIZATION"))

    tests.append(("(x+y)+(x+y)+(x+y)", "cse", "OPTIMIZATION"))
    tests.append(("(a*b)+(a*b)+(a*b)", "cse", "OPTIMIZATION"))
    tests.append(("(p+q)+(p+q)+(p+q)+(p+q)", "cse", "OPTIMIZATION"))

    # More complex CSE
    tests.append(("(x*y)+(x*y)", "cse", "OPTIMIZATION"))
    tests.append(("(x*y)*(x*y)", "cse", "OPTIMIZATION"))
    tests.append(("(a-b)+(a-b)", "cse", "OPTIMIZATION"))
    tests.append(("(a-b)*(a-b)", "cse", "OPTIMIZATION"))

    # Triple repetitions
    tests.append(("(x+y)+(x+y)+(x+y)", "cse", "OPTIMIZATION"))
    tests.append(("(x*y)+(x*y)+(x*y)", "cse", "OPTIMIZATION"))
    tests.append(("(a+b)+(a+b)+(a+b)", "cse", "OPTIMIZATION"))

    # Quadruple repetitions
    tests.append(("(x+y)+(x+y)+(x+y)+(x+y)", "cse", "OPTIMIZATION"))
    tests.append(("(a*b)+(a*b)+(a*b)+(a*b)", "cse", "OPTIMIZATION"))

    # Different structures
    tests.append(("(m+n)+(m+n)", "cse", "OPTIMIZATION"))
    tests.append(("(m*n)+(m*n)", "cse", "OPTIMIZATION"))
    tests.append(("(p-q)+(p-q)", "cse", "OPTIMIZATION"))

    # More CSE patterns
    tests.append(("(x+1)+(x+1)", "cse", "OPTIMIZATION"))
    tests.append(("(y+2)+(y+2)", "cse", "OPTIMIZATION"))
    tests.append(("(z*3)+(z*3)", "cse", "OPTIMIZATION"))
    tests.append(("(a+b+c)+(a+b+c)", "cse", "OPTIMIZATION"))
    tests.append(("(x*y*z)+(x*y*z)", "cse", "OPTIMIZATION"))

    return tests


def run_tests():
    print("=" * 80)
    print("  100 TEST CASES: STUCK CLASSIFIER")
    print("=" * 80)

    tests = generate_tests()
    print(f"\nRunning {len(tests)} tests...\n")

    results = {
        "pass": 0,
        "fail": 0,
        "by_category": {},
        "by_expected": {"TRUE_PRIMITIVE": {"pass": 0, "fail": 0},
                        "ISOMORPHISM": {"pass": 0, "fail": 0},
                        "OPTIMIZATION": {"pass": 0, "fail": 0}}
    }

    failures = []

    for i, (expr, category, expected) in enumerate(tests):
        analysis = classify_stuck(expr)
        actual = analysis.stuck_type.value

        passed = actual == expected
        if passed:
            results["pass"] += 1
            results["by_expected"][expected]["pass"] += 1
            status = "OK"
        else:
            results["fail"] += 1
            results["by_expected"][expected]["fail"] += 1
            status = "FAIL"
            failures.append((expr, category, expected, actual))

        # Track by category
        if category not in results["by_category"]:
            results["by_category"][category] = {"pass": 0, "fail": 0}
        if passed:
            results["by_category"][category]["pass"] += 1
        else:
            results["by_category"][category]["fail"] += 1

        # Print progress every 10 tests
        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/{len(tests)}] {results['pass']} passed, {results['fail']} failed")

    # Summary
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    total = results["pass"] + results["fail"]
    pct = results["pass"] / total * 100

    print(f"\n  Total: {results['pass']}/{total} passed ({pct:.1f}%)\n")

    print("  By Expected Type:")
    print("  " + "-" * 40)
    for typ, counts in results["by_expected"].items():
        total_typ = counts["pass"] + counts["fail"]
        pct_typ = counts["pass"] / total_typ * 100 if total_typ > 0 else 0
        bar = "#" * int(pct_typ / 5)
        print(f"    {typ:<16}: {counts['pass']:>2}/{total_typ:<2} ({pct_typ:5.1f}%) {bar}")

    print("\n  By Category:")
    print("  " + "-" * 40)
    for cat, counts in sorted(results["by_category"].items()):
        total_cat = counts["pass"] + counts["fail"]
        pct_cat = counts["pass"] / total_cat * 100 if total_cat > 0 else 0
        bar = "#" * int(pct_cat / 5)
        print(f"    {cat:<16}: {counts['pass']:>2}/{total_cat:<2} ({pct_cat:5.1f}%) {bar}")

    # Show failures
    if failures:
        print("\n" + "=" * 80)
        print("  FAILURES")
        print("=" * 80)
        for expr, cat, expected, actual in failures[:10]:  # Show first 10
            print(f"    {expr:<30} expected {expected:<16} got {actual}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more failures")

    # Final verdict
    print("\n" + "=" * 80)
    if results["fail"] == 0:
        print("  ALL TESTS PASSED!")
    else:
        print(f"  {results['fail']} TESTS FAILED")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_tests()
