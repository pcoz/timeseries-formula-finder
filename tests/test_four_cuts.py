"""
Comprehensive Testing of Four Cuts
==================================

Run all test cases through all four cuts and analyze patterns.
"""

import sys
sys.path.insert(0, '..')

from core.four_cuts import (
    RegMachine, RegInstr, RegOp,
    reg_curvature, reg_flow, reg_to_alg, alg_flow, alg_to_reg,
    test_all_cuts
)
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class TestResult:
    name: str
    category: str
    initial_instrs: int
    reg_instrs: int
    lam_instrs: int
    df_instrs: int
    alg_instrs: int
    best_cut: str
    best_instrs: int
    reduction_pct: float


def test_program(name: str, category: str, prog: RegMachine) -> TestResult:
    """Test a program in all four cuts."""
    initial = len(prog.instrs)
    results = test_all_cuts(prog)

    best = min(results.keys(), key=lambda c: results[c])

    return TestResult(
        name=name,
        category=category,
        initial_instrs=initial,
        reg_instrs=results["register"][0],
        lam_instrs=results["lambda"][0],
        df_instrs=results["dataflow"][0],
        alg_instrs=results["algebraic"][0],
        best_cut=best,
        best_instrs=results[best][0],
        reduction_pct=(1 - results[best][0] / max(initial, 1)) * 100
    )


def make_all_tests() -> List[Tuple[str, str, RegMachine]]:
    """Create comprehensive test suite."""
    tests = []

    # === CONSTANTS ===
    tests.append(("2+3", "constants", RegMachine([
        RegInstr(RegOp.CONST, "a", 2),
        RegInstr(RegOp.CONST, "b", 3),
        RegInstr(RegOp.ADD, "result", "a", "b"),
    ], [], "result")))

    tests.append(("(2+3)*(4+5)", "constants", RegMachine([
        RegInstr(RegOp.CONST, "a", 2),
        RegInstr(RegOp.CONST, "b", 3),
        RegInstr(RegOp.ADD, "c", "a", "b"),
        RegInstr(RegOp.CONST, "d", 4),
        RegInstr(RegOp.CONST, "e", 5),
        RegInstr(RegOp.ADD, "f", "d", "e"),
        RegInstr(RegOp.MUL, "result", "c", "f"),
    ], [], "result")))

    tests.append(("2*3*4*5", "constants", RegMachine([
        RegInstr(RegOp.CONST, "a", 2),
        RegInstr(RegOp.CONST, "b", 3),
        RegInstr(RegOp.MUL, "c", "a", "b"),
        RegInstr(RegOp.CONST, "d", 4),
        RegInstr(RegOp.MUL, "e", "c", "d"),
        RegInstr(RegOp.CONST, "f", 5),
        RegInstr(RegOp.MUL, "result", "e", "f"),
    ], [], "result")))

    # === CONSTANTS + VARIABLE ===
    tests.append(("5*x", "const_var", RegMachine([
        RegInstr(RegOp.CONST, "five", 5),
        RegInstr(RegOp.MUL, "result", "five", "x"),
    ], ["x"], "result")))

    tests.append(("(2+3)*x", "const_var", RegMachine([
        RegInstr(RegOp.CONST, "a", 2),
        RegInstr(RegOp.CONST, "b", 3),
        RegInstr(RegOp.ADD, "c", "a", "b"),
        RegInstr(RegOp.MUL, "result", "c", "x"),
    ], ["x"], "result")))

    tests.append(("(2*3*4)*x", "const_var", RegMachine([
        RegInstr(RegOp.CONST, "a", 2),
        RegInstr(RegOp.CONST, "b", 3),
        RegInstr(RegOp.MUL, "c", "a", "b"),
        RegInstr(RegOp.CONST, "d", 4),
        RegInstr(RegOp.MUL, "e", "c", "d"),
        RegInstr(RegOp.MUL, "result", "e", "x"),
    ], ["x"], "result")))

    # === IDENTITY OPERATIONS ===
    tests.append(("x+0", "identity", RegMachine([
        RegInstr(RegOp.CONST, "zero", 0),
        RegInstr(RegOp.ADD, "result", "x", "zero"),
    ], ["x"], "result")))

    tests.append(("x*1", "identity", RegMachine([
        RegInstr(RegOp.CONST, "one", 1),
        RegInstr(RegOp.MUL, "result", "x", "one"),
    ], ["x"], "result")))

    tests.append(("x+0+0+0", "identity", RegMachine([
        RegInstr(RegOp.CONST, "z", 0),
        RegInstr(RegOp.ADD, "t1", "x", "z"),
        RegInstr(RegOp.ADD, "t2", "t1", "z"),
        RegInstr(RegOp.ADD, "result", "t2", "z"),
    ], ["x"], "result")))

    tests.append(("x*1*1*1", "identity", RegMachine([
        RegInstr(RegOp.CONST, "one", 1),
        RegInstr(RegOp.MUL, "t1", "x", "one"),
        RegInstr(RegOp.MUL, "t2", "t1", "one"),
        RegInstr(RegOp.MUL, "result", "t2", "one"),
    ], ["x"], "result")))

    # === DEAD CODE ===
    tests.append(("dead_simple", "dead_code", RegMachine([
        RegInstr(RegOp.ADD, "dead", "x", "y"),
        RegInstr(RegOp.MUL, "result", "x", "y"),
    ], ["x", "y"], "result")))

    tests.append(("dead_chain", "dead_code", RegMachine([
        RegInstr(RegOp.ADD, "d1", "x", "y"),
        RegInstr(RegOp.MUL, "d2", "d1", "x"),
        RegInstr(RegOp.ADD, "d3", "d2", "y"),
        RegInstr(RegOp.MUL, "result", "x", "y"),
    ], ["x", "y"], "result")))

    # === ADDITIVE CHAINS (Algebraic should win!) ===
    tests.append(("x+x", "add_chain", RegMachine([
        RegInstr(RegOp.ADD, "result", "x", "x"),
    ], ["x"], "result")))

    tests.append(("x+x+x", "add_chain", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "x"),
        RegInstr(RegOp.ADD, "result", "t1", "x"),
    ], ["x"], "result")))

    tests.append(("x+x+x+x", "add_chain", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "x"),
        RegInstr(RegOp.ADD, "t2", "t1", "x"),
        RegInstr(RegOp.ADD, "result", "t2", "x"),
    ], ["x"], "result")))

    tests.append(("x+x+x+x+x", "add_chain", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "x"),
        RegInstr(RegOp.ADD, "t2", "t1", "x"),
        RegInstr(RegOp.ADD, "t3", "t2", "x"),
        RegInstr(RegOp.ADD, "result", "t3", "x"),
    ], ["x"], "result")))

    # === MULTIPLICATIVE CHAINS ===
    tests.append(("x*x", "mul_chain", RegMachine([
        RegInstr(RegOp.MUL, "result", "x", "x"),
    ], ["x"], "result")))

    tests.append(("x*x*x", "mul_chain", RegMachine([
        RegInstr(RegOp.MUL, "t1", "x", "x"),
        RegInstr(RegOp.MUL, "result", "t1", "x"),
    ], ["x"], "result")))

    tests.append(("x*x*x*x", "mul_chain", RegMachine([
        RegInstr(RegOp.MUL, "t1", "x", "x"),
        RegInstr(RegOp.MUL, "t2", "t1", "x"),
        RegInstr(RegOp.MUL, "result", "t2", "x"),
    ], ["x"], "result")))

    # === POLYNOMIALS ===
    tests.append(("x^2", "polynomial", RegMachine([
        RegInstr(RegOp.MUL, "result", "x", "x"),
    ], ["x"], "result")))

    tests.append(("2x+1", "polynomial", RegMachine([
        RegInstr(RegOp.CONST, "two", 2),
        RegInstr(RegOp.MUL, "t1", "two", "x"),
        RegInstr(RegOp.CONST, "one", 1),
        RegInstr(RegOp.ADD, "result", "t1", "one"),
    ], ["x"], "result")))

    tests.append(("x^2+x+1", "polynomial", RegMachine([
        RegInstr(RegOp.MUL, "x2", "x", "x"),
        RegInstr(RegOp.ADD, "t1", "x2", "x"),
        RegInstr(RegOp.CONST, "one", 1),
        RegInstr(RegOp.ADD, "result", "t1", "one"),
    ], ["x"], "result")))

    tests.append(("2x^2+3x+1", "polynomial", RegMachine([
        RegInstr(RegOp.MUL, "x2", "x", "x"),
        RegInstr(RegOp.CONST, "two", 2),
        RegInstr(RegOp.MUL, "t1", "two", "x2"),
        RegInstr(RegOp.CONST, "three", 3),
        RegInstr(RegOp.MUL, "t2", "three", "x"),
        RegInstr(RegOp.ADD, "t3", "t1", "t2"),
        RegInstr(RegOp.CONST, "one", 1),
        RegInstr(RegOp.ADD, "result", "t3", "one"),
    ], ["x"], "result")))

    # === MULTI-VARIABLE ===
    tests.append(("x*y", "multivar", RegMachine([
        RegInstr(RegOp.MUL, "result", "x", "y"),
    ], ["x", "y"], "result")))

    tests.append(("x*y+z", "multivar", RegMachine([
        RegInstr(RegOp.MUL, "t1", "x", "y"),
        RegInstr(RegOp.ADD, "result", "t1", "z"),
    ], ["x", "y", "z"], "result")))

    tests.append(("x+y+z", "multivar", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "y"),
        RegInstr(RegOp.ADD, "result", "t1", "z"),
    ], ["x", "y", "z"], "result")))

    tests.append(("(x+y)*(x-y)", "multivar", RegMachine([
        RegInstr(RegOp.ADD, "sum", "x", "y"),
        RegInstr(RegOp.SUB, "diff", "x", "y"),
        RegInstr(RegOp.MUL, "result", "sum", "diff"),
    ], ["x", "y"], "result")))

    # === REDUNDANT COMPUTATION ===
    tests.append(("redundant_add", "redundant", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "y"),
        RegInstr(RegOp.ADD, "t2", "x", "y"),
        RegInstr(RegOp.MUL, "result", "t1", "t2"),
    ], ["x", "y"], "result")))

    tests.append(("redundant_mul", "redundant", RegMachine([
        RegInstr(RegOp.MUL, "a", "x", "x"),
        RegInstr(RegOp.MUL, "b", "x", "x"),
        RegInstr(RegOp.ADD, "result", "a", "b"),
    ], ["x"], "result")))

    # === WIDE/PARALLEL ===
    tests.append(("(a+b)+(c+d)", "wide", RegMachine([
        RegInstr(RegOp.ADD, "t1", "a", "b"),
        RegInstr(RegOp.ADD, "t2", "c", "d"),
        RegInstr(RegOp.ADD, "result", "t1", "t2"),
    ], ["a", "b", "c", "d"], "result")))

    tests.append(("(a*b)+(c*d)", "wide", RegMachine([
        RegInstr(RegOp.MUL, "t1", "a", "b"),
        RegInstr(RegOp.MUL, "t2", "c", "d"),
        RegInstr(RegOp.ADD, "result", "t1", "t2"),
    ], ["a", "b", "c", "d"], "result")))

    # === MIXED COMPLEXITY ===
    tests.append(("mixed_1", "mixed", RegMachine([
        RegInstr(RegOp.CONST, "c", 5),
        RegInstr(RegOp.ADD, "t1", "x", "c"),
        RegInstr(RegOp.MUL, "t2", "t1", "y"),
        RegInstr(RegOp.CONST, "zero", 0),
        RegInstr(RegOp.ADD, "result", "t2", "zero"),
    ], ["x", "y"], "result")))

    tests.append(("mixed_2", "mixed", RegMachine([
        RegInstr(RegOp.CONST, "a", 2),
        RegInstr(RegOp.CONST, "b", 3),
        RegInstr(RegOp.ADD, "c", "a", "b"),
        RegInstr(RegOp.MUL, "t1", "x", "x"),
        RegInstr(RegOp.MUL, "t2", "c", "t1"),
        RegInstr(RegOp.ADD, "dead", "x", "y"),
        RegInstr(RegOp.CONST, "one", 1),
        RegInstr(RegOp.MUL, "t3", "t2", "one"),
        RegInstr(RegOp.ADD, "result", "t3", "y"),
    ], ["x", "y"], "result")))

    # === DISTRIBUTION TESTS ===
    tests.append(("2*(x+y)", "distrib", RegMachine([
        RegInstr(RegOp.ADD, "sum", "x", "y"),
        RegInstr(RegOp.CONST, "two", 2),
        RegInstr(RegOp.MUL, "result", "two", "sum"),
    ], ["x", "y"], "result")))

    tests.append(("x*(y+z)", "distrib", RegMachine([
        RegInstr(RegOp.ADD, "sum", "y", "z"),
        RegInstr(RegOp.MUL, "result", "x", "sum"),
    ], ["x", "y", "z"], "result")))

    return tests


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("  COMPREHENSIVE TEST: FOUR CUTS")
    print("=" * 100)

    tests = make_all_tests()
    results = []

    for name, category, prog in tests:
        result = test_program(name, category, prog)
        results.append(result)

    # Display results table
    print(f"\n{'Name':<18} {'Category':<12} {'Init':<5} {'Reg':<5} {'Lam':<5} {'DF':<5} {'Alg':<5} {'Best':<12} {'Red%':<6}")
    print("-" * 100)

    for r in results:
        print(f"{r.name:<18} {r.category:<12} {r.initial_instrs:<5} "
              f"{r.reg_instrs:<5} {r.lam_instrs:<5} {r.df_instrs:<5} {r.alg_instrs:<5} "
              f"{r.best_cut:<12} {r.reduction_pct:>5.1f}%")

    # === ANALYSIS BY CATEGORY ===
    print("\n" + "=" * 100)
    print("  ANALYSIS BY CATEGORY")
    print("=" * 100)

    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"register": 0, "lambda": 0, "dataflow": 0, "algebraic": 0}
        categories[r.category][r.best_cut] += 1

    print(f"\n{'Category':<15} {'Register':<10} {'Lambda':<10} {'Dataflow':<10} {'Algebraic':<10}")
    print("-" * 60)
    for cat, counts in sorted(categories.items()):
        print(f"{cat:<15} {counts['register']:<10} {counts['lambda']:<10} "
              f"{counts['dataflow']:<10} {counts['algebraic']:<10}")

    # === OVERALL WINNER COUNT ===
    print("\n" + "=" * 100)
    print("  OVERALL WINNER COUNT")
    print("=" * 100)

    winner_counts = {"register": 0, "lambda": 0, "dataflow": 0, "algebraic": 0}
    for r in results:
        winner_counts[r.best_cut] += 1

    total = len(results)
    for cut, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {cut:<12}: {count:>3} wins ({pct:5.1f}%) {bar}")

    # === SINGULARITIES ===
    print("\n" + "=" * 100)
    print("  SINGULARITIES (No Reduction Achieved)")
    print("=" * 100)

    singularities = [r for r in results if r.best_instrs >= r.initial_instrs]
    if singularities:
        for r in singularities:
            print(f"  {r.name:<20} ({r.category}): {r.initial_instrs} instrs - already minimal")
    else:
        print("  None! All problems were reduced by at least one cut.")

    # === WHERE ALGEBRAIC WINS ===
    print("\n" + "=" * 100)
    print("  WHERE ALGEBRAIC WINS")
    print("=" * 100)

    alg_wins = [r for r in results if r.best_cut == "algebraic"]
    if alg_wins:
        for r in alg_wins:
            print(f"  {r.name:<20}: {r.initial_instrs} -> {r.alg_instrs} (others: Reg={r.reg_instrs}, Lam={r.lam_instrs}, DF={r.df_instrs})")
    else:
        print("  None in this test set.")

    # === WHERE CUTS DISAGREE ===
    print("\n" + "=" * 100)
    print("  INTERESTING CASES (Cuts Disagree)")
    print("=" * 100)

    disagreements = [r for r in results
                     if len(set([r.reg_instrs, r.lam_instrs, r.df_instrs, r.alg_instrs])) > 1]

    for r in disagreements:
        print(f"  {r.name:<20}: Reg={r.reg_instrs}, Lam={r.lam_instrs}, DF={r.df_instrs}, Alg={r.alg_instrs} -> {r.best_cut}")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 100)
    print("  SUMMARY: WHICH CUT FOR WHICH PROBLEM?")
    print("=" * 100)

    print("""
    PROBLEM TYPE              BEST CUT        WHY
    ────────────────────────────────────────────────────────────────
    Pure constants            register        constant folding
    Constants + variable      register        constant folding
    Identity (x+0, x*1)       lambda/alg      evaluation/simplification
    Dead code                 register        liveness analysis
    Additive chains           ALGEBRAIC       x+x+x+x -> 4*x
    Multiplicative chains     register        x^n stays as n-1 muls
    Polynomials               register        mixed, no clear winner
    Multi-variable            varies          depends on structure
    Redundant computation     lambda/df       CSE
    Wide/parallel             register        already optimal
    Mixed                     lambda          composition helps
    Distribution              varies          algebraic could help
    """)

    # === RECOMMENDATIONS ===
    print("\n" + "=" * 100)
    print("  RECOMMENDATIONS")
    print("=" * 100)

    print("""
    1. ALGEBRAIC cut excels at ADDITIVE CHAINS (x+x+x -> 3*x)
       - This is impossible for other cuts without this knowledge

    2. REGISTER cut is the reliable baseline
       - Constant folding, dead code elimination work well

    3. LAMBDA/DATAFLOW shine on REDUNDANT computation
       - CSE catches repeated subexpressions

    4. SINGULARITIES reveal TRUE PRIMITIVES
       - x*y, x^2, (a+b)+(c+d) cannot be simplified further
       - These are the atoms of computation for this domain

    5. META-LEARNER VALUE is clear in disagreement cases
       - When cuts give different results, choosing matters

    6. NEXT STEPS:
       - Improve algebraic handling of multiplication (x*x*x -> x^3)
       - Add distributive law properly
       - Consider factoring (5x + 5y -> 5*(x+y))
    """)
