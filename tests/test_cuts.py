"""
Comprehensive Testing of the Three Cuts
========================================

Run diverse problems through all cuts and analyze:
1. Which cut actually produces the simplest FINAL result?
2. What problem characteristics predict best cut?
3. Where do singularities (stuck points) occur?
"""

import sys
sys.path.insert(0, '..')

from core.three_cuts import (
    RegMachine, RegInstr, RegOp,
    reg_curvature, reg_flow, reg_to_lam, reg_to_df,
    lam_flow, lam_curvature, lam_size, lam_to_df,
    df_flow, df_curvature, df_to_reg,
    DataflowGraph, DFNode
)
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class TestResult:
    name: str
    category: str

    # Initial state
    initial_instrs: int
    initial_reg_curv: float

    # After flow in each cut (measured in register machine form)
    reg_final_instrs: int
    reg_final_curv: float
    reg_reduction: float

    lam_final_instrs: int
    lam_final_curv: float
    lam_reduction: float

    df_final_instrs: int
    df_final_curv: float
    df_reduction: float

    # Winner
    best_cut: str
    best_instrs: int


def test_program(name: str, category: str, prog: RegMachine) -> TestResult:
    """Test a program in all three cuts, measuring final register machine size."""

    initial_instrs = len(prog.instrs)
    initial_curv = reg_curvature(prog)

    # Cut 1: Direct register flow
    reg_result = reg_flow(prog)
    reg_instrs = len(reg_result.instrs)
    reg_curv = reg_curvature(reg_result)

    # Cut 2: Via lambda
    try:
        lam = reg_to_lam(prog)
        lam_flowed = lam_flow(lam)
        df_from_lam = lam_to_df(lam_flowed)
        df_from_lam_flowed = df_flow(df_from_lam)
        reg_from_lam = df_to_reg(df_from_lam_flowed)
        reg_from_lam = reg_flow(reg_from_lam)
        lam_instrs = len(reg_from_lam.instrs)
        lam_curv = reg_curvature(reg_from_lam)
    except Exception as e:
        lam_instrs = initial_instrs
        lam_curv = initial_curv

    # Cut 3: Via dataflow
    try:
        df = reg_to_df(prog)
        df_flowed = df_flow(df)
        reg_from_df = df_to_reg(df_flowed)
        reg_from_df = reg_flow(reg_from_df)
        df_instrs = len(reg_from_df.instrs)
        df_curv = reg_curvature(reg_from_df)
    except Exception as e:
        df_instrs = initial_instrs
        df_curv = initial_curv

    # Determine winner (lowest instruction count, tie-break by curvature)
    results = [
        ("register", reg_instrs, reg_curv),
        ("lambda", lam_instrs, lam_curv),
        ("dataflow", df_instrs, df_curv),
    ]
    best = min(results, key=lambda x: (x[1], x[2]))

    return TestResult(
        name=name,
        category=category,
        initial_instrs=initial_instrs,
        initial_reg_curv=initial_curv,
        reg_final_instrs=reg_instrs,
        reg_final_curv=reg_curv,
        reg_reduction=(1 - reg_instrs/max(initial_instrs, 1)) * 100,
        lam_final_instrs=lam_instrs,
        lam_final_curv=lam_curv,
        lam_reduction=(1 - lam_instrs/max(initial_instrs, 1)) * 100,
        df_final_instrs=df_instrs,
        df_final_curv=df_curv,
        df_reduction=(1 - df_instrs/max(initial_instrs, 1)) * 100,
        best_cut=best[0],
        best_instrs=best[1],
    )


# =============================================================================
# TEST PROGRAMS BY CATEGORY
# =============================================================================

def make_tests() -> List[Tuple[str, str, RegMachine]]:
    """Create diverse test programs."""
    tests = []

    # === CATEGORY: Pure Constants ===
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

    # === CATEGORY: Constants with Variable ===
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

    # === CATEGORY: Algebraic Identity ===
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

    # === CATEGORY: Dead Code ===
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

    # === CATEGORY: Deep Chains ===
    tests.append(("chain_2", "deep_chain", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "x"),
        RegInstr(RegOp.ADD, "result", "t1", "x"),
    ], ["x"], "result")))

    tests.append(("chain_4", "deep_chain", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "x"),
        RegInstr(RegOp.ADD, "t2", "t1", "x"),
        RegInstr(RegOp.ADD, "t3", "t2", "x"),
        RegInstr(RegOp.ADD, "result", "t3", "x"),
    ], ["x"], "result")))

    tests.append(("mul_chain", "deep_chain", RegMachine([
        RegInstr(RegOp.MUL, "t1", "x", "x"),
        RegInstr(RegOp.MUL, "t2", "t1", "x"),
        RegInstr(RegOp.MUL, "result", "t2", "x"),
    ], ["x"], "result")))

    # === CATEGORY: Wide/Parallel ===
    tests.append(("wide_2", "wide", RegMachine([
        RegInstr(RegOp.ADD, "t1", "a", "b"),
        RegInstr(RegOp.ADD, "t2", "c", "d"),
        RegInstr(RegOp.ADD, "result", "t1", "t2"),
    ], ["a", "b", "c", "d"], "result")))

    tests.append(("wide_3", "wide", RegMachine([
        RegInstr(RegOp.ADD, "t1", "a", "b"),
        RegInstr(RegOp.ADD, "t2", "c", "d"),
        RegInstr(RegOp.ADD, "t3", "e", "f"),
        RegInstr(RegOp.ADD, "t4", "t1", "t2"),
        RegInstr(RegOp.ADD, "result", "t4", "t3"),
    ], ["a", "b", "c", "d", "e", "f"], "result")))

    # === CATEGORY: Polynomials ===
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

    # === CATEGORY: Multi-variable ===
    tests.append(("x*y", "multivar", RegMachine([
        RegInstr(RegOp.MUL, "result", "x", "y"),
    ], ["x", "y"], "result")))

    tests.append(("x*y+z", "multivar", RegMachine([
        RegInstr(RegOp.MUL, "t1", "x", "y"),
        RegInstr(RegOp.ADD, "result", "t1", "z"),
    ], ["x", "y", "z"], "result")))

    tests.append(("(x+y)*(x-y)", "multivar", RegMachine([
        RegInstr(RegOp.ADD, "sum", "x", "y"),
        RegInstr(RegOp.SUB, "diff", "x", "y"),
        RegInstr(RegOp.MUL, "result", "sum", "diff"),
    ], ["x", "y"], "result")))

    # === CATEGORY: Redundant Computation ===
    tests.append(("redundant_add", "redundant", RegMachine([
        RegInstr(RegOp.ADD, "t1", "x", "y"),
        RegInstr(RegOp.ADD, "t2", "x", "y"),  # Same as t1!
        RegInstr(RegOp.MUL, "result", "t1", "t2"),
    ], ["x", "y"], "result")))

    tests.append(("redundant_complex", "redundant", RegMachine([
        RegInstr(RegOp.MUL, "a", "x", "x"),
        RegInstr(RegOp.MUL, "b", "x", "x"),  # Same as a!
        RegInstr(RegOp.ADD, "result", "a", "b"),
    ], ["x"], "result")))

    # === CATEGORY: Mixed Complexity ===
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
        RegInstr(RegOp.ADD, "c", "a", "b"),  # 5
        RegInstr(RegOp.MUL, "t1", "x", "x"),  # x^2
        RegInstr(RegOp.MUL, "t2", "c", "t1"),  # 5x^2
        RegInstr(RegOp.ADD, "dead", "x", "y"),  # dead code
        RegInstr(RegOp.CONST, "one", 1),
        RegInstr(RegOp.MUL, "t3", "t2", "one"),  # identity
        RegInstr(RegOp.ADD, "result", "t3", "y"),
    ], ["x", "y"], "result")))

    return tests


# =============================================================================
# MAIN: Run All Tests
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  COMPREHENSIVE TEST OF THREE CUTS")
    print("=" * 90)

    tests = make_tests()
    results = []

    for name, category, prog in tests:
        result = test_program(name, category, prog)
        results.append(result)

    # Display results
    print(f"\n{'Name':<20} {'Cat':<12} {'Init':<5} {'Reg':<5} {'Lam':<5} {'DF':<5} {'Best':<10} {'Reduce':<8}")
    print("-" * 90)

    for r in results:
        best_reduce = (1 - r.best_instrs / max(r.initial_instrs, 1)) * 100
        print(f"{r.name:<20} {r.category:<12} {r.initial_instrs:<5} "
              f"{r.reg_final_instrs:<5} {r.lam_final_instrs:<5} {r.df_final_instrs:<5} "
              f"{r.best_cut:<10} {best_reduce:>6.1f}%")

    # Analyze by category
    print("\n" + "=" * 90)
    print("  ANALYSIS BY CATEGORY")
    print("=" * 90)

    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"register": 0, "lambda": 0, "dataflow": 0, "total": 0}
        categories[r.category][r.best_cut] += 1
        categories[r.category]["total"] += 1

    print(f"\n{'Category':<15} {'Register':<10} {'Lambda':<10} {'Dataflow':<10} {'Total':<8}")
    print("-" * 55)
    for cat, counts in categories.items():
        print(f"{cat:<15} {counts['register']:<10} {counts['lambda']:<10} "
              f"{counts['dataflow']:<10} {counts['total']:<8}")

    # Overall winner count
    print("\n" + "=" * 90)
    print("  OVERALL WINNER COUNT")
    print("=" * 90)

    winner_counts = {"register": 0, "lambda": 0, "dataflow": 0}
    for r in results:
        winner_counts[r.best_cut] += 1

    total = len(results)
    for cut, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"  {cut:<12}: {count:>3} wins ({count/total*100:.1f}%)")

    # Find singularities (where no cut improved)
    print("\n" + "=" * 90)
    print("  SINGULARITIES (No Reduction Possible)")
    print("=" * 90)

    singularities = [r for r in results if r.reg_final_instrs >= r.initial_instrs
                     and r.lam_final_instrs >= r.initial_instrs
                     and r.df_final_instrs >= r.initial_instrs]

    if singularities:
        for r in singularities:
            print(f"  {r.name}: {r.initial_instrs} instructions, category={r.category}")
    else:
        print("  None found - all problems were reduced by at least one cut!")

    # Find disagreements (where cuts give different results)
    print("\n" + "=" * 90)
    print("  INTERESTING CASES (Cuts Disagree)")
    print("=" * 90)

    disagreements = [r for r in results if len(set([r.reg_final_instrs, r.lam_final_instrs, r.df_final_instrs])) > 1]

    for r in disagreements[:10]:  # Show top 10
        print(f"  {r.name:<20}: Reg={r.reg_final_instrs}, Lam={r.lam_final_instrs}, DF={r.df_final_instrs} -> {r.best_cut}")

    # Recommendations
    print("\n" + "=" * 90)
    print("  RECOMMENDATIONS")
    print("=" * 90)

    print("""
Based on the test results:

1. PATTERN: Which cut works best for which problem type?
""")

    for cat, counts in categories.items():
        best_for_cat = max(["register", "lambda", "dataflow"], key=lambda c: counts[c])
        print(f"   {cat:<15} -> {best_for_cat}")

    print("""
2. SINGULARITIES: Problems that resist ALL flows are candidates for:
   - New rewrite rules (surgery)
   - New cuts/perspectives
   - Fundamental irreducibility (true complexity)

3. DISAGREEMENTS: When cuts disagree, the meta-learner adds value.
   These are where "choosing the right view" matters most.

4. NEXT DIRECTIONS:
   - Add common subexpression elimination to dataflow
   - Add more algebraic identities (distributive law, etc.)
   - Try a fourth cut (stack machine? combinators?)
   - Train a neural meta-learner on these patterns
""")
