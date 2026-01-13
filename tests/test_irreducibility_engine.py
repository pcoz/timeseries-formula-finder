"""
100 Real-World Test Cases for Irreducibility Engine
====================================================

This test suite verifies the engine can:
1. Correctly classify inputs (TRUE_PRIMITIVE, ISOMORPHISM, OPTIMIZATION)
2. Compute minimal forms
3. Verify equivalence between original and minimal forms

Test domains:
- Mathematical expressions (40 cases)
- Data transformations (20 cases)
- Business rules (20 cases)
- State machines (20 cases)
"""

import sys
sys.path.insert(0, '..')

from core.irreducibility_engine import (
    IrreducibilityEngine, IrreducibilityType,
    MathExpression, DataTransformation, BusinessRule, StateMachine,
    analyze, minimize, is_irreducible
)


# =============================================================================
# TEST CASE GENERATORS
# =============================================================================

def generate_math_tests():
    """40 mathematical expression test cases."""
    tests = []

    # --- TRUE_PRIMITIVES (15 cases) ---
    # Simple binary ops on distinct variables
    tests.append(("x+y", "TRUE_PRIMITIVE", "2 inputs, 1 op = minimum"))
    tests.append(("x-y", "TRUE_PRIMITIVE", "subtraction is irreducible"))
    tests.append(("x*y", "TRUE_PRIMITIVE", "multiplication is irreducible"))
    tests.append(("a/b", "TRUE_PRIMITIVE", "division is irreducible"))

    # Three-variable minimal
    tests.append(("x+y+z", "TRUE_PRIMITIVE", "3 inputs need 2 ops"))
    tests.append(("x*y*z", "TRUE_PRIMITIVE", "3 multiplicands"))
    tests.append(("a+b-c", "TRUE_PRIMITIVE", "mixed ops, 3 vars"))
    tests.append(("(x+y)*z", "TRUE_PRIMITIVE", "compound, minimal"))

    # Four-variable minimal
    tests.append(("a+b+c+d", "TRUE_PRIMITIVE", "4 inputs need 3 ops"))
    tests.append(("(a+b)*(c+d)", "TRUE_PRIMITIVE", "balanced tree"))
    tests.append(("a*b+c*d", "TRUE_PRIMITIVE", "two products summed"))

    # Minimal powers
    tests.append(("x*x", "TRUE_PRIMITIVE", "x^2 in 1 op = optimal"))
    tests.append(("x*x*x", "TRUE_PRIMITIVE", "x^3 in 2 ops = optimal"))
    tests.append(("y*y", "TRUE_PRIMITIVE", "y squared"))
    tests.append(("a*a+b*b", "TRUE_PRIMITIVE", "sum of squares"))

    # --- ISOMORPHISMS (10 cases) ---
    # Difference of squares
    tests.append(("(x+y)*(x-y)", "ISOMORPHISM", "equals x^2 - y^2"))
    tests.append(("(a+b)*(a-b)", "ISOMORPHISM", "difference of squares"))
    tests.append(("(p-q)*(p+q)", "ISOMORPHISM", "commuted diff of squares"))

    # Repeated addition = multiplication
    tests.append(("x+x", "ISOMORPHISM", "equals 2*x"))
    tests.append(("x+x+x", "ISOMORPHISM", "equals 3*x"))
    tests.append(("y+y+y+y", "ISOMORPHISM", "equals 4*y"))
    tests.append(("a+a+a+a+a", "ISOMORPHISM", "equals 5*a"))
    tests.append(("z+z+z+z+z+z", "ISOMORPHISM", "equals 6*z"))
    tests.append(("b+b", "ISOMORPHISM", "equals 2*b"))
    tests.append(("c+c+c", "ISOMORPHISM", "equals 3*c"))

    # --- OPTIMIZATIONS (15 cases) ---
    # Power optimizations (x^4 and above)
    tests.append(("x*x*x*x", "OPTIMIZATION", "x^4: 3 ops -> 2 ops"))
    tests.append(("x*x*x*x*x", "OPTIMIZATION", "x^5: 4 ops -> 3 ops"))
    tests.append(("x*x*x*x*x*x", "OPTIMIZATION", "x^6: 5 ops -> 3 ops"))
    tests.append(("x*x*x*x*x*x*x*x", "OPTIMIZATION", "x^8: 7 ops -> 3 ops"))
    tests.append(("y*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y", "OPTIMIZATION", "y^16: 15 ops -> 4 ops"))

    # Common subexpression elimination
    tests.append(("(x+y)+(x+y)", "OPTIMIZATION", "CSE: compute x+y once"))
    tests.append(("(x+y)*(x+y)", "OPTIMIZATION", "CSE: square of sum"))
    tests.append(("(a*b)+(a*b)", "OPTIMIZATION", "CSE: repeated product"))
    tests.append(("(a+b)+(a+b)+(a+b)", "OPTIMIZATION", "CSE: triple repetition"))
    tests.append(("(x*y)+(x*y)+(x*y)+(x*y)", "OPTIMIZATION", "CSE: quadruple"))

    # Mixed optimizations
    tests.append(("x*x*x*x+y*y*y*y", "OPTIMIZATION", "two powers"))
    tests.append(("(x+y)*(x+y)+(a+b)*(a+b)", "OPTIMIZATION", "two CSE opportunities"))
    tests.append(("z*z*z*z*z*z*z", "OPTIMIZATION", "z^7 optimization"))
    tests.append(("(p+q)+(p+q)+(r*r*r*r)", "OPTIMIZATION", "CSE + power"))
    tests.append(("a*a*a*a*a*a*a*a*a*a", "OPTIMIZATION", "a^10"))

    return tests


def generate_data_transform_tests():
    """20 data transformation test cases."""
    tests = []

    # --- TRUE_PRIMITIVES (7 cases) ---
    tests.append((
        DataTransformation(
            raw="SELECT name FROM users",
            domain="data",
            inputs=["users"],
            outputs=["name"],
            operations=["SELECT"]
        ),
        "TRUE_PRIMITIVE",
        "single select = minimum"
    ))
    tests.append((
        DataTransformation(
            raw="SELECT a, b FROM t",
            domain="data",
            inputs=["t"],
            outputs=["a", "b"],
            operations=["SELECT"]
        ),
        "TRUE_PRIMITIVE",
        "multi-column select"
    ))
    tests.append((
        DataTransformation(
            raw="filter x > 5",
            domain="data",
            inputs=["data"],
            outputs=["filtered"],
            operations=["FILTER"]
        ),
        "TRUE_PRIMITIVE",
        "single filter"
    ))
    tests.append((
        DataTransformation(
            raw="map x -> x*2",
            domain="data",
            inputs=["list"],
            outputs=["mapped"],
            operations=["MAP"]
        ),
        "TRUE_PRIMITIVE",
        "single map"
    ))
    tests.append((
        DataTransformation(
            raw="reduce sum",
            domain="data",
            inputs=["numbers"],
            outputs=["total"],
            operations=["REDUCE"]
        ),
        "TRUE_PRIMITIVE",
        "single reduce"
    ))
    tests.append((
        DataTransformation(
            raw="SELECT * FROM a JOIN b",
            domain="data",
            inputs=["a", "b"],
            outputs=["joined"],
            operations=["SELECT", "JOIN"]
        ),
        "TRUE_PRIMITIVE",
        "join requires 2 ops minimum"
    ))
    tests.append((
        DataTransformation(
            raw="GROUP BY category",
            domain="data",
            inputs=["sales"],
            outputs=["grouped"],
            operations=["GROUP BY"]
        ),
        "TRUE_PRIMITIVE",
        "single grouping"
    ))

    # --- OPTIMIZATIONS (13 cases) ---
    tests.append((
        DataTransformation(
            raw="SELECT a WHERE x > 5 WHERE x > 3",
            domain="data",
            inputs=["t"],
            outputs=["a"],
            operations=["SELECT", "WHERE", "WHERE"]
        ),
        "OPTIMIZATION",
        "redundant WHERE clauses"
    ))
    tests.append((
        DataTransformation(
            raw="map f | map g",
            domain="data",
            inputs=["list"],
            outputs=["result"],
            operations=["MAP", "MAP"]
        ),
        "OPTIMIZATION",
        "fusible maps"
    ))
    tests.append((
        DataTransformation(
            raw="filter p | filter q",
            domain="data",
            inputs=["data"],
            outputs=["result"],
            operations=["FILTER", "FILTER"]
        ),
        "OPTIMIZATION",
        "fusible filters"
    ))
    tests.append((
        DataTransformation(
            raw="SELECT * WHERE a WHERE a",
            domain="data",
            inputs=["t"],
            outputs=["*"],
            operations=["SELECT", "WHERE", "WHERE"]
        ),
        "OPTIMIZATION",
        "duplicate where conditions"
    ))
    tests.append((
        DataTransformation(
            raw="SELECT SELECT a FROM t",
            domain="data",
            inputs=["t"],
            outputs=["a"],
            operations=["SELECT", "SELECT"]
        ),
        "OPTIMIZATION",
        "redundant select"
    ))
    tests.append((
        DataTransformation(
            raw="map square | map sqrt",
            domain="data",
            inputs=["nums"],
            outputs=["nums"],
            operations=["MAP", "MAP"]
        ),
        "OPTIMIZATION",
        "potentially canceling maps"
    ))
    tests.append((
        DataTransformation(
            raw="filter x > 0 | filter x > 0",
            domain="data",
            inputs=["data"],
            outputs=["positive"],
            operations=["FILTER", "FILTER"]
        ),
        "OPTIMIZATION",
        "identical filters"
    ))
    tests.append((
        DataTransformation(
            raw="ORDER BY a ORDER BY a",
            domain="data",
            inputs=["t"],
            outputs=["sorted"],
            operations=["ORDER BY", "ORDER BY"]
        ),
        "OPTIMIZATION",
        "redundant ordering"
    ))
    tests.append((
        DataTransformation(
            raw="map f | map f",
            domain="data",
            inputs=["list"],
            outputs=["result"],
            operations=["MAP", "MAP"]
        ),
        "OPTIMIZATION",
        "identical maps (could be f^2)"
    ))
    tests.append((
        DataTransformation(
            raw="SELECT a, b WHERE x > 5 WHERE x > 5",
            domain="data",
            inputs=["t"],
            outputs=["a", "b"],
            operations=["SELECT", "WHERE", "WHERE"]
        ),
        "OPTIMIZATION",
        "duplicate predicate"
    ))
    tests.append((
        DataTransformation(
            raw="map double | filter positive | map double | filter positive",
            domain="data",
            inputs=["nums"],
            outputs=["result"],
            operations=["MAP", "FILTER", "MAP", "FILTER"]
        ),
        "OPTIMIZATION",
        "repeated pattern"
    ))
    tests.append((
        DataTransformation(
            raw="WHERE age > 21 WHERE age > 18",
            domain="data",
            inputs=["users"],
            outputs=["adults"],
            operations=["WHERE", "WHERE"]
        ),
        "OPTIMIZATION",
        "subsumed predicates"
    ))
    tests.append((
        DataTransformation(
            raw="filter | filter | filter",
            domain="data",
            inputs=["data"],
            outputs=["result"],
            operations=["FILTER", "FILTER", "FILTER"]
        ),
        "OPTIMIZATION",
        "triple filter - fusible"
    ))

    return tests


def generate_business_rule_tests():
    """20 business rule test cases."""
    tests = []

    # --- TRUE_PRIMITIVES (8 cases) ---
    tests.append((
        BusinessRule(
            raw="IF age >= 18 THEN allow_purchase",
            domain="rule",
            conditions=["age >= 18"],
            actions=["allow_purchase"]
        ),
        "TRUE_PRIMITIVE",
        "single condition, single action"
    ))
    tests.append((
        BusinessRule(
            raw="IF verified THEN grant_access",
            domain="rule",
            conditions=["verified"],
            actions=["grant_access"]
        ),
        "TRUE_PRIMITIVE",
        "boolean condition"
    ))
    tests.append((
        BusinessRule(
            raw="IF credit_score > 700 AND income > 50000 THEN approve_loan",
            domain="rule",
            conditions=["credit_score > 700", "income > 50000"],
            actions=["approve_loan"]
        ),
        "TRUE_PRIMITIVE",
        "two independent conditions"
    ))
    tests.append((
        BusinessRule(
            raw="IF stock > 0 AND price < budget THEN purchase",
            domain="rule",
            conditions=["stock > 0", "price < budget"],
            actions=["purchase"]
        ),
        "TRUE_PRIMITIVE",
        "availability + affordability"
    ))
    tests.append((
        BusinessRule(
            raw="IF kyc_complete THEN enable_trading",
            domain="rule",
            conditions=["kyc_complete"],
            actions=["enable_trading"]
        ),
        "TRUE_PRIMITIVE",
        "compliance rule"
    ))
    tests.append((
        BusinessRule(
            raw="IF order_total > 100 THEN apply_discount; send_email",
            domain="rule",
            conditions=["order_total > 100"],
            actions=["apply_discount", "send_email"]
        ),
        "TRUE_PRIMITIVE",
        "one condition, two actions"
    ))
    tests.append((
        BusinessRule(
            raw="IF is_premium AND has_balance THEN allow_withdrawal",
            domain="rule",
            conditions=["is_premium", "has_balance"],
            actions=["allow_withdrawal"]
        ),
        "TRUE_PRIMITIVE",
        "membership + balance check"
    ))
    tests.append((
        BusinessRule(
            raw="IF submitted AND reviewed AND approved THEN process",
            domain="rule",
            conditions=["submitted", "reviewed", "approved"],
            actions=["process"]
        ),
        "TRUE_PRIMITIVE",
        "three-stage approval"
    ))

    # --- OPTIMIZATIONS (12 cases) ---
    tests.append((
        BusinessRule(
            raw="IF x > 5 AND x > 3 THEN action",
            domain="rule",
            conditions=["x > 5", "x > 3"],
            actions=["action"]
        ),
        "OPTIMIZATION",
        "x > 5 subsumes x > 3"
    ))
    tests.append((
        BusinessRule(
            raw="IF age >= 21 AND age >= 18 THEN serve_alcohol",
            domain="rule",
            conditions=["age >= 21", "age >= 18"],
            actions=["serve_alcohol"]
        ),
        "OPTIMIZATION",
        "redundant age check"
    ))
    tests.append((
        BusinessRule(
            raw="IF active AND active THEN proceed",
            domain="rule",
            conditions=["active", "active"],
            actions=["proceed"]
        ),
        "OPTIMIZATION",
        "duplicate condition"
    ))
    tests.append((
        BusinessRule(
            raw="IF verified AND verified AND verified THEN allow",
            domain="rule",
            conditions=["verified", "verified", "verified"],
            actions=["allow"]
        ),
        "OPTIMIZATION",
        "triple duplicate"
    ))
    tests.append((
        BusinessRule(
            raw="IF balance > 1000 AND balance > 500 AND balance > 100 THEN gold_status",
            domain="rule",
            conditions=["balance > 1000", "balance > 500", "balance > 100"],
            actions=["gold_status"]
        ),
        "OPTIMIZATION",
        "cascading subsumption"
    ))
    tests.append((
        BusinessRule(
            raw="IF temp > 100 AND temp > 50 THEN overheat_alert",
            domain="rule",
            conditions=["temp > 100", "temp > 50"],
            actions=["overheat_alert"]
        ),
        "OPTIMIZATION",
        "temp threshold redundancy"
    ))
    tests.append((
        BusinessRule(
            raw="IF score >= 90 AND score >= 80 AND score >= 70 THEN A_grade",
            domain="rule",
            conditions=["score >= 90", "score >= 80", "score >= 70"],
            actions=["A_grade"]
        ),
        "OPTIMIZATION",
        "grade threshold redundancy"
    ))
    tests.append((
        BusinessRule(
            raw="IF enabled AND enabled THEN start; start",
            domain="rule",
            conditions=["enabled", "enabled"],
            actions=["start", "start"]
        ),
        "OPTIMIZATION",
        "duplicate condition + action"
    ))
    tests.append((
        BusinessRule(
            raw="IF count > 10 AND count > 5 AND count > 1 THEN batch_process",
            domain="rule",
            conditions=["count > 10", "count > 5", "count > 1"],
            actions=["batch_process"]
        ),
        "OPTIMIZATION",
        "triple threshold redundancy"
    ))
    tests.append((
        BusinessRule(
            raw="IF valid AND valid THEN submit",
            domain="rule",
            conditions=["valid", "valid"],
            actions=["submit"]
        ),
        "OPTIMIZATION",
        "duplicate validity check"
    ))
    tests.append((
        BusinessRule(
            raw="IF amount > 500 AND amount > 200 THEN review_required",
            domain="rule",
            conditions=["amount > 500", "amount > 200"],
            actions=["review_required"]
        ),
        "OPTIMIZATION",
        "amount threshold redundancy"
    ))
    tests.append((
        BusinessRule(
            raw="IF priority > 8 AND priority > 5 AND priority > 3 THEN escalate",
            domain="rule",
            conditions=["priority > 8", "priority > 5", "priority > 3"],
            actions=["escalate"]
        ),
        "OPTIMIZATION",
        "priority level redundancy"
    ))

    return tests


def generate_state_machine_tests():
    """20 state machine test cases."""
    tests = []

    # --- TRUE_PRIMITIVES (8 cases) ---
    tests.append((
        StateMachine(
            raw="S0 -> S1",
            domain="state",
            states=["S0", "S1"],
            transitions=[("S0", "go", "S1")]
        ),
        "TRUE_PRIMITIVE",
        "minimal 2-state machine"
    ))
    tests.append((
        StateMachine(
            raw="idle -> running -> stopped",
            domain="state",
            states=["idle", "running", "stopped"],
            transitions=[("idle", "start", "running"), ("running", "stop", "stopped")]
        ),
        "TRUE_PRIMITIVE",
        "linear 3-state machine"
    ))
    tests.append((
        StateMachine(
            raw="login flow",
            domain="state",
            states=["logged_out", "authenticating", "logged_in"],
            transitions=[
                ("logged_out", "login", "authenticating"),
                ("authenticating", "success", "logged_in"),
                ("authenticating", "fail", "logged_out")
            ]
        ),
        "TRUE_PRIMITIVE",
        "auth state machine"
    ))
    tests.append((
        StateMachine(
            raw="order states",
            domain="state",
            states=["pending", "confirmed", "shipped", "delivered"],
            transitions=[
                ("pending", "confirm", "confirmed"),
                ("confirmed", "ship", "shipped"),
                ("shipped", "deliver", "delivered")
            ]
        ),
        "TRUE_PRIMITIVE",
        "order lifecycle"
    ))
    tests.append((
        StateMachine(
            raw="traffic light",
            domain="state",
            states=["red", "green", "yellow"],
            transitions=[
                ("red", "timer", "green"),
                ("green", "timer", "yellow"),
                ("yellow", "timer", "red")
            ]
        ),
        "TRUE_PRIMITIVE",
        "cyclic traffic light"
    ))
    tests.append((
        StateMachine(
            raw="door",
            domain="state",
            states=["open", "closed"],
            transitions=[
                ("open", "close", "closed"),
                ("closed", "open", "open")
            ]
        ),
        "TRUE_PRIMITIVE",
        "binary toggle"
    ))
    tests.append((
        StateMachine(
            raw="KYB flow",
            domain="state",
            states=["NoDocs", "Started", "PendingApproval", "Approved", "Rejected"],
            transitions=[
                ("NoDocs", "upload", "Started"),
                ("Started", "complete", "PendingApproval"),
                ("PendingApproval", "approve", "Approved"),
                ("PendingApproval", "reject", "Rejected")
            ]
        ),
        "TRUE_PRIMITIVE",
        "KYB verification flow"
    ))
    tests.append((
        StateMachine(
            raw="payment",
            domain="state",
            states=["initiated", "processing", "completed", "failed"],
            transitions=[
                ("initiated", "process", "processing"),
                ("processing", "success", "completed"),
                ("processing", "error", "failed")
            ]
        ),
        "TRUE_PRIMITIVE",
        "payment processing"
    ))

    # --- OPTIMIZATIONS (12 cases) ---
    tests.append((
        StateMachine(
            raw="with unreachable",
            domain="state",
            states=["S0", "S1", "UNREACHABLE"],
            transitions=[("S0", "go", "S1")]
        ),
        "OPTIMIZATION",
        "unreachable state"
    ))
    tests.append((
        StateMachine(
            raw="duplicate transitions",
            domain="state",
            states=["A", "B"],
            transitions=[("A", "go", "B"), ("A", "go", "B")]
        ),
        "OPTIMIZATION",
        "duplicate transition"
    ))
    tests.append((
        StateMachine(
            raw="orphan states",
            domain="state",
            states=["start", "middle", "end", "orphan1", "orphan2"],
            transitions=[
                ("start", "next", "middle"),
                ("middle", "next", "end")
            ]
        ),
        "OPTIMIZATION",
        "two orphan states"
    ))
    tests.append((
        StateMachine(
            raw="redundant self-loop machine",
            domain="state",
            states=["A", "B", "DEAD"],
            transitions=[
                ("A", "go", "B"),
                ("DEAD", "loop", "DEAD")  # unreachable self-loop
            ]
        ),
        "OPTIMIZATION",
        "unreachable self-loop"
    ))
    tests.append((
        StateMachine(
            raw="equivalent states",
            domain="state",
            states=["A", "B", "C"],
            transitions=[
                ("A", "x", "C"),
                ("B", "x", "C")  # A and B have same behavior
            ]
        ),
        "OPTIMIZATION",
        "equivalent states (same outgoing)"
    ))
    tests.append((
        StateMachine(
            raw="dead branch",
            domain="state",
            states=["main", "branch", "dead_end"],
            transitions=[
                ("main", "go", "branch"),
                ("dead_end", "stuck", "dead_end")
            ]
        ),
        "OPTIMIZATION",
        "unreachable dead branch"
    ))
    tests.append((
        StateMachine(
            raw="triple duplicate",
            domain="state",
            states=["X", "Y"],
            transitions=[
                ("X", "a", "Y"),
                ("X", "a", "Y"),
                ("X", "a", "Y")
            ]
        ),
        "OPTIMIZATION",
        "triple duplicate transition"
    ))
    tests.append((
        StateMachine(
            raw="orphan cluster",
            domain="state",
            states=["active", "idle", "orphan1", "orphan2", "orphan3"],
            transitions=[
                ("active", "pause", "idle"),
                ("idle", "resume", "active"),
                ("orphan1", "x", "orphan2"),  # unreachable cluster
                ("orphan2", "y", "orphan3")
            ]
        ),
        "OPTIMIZATION",
        "orphan state cluster"
    ))
    tests.append((
        StateMachine(
            raw="single orphan",
            domain="state",
            states=["A", "B", "C", "ORPHAN"],
            transitions=[
                ("A", "step1", "B"),
                ("B", "step2", "C")
            ]
        ),
        "OPTIMIZATION",
        "single orphan state"
    ))
    tests.append((
        StateMachine(
            raw="redundant path",
            domain="state",
            states=["start", "end", "unused"],
            transitions=[
                ("start", "done", "end"),
                ("unused", "never", "end")
            ]
        ),
        "OPTIMIZATION",
        "unused state with transition"
    ))
    tests.append((
        StateMachine(
            raw="dup with extra",
            domain="state",
            states=["P", "Q", "R", "DEAD"],
            transitions=[
                ("P", "go", "Q"),
                ("P", "go", "Q"),
                ("Q", "next", "R")
            ]
        ),
        "OPTIMIZATION",
        "duplicate + unreachable"
    ))
    tests.append((
        StateMachine(
            raw="complex orphan",
            domain="state",
            states=["S1", "S2", "S3", "LOST1", "LOST2"],
            transitions=[
                ("S1", "a", "S2"),
                ("S2", "b", "S3"),
                ("LOST1", "c", "LOST2")
            ]
        ),
        "OPTIMIZATION",
        "unreachable pair"
    ))

    return tests


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all 100 test cases."""
    print("=" * 80)
    print("  IRREDUCIBILITY ENGINE: 100 REAL-WORLD TEST CASES")
    print("=" * 80)

    engine = IrreducibilityEngine()

    # Gather all tests
    all_tests = []

    math_tests = generate_math_tests()
    for expr, expected, desc in math_tests:
        all_tests.append(("math", expr, expected, desc))

    data_tests = generate_data_transform_tests()
    for data, expected, desc in data_tests:
        all_tests.append(("data", data, expected, desc))

    rule_tests = generate_business_rule_tests()
    for rule, expected, desc in rule_tests:
        all_tests.append(("rule", rule, expected, desc))

    state_tests = generate_state_machine_tests()
    for sm, expected, desc in state_tests:
        all_tests.append(("state", sm, expected, desc))

    print(f"\nTotal test cases: {len(all_tests)}")
    print(f"  - Math expressions: {len(math_tests)}")
    print(f"  - Data transforms:  {len(data_tests)}")
    print(f"  - Business rules:   {len(rule_tests)}")
    print(f"  - State machines:   {len(state_tests)}")
    print()

    # Run tests
    results = {
        "pass": 0,
        "fail": 0,
        "by_domain": {
            "math": {"pass": 0, "fail": 0},
            "data": {"pass": 0, "fail": 0},
            "rule": {"pass": 0, "fail": 0},
            "state": {"pass": 0, "fail": 0}
        },
        "by_expected": {
            "TRUE_PRIMITIVE": {"pass": 0, "fail": 0},
            "ISOMORPHISM": {"pass": 0, "fail": 0},
            "OPTIMIZATION": {"pass": 0, "fail": 0}
        }
    }

    failures = []

    for i, (domain, input_data, expected, desc) in enumerate(all_tests):
        try:
            result = engine.analyze(input_data)
            actual = result.classification.value

            passed = actual == expected

            if passed:
                results["pass"] += 1
                results["by_domain"][domain]["pass"] += 1
                results["by_expected"][expected]["pass"] += 1
            else:
                results["fail"] += 1
                results["by_domain"][domain]["fail"] += 1
                results["by_expected"][expected]["fail"] += 1

                if isinstance(input_data, str):
                    input_str = input_data[:40]
                else:
                    input_str = input_data.raw[:40] if hasattr(input_data, 'raw') else str(input_data)[:40]

                failures.append({
                    "domain": domain,
                    "input": input_str,
                    "expected": expected,
                    "actual": actual,
                    "desc": desc
                })

        except Exception as e:
            results["fail"] += 1
            results["by_domain"][domain]["fail"] += 1
            results["by_expected"][expected]["fail"] += 1
            failures.append({
                "domain": domain,
                "input": str(input_data)[:40],
                "expected": expected,
                "actual": f"ERROR: {e}",
                "desc": desc
            })

        # Progress
        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/{len(all_tests)}] {results['pass']} passed, {results['fail']} failed")

    # Final summary
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    total = results["pass"] + results["fail"]
    pct = results["pass"] / total * 100

    print(f"\n  Overall: {results['pass']}/{total} passed ({pct:.1f}%)")

    print("\n  By Domain:")
    print("  " + "-" * 50)
    for domain, counts in results["by_domain"].items():
        t = counts["pass"] + counts["fail"]
        p = counts["pass"] / t * 100 if t > 0 else 0
        bar = "#" * int(p / 5)
        print(f"    {domain:<10}: {counts['pass']:>2}/{t:<2} ({p:5.1f}%) {bar}")

    print("\n  By Classification:")
    print("  " + "-" * 50)
    for typ, counts in results["by_expected"].items():
        t = counts["pass"] + counts["fail"]
        p = counts["pass"] / t * 100 if t > 0 else 0
        bar = "#" * int(p / 5)
        print(f"    {typ:<16}: {counts['pass']:>2}/{t:<2} ({p:5.1f}%) {bar}")

    # Show failures
    if failures:
        print("\n" + "=" * 80)
        print("  FAILURES (first 15)")
        print("=" * 80)
        for f in failures[:15]:
            print(f"    [{f['domain']}] {f['input'][:30]:<30}")
            print(f"           expected: {f['expected']}, got: {f['actual']}")
            print(f"           ({f['desc']})")
        if len(failures) > 15:
            print(f"    ... and {len(failures) - 15} more failures")

    print("\n" + "=" * 80)
    if results["fail"] == 0:
        print("  ALL 100 TESTS PASSED!")
    else:
        print(f"  {results['pass']}/{total} TESTS PASSED")
    print("=" * 80)

    return results


def run_verification_tests():
    """Run verification tests to ensure minimal forms are equivalent."""
    print("\n" + "=" * 80)
    print("  VERIFICATION TESTS")
    print("=" * 80)

    engine = IrreducibilityEngine()

    # Test cases with known test inputs for verification
    verification_cases = [
        # (expression, test_inputs)
        ("x*x*x*x", [{"x": 2}, {"x": 3}, {"x": -1}, {"x": 0.5}]),
        ("x+y", [{"x": 1, "y": 2}, {"x": -5, "y": 3}, {"x": 0, "y": 0}]),
        ("(x+y)+(x+y)", [{"x": 2, "y": 3}, {"x": -1, "y": 4}]),
        ("a*a*a*a*a*a*a*a", [{"a": 2}, {"a": -1}, {"a": 1.5}]),
        ("x+y+z", [{"x": 1, "y": 2, "z": 3}, {"x": -1, "y": -2, "z": -3}]),
    ]

    print(f"\n  Running {len(verification_cases)} verification tests...\n")

    passed = 0
    failed = 0

    for expr, test_inputs in verification_cases:
        minimal = engine.compute_minimal(expr)
        verified = engine.verify(expr, minimal, test_inputs)

        if verified:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"    [{status}] {expr[:30]:<30} -> {str(minimal)[:30]}")

    print(f"\n  Verification: {passed}/{passed + failed} passed")

    return {"pass": passed, "fail": failed}


def run_batch_analysis_demo():
    """Demo batch analysis and summary statistics."""
    print("\n" + "=" * 80)
    print("  BATCH ANALYSIS DEMO")
    print("=" * 80)

    engine = IrreducibilityEngine()

    # Sample of real-world expressions
    sample = [
        "x+y",
        "x*x*x*x",
        "(a+b)*(a-b)",
        "x+x+x+x",
        "(x+y)+(x+y)",
        "a*b+c*d",
        "z*z*z*z*z*z*z*z",
        "p+q+r+s",
        "(m*n)+(m*n)+(m*n)",
        "x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x",
    ]

    print(f"\n  Analyzing {len(sample)} expressions...")

    results = engine.batch_analyze(sample)
    summary = engine.summary(results)

    print(f"\n  Summary Statistics:")
    print(f"    Total analyzed:        {summary['total']}")
    print(f"    Optimal count:         {summary['optimal_count']}")
    print(f"    Optimal percentage:    {summary['optimal_percentage']:.1f}%")
    print(f"    Average confidence:    {summary['average_confidence']:.2f}")
    print(f"    Total potential savings: {summary['total_potential_savings']} operations")

    print(f"\n  By Type:")
    for typ, count in summary['by_type'].items():
        if count > 0:
            print(f"    {typ}: {count}")

    return summary


if __name__ == "__main__":
    # Run main test suite
    main_results = run_all_tests()

    # Run verification tests
    verify_results = run_verification_tests()

    # Run batch demo
    batch_summary = run_batch_analysis_demo()

    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    total_tests = main_results["pass"] + main_results["fail"] + verify_results["pass"] + verify_results["fail"]
    total_passed = main_results["pass"] + verify_results["pass"]
    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    print("=" * 80)
