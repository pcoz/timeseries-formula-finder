"""
TOLERANT FORM DETECTION TEST
============================

Test if our framework can detect forms in NN outputs when we:
1. Round values to nearest plausible form predictions
2. Search for forms that fit within error bounds
3. Use the actual form detector (not regression)
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import List, Tuple, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from core.form_validator import FormValidator, ValidationResult
    from core.recursive_form_flow import RecursiveFormFlow
    from core.numerical_form_analyzer import NumericalFormAnalyzer, FormType
except ImportError:
    sys.path.insert(0, '../core')
    from form_validator import FormValidator, ValidationResult
    from recursive_form_flow import RecursiveFormFlow
    from numerical_form_analyzer import NumericalFormAnalyzer, FormType

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn required")
    sys.exit(1)


def snap_to_nearest_integer(values: List[float]) -> List[float]:
    """Round to nearest integer"""
    return [round(v) for v in values]


def snap_to_form_candidates(values: List[float], tolerance: float = 0.5) -> List[List[float]]:
    """
    Generate candidate sequences by snapping to plausible integer values.
    For each value, try rounding up and down if within tolerance.
    Returns multiple candidate sequences.
    """
    candidates = [[]]

    for v in values:
        new_candidates = []
        floor_v = int(np.floor(v))
        ceil_v = int(np.ceil(v))

        for cand in candidates:
            # Always include rounded value
            new_candidates.append(cand + [round(v)])

            # If close to integer boundaries, try both
            if abs(v - floor_v) < tolerance and floor_v != round(v):
                new_candidates.append(cand + [float(floor_v)])
            if abs(v - ceil_v) < tolerance and ceil_v != round(v):
                new_candidates.append(cand + [float(ceil_v)])

        candidates = new_candidates

        # Limit explosion - keep only 50 best candidates
        if len(candidates) > 50:
            candidates = candidates[:50]

    return candidates


def try_detect_form(values: List[float], analyzer: RecursiveFormFlow,
                    validator: FormValidator) -> Tuple[Optional[str], str]:
    """Try to detect a form in the values"""
    form = analyzer.analyze(values)
    if form is None:
        return None, "NO_FORM"

    result = validator.validate(values)
    return form.base_form.form_type, result.validation.value


def search_forms_with_tolerance(nn_outputs: List[float],
                                 max_attempts: int = 100) -> dict:
    """
    Search for a valid form by trying different roundings of NN outputs.
    """
    analyzer = RecursiveFormFlow(max_depth=4)
    validator = FormValidator(strict_mode=False)  # Relaxed mode

    # Method 1: Direct integer rounding
    rounded = snap_to_nearest_integer(nn_outputs)
    form_type, validation = try_detect_form(rounded, analyzer, validator)

    if validation == "true_form":
        return {
            'method': 'direct_rounding',
            'form_type': form_type,
            'validation': validation,
            'values': rounded
        }

    # Method 2: Try multiple candidate sequences
    candidates = snap_to_form_candidates(nn_outputs, tolerance=0.5)

    for cand in candidates[:max_attempts]:
        form_type, validation = try_detect_form(cand, analyzer, validator)
        if validation == "true_form":
            return {
                'method': 'candidate_search',
                'form_type': form_type,
                'validation': validation,
                'values': cand
            }

    # Method 3: Try with more tolerance (as floats)
    form_type, validation = try_detect_form(nn_outputs, analyzer, validator)

    return {
        'method': 'none_found',
        'form_type': form_type,
        'validation': validation,
        'values': nn_outputs
    }


def train_and_test(name: str, true_func: Callable,
                   x_range: Tuple[float, float]) -> dict:
    """Train NN and test form detection with tolerance"""

    # Train NN
    X_train = np.linspace(x_range[0], x_range[1], 200).reshape(-1, 1)
    y_train = np.array([true_func(x[0]) for x in X_train])

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    nn = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        max_iter=5000,
        early_stopping=True,
        random_state=42
    )
    nn.fit(X_scaled, y_scaled)

    # Sample
    n_sample = 15
    sample_points = np.array([[float(i)] for i in range(n_sample)])
    sample_scaled = scaler_x.transform(sample_points)
    nn_scaled = nn.predict(sample_scaled)
    nn_outputs = scaler_y.inverse_transform(nn_scaled.reshape(-1, 1)).ravel().tolist()

    true_outputs = [true_func(i) for i in range(n_sample)]

    # Baseline: what form does true data have?
    analyzer = RecursiveFormFlow(max_depth=4)
    validator = FormValidator(strict_mode=True)

    true_form = analyzer.analyze(true_outputs)
    true_result = validator.validate(true_outputs)
    true_type = true_form.base_form.form_type if true_form else None

    # Search for form in NN outputs
    search_result = search_forms_with_tolerance(nn_outputs)

    # Calculate error
    nn_error = np.mean(np.abs(np.array(nn_outputs) - np.array(true_outputs)))

    return {
        'name': name,
        'nn_outputs': nn_outputs,
        'true_outputs': true_outputs,
        'nn_error': nn_error,
        'true_type': str(true_type),
        'true_validation': true_result.validation.value,
        'nn_search_result': search_result,
        'success': (search_result['validation'] == 'true_form' and
                   search_result['form_type'] == true_type)
    }


def run_tests():
    """Run tolerant form detection tests"""

    print("=" * 70)
    print("  TOLERANT FORM DETECTION")
    print("  Using OUR framework's form detection with error tolerance")
    print("=" * 70)

    tests = [
        ("Linear: 2x + 3", lambda x: 2*x + 3, (0, 20)),
        ("Quadratic: x²", lambda x: x**2, (0, 15)),
        ("Triangular", lambda x: x*(x+1)/2, (0, 15)),
        ("Powers of 2", lambda x: 2**x, (0, 12)),
        ("Fibonacci", lambda x: [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377][int(x)] if x < 15 else 0, (0, 12)),
        ("Cubic: x³", lambda x: x**3, (0, 10)),
    ]

    results = []

    for name, func, x_range in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        try:
            result = train_and_test(name, func, x_range)

            print(f"\n  True outputs:  {[int(x) if x == int(x) else f'{x:.1f}' for x in result['true_outputs'][:8]]}")
            print(f"  NN outputs:    {[f'{x:.1f}' for x in result['nn_outputs'][:8]]}")
            print(f"  NN error:      {result['nn_error']:.2f}")

            print(f"\n  True form:     {result['true_type']} ({result['true_validation']})")

            sr = result['nn_search_result']
            print(f"\n  NN form search:")
            print(f"    Method:      {sr['method']}")
            print(f"    Form type:   {sr['form_type']}")
            print(f"    Validation:  {sr['validation']}")
            if sr['method'] != 'none_found':
                print(f"    Snapped to:  {[int(x) if x == int(x) else x for x in sr['values'][:8]]}")

            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"\n  Result: {status}")

            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'name': name, 'success': False})

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    success = sum(1 for r in results if r.get('success', False))

    print(f"  {'Function':<20} {'NN Error':<10} {'True Type':<15} {'Found':<15} {'Status'}")
    print("  " + "-" * 70)

    for r in results:
        if 'nn_error' not in r:
            continue
        true_t = r['true_type'][:12]
        found = r['nn_search_result']['form_type']
        found_t = str(found)[:12] if found else 'None'
        status = "SUCCESS" if r['success'] else "FAILED"
        print(f"  {r['name']:<20} {r['nn_error']:<10.2f} {true_t:<15} {found_t:<15} {status}")

    print()
    print(f"  Successful extractions: {success}/{len(results)}")

    print()
    print("=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print()

    if success >= len(results) * 0.5:
        print(f"  POSITIVE: {success}/{len(results)} forms extracted with tolerance.")
        print("  Tolerant form detection can recover forms from NN approximations.")
    else:
        print(f"  NEGATIVE: Only {success}/{len(results)} forms extracted.")
        print("  Even with tolerance, NN outputs diverge too much from true forms.")

    return results


if __name__ == "__main__":
    run_tests()
