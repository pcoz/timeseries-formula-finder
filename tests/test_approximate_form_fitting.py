"""
APPROXIMATE FORM FITTING TEST
=============================

Test if we can find approximate closed forms that fit NN outputs,
even if not exact. This is a different approach:
- Fit form parameters using regression
- Accept approximate matches within error bounds
- Report how well the form approximates the NN behavior
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn required for this test")
    sys.exit(1)


@dataclass
class ApproximateForm:
    """A form fitted to approximate data"""
    form_type: str
    parameters: dict
    formula: str
    r_squared: float  # How well it fits (0-1)
    max_error: float  # Maximum absolute error
    mean_error: float  # Mean absolute error


def fit_linear(x: np.ndarray, y: np.ndarray) -> ApproximateForm:
    """Fit y = ax + b"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-10:
        return None

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n

    predicted = a * x + b
    return _make_form("linear", {"a": a, "b": b}, f"{a:.4f}*x + {b:.4f}", x, y, predicted)


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> ApproximateForm:
    """Fit y = ax^2 + bx + c"""
    try:
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        predicted = a * x**2 + b * x + c
        return _make_form("quadratic", {"a": a, "b": b, "c": c},
                         f"{a:.4f}*x² + {b:.4f}*x + {c:.4f}", x, y, predicted)
    except:
        return None


def fit_cubic(x: np.ndarray, y: np.ndarray) -> ApproximateForm:
    """Fit y = ax^3 + bx^2 + cx + d"""
    try:
        coeffs = np.polyfit(x, y, 3)
        a, b, c, d = coeffs
        predicted = a * x**3 + b * x**2 + c * x + d
        return _make_form("cubic", {"a": a, "b": b, "c": c, "d": d},
                         f"{a:.4f}*x³ + {b:.4f}*x² + {c:.4f}*x + {d:.4f}", x, y, predicted)
    except:
        return None


def fit_exponential(x: np.ndarray, y: np.ndarray) -> ApproximateForm:
    """Fit y = a * b^x (using log transform)"""
    try:
        # Filter positive values for log
        mask = y > 0
        if np.sum(mask) < 3:
            return None

        x_pos = x[mask]
        y_pos = y[mask]

        # Log transform: log(y) = log(a) + x*log(b)
        log_y = np.log(y_pos)
        coeffs = np.polyfit(x_pos, log_y, 1)
        log_b, log_a = coeffs

        a = np.exp(log_a)
        b = np.exp(log_b)

        predicted = a * (b ** x)
        return _make_form("exponential", {"a": a, "b": b},
                         f"{a:.4f} * {b:.4f}^x", x, y, predicted)
    except:
        return None


def fit_power(x: np.ndarray, y: np.ndarray) -> ApproximateForm:
    """Fit y = a * x^b (using log transform)"""
    try:
        # Filter positive values
        mask = (x > 0) & (y > 0)
        if np.sum(mask) < 3:
            return None

        x_pos = x[mask]
        y_pos = y[mask]

        # Log transform: log(y) = log(a) + b*log(x)
        log_x = np.log(x_pos)
        log_y = np.log(y_pos)
        coeffs = np.polyfit(log_x, log_y, 1)
        b, log_a = coeffs

        a = np.exp(log_a)

        # Predict on all x (set 0 prediction for x=0)
        predicted = np.where(x > 0, a * (x ** b), 0)
        return _make_form("power", {"a": a, "b": b},
                         f"{a:.4f} * x^{b:.4f}", x, y, predicted)
    except:
        return None


def _make_form(form_type: str, params: dict, formula: str,
               x: np.ndarray, y: np.ndarray, predicted: np.ndarray) -> ApproximateForm:
    """Calculate fit metrics and create ApproximateForm"""
    errors = np.abs(y - predicted)
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return ApproximateForm(
        form_type=form_type,
        parameters=params,
        formula=formula,
        r_squared=max(0, r_squared),
        max_error=np.max(errors),
        mean_error=np.mean(errors)
    )


def find_best_form(nn_outputs: List[float]) -> Tuple[ApproximateForm, List[ApproximateForm]]:
    """
    Find the best-fitting closed form for NN outputs.
    Returns (best_form, all_forms_sorted_by_fit)
    """
    x = np.array(list(range(len(nn_outputs))), dtype=float)
    y = np.array(nn_outputs)

    forms = []
    for fitter in [fit_linear, fit_quadratic, fit_cubic, fit_exponential, fit_power]:
        form = fitter(x, y)
        if form is not None:
            forms.append(form)

    # Sort by R² (higher is better)
    forms.sort(key=lambda f: f.r_squared, reverse=True)

    return (forms[0] if forms else None, forms)


def train_nn_and_extract(true_func: Callable, x_range: Tuple[float, float],
                         n_train: int = 200, n_sample: int = 15) -> dict:
    """Train NN on function and try to extract approximate form"""

    # Generate training data
    X_train = np.linspace(x_range[0], x_range[1], n_train).reshape(-1, 1)
    y_train = np.array([true_func(x[0]) for x in X_train])

    # Normalize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Train
    nn = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        max_iter=5000,
        early_stopping=True,
        random_state=42
    )
    nn.fit(X_scaled, y_scaled)

    # Sample at integer points
    sample_points = np.array([[float(i)] for i in range(n_sample)])
    sample_scaled = scaler_x.transform(sample_points)
    nn_scaled = nn.predict(sample_scaled)
    nn_outputs = scaler_y.inverse_transform(nn_scaled.reshape(-1, 1)).ravel().tolist()

    true_outputs = [true_func(i) for i in range(n_sample)]

    # Find best approximate form
    best_form, all_forms = find_best_form(nn_outputs)

    # Also find best form for true outputs (baseline)
    true_best, _ = find_best_form(true_outputs)

    return {
        'nn_outputs': nn_outputs,
        'true_outputs': true_outputs,
        'nn_best_form': best_form,
        'true_best_form': true_best,
        'all_forms': all_forms
    }


def run_tests():
    """Run approximate form fitting tests"""

    print("=" * 70)
    print("  APPROXIMATE FORM FITTING TEST")
    print("  Can we find closed forms that APPROXIMATE NN outputs?")
    print("=" * 70)

    tests = [
        ("Linear: 2x + 3", lambda x: 2*x + 3, (0, 20)),
        ("Quadratic: x²", lambda x: x**2, (0, 15)),
        ("Quadratic: x² + x", lambda x: x**2 + x, (0, 12)),
        ("Powers of 2", lambda x: 2**x, (0, 12)),
        ("Triangular: n(n+1)/2", lambda x: x*(x+1)/2, (0, 15)),
        ("Cubic: x³", lambda x: x**3, (0, 10)),
    ]

    results = []

    for name, func, x_range in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        try:
            result = train_nn_and_extract(func, x_range)

            nn_form = result['nn_best_form']
            true_form = result['true_best_form']

            print(f"\n  True outputs:  {[f'{x:.1f}' for x in result['true_outputs'][:8]]}")
            print(f"  NN outputs:    {[f'{x:.1f}' for x in result['nn_outputs'][:8]]}")

            print(f"\n  Best fit for TRUE data:")
            print(f"    Type: {true_form.form_type}")
            print(f"    Formula: {true_form.formula}")
            print(f"    R²: {true_form.r_squared:.4f}")

            print(f"\n  Best fit for NN data:")
            print(f"    Type: {nn_form.form_type}")
            print(f"    Formula: {nn_form.formula}")
            print(f"    R²: {nn_form.r_squared:.4f}")
            print(f"    Mean error: {nn_form.mean_error:.2f}")
            print(f"    Max error: {nn_form.max_error:.2f}")

            # Check if same form type found
            same_type = nn_form.form_type == true_form.form_type
            good_fit = nn_form.r_squared >= 0.95

            status = "SUCCESS" if same_type and good_fit else "PARTIAL" if same_type or good_fit else "FAILED"
            print(f"\n  Status: {status} (same type: {same_type}, R² >= 0.95: {good_fit})")

            results.append({
                'name': name,
                'same_type': same_type,
                'nn_r2': nn_form.r_squared,
                'true_type': true_form.form_type,
                'nn_type': nn_form.form_type,
                'mean_error': nn_form.mean_error,
                'status': status
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': name,
                'same_type': False,
                'nn_r2': 0,
                'status': 'ERROR'
            })

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Function':<25} {'True Type':<12} {'NN Type':<12} {'R²':<8} {'Status'}")
    print("  " + "-" * 70)

    success = 0
    partial = 0

    for r in results:
        true_t = r.get('true_type', 'N/A')[:10]
        nn_t = r.get('nn_type', 'N/A')[:10]
        r2 = r.get('nn_r2', 0)
        status = r.get('status', 'ERROR')

        if status == 'SUCCESS':
            success += 1
        elif status == 'PARTIAL':
            partial += 1

        print(f"  {r['name']:<25} {true_t:<12} {nn_t:<12} {r2:.4f}   {status}")

    print()
    print(f"  Full success (same type + R² >= 0.95): {success}/{len(results)}")
    print(f"  Partial success: {partial}/{len(results)}")

    print()
    print("=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print()

    if success >= len(results) * 0.7:
        print("  POSITIVE: Approximate form fitting works well.")
        print("  We can find closed forms that approximate NN behavior.")
    elif success + partial >= len(results) * 0.5:
        print("  MIXED: Approximate fitting works for some functions.")
        print("  Linear/quadratic work well; exponential struggles.")
    else:
        print("  NEGATIVE: Approximate fitting has limited success.")

    return results


if __name__ == "__main__":
    run_tests()
