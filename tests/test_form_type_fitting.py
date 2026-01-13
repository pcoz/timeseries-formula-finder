"""
FORM TYPE FITTING TEST
======================

Instead of detecting exact forms, FIT our framework's specific form types
to NN outputs and find which form type best explains the NN behavior.

This answers: "Which closed form does this NN approximate?"
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
    sys.exit(1)


@dataclass
class FormFit:
    """A form type fitted to data"""
    form_type: str
    parameters: dict
    formula: str
    r_squared: float
    residual_std: float
    generator: Callable  # Function to generate values


def fit_arithmetic_seq(x: np.ndarray, y: np.ndarray) -> Optional[FormFit]:
    """Fit y = a + b*x (arithmetic sequence starting from x=0)"""
    try:
        b = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0
        a = y[0]

        # Refine with least squares
        A = np.column_stack([np.ones_like(x), x])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = params

        predicted = a + b * x
        r2 = _r_squared(y, predicted)

        return FormFit(
            form_type="ARITHMETIC_SEQ",
            parameters={"a": a, "b": b},
            formula=f"a(n) = {a:.2f} + {b:.2f}*n",
            r_squared=r2,
            residual_std=np.std(y - predicted),
            generator=lambda n, a=a, b=b: a + b * n
        )
    except:
        return None


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> Optional[FormFit]:
    """Fit y = a*x^2 + b*x + c"""
    try:
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        predicted = a * x**2 + b * x + c
        r2 = _r_squared(y, predicted)

        return FormFit(
            form_type="QUADRATIC",
            parameters={"a": a, "b": b, "c": c},
            formula=f"a(n) = {a:.4f}*n² + {b:.4f}*n + {c:.4f}",
            r_squared=r2,
            residual_std=np.std(y - predicted),
            generator=lambda n, a=a, b=b, c=c: a * n**2 + b * n + c
        )
    except:
        return None


def fit_cubic(x: np.ndarray, y: np.ndarray) -> Optional[FormFit]:
    """Fit y = a*x^3 + b*x^2 + c*x + d"""
    try:
        coeffs = np.polyfit(x, y, 3)
        a, b, c, d = coeffs
        predicted = a * x**3 + b * x**2 + c * x + d
        r2 = _r_squared(y, predicted)

        return FormFit(
            form_type="CUBIC",
            parameters={"a": a, "b": b, "c": c, "d": d},
            formula=f"a(n) = {a:.4f}*n³ + {b:.4f}*n² + {c:.4f}*n + {d:.4f}",
            r_squared=r2,
            residual_std=np.std(y - predicted),
            generator=lambda n, a=a, b=b, c=c, d=d: a * n**3 + b * n**2 + c * n + d
        )
    except:
        return None


def fit_geometric_seq(x: np.ndarray, y: np.ndarray) -> Optional[FormFit]:
    """Fit y = a * r^x"""
    try:
        mask = y > 0
        if np.sum(mask) < 3:
            return None

        x_pos = x[mask]
        y_pos = y[mask]

        log_y = np.log(y_pos)
        coeffs = np.polyfit(x_pos, log_y, 1)
        log_r, log_a = coeffs

        a = np.exp(log_a)
        r = np.exp(log_r)

        predicted = a * (r ** x)
        # Only score on positive values
        r2 = _r_squared(y[mask], predicted[mask])

        return FormFit(
            form_type="GEOMETRIC_SEQ",
            parameters={"a": a, "r": r},
            formula=f"a(n) = {a:.4f} * {r:.4f}^n",
            r_squared=r2,
            residual_std=np.std(y[mask] - predicted[mask]),
            generator=lambda n, a=a, r=r: a * (r ** n)
        )
    except:
        return None


def fit_triangular(x: np.ndarray, y: np.ndarray) -> Optional[FormFit]:
    """Fit y = a * n * (n+1) / 2 + b (scaled triangular)"""
    try:
        # Triangular numbers: T(n) = n*(n+1)/2
        triangular = x * (x + 1) / 2

        # Fit y = a * T(n) + b
        A = np.column_stack([triangular, np.ones_like(x)])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = params

        predicted = a * triangular + b
        r2 = _r_squared(y, predicted)

        return FormFit(
            form_type="TRIANGULAR",
            parameters={"a": a, "b": b},
            formula=f"a(n) = {a:.4f} * n(n+1)/2 + {b:.4f}",
            r_squared=r2,
            residual_std=np.std(y - predicted),
            generator=lambda n, a=a, b=b: a * n * (n + 1) / 2 + b
        )
    except:
        return None


def fit_fibonacci(x: np.ndarray, y: np.ndarray) -> Optional[FormFit]:
    """Try to fit a scaled Fibonacci-like sequence"""
    try:
        # Generate Fibonacci: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)
        n_max = int(max(x)) + 1
        fib = [0, 1]
        for i in range(2, n_max):
            fib.append(fib[-1] + fib[-2])
        fib = np.array(fib[:len(x)])

        if len(fib) != len(x):
            return None

        # Fit y = a * fib + b
        A = np.column_stack([fib, np.ones_like(x)])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = params

        predicted = a * fib + b
        r2 = _r_squared(y, predicted)

        def fib_gen(n, a=a, b=b):
            if n == 0:
                return b
            if n == 1:
                return a + b
            f0, f1 = 0, 1
            for _ in range(2, n + 1):
                f0, f1 = f1, f0 + f1
            return a * f1 + b

        return FormFit(
            form_type="FIBONACCI",
            parameters={"a": a, "b": b},
            formula=f"a(n) = {a:.4f} * fib(n) + {b:.4f}",
            r_squared=r2,
            residual_std=np.std(y - predicted),
            generator=fib_gen
        )
    except:
        return None


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0


def find_best_form_type(values: List[float]) -> Tuple[FormFit, List[FormFit]]:
    """Find which of our framework's form types best fits the data"""
    x = np.arange(len(values), dtype=float)
    y = np.array(values)

    fitters = [
        fit_arithmetic_seq,
        fit_quadratic,
        fit_cubic,
        fit_geometric_seq,
        fit_triangular,
        fit_fibonacci,
    ]

    fits = []
    for fitter in fitters:
        fit = fitter(x, y)
        if fit is not None:
            fits.append(fit)

    # Sort by R² (prefer simpler forms when R² is similar)
    # Penalize complexity slightly
    def score(fit):
        complexity_penalty = {
            "ARITHMETIC_SEQ": 0.0,
            "TRIANGULAR": 0.01,
            "QUADRATIC": 0.02,
            "FIBONACCI": 0.02,
            "GEOMETRIC_SEQ": 0.03,
            "CUBIC": 0.04,
        }
        return fit.r_squared - complexity_penalty.get(fit.form_type, 0.05)

    fits.sort(key=score, reverse=True)

    return (fits[0] if fits else None, fits)


def train_and_analyze(name: str, true_func: Callable,
                      true_type: str, x_range: Tuple[float, float]) -> dict:
    """Train NN and analyze which form type it learned"""

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

    # Find best form type for NN outputs
    best_fit, all_fits = find_best_form_type(nn_outputs)

    # Also fit true outputs
    true_best, _ = find_best_form_type(true_outputs)

    # Calculate how well the fitted form predicts beyond training
    if best_fit:
        # Test extrapolation
        extrap_indices = range(n_sample, n_sample + 5)
        extrap_true = [true_func(i) for i in extrap_indices]
        extrap_pred = [best_fit.generator(i) for i in extrap_indices]
        extrap_error = np.mean(np.abs(np.array(extrap_pred) - np.array(extrap_true)))
    else:
        extrap_error = float('inf')

    return {
        'name': name,
        'nn_outputs': nn_outputs,
        'true_outputs': true_outputs,
        'true_type': true_type,
        'best_fit': best_fit,
        'true_best': true_best,
        'all_fits': all_fits,
        'correct_type': best_fit.form_type == true_type if best_fit else False,
        'extrap_error': extrap_error
    }


def run_tests():
    """Run form type fitting tests"""

    print("=" * 70)
    print("  FORM TYPE FITTING")
    print("  Which of our framework's forms best approximates NN behavior?")
    print("=" * 70)

    tests = [
        ("Linear: 2x + 3", lambda x: 2*x + 3, "ARITHMETIC_SEQ", (0, 20)),
        ("Quadratic: x²", lambda x: x**2, "QUADRATIC", (0, 15)),
        ("Triangular", lambda x: x*(x+1)/2, "TRIANGULAR", (0, 15)),
        ("Powers of 2", lambda x: 2**x, "GEOMETRIC_SEQ", (0, 12)),
        ("Fibonacci", lambda x: [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377][int(x)] if x < 15 else 0, "FIBONACCI", (0, 12)),
        ("Cubic: x³", lambda x: x**3, "CUBIC", (0, 10)),
    ]

    results = []

    for name, func, true_type, x_range in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        try:
            result = train_and_analyze(name, func, true_type, x_range)

            print(f"\n  True outputs:  {[int(x) if x == int(x) else f'{x:.1f}' for x in result['true_outputs'][:8]]}")
            print(f"  NN outputs:    {[f'{x:.1f}' for x in result['nn_outputs'][:8]]}")

            print(f"\n  Expected form: {true_type}")

            if result['best_fit']:
                bf = result['best_fit']
                print(f"\n  Best fitting form type:")
                print(f"    Type:      {bf.form_type}")
                print(f"    Formula:   {bf.formula}")
                print(f"    R²:        {bf.r_squared:.4f}")
                print(f"    Residual:  {bf.residual_std:.2f}")
                print(f"    Extrap err:{result['extrap_error']:.2f}")

                print(f"\n  All fits (by R2):")
                for fit in result['all_fits'][:4]:
                    match = "*" if fit.form_type == true_type else ""
                    print(f"    {fit.form_type:<15} R2={fit.r_squared:.4f} {match}")

            status = "CORRECT" if result['correct_type'] else "WRONG"
            print(f"\n  Result: {status} form type identified")

            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'name': name, 'correct_type': False})

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    correct = sum(1 for r in results if r.get('correct_type', False))

    print(f"  {'Function':<20} {'True Type':<15} {'NN Best Fit':<15} {'R²':<8} {'Status'}")
    print("  " + "-" * 75)

    for r in results:
        if 'best_fit' not in r:
            continue
        true_t = r['true_type'][:12]
        nn_t = r['best_fit'].form_type[:12] if r['best_fit'] else 'None'
        r2 = r['best_fit'].r_squared if r['best_fit'] else 0
        status = "CORRECT" if r['correct_type'] else "WRONG"
        print(f"  {r['name']:<20} {true_t:<15} {nn_t:<15} {r2:.4f}   {status}")

    print()
    print(f"  Correct form type identified: {correct}/{len(results)}")

    # Also check functional class (grouping similar forms)
    FORM_CLASS = {
        "ARITHMETIC_SEQ": "linear",
        "QUADRATIC": "polynomial-2",
        "TRIANGULAR": "polynomial-2",  # n(n+1)/2 is degree-2
        "CUBIC": "polynomial-3",
        "GEOMETRIC_SEQ": "exponential",
        "FIBONACCI": "recursive",
    }

    class_correct = 0
    for r in results:
        if 'best_fit' not in r or r['best_fit'] is None:
            continue
        true_class = FORM_CLASS.get(r['true_type'], 'unknown')
        nn_class = FORM_CLASS.get(r['best_fit'].form_type, 'unknown')
        if true_class == nn_class:
            class_correct += 1

    print(f"  Correct functional CLASS: {class_correct}/{len(results)}")
    print()
    print("  (Classes: linear, polynomial-2, polynomial-3, exponential, recursive)")

    print()
    print("=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print()

    if correct >= len(results) * 0.7:
        print(f"  POSITIVE: {correct}/{len(results)} correct form types identified.")
        print("  We CAN determine what form type an NN approximates.")
        print("  This enables form-aware NN compression and interpretability.")
    elif correct >= len(results) * 0.4:
        print(f"  MIXED: {correct}/{len(results)} correct form types.")
        print("  Works for some forms (polynomial), struggles with others (exponential).")
    else:
        print(f"  NEGATIVE: Only {correct}/{len(results)} correct.")
        print("  Cannot reliably identify what form an NN learned.")

    return results


if __name__ == "__main__":
    run_tests()
