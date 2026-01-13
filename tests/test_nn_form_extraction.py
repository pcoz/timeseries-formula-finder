"""
NN FORM EXTRACTION VIA INTERNAL ANALYSIS
========================================

Since neurons compute closed forms internally, can we:
1. Identify which neurons implement which forms
2. Extract the closed-form parameters
3. Use the framework to validate these as TRUE forms
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
except ImportError:
    print("sklearn required")
    sys.exit(1)

try:
    from core.form_validator import FormValidator, ValidationResult
    from core.recursive_form_flow import RecursiveFormFlow
except ImportError:
    sys.path.insert(0, '../core')
    from form_validator import FormValidator, ValidationResult
    from recursive_form_flow import RecursiveFormFlow


@dataclass
class NeuronForm:
    """Closed form extracted from a neuron"""
    layer: int
    neuron_idx: int
    form_type: str
    coefficients: list
    r_squared: float
    formula: str


def extract_neuron_forms(nn, scaler_x, scaler_y, n_samples: int = 15) -> List[NeuronForm]:
    """
    Extract closed-form descriptions of what each neuron computes.

    Key insight: We sample the neuron outputs for inputs [0, 1, 2, ...n]
    and fit polynomial forms to characterize the neuron's behavior.
    """
    weights = nn.coefs_
    biases = nn.intercepts_

    # Sample inputs
    sample_raw = np.array([[float(i)] for i in range(n_samples)])
    sample_scaled = scaler_x.transform(sample_raw)

    forms = []
    x = np.arange(n_samples, dtype=float)

    current_input = sample_scaled

    for layer_idx, (W, b) in enumerate(zip(weights, biases)):
        # Compute this layer's output
        outputs = current_input @ W + b

        # Analyze each neuron
        for neuron_idx in range(outputs.shape[1]):
            y = outputs[:, neuron_idx]

            # Try polynomial fits, choose simplest adequate one
            best_form = None
            best_r2 = 0
            best_degree = 0

            for degree in [1, 2, 3]:
                try:
                    coeffs = np.polyfit(x, y, degree)
                    pred = np.polyval(coeffs, x)
                    ss_res = np.sum((y - pred)**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)

                    # Accept if R2 > 0.99 (very high bar)
                    if r2 > 0.99 and (best_form is None or degree < best_degree):
                        best_form = coeffs
                        best_r2 = r2
                        best_degree = degree
                except:
                    pass

            if best_form is not None:
                form_type = ['constant', 'linear', 'quadratic', 'cubic'][best_degree]

                # Build formula string
                terms = []
                for i, c in enumerate(best_form):
                    power = best_degree - i
                    if abs(c) > 0.001:
                        if power == 0:
                            terms.append(f"{c:.3f}")
                        elif power == 1:
                            terms.append(f"{c:.3f}*n")
                        else:
                            terms.append(f"{c:.3f}*n^{power}")
                formula = " + ".join(terms) if terms else "0"

                forms.append(NeuronForm(
                    layer=layer_idx + 1,
                    neuron_idx=neuron_idx,
                    form_type=form_type,
                    coefficients=best_form.tolist(),
                    r_squared=best_r2,
                    formula=formula
                ))

        # Propagate through layer (with ReLU except last)
        current_input = outputs
        if layer_idx < len(weights) - 1:
            current_input = np.maximum(0, current_input)

    return forms


def validate_extracted_forms(forms: List[NeuronForm], nn,
                             scaler_x, scaler_y,
                             true_func: Callable) -> dict:
    """
    Validate extracted forms:
    1. Do they accurately describe neuron behavior?
    2. Can they predict neuron outputs for new inputs?
    3. Do they help explain the overall function?
    """
    # Test on new inputs (beyond training range)
    test_inputs = np.array([[float(i)] for i in range(15, 20)])
    test_scaled = scaler_x.transform(test_inputs)

    weights = nn.coefs_
    biases = nn.intercepts_

    results = {
        'total_forms': len(forms),
        'linear_neurons': sum(1 for f in forms if f.form_type == 'linear'),
        'quadratic_neurons': sum(1 for f in forms if f.form_type == 'quadratic'),
        'cubic_neurons': sum(1 for f in forms if f.form_type == 'cubic'),
        'extrapolation_tests': []
    }

    # Test if forms extrapolate correctly
    x_test = np.array([15, 16, 17, 18, 19], dtype=float)

    current_input = test_scaled
    form_idx = 0

    for layer_idx, (W, b) in enumerate(zip(weights, biases)):
        actual_outputs = current_input @ W + b

        for neuron_idx in range(actual_outputs.shape[1]):
            # Find corresponding form
            matching_forms = [f for f in forms
                            if f.layer == layer_idx + 1 and f.neuron_idx == neuron_idx]

            if matching_forms:
                form = matching_forms[0]
                actual = actual_outputs[:, neuron_idx]
                predicted = np.polyval(form.coefficients, x_test)

                error = np.mean(np.abs(actual - predicted))
                rel_error = error / (np.mean(np.abs(actual)) + 1e-10)

                results['extrapolation_tests'].append({
                    'layer': layer_idx + 1,
                    'neuron': neuron_idx,
                    'form_type': form.form_type,
                    'mean_error': error,
                    'rel_error': rel_error,
                    'extrapolates': rel_error < 0.1  # < 10% error
                })

        current_input = actual_outputs
        if layer_idx < len(weights) - 1:
            current_input = np.maximum(0, current_input)

    # Count how many forms extrapolate correctly
    extrap_results = results['extrapolation_tests']
    successful = sum(1 for r in extrap_results if r['extrapolates'])
    results['extrapolation_success_rate'] = successful / len(extrap_results) if extrap_results else 0

    return results


def test_form_extraction(name: str, true_func: Callable,
                         x_range: Tuple[float, float]) -> dict:
    """Full pipeline: train NN, extract forms, validate"""

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Train NN
    X_train = np.linspace(x_range[0], x_range[1], 200).reshape(-1, 1)
    y_train = np.array([true_func(x[0]) for x in X_train])

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    nn = MLPRegressor(
        hidden_layer_sizes=(16, 16),  # Smaller for clearer analysis
        activation='relu',
        solver='adam',
        max_iter=5000,
        early_stopping=True,
        random_state=42
    )
    nn.fit(X_scaled, y_scaled)

    # Extract forms
    forms = extract_neuron_forms(nn, scaler_x, scaler_y)

    print(f"\n  Extracted forms by layer:")
    for layer in range(1, 4):
        layer_forms = [f for f in forms if f.layer == layer]
        print(f"    Layer {layer}: {len(layer_forms)} neurons with closed forms")
        by_type = {}
        for f in layer_forms:
            by_type[f.form_type] = by_type.get(f.form_type, 0) + 1
        for ftype, count in by_type.items():
            print(f"      {ftype}: {count}")

    # Show some example forms
    print(f"\n  Example extracted formulas:")
    for form in forms[:5]:
        print(f"    Layer {form.layer}, Neuron {form.neuron_idx}: {form.formula}")
        print(f"      (R2={form.r_squared:.4f}, type={form.form_type})")

    # Validate
    validation = validate_extracted_forms(forms, nn, scaler_x, scaler_y, true_func)

    print(f"\n  Validation results:")
    print(f"    Forms that extrapolate correctly: {validation['extrapolation_success_rate']:.0%}")

    # Now the key question: can we use these forms with our framework?
    # Let's try validating one of the extracted neuron outputs
    validator = FormValidator(strict_mode=False)

    sample_raw = np.array([[float(i)] for i in range(15)])
    sample_scaled = scaler_x.transform(sample_raw)

    # Get layer 2 output (after ReLU)
    layer1_out = np.maximum(0, sample_scaled @ nn.coefs_[0] + nn.intercepts_[0])
    layer2_out = np.maximum(0, layer1_out @ nn.coefs_[1] + nn.intercepts_[1])

    # Try to validate some neurons with our framework
    print(f"\n  Framework validation of neuron outputs:")
    framework_validated = 0
    for i in range(min(5, layer2_out.shape[1])):
        neuron_out = layer2_out[:, i].tolist()
        result = validator.validate(neuron_out)
        status = result.validation.value
        if status == 'true_form':
            framework_validated += 1
        print(f"    Neuron {i}: {status} (gen={result.generalization_score:.0%})")

    return {
        'name': name,
        'forms_extracted': len(forms),
        'extrapolation_success': validation['extrapolation_success_rate'],
        'framework_validated': framework_validated
    }


def run_tests():
    """Run form extraction tests"""

    print("=" * 70)
    print("  NN FORM EXTRACTION VIA INTERNAL ANALYSIS")
    print("  Extracting closed forms from inside the neural network")
    print("=" * 70)

    tests = [
        ("Linear: 2x + 3", lambda x: 2*x + 3, (0, 20)),
        ("Quadratic: x^2", lambda x: x**2, (0, 15)),
        ("Triangular", lambda x: x*(x+1)/2, (0, 15)),
        ("Powers of 2", lambda x: 2**x, (0, 12)),
    ]

    results = []

    for name, func, x_range in tests:
        try:
            result = test_form_extraction(name, func, x_range)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    print(f"  {'Function':<20} {'Forms':<10} {'Extrap%':<12} {'Framework'}")
    print("  " + "-" * 55)

    for r in results:
        print(f"  {r['name']:<20} {r['forms_extracted']:<10} "
              f"{r['extrapolation_success']*100:.0f}%          {r['framework_validated']}")

    print()
    print("=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print()

    avg_extrap = np.mean([r['extrapolation_success'] for r in results])
    if avg_extrap > 0.7:
        print(f"  POSITIVE: {avg_extrap*100:.0f}% of neuron forms extrapolate correctly.")
        print("  Internal NN forms CAN be extracted and validated.")
        print("  This enables form-aware compression and interpretability.")
    else:
        print(f"  MIXED: {avg_extrap*100:.0f}% extrapolation success.")
        print("  Some forms work, but ReLU/composition breaks others.")

    return results


if __name__ == "__main__":
    run_tests()
