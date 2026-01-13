"""
NN INTERNAL FORM DETECTION
==========================

Analyze the INTERNAL transforms of a neural network to find
closed-form sub-computations within the weight matrices.

The idea: Maybe the NN has learned weights that implement
closed forms - we just need to find them inside the network.
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


def analyze_weight_matrix(W: np.ndarray, name: str) -> dict:
    """
    Analyze a weight matrix for patterns that suggest closed forms.

    Look for:
    - Rows/columns with arithmetic progressions (linear)
    - Rows/columns with geometric progressions (exponential)
    - Repeated patterns suggesting polynomial structure
    """
    results = {
        'name': name,
        'shape': W.shape,
        'patterns_found': []
    }

    # Check rows for patterns
    for i, row in enumerate(W):
        if len(row) < 3:
            continue

        # Check for arithmetic progression (constant difference)
        diffs = np.diff(row)
        if np.std(diffs) < 0.1 * np.abs(np.mean(diffs)) + 0.01:
            results['patterns_found'].append({
                'type': 'linear_row',
                'index': i,
                'slope': np.mean(diffs),
                'intercept': row[0]
            })

        # Check for geometric progression (constant ratio)
        nonzero = row[row != 0]
        if len(nonzero) >= 3:
            ratios = nonzero[1:] / nonzero[:-1]
            if np.std(ratios) < 0.1 * np.abs(np.mean(ratios)) + 0.01:
                results['patterns_found'].append({
                    'type': 'geometric_row',
                    'index': i,
                    'ratio': np.mean(ratios),
                    'initial': nonzero[0]
                })

    # Check for diagonal patterns
    if W.shape[0] == W.shape[1]:
        diag = np.diag(W)
        diffs = np.diff(diag)
        if np.std(diffs) < 0.1 * np.abs(np.mean(diffs)) + 0.01:
            results['patterns_found'].append({
                'type': 'diagonal_linear',
                'slope': np.mean(diffs),
                'values': diag.tolist()[:5]
            })

    return results


def analyze_layer_computation(W: np.ndarray, b: np.ndarray,
                              sample_inputs: np.ndarray) -> dict:
    """
    Analyze what a layer computes for specific inputs.

    For each neuron, check if its output follows a closed form
    as a function of the input.
    """
    # For inputs [0, 1, 2, ..., n], check what each neuron computes
    outputs = sample_inputs @ W + b  # Shape: (n_samples, n_neurons)

    results = {
        'neuron_forms': [],
        'n_neurons': W.shape[1]
    }

    n_samples = len(sample_inputs)
    x = np.arange(n_samples, dtype=float)

    for j in range(outputs.shape[1]):
        y = outputs[:, j]

        # Try to fit different forms
        best_form = None
        best_r2 = 0

        # Linear: y = ax + b
        try:
            coeffs = np.polyfit(x, y, 1)
            pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - pred)**2) / (np.sum((y - np.mean(y))**2) + 1e-10)
            if r2 > best_r2 and r2 > 0.95:
                best_r2 = r2
                best_form = {'type': 'linear', 'coeffs': coeffs.tolist(), 'r2': r2}
        except:
            pass

        # Quadratic: y = ax^2 + bx + c
        try:
            coeffs = np.polyfit(x, y, 2)
            pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - pred)**2) / (np.sum((y - np.mean(y))**2) + 1e-10)
            if r2 > best_r2 and r2 > 0.95:
                best_r2 = r2
                best_form = {'type': 'quadratic', 'coeffs': coeffs.tolist(), 'r2': r2}
        except:
            pass

        if best_form:
            results['neuron_forms'].append({
                'neuron': j,
                'form': best_form
            })

    return results


def extract_nn_internals(nn, scaler_x, scaler_y) -> dict:
    """Extract and analyze the internal structure of a trained NN"""

    results = {
        'layers': [],
        'layer_computations': []
    }

    # Get weights and biases
    weights = nn.coefs_  # List of weight matrices
    biases = nn.intercepts_  # List of bias vectors

    for i, (W, b) in enumerate(zip(weights, biases)):
        # Analyze weight matrix structure
        weight_analysis = analyze_weight_matrix(W, f"Layer {i+1}")
        results['layers'].append({
            'layer': i + 1,
            'weight_shape': W.shape,
            'bias_shape': b.shape,
            'weight_patterns': weight_analysis['patterns_found'],
            'weight_stats': {
                'mean': float(np.mean(W)),
                'std': float(np.std(W)),
                'max': float(np.max(np.abs(W)))
            }
        })

    # Analyze what each layer computes for standard inputs
    # Sample inputs: [0, 1, 2, ..., 14]
    sample_raw = np.array([[float(i)] for i in range(15)])
    sample_scaled = scaler_x.transform(sample_raw)

    current_input = sample_scaled
    for i, (W, b) in enumerate(zip(weights, biases)):
        layer_analysis = analyze_layer_computation(W, b, current_input)
        results['layer_computations'].append({
            'layer': i + 1,
            'neurons_with_forms': len(layer_analysis['neuron_forms']),
            'total_neurons': layer_analysis['n_neurons'],
            'forms_found': layer_analysis['neuron_forms'][:5]  # First 5
        })

        # Compute next layer input (with ReLU except last layer)
        current_input = current_input @ W + b
        if i < len(weights) - 1:
            current_input = np.maximum(0, current_input)  # ReLU

    return results


def test_internal_forms(name: str, true_func: Callable,
                        x_range: Tuple[float, float]) -> dict:
    """Train NN and analyze its internal structure for forms"""

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
        hidden_layer_sizes=(32, 32),
        activation='relu',
        solver='adam',
        max_iter=5000,
        early_stopping=True,
        random_state=42
    )
    nn.fit(X_scaled, y_scaled)

    # Analyze internal structure
    internals = extract_nn_internals(nn, scaler_x, scaler_y)

    # Report findings
    print(f"\n  Network structure:")
    for layer in internals['layers']:
        print(f"    Layer {layer['layer']}: {layer['weight_shape']} weights")
        if layer['weight_patterns']:
            for p in layer['weight_patterns'][:2]:
                print(f"      Pattern: {p['type']}")

    print(f"\n  Layer computations (forms found in neuron outputs):")
    total_forms = 0
    for comp in internals['layer_computations']:
        n_forms = comp['neurons_with_forms']
        total_forms += n_forms
        print(f"    Layer {comp['layer']}: {n_forms}/{comp['total_neurons']} neurons have closed forms")
        for form in comp['forms_found'][:3]:
            f = form['form']
            print(f"      Neuron {form['neuron']}: {f['type']} (R2={f['r2']:.3f})")

    return {
        'name': name,
        'internals': internals,
        'total_forms_found': total_forms
    }


def run_tests():
    """Test internal form detection"""

    print("=" * 70)
    print("  NN INTERNAL FORM DETECTION")
    print("  Analyzing weight matrices and neuron computations")
    print("=" * 70)

    tests = [
        ("Linear: 2x + 3", lambda x: 2*x + 3, (0, 20)),
        ("Quadratic: x^2", lambda x: x**2, (0, 15)),
        ("Triangular", lambda x: x*(x+1)/2, (0, 15)),
        ("Powers of 2", lambda x: 2**x, (0, 12)),
        ("Cubic: x^3", lambda x: x**3, (0, 10)),
    ]

    results = []

    for name, func, x_range in tests:
        try:
            result = test_internal_forms(name, func, x_range)
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

    for r in results:
        print(f"  {r['name']:<25} Forms found: {r['total_forms_found']}")

    print()
    print("=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print()

    total = sum(r['total_forms_found'] for r in results)
    if total > len(results) * 5:
        print(f"  POSITIVE: Found {total} closed-form neuron computations.")
        print("  NN internals DO contain closed-form sub-computations.")
    else:
        print(f"  NEGATIVE: Only {total} closed-form neurons found.")
        print("  NN internals are distributed, not cleanly decomposable.")

    return results


if __name__ == "__main__":
    run_tests()
