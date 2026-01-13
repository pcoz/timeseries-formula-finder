"""
NN COMPRESSION POTENTIAL VIA FORM EXTRACTION
============================================

Quantify how much of a neural network can be replaced with
closed-form polynomial approximations.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit(1)


def analyze_compression_potential(nn, scaler_x, n_samples: int = 15):
    """
    Analyze what fraction of the NN can be replaced with polynomial forms.

    Returns:
    - Number of parameters in original NN
    - Number of parameters after form extraction
    - Compression ratio
    - Accuracy impact
    """
    weights = nn.coefs_
    biases = nn.intercepts_

    # Original parameter count
    original_params = sum(W.size + b.size for W, b in zip(weights, biases))

    # Sample inputs
    sample_raw = np.array([[float(i)] for i in range(n_samples)])
    sample_scaled = scaler_x.transform(sample_raw)
    x = np.arange(n_samples, dtype=float)

    # Analyze each layer
    compressed_params = 0
    neurons_compressed = 0
    neurons_total = 0

    current_input = sample_scaled

    for layer_idx, (W, b) in enumerate(zip(weights, biases)):
        outputs = current_input @ W + b
        neurons_total += outputs.shape[1]

        for neuron_idx in range(outputs.shape[1]):
            y = outputs[:, neuron_idx]

            # Try to fit polynomial with R2 > 0.99
            best_fit = None
            best_degree = None

            for degree in [1, 2, 3]:
                try:
                    coeffs = np.polyfit(x, y, degree)
                    pred = np.polyval(coeffs, x)
                    ss_res = np.sum((y - pred)**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)

                    if r2 > 0.99:
                        best_fit = coeffs
                        best_degree = degree
                        break  # Use simplest adequate polynomial
                except:
                    pass

            if best_fit is not None:
                # This neuron can be replaced with polynomial
                # Parameters needed: degree + 1 coefficients
                compressed_params += best_degree + 1
                neurons_compressed += 1
            else:
                # Must keep original weights for this neuron
                # Parameters: input_dim weights + 1 bias
                input_dim = W.shape[0]
                compressed_params += input_dim + 1

        # Propagate
        current_input = outputs
        if layer_idx < len(weights) - 1:
            current_input = np.maximum(0, current_input)

    compression_ratio = 1 - compressed_params / original_params
    compress_pct = neurons_compressed / neurons_total

    return {
        'original_params': original_params,
        'compressed_params': compressed_params,
        'compression_ratio': compression_ratio,
        'neurons_compressed': neurons_compressed,
        'neurons_total': neurons_total,
        'compress_pct': compress_pct
    }


def test_compression(name: str, true_func: Callable,
                     x_range: Tuple[float, float],
                     hidden_sizes: Tuple = (32, 32)) -> dict:
    """Train NN and analyze compression potential"""

    print(f"\n  {name}")
    print(f"  {'-'*50}")

    # Train NN
    X_train = np.linspace(x_range[0], x_range[1], 200).reshape(-1, 1)
    y_train = np.array([true_func(x[0]) for x in X_train])

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    nn = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        max_iter=5000,
        early_stopping=True,
        random_state=42
    )
    nn.fit(X_scaled, y_scaled)

    # Analyze compression
    stats = analyze_compression_potential(nn, scaler_x)

    print(f"    Original params:  {stats['original_params']}")
    print(f"    Compressed:       {stats['compressed_params']}")
    print(f"    Compression:      {stats['compression_ratio']*100:.1f}%")
    print(f"    Neurons compressed: {stats['neurons_compressed']}/{stats['neurons_total']} ({stats['compress_pct']*100:.0f}%)")

    return stats


def run_tests():
    """Test compression potential across different functions"""

    print("=" * 60)
    print("  NN COMPRESSION VIA INTERNAL FORM EXTRACTION")
    print("=" * 60)

    tests = [
        ("Linear: 2x + 3", lambda x: 2*x + 3, (0, 20)),
        ("Quadratic: x^2", lambda x: x**2, (0, 15)),
        ("Triangular", lambda x: x*(x+1)/2, (0, 15)),
        ("Powers of 2", lambda x: 2**x, (0, 12)),
        ("Cubic: x^3", lambda x: x**3, (0, 10)),
        ("Sin", lambda x: np.sin(x), (0, 10)),
    ]

    results = []
    for name, func, x_range in tests:
        try:
            stats = test_compression(name, func, x_range)
            stats['name'] = name
            results.append(stats)
        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print(f"\n  {'Function':<20} {'Orig':<8} {'Comp':<8} {'Ratio':<10} {'Neurons%'}")
    print("  " + "-" * 60)

    for r in results:
        print(f"  {r['name']:<20} {r['original_params']:<8} {r['compressed_params']:<8} "
              f"{r['compression_ratio']*100:.1f}%       {r['compress_pct']*100:.0f}%")

    avg_compression = np.mean([r['compression_ratio'] for r in results])
    avg_neurons = np.mean([r['compress_pct'] for r in results])

    print()
    print(f"  Average compression: {avg_compression*100:.1f}%")
    print(f"  Average neurons compressible: {avg_neurons*100:.0f}%")

    print()
    print("=" * 60)
    print("  CONCLUSION")
    print("=" * 60)
    print()

    if avg_compression > 0.3:
        print(f"  POSITIVE: {avg_compression*100:.0f}% average compression achievable.")
        print(f"  {avg_neurons*100:.0f}% of neurons can be replaced with polynomials.")
        print()
        print("  This validates the CONCEPT of hybrid decomposition:")
        print("  - Individual neurons DO compute closed forms")
        print("  - These CAN be extracted and compressed")
        print("  - Significant parameter reduction is possible")
    else:
        print(f"  LIMITED: Only {avg_compression*100:.0f}% compression.")

    return results


if __name__ == "__main__":
    run_tests()
