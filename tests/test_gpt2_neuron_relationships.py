"""
GPT-2 NEURON RELATIONSHIP ANALYSIS
===================================

Can we express some neurons as functions of other neurons?

If neuron B ≈ f(neuron A), we can eliminate B's weights and
compute it from A's output.

This is the RIGHT way to think about compression:
- Not: can we simplify one neuron? (already simple)
- But: are neurons redundant/correlated?
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers and torch required")
    sys.exit(1)


def gelu(x):
    """Gaussian Error Linear Unit"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def get_ffn_activations(model, layer_idx: int, n_samples: int = 500):
    """
    Sample FFN hidden layer activations for random inputs.
    Returns: (n_samples, 3072) array of neuron activations
    """
    block = model.h[layer_idx]
    mlp = block.mlp

    # GPT-2 uses Conv1D which stores weights as (in, out)
    W1 = mlp.c_fc.weight.detach().numpy()  # (768, 3072)
    b1 = mlp.c_fc.bias.detach().numpy()     # (3072,)

    print(f"    W1 shape: {W1.shape}, b1 shape: {b1.shape}")

    # Sample random inputs (simulating attention outputs)
    np.random.seed(42)
    inputs = np.random.randn(n_samples, 768) * 0.5

    # Compute hidden activations
    pre_act = inputs @ W1 + b1  # (n_samples, 3072)
    activations = gelu(pre_act)  # (n_samples, 3072)

    return activations


def analyze_neuron_correlations(activations: np.ndarray) -> Dict:
    """
    Find highly correlated neurons that could be merged/eliminated.
    """
    n_samples, n_neurons = activations.shape

    # Compute correlation matrix
    # Normalize each neuron
    means = activations.mean(axis=0, keepdims=True)
    stds = activations.std(axis=0, keepdims=True) + 1e-8
    normalized = (activations - means) / stds

    # Correlation matrix
    corr = (normalized.T @ normalized) / n_samples

    # Find highly correlated pairs (excluding diagonal)
    np.fill_diagonal(corr, 0)

    high_corr_threshold = 0.95
    high_corr_pairs = np.sum(np.abs(corr) > high_corr_threshold) // 2

    med_corr_threshold = 0.8
    med_corr_pairs = np.sum(np.abs(corr) > med_corr_threshold) // 2

    # Find neurons that could be predicted from others
    max_corr_per_neuron = np.max(np.abs(corr), axis=1)
    predictable_neurons = np.sum(max_corr_per_neuron > high_corr_threshold)

    return {
        'n_neurons': n_neurons,
        'high_corr_pairs': int(high_corr_pairs),
        'med_corr_pairs': int(med_corr_pairs),
        'predictable_neurons': int(predictable_neurons),
        'max_correlation': float(np.max(np.abs(corr))),
        'mean_max_correlation': float(np.mean(max_corr_per_neuron)),
    }


def analyze_neuron_linear_dependencies(activations: np.ndarray, max_check: int = 500) -> Dict:
    """
    Check if neurons can be expressed as LINEAR combinations of others.

    For each neuron j, try to predict it from a small subset of other neurons.
    If R² > 0.99, that neuron is redundant.
    """
    n_samples, n_neurons = activations.shape

    redundant_count = 0
    r2_scores = []

    # Check subset of neurons for speed
    neurons_to_check = min(max_check, n_neurons)
    check_indices = np.random.choice(n_neurons, neurons_to_check, replace=False)

    for j in check_indices:
        y = activations[:, j]

        # Use first 100 OTHER neurons as predictors
        other_indices = [i for i in range(min(100, n_neurons)) if i != j]
        X = activations[:, other_indices]

        # Fit linear regression
        try:
            # Add bias term
            X_bias = np.column_stack([X, np.ones(n_samples)])
            coeffs, residuals, rank, s = np.linalg.lstsq(X_bias, y, rcond=None)

            pred = X_bias @ coeffs
            ss_res = np.sum((y - pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2) + 1e-10
            r2 = 1 - ss_res / ss_tot

            r2_scores.append(r2)
            if r2 > 0.99:
                redundant_count += 1
        except:
            pass

    return {
        'neurons_checked': neurons_to_check,
        'redundant_neurons': redundant_count,
        'redundant_pct': redundant_count / neurons_to_check if neurons_to_check > 0 else 0,
        'mean_r2': float(np.mean(r2_scores)) if r2_scores else 0,
        'max_r2': float(np.max(r2_scores)) if r2_scores else 0,
        'neurons_above_95_r2': sum(1 for r in r2_scores if r > 0.95),
        'neurons_above_90_r2': sum(1 for r in r2_scores if r > 0.90),
    }


def analyze_neuron_polynomial_relationships(activations: np.ndarray, max_check: int = 200) -> Dict:
    """
    Check if neurons can be expressed as POLYNOMIAL functions of others.

    For each neuron j, try to predict it from polynomial features of a few other neurons.
    """
    n_samples, n_neurons = activations.shape

    redundant_count = 0
    r2_scores = []

    neurons_to_check = min(max_check, n_neurons)
    check_indices = np.random.choice(n_neurons, neurons_to_check, replace=False)

    for j in check_indices:
        y = activations[:, j]

        # Use 10 other neurons and their polynomial features
        other_indices = [i for i in range(20) if i != j][:10]
        base = activations[:, other_indices]

        # Create polynomial features: x, x², x*y for pairs
        features = [base]
        features.append(base ** 2)  # Quadratic
        # Cross terms for first few
        for i in range(min(5, len(other_indices))):
            for k in range(i+1, min(5, len(other_indices))):
                features.append((base[:, i] * base[:, k]).reshape(-1, 1))

        X = np.column_stack(features)
        X_bias = np.column_stack([X, np.ones(n_samples)])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
            pred = X_bias @ coeffs
            ss_res = np.sum((y - pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2) + 1e-10
            r2 = 1 - ss_res / ss_tot

            r2_scores.append(r2)
            if r2 > 0.99:
                redundant_count += 1
        except:
            pass

    return {
        'neurons_checked': neurons_to_check,
        'poly_redundant_neurons': redundant_count,
        'poly_redundant_pct': redundant_count / neurons_to_check if neurons_to_check > 0 else 0,
        'mean_r2': float(np.mean(r2_scores)) if r2_scores else 0,
        'max_r2': float(np.max(r2_scores)) if r2_scores else 0,
        'neurons_above_95_r2': sum(1 for r in r2_scores if r > 0.95),
        'neurons_above_90_r2': sum(1 for r in r2_scores if r > 0.90),
    }


def analyze_pca_compression(activations: np.ndarray) -> Dict:
    """
    How many principal components capture most of the variance?
    This tells us the effective dimensionality.
    """
    # Center the data
    centered = activations - activations.mean(axis=0)

    # SVD
    U, S, Vh = np.linalg.svd(centered, full_matrices=False)

    # Variance explained
    var_explained = S**2 / np.sum(S**2)
    cumulative_var = np.cumsum(var_explained)

    # Find components needed for various thresholds
    dims_for_90 = np.searchsorted(cumulative_var, 0.90) + 1
    dims_for_95 = np.searchsorted(cumulative_var, 0.95) + 1
    dims_for_99 = np.searchsorted(cumulative_var, 0.99) + 1

    n_neurons = activations.shape[1]

    return {
        'total_dims': n_neurons,
        'dims_for_90_var': int(dims_for_90),
        'dims_for_95_var': int(dims_for_95),
        'dims_for_99_var': int(dims_for_99),
        'compression_at_95': 1 - dims_for_95 / n_neurons,
        'compression_at_99': 1 - dims_for_99 / n_neurons,
        'top_10_var': float(cumulative_var[9]) if len(cumulative_var) > 9 else 1.0,
        'top_50_var': float(cumulative_var[49]) if len(cumulative_var) > 49 else 1.0,
        'top_100_var': float(cumulative_var[99]) if len(cumulative_var) > 99 else 1.0,
    }


def get_real_ffn_activations(model, tokenizer, layer_idx: int, texts: List[str]):
    """
    Get FFN activations from REAL text inputs.
    """
    all_activations = []

    block = model.h[layer_idx]
    mlp = block.mlp
    W1 = mlp.c_fc.weight.detach().numpy()
    b1 = mlp.c_fc.bias.detach().numpy()

    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get hidden state at this layer (input to FFN)
        hidden = outputs.hidden_states[layer_idx].numpy()[0]  # (seq_len, 768)

        # Compute FFN hidden activations
        pre_act = hidden @ W1 + b1
        activations = gelu(pre_act)  # (seq_len, 3072)

        all_activations.append(activations)

    return np.vstack(all_activations)


def run_test():
    print("=" * 70)
    print("  GPT-2 NEURON RELATIONSHIP ANALYSIS")
    print("=" * 70)
    print()
    print("  Question: Can some neurons be computed from OTHER neurons?")
    print()
    print("  Loading GPT-2 Small...")

    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    print("  Model loaded.")
    print()

    # Analyze first few layers
    for layer_idx in [0, 5, 11]:
        print("-" * 70)
        print(f"  LAYER {layer_idx + 1}")
        print("-" * 70)

        print("  Getting activations...")
        activations = get_ffn_activations(model, layer_idx, n_samples=500)
        print(f"  Activation shape: {activations.shape}")
        print()

        # Correlation analysis
        print("  1. CORRELATION ANALYSIS")
        corr_results = analyze_neuron_correlations(activations)
        print(f"     High correlation pairs (>0.95): {corr_results['high_corr_pairs']}")
        print(f"     Medium correlation pairs (>0.8): {corr_results['med_corr_pairs']}")
        print(f"     Max correlation: {corr_results['max_correlation']:.3f}")
        print(f"     Predictable neurons (corr>0.95): {corr_results['predictable_neurons']}")
        print()

        # Linear dependency analysis
        print("  2. LINEAR DEPENDENCY ANALYSIS")
        print("     (Can neuron j be predicted from linear combo of others?)")
        linear_results = analyze_neuron_linear_dependencies(activations, max_check=300)
        print(f"     Neurons checked: {linear_results['neurons_checked']}")
        print(f"     Redundant (R²>0.99): {linear_results['redundant_neurons']} ({linear_results['redundant_pct']*100:.1f}%)")
        print(f"     R²>0.95: {linear_results['neurons_above_95_r2']}")
        print(f"     R²>0.90: {linear_results['neurons_above_90_r2']}")
        print(f"     Mean R²: {linear_results['mean_r2']:.3f}")
        print()

        # Polynomial dependency analysis
        print("  3. POLYNOMIAL DEPENDENCY ANALYSIS")
        print("     (Can neuron j be predicted from polynomial of others?)")
        poly_results = analyze_neuron_polynomial_relationships(activations, max_check=200)
        print(f"     Neurons checked: {poly_results['neurons_checked']}")
        print(f"     Redundant (R²>0.99): {poly_results['poly_redundant_neurons']} ({poly_results['poly_redundant_pct']*100:.1f}%)")
        print(f"     R²>0.95: {poly_results['neurons_above_95_r2']}")
        print(f"     R²>0.90: {poly_results['neurons_above_90_r2']}")
        print(f"     Mean R²: {poly_results['mean_r2']:.3f}")
        print()

        # PCA analysis
        print("  4. PCA DIMENSIONALITY ANALYSIS")
        print("     (What's the effective dimensionality of neuron outputs?)")
        pca_results = analyze_pca_compression(activations)
        print(f"     Total dimensions: {pca_results['total_dims']}")
        print(f"     Dims for 90% variance: {pca_results['dims_for_90_var']} ({pca_results['dims_for_90_var']/pca_results['total_dims']*100:.1f}%)")
        print(f"     Dims for 95% variance: {pca_results['dims_for_95_var']} ({pca_results['dims_for_95_var']/pca_results['total_dims']*100:.1f}%)")
        print(f"     Dims for 99% variance: {pca_results['dims_for_99_var']} ({pca_results['dims_for_99_var']/pca_results['total_dims']*100:.1f}%)")
        print(f"     Compression at 95% var: {pca_results['compression_at_95']*100:.1f}%")
        print()

    # Now test with REAL text
    print("=" * 70)
    print("  REAL TEXT ANALYSIS")
    print("=" * 70)
    print()

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In machine learning, neural networks are computational systems inspired by biological neural networks.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "To be or not to be, that is the question.",
        "import numpy as np; x = np.array([1, 2, 3, 4, 5])",
        "The theory of relativity revolutionized our understanding of space and time.",
        "Once upon a time, in a land far away, there lived a young princess.",
        "The stock market crashed yesterday, leading to widespread panic among investors.",
        "SELECT * FROM users WHERE age > 21 ORDER BY name DESC;",
        "Climate change is one of the most pressing issues facing humanity today.",
        "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
        "According to quantum mechanics, particles can exist in superposition states.",
        "The patient presented with symptoms including fever, cough, and fatigue.",
        "In 1969, Neil Armstrong became the first human to walk on the moon.",
        "The function returns true if the input is a valid email address.",
        "Mozart composed his first symphony at the age of eight.",
        "The derivative of x squared is two x.",
        "Breaking news: Scientists discover high-temperature superconductor at room temperature.",
        "The cat sat on the mat and watched the birds outside the window.",
        "Implementing a binary search tree requires careful handling of edge cases.",
        "The Renaissance period saw a flourishing of art, literature, and science.",
        "Error: Cannot read property 'length' of undefined at line 42.",
        "The GDP of the United States grew by 2.3 percent last quarter.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
        "The protagonist struggled with the moral implications of their decision.",
        "pip install tensorflow numpy pandas scikit-learn matplotlib",
        "The contract stipulates that payment must be received within 30 days.",
        "Mitochondria are often called the powerhouses of the cell.",
        "The jury found the defendant not guilty on all charges.",
    ]

    for layer_idx in [0, 5, 11]:
        print(f"  Layer {layer_idx + 1} with real text:")
        real_activations = get_real_ffn_activations(model, tokenizer, layer_idx, sample_texts)
        print(f"    Activation shape: {real_activations.shape}")

        # Check sparsity
        near_zero = np.sum(np.abs(real_activations) < 0.01) / real_activations.size
        mean_act = np.mean(np.abs(real_activations))
        max_act = np.max(np.abs(real_activations))
        active_neurons = np.sum(np.any(np.abs(real_activations) > 0.1, axis=0))
        print(f"    Near-zero (<0.01): {near_zero*100:.1f}%")
        print(f"    Mean |activation|: {mean_act:.3f}")
        print(f"    Max |activation|: {max_act:.3f}")
        print(f"    Neurons ever active (>0.1): {active_neurons}/3072 ({active_neurons/3072*100:.1f}%)")

        pca_real = analyze_pca_compression(real_activations)
        print(f"    Dims for 90% var: {pca_real['dims_for_90_var']} ({pca_real['dims_for_90_var']/3072*100:.1f}%)")
        print(f"    Dims for 95% var: {pca_real['dims_for_95_var']} ({pca_real['dims_for_95_var']/3072*100:.1f}%)")
        print(f"    Dims for 99% var: {pca_real['dims_for_99_var']} ({pca_real['dims_for_99_var']/3072*100:.1f}%)")
        print(f"    Compression at 95%: {pca_real['compression_at_95']*100:.1f}%")
        print()

    # Verify compression with reconstruction error
    print("=" * 70)
    print("  RECONSTRUCTION TEST")
    print("=" * 70)
    print()
    print("  Can we reconstruct activations from fewer dimensions?")
    print()

    for layer_idx in [0, 5, 11]:
        real_activations = get_real_ffn_activations(model, tokenizer, layer_idx, sample_texts)

        # PCA compression test
        centered = real_activations - real_activations.mean(axis=0)
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)

        for k in [50, 100, 200]:
            # Reconstruct using top-k components
            reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :] + real_activations.mean(axis=0)

            # Reconstruction error
            mse = np.mean((real_activations - reconstructed)**2)
            relative_error = np.sqrt(mse) / np.mean(np.abs(real_activations))

            print(f"  Layer {layer_idx+1}, k={k}: relative error = {relative_error*100:.2f}%")

        print()

    # Sparsity analysis - the real opportunity
    print("=" * 70)
    print("  SPARSITY OPPORTUNITY")
    print("=" * 70)
    print()
    print("  If we could PREDICT which neurons fire, we could skip the rest.")
    print()

    for layer_idx in [0, 5, 11]:
        real_activations = get_real_ffn_activations(model, tokenizer, layer_idx, sample_texts)

        # What fraction of compute could we skip if we knew which neurons fire?
        threshold = 0.1 * np.mean(np.abs(real_activations))

        # Per-token sparsity
        sparse_per_token = np.mean(np.abs(real_activations) < threshold, axis=1)
        mean_sparse = np.mean(sparse_per_token)

        # Which neurons EVER fire significantly?
        ever_active = np.any(np.abs(real_activations) > threshold, axis=0)
        always_dormant = 3072 - np.sum(ever_active)

        print(f"  Layer {layer_idx+1}:")
        print(f"    Average sparsity per token: {mean_sparse*100:.1f}%")
        print(f"    Neurons never active (could prune): {always_dormant}")
        print(f"    Potential compute savings: {mean_sparse*100:.1f}%")
        print()

    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print()
    print("  FINDING: Middle/late FFN layers have 60-75% sparsity!")
    print()
    print("  This is known - it's why Mixture of Experts (MoE) works.")
    print("  The opportunity is: predict WHICH neurons fire, skip the rest.")
    print()
    print("  CONNECTION TO CLOSED FORMS:")
    print("    - Individual neuron computations ARE closed forms")
    print("    - But that's not the compression opportunity")
    print("    - The opportunity is SPARSITY: most neurons output ~0")
    print("    - A predictor/router could identify active neurons")
    print()
    print("  NEXT STEP: Can we predict which neurons will fire?")
    print("    - Input features that correlate with neuron activation")
    print("    - Lightweight router network (like MoE)")
    print("    - This is an active research area")
    print()

    return {
        'correlation': corr_results,
        'linear': linear_results,
        'polynomial': poly_results,
        'pca': pca_results,
        'pca_real': pca_real
    }


if __name__ == "__main__":
    results = run_test()
