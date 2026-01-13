"""
GPT-2 SMALL FFN COMPRESSION ANALYSIS
=====================================

Analyze compression potential in GPT-2's FFN layers using:
1. Weight matrix rank analysis (SVD-based)
2. Neuron activation patterns (dead/linear/nonlinear)
3. Effective dimensionality

GPT-2 Small architecture:
- 12 transformer layers
- Each layer has FFN: GELU(xW1 + b1)W2 + b2
- W1: (768, 3072) - 4x expansion
- W2: (3072, 768) - contraction
- Total FFN params per layer: 768*3072 + 3072 + 3072*768 + 768 = ~4.7M
- Total FFN params: ~56M (about 45% of model)
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from transformers import GPT2Model, GPT2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers and torch required")
    print("Install with: pip install transformers torch")
    sys.exit(1)


def analyze_ffn_neuron_forms_1d(
    W1: np.ndarray,  # (in_dim, hidden_dim)
    b1: np.ndarray,  # (hidden_dim,)
    W2: np.ndarray,  # (hidden_dim, out_dim)
    b2: np.ndarray,  # (out_dim,)
    n_samples: int = 20,
    max_neurons: int = None
) -> Dict:
    """
    [FLAWED] Original test - sweeps 1D slice through high-dim space.
    This will ALWAYS show high compression because GELU is smooth.
    Kept for comparison.
    """
    in_dim, hidden_dim = W1.shape
    if max_neurons is None:
        max_neurons = hidden_dim

    sample_inputs = np.zeros((n_samples, in_dim))
    for i in range(n_samples):
        scale = -2 + 4 * i / (n_samples - 1)
        np.random.seed(42)
        direction = np.random.randn(in_dim)
        direction /= np.linalg.norm(direction)
        sample_inputs[i] = scale * direction

    x = np.arange(n_samples, dtype=float)
    neurons_compressed = 0
    original_params = 0
    compressed_params = 0
    form_types = {'linear': 0, 'quadratic': 0, 'cubic': 0, 'none': 0}

    for j in range(min(hidden_dim, max_neurons)):
        pre_activation = sample_inputs @ W1[:, j] + b1[j]
        y = gelu(pre_activation)
        neuron_original_params = in_dim + 1
        original_params += neuron_original_params

        best_fit = None
        best_degree = None
        for degree in [1, 2, 3]:
            try:
                coeffs = np.polyfit(x, y, degree)
                pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2) + 1e-10
                r2 = 1 - ss_res / ss_tot
                if r2 > 0.99:
                    best_fit = coeffs
                    best_degree = degree
                    break
            except:
                pass

        if best_fit is not None:
            compressed_params += best_degree + 1
            neurons_compressed += 1
            form_types[['linear', 'quadratic', 'cubic'][best_degree - 1]] += 1
        else:
            compressed_params += neuron_original_params
            form_types['none'] += 1

    neurons_analyzed = min(hidden_dim, max_neurons)
    compression_ratio = 1 - compressed_params / original_params if original_params > 0 else 0

    return {
        'neurons_analyzed': neurons_analyzed,
        'neurons_compressed': neurons_compressed,
        'compress_pct': neurons_compressed / neurons_analyzed if neurons_analyzed > 0 else 0,
        'original_params': original_params,
        'compressed_params': compressed_params,
        'compression_ratio': compression_ratio,
        'form_types': form_types
    }


def analyze_weight_matrix_rank(W: np.ndarray, threshold: float = 0.99) -> Dict:
    """
    Analyze effective rank of weight matrix via SVD.
    Low effective rank = compression opportunity via factorization.
    """
    U, S, Vh = np.linalg.svd(W, full_matrices=False)

    # Compute energy in each singular value
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy

    # Find effective rank at threshold
    effective_rank = np.searchsorted(cumulative_energy, threshold) + 1

    # Compression ratio if we keep only effective_rank components
    # Original: m*n params
    # Low-rank: m*k + k*n params (where k = effective_rank)
    m, n = W.shape
    original_params = m * n
    compressed_params = m * effective_rank + effective_rank * n

    return {
        'shape': (m, n),
        'full_rank': min(m, n),
        'effective_rank': effective_rank,
        'effective_rank_ratio': effective_rank / min(m, n),
        'original_params': original_params,
        'compressed_params': compressed_params,
        'compression_ratio': 1 - compressed_params / original_params if compressed_params < original_params else 0,
        'top_10_singular_values': S[:10].tolist(),
        'energy_at_ranks': {
            10: float(cumulative_energy[9]) if len(cumulative_energy) > 9 else 1.0,
            50: float(cumulative_energy[49]) if len(cumulative_energy) > 49 else 1.0,
            100: float(cumulative_energy[99]) if len(cumulative_energy) > 99 else 1.0,
            200: float(cumulative_energy[199]) if len(cumulative_energy) > 199 else 1.0,
        }
    }


def analyze_pre_activation_distribution(
    W: np.ndarray,
    b: np.ndarray,
    n_samples: int = 1000
) -> Dict:
    """
    Analyze the distribution of pre-activation values.
    If pre-activations cluster, the effective function is simpler.
    """
    in_dim, out_dim = W.shape

    # Sample random inputs from typical activation range
    np.random.seed(42)
    inputs = np.random.randn(n_samples, in_dim) * 0.5  # Typical activation scale

    # Compute pre-activations for all neurons
    pre_activations = inputs @ W + b  # (n_samples, out_dim)

    # Analyze distribution
    means = np.mean(pre_activations, axis=0)
    stds = np.std(pre_activations, axis=0)

    # Count "dead" neurons (always negative -> GELU ≈ 0)
    # and "linear" neurons (always positive -> GELU ≈ x)
    always_negative = np.sum(np.all(pre_activations < -3, axis=0))
    always_positive = np.sum(np.all(pre_activations > 3, axis=0))
    mixed = out_dim - always_negative - always_positive

    return {
        'total_neurons': out_dim,
        'dead_neurons': int(always_negative),  # Always in saturation
        'linear_neurons': int(always_positive),  # Always in linear regime
        'nonlinear_neurons': int(mixed),  # Actually using nonlinearity
        'dead_ratio': always_negative / out_dim,
        'linear_ratio': always_positive / out_dim,
        'mean_std': float(np.mean(stds)),
    }


def analyze_ffn_neuron_forms(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    n_samples: int = 20,
    max_neurons: int = None
) -> Dict:
    """
    Comprehensive FFN analysis combining multiple metrics.
    """
    in_dim = W1.shape[0]
    hidden_dim = W1.shape[1]

    # 1. Weight matrix rank analysis (actual compression metric)
    rank_w1 = analyze_weight_matrix_rank(W1.T)  # Transpose to (hidden, in)
    rank_w2 = analyze_weight_matrix_rank(W2.T)  # Transpose to (out, hidden)

    # 2. Pre-activation distribution
    dist = analyze_pre_activation_distribution(W1.T, b1)

    # Combined compression estimate
    # We can compress via:
    # a) Low-rank factorization of weights
    # b) Removing dead neurons
    # c) Linearizing always-positive neurons

    original_params = (in_dim * hidden_dim + hidden_dim +  # W1, b1
                      hidden_dim * in_dim + in_dim)         # W2, b2 (assuming out=in)

    # Effective compression from rank + dead neurons
    effective_hidden = hidden_dim - dist['dead_neurons']
    rank_compression = (rank_w1['compression_ratio'] + rank_w2['compression_ratio']) / 2

    return {
        'neurons_analyzed': hidden_dim,
        'neurons_compressed': dist['dead_neurons'] + dist['linear_neurons'],
        'compress_pct': (dist['dead_neurons'] + dist['linear_neurons']) / hidden_dim,
        'original_params': original_params,
        'compressed_params': int(original_params * (1 - rank_compression)),
        'compression_ratio': rank_compression,
        'form_types': {
            'dead': dist['dead_neurons'],
            'linear': dist['linear_neurons'],
            'nonlinear': dist['nonlinear_neurons']
        },
        'rank_analysis': {
            'W1_effective_rank': rank_w1['effective_rank'],
            'W1_rank_ratio': rank_w1['effective_rank_ratio'],
            'W2_effective_rank': rank_w2['effective_rank'],
            'W2_rank_ratio': rank_w2['effective_rank_ratio'],
            'W1_energy_at_100': rank_w1['energy_at_ranks'].get(100, 1.0),
            'W2_energy_at_100': rank_w2['energy_at_ranks'].get(100, 1.0),
        }
    }


def gelu(x):
    """Gaussian Error Linear Unit activation"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def analyze_gpt2_compression(model, n_layers: int = None):
    """
    Analyze compression potential of GPT-2's FFN layers.
    """
    results = {
        'layers': [],
        'totals': {
            'original_params': 0,
            'compressed_params': 0,
            'neurons_analyzed': 0,
            'neurons_compressed': 0,
            'W1_rank_sum': 0,
            'W2_rank_sum': 0,
        }
    }

    # Access transformer blocks
    blocks = model.h
    if n_layers is None:
        n_layers = len(blocks)

    print(f"\n  Analyzing {n_layers} transformer layers...")
    print()

    for layer_idx in range(n_layers):
        block = blocks[layer_idx]

        # Extract FFN weights (GPT-2 calls it 'mlp')
        mlp = block.mlp

        # c_fc: first linear layer (768 -> 3072)
        # c_proj: second linear layer (3072 -> 768)
        W1 = mlp.c_fc.weight.detach().numpy().T  # (768, 3072)
        b1 = mlp.c_fc.bias.detach().numpy()      # (3072,)
        W2 = mlp.c_proj.weight.detach().numpy().T  # (3072, 768)
        b2 = mlp.c_proj.bias.detach().numpy()      # (768,)

        # Analyze compression potential
        stats = analyze_ffn_neuron_forms(W1, b1, W2, b2)

        rank = stats['rank_analysis']
        forms = stats['form_types']

        print(f"  Layer {layer_idx + 1}:")
        print(f"    W1 effective rank: {rank['W1_effective_rank']}/768 ({rank['W1_rank_ratio']*100:.1f}%)")
        print(f"    W2 effective rank: {rank['W2_effective_rank']}/768 ({rank['W2_rank_ratio']*100:.1f}%)")
        print(f"    Energy @ rank 100: W1={rank['W1_energy_at_100']*100:.1f}%, W2={rank['W2_energy_at_100']*100:.1f}%")
        print(f"    Neurons: {forms['dead']} dead, {forms['linear']} linear, {forms['nonlinear']} nonlinear")

        results['layers'].append({
            'layer': layer_idx + 1,
            **stats
        })

        # Accumulate totals
        results['totals']['original_params'] += stats['original_params']
        results['totals']['compressed_params'] += stats['compressed_params']
        results['totals']['neurons_analyzed'] += stats['neurons_analyzed']
        results['totals']['neurons_compressed'] += stats['neurons_compressed']
        results['totals']['W1_rank_sum'] += rank['W1_effective_rank']
        results['totals']['W2_rank_sum'] += rank['W2_effective_rank']

    # Compute overall stats
    totals = results['totals']
    totals['compression_ratio'] = 1 - totals['compressed_params'] / totals['original_params']
    totals['compress_pct'] = totals['neurons_compressed'] / totals['neurons_analyzed']
    totals['avg_W1_rank'] = totals['W1_rank_sum'] / n_layers
    totals['avg_W2_rank'] = totals['W2_rank_sum'] / n_layers

    return results


def run_gpt2_test():
    """Main test function"""

    print("=" * 70)
    print("  GPT-2 SMALL FFN COMPRESSION ANALYSIS")
    print("=" * 70)
    print()
    print("  Loading GPT-2 Small (~500MB download on first run)...")

    # Load GPT-2 Small
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")

    # Analyze all 12 layers
    print()
    print("-" * 70)
    print("  LAYER-BY-LAYER ANALYSIS")
    print("-" * 70)

    results = analyze_gpt2_compression(model, n_layers=12)

    # Summary
    totals = results['totals']

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print(f"  Average W1 effective rank: {totals['avg_W1_rank']:.0f}/768 ({totals['avg_W1_rank']/768*100:.1f}%)")
    print(f"  Average W2 effective rank: {totals['avg_W2_rank']:.0f}/768 ({totals['avg_W2_rank']/768*100:.1f}%)")
    print()
    print(f"  Total neurons analyzed: {totals['neurons_analyzed']:,}")
    print(f"  Dead + Linear neurons:  {totals['neurons_compressed']:,} ({totals['compress_pct']*100:.1f}%)")

    # What does effective rank mean for compression?
    avg_rank_ratio = (totals['avg_W1_rank'] + totals['avg_W2_rank']) / 2 / 768

    print()
    print("=" * 70)
    print("  COMPRESSION INTERPRETATION")
    print("=" * 70)
    print()
    print("  LOW-RANK FACTORIZATION (like LoRA):")
    print(f"    If we use rank-{int(totals['avg_W1_rank'])} approximation:")

    # Original FFN params per layer: 768*3072 + 3072 + 3072*768 + 768 = 4,722,432
    # Low-rank: 768*r + r*3072 + 3072*r + r*768 = r*(768+3072)*2 = r*7680
    original_per_layer = 768*3072*2 + 3072 + 768
    avg_rank = int((totals['avg_W1_rank'] + totals['avg_W2_rank']) / 2)
    lowrank_per_layer = avg_rank * (768 + 3072) * 2

    compression_from_rank = 1 - lowrank_per_layer / original_per_layer

    print(f"    Original params/layer: {original_per_layer:,}")
    print(f"    Low-rank params/layer: {lowrank_per_layer:,}")
    print(f"    Compression: {compression_from_rank*100:.1f}%")

    print()
    print("  BUT NOTE:")
    print("    - This is SVD-based compression (well-known technique)")
    print("    - LoRA already exploits this for fine-tuning")
    print("    - The 61% result from MLPs was about polynomial forms")
    print("    - Polynomial extraction doesn't directly apply to high-dim inputs")

    print()
    print("=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print()

    if compression_from_rank > 0.5:
        print("  HIGH compression potential via low-rank factorization.")
    elif compression_from_rank > 0.2:
        print("  MODERATE compression potential via low-rank factorization.")
    else:
        print("  LIMITED compression potential - weights are nearly full rank.")

    print()
    print("  KEY INSIGHT:")
    print("    The original MLP polynomial extraction worked because inputs were 1D.")
    print("    For transformers with 768-dim inputs, different techniques apply:")
    print("    - Low-rank factorization (SVD, LoRA)")
    print("    - Pruning (remove dead/redundant neurons)")
    print("    - Quantization (reduce precision)")
    print()
    print("    The 'closed form' insight may apply to:")
    print("    - Attention patterns (sparse attention)")
    print("    - Token-level predictions (if output is structured)")
    print("    - Not directly to FFN weight matrices")

    return results


if __name__ == "__main__":
    results = run_gpt2_test()
