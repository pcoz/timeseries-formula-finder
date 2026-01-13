"""
NUMERICAL SEQUENCE -> NEURON ACTIVATION
========================================

Feed GPT-2 actual numerical sequences with different closed-form structures.
See if the STRUCTURE activates different neurons.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from typing import Dict, List, Set
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError:
    print("ERROR: transformers and torch required")
    sys.exit(1)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def generate_sequences() -> Dict[str, List[str]]:
    """
    Generate numerical sequences from different closed forms.
    """
    sequences = {
        'constant': [],
        'linear': [],
        'quadratic': [],
        'cubic': [],
        'exponential': [],
        'fibonacci': [],
        'triangular': [],
        'factorial': [],
    }

    # Generate multiple examples of each form
    for offset in range(5):
        # Constant: c
        c = offset + 3
        sequences['constant'].append(', '.join(str(c) for _ in range(8)))

        # Linear: an + b
        a, b = offset + 1, offset * 2
        sequences['linear'].append(', '.join(str(a*n + b) for n in range(8)))

        # Quadratic: n^2 + offset
        sequences['quadratic'].append(', '.join(str(n*n + offset) for n in range(8)))

        # Cubic: n^3
        sequences['cubic'].append(', '.join(str(n**3 + offset) for n in range(8)))

        # Exponential: 2^n * (offset+1)
        base = offset + 2
        sequences['exponential'].append(', '.join(str(base**n) for n in range(8)))

        # Fibonacci-like: F(n) with different starts
        fib = [offset + 1, offset + 2]
        for _ in range(6):
            fib.append(fib[-1] + fib[-2])
        sequences['fibonacci'].append(', '.join(str(x) for x in fib))

        # Triangular: n(n+1)/2 + offset
        sequences['triangular'].append(', '.join(str(n*(n+1)//2 + offset) for n in range(8)))

        # Factorial: n!
        factorials = [1]
        for i in range(1, 8):
            factorials.append(factorials[-1] * i)
        # Add offset to make them different
        sequences['factorial'].append(', '.join(str(x + offset) for x in factorials))

    return sequences


def get_active_neurons(model, tokenizer, text: str, layer_idx: int, threshold_factor: float = 0.5) -> Set[int]:
    """
    Which neurons fire for this input?
    """
    block = model.h[layer_idx]
    mlp = block.mlp
    W1 = mlp.c_fc.weight.detach().numpy()
    b1 = mlp.c_fc.bias.detach().numpy()

    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden state at this layer
    hidden = outputs.hidden_states[layer_idx].numpy()[0]  # (seq_len, 768)

    # Compute FFN activations for each position, then aggregate
    all_active = set()
    for pos in range(hidden.shape[0]):
        pre_act = hidden[pos] @ W1 + b1
        activation = gelu(pre_act)

        # Neurons above threshold
        threshold = threshold_factor * np.mean(np.abs(activation))
        active = np.where(np.abs(activation) > threshold)[0]
        all_active.update(active)

    return all_active


def get_consistent_neurons(model, tokenizer, texts: List[str], layer_idx: int) -> Set[int]:
    """
    Neurons that fire for ALL examples in the list.
    """
    all_sets = [get_active_neurons(model, tokenizer, t, layer_idx) for t in texts]
    return set.intersection(*all_sets) if all_sets else set()


def run_test():
    print("=" * 70)
    print("  NUMERICAL SEQUENCE -> NEURON ACTIVATION")
    print("=" * 70)
    print()
    print("  Question: Do sequences with different closed-form structures")
    print("            activate different neurons?")
    print()

    print("  Loading GPT-2...")
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    print("  Generating sequences...")
    sequences = generate_sequences()

    print()
    print("  Example sequences:")
    for form, seqs in sequences.items():
        print(f"    {form:12}: {seqs[0]}")

    for layer_idx in [0, 5, 11]:
        print()
        print("-" * 70)
        print(f"  LAYER {layer_idx + 1}")
        print("-" * 70)

        form_neurons = {}
        for form, seqs in sequences.items():
            consistent = get_consistent_neurons(model, tokenizer, seqs, layer_idx)
            any_active = set.union(*[get_active_neurons(model, tokenizer, s, layer_idx) for s in seqs])
            form_neurons[form] = {
                'consistent': consistent,
                'any': any_active,
            }

        print()
        print("  Consistent neurons per form:")
        for form, data in form_neurons.items():
            print(f"    {form:12}: {len(data['consistent']):4} consistent, {len(data['any']):4} any")

        # Form-specific neurons
        print()
        print("  Form-SPECIFIC neurons:")
        for form in form_neurons:
            specific = form_neurons[form]['consistent'].copy()
            for other in form_neurons:
                if other != form:
                    specific -= form_neurons[other]['consistent']
            print(f"    {form:12}: {len(specific):4} specific")

        # Overlap analysis
        print()
        print("  Pairwise overlap (consistent):")
        forms = list(form_neurons.keys())

        # Find pairs with LOW overlap (most different)
        overlaps = []
        for i, f1 in enumerate(forms):
            for f2 in forms[i+1:]:
                n1 = form_neurons[f1]['consistent']
                n2 = form_neurons[f2]['consistent']
                if n1 and n2:
                    jaccard = len(n1 & n2) / len(n1 | n2)
                    overlaps.append((f1, f2, jaccard, len(n1 & n2)))

        overlaps.sort(key=lambda x: x[2])
        print("    Most different pairs:")
        for f1, f2, jaccard, shared in overlaps[:5]:
            print(f"      {f1:12} vs {f2:12}: {jaccard*100:.1f}% Jaccard, {shared} shared")

        print("    Most similar pairs:")
        for f1, f2, jaccard, shared in overlaps[-5:]:
            print(f"      {f1:12} vs {f2:12}: {jaccard*100:.1f}% Jaccard, {shared} shared")

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    return sequences, form_neurons


if __name__ == "__main__":
    results = run_test()
