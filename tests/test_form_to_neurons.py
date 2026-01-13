"""
CLOSED FORM -> NEURON ACTIVATION MAPPING
========================================

Question: Do different closed-form input patterns activate different neurons?

If yes, we can use form detection as a router:
1. Detect input form (linear, quadratic, etc.)
2. Activate only the neurons that respond to that form
3. Skip the rest
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


def get_neurons_for_embedding(model, embedding: np.ndarray, layer_idx: int, threshold: float = 0.1) -> Set[int]:
    """
    Given an embedding vector, which neurons fire in the FFN?
    """
    block = model.h[layer_idx]
    mlp = block.mlp

    W1 = mlp.c_fc.weight.detach().numpy()  # (768, 3072)
    b1 = mlp.c_fc.bias.detach().numpy()

    pre_act = embedding @ W1 + b1
    activation = gelu(pre_act)

    # Which neurons fire above threshold?
    mean_act = np.mean(np.abs(activation))
    active = np.where(np.abs(activation) > threshold * mean_act)[0]

    return set(active)


def generate_form_embeddings(model, tokenizer) -> Dict[str, List[np.ndarray]]:
    """
    Generate embeddings for different closed-form patterns.
    """
    forms = {
        'linear': [
            "2x + 3",
            "5x - 7",
            "x + 1",
            "3x",
            "10x + 20",
            "The slope is constant",
            "y = mx + b",
            "linear function",
        ],
        'quadratic': [
            "x squared",
            "x^2 + 2x + 1",
            "parabola",
            "quadratic equation",
            "y = ax^2 + bx + c",
            "second degree polynomial",
            "the area of a square",
            "x times x",
        ],
        'exponential': [
            "2^x",
            "e^x",
            "exponential growth",
            "doubling time",
            "compound interest",
            "powers of 2",
            "geometric sequence",
            "exponential function",
        ],
        'periodic': [
            "sin(x)",
            "cos(x)",
            "sine wave",
            "oscillation",
            "periodic function",
            "trigonometric",
            "cycles per second",
            "harmonic motion",
        ],
        'code': [
            "def foo():",
            "for i in range(10):",
            "if x > 0:",
            "return value",
            "import numpy",
            "class MyClass:",
            "print(hello)",
            "x = 42",
        ],
        'prose': [
            "The cat sat on the mat.",
            "Once upon a time",
            "In the beginning",
            "She walked to the store",
            "The weather was nice",
            "He said hello",
            "They went home",
            "It was a dark night",
        ],
    }

    embeddings = {}

    for form_name, texts in forms.items():
        form_embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state as embedding
            hidden = outputs.hidden_states[-1].numpy()[0]  # (seq_len, 768)
            embedding = hidden.mean(axis=0)  # (768,)
            form_embeddings.append(embedding)
        embeddings[form_name] = form_embeddings

    return embeddings


def analyze_form_neuron_mapping(model, embeddings: Dict[str, List[np.ndarray]], layer_idx: int):
    """
    For each form type, which neurons consistently fire?
    """
    form_neurons = {}

    for form_name, form_embeds in embeddings.items():
        # Get active neurons for each example of this form
        all_active = [get_neurons_for_embedding(model, emb, layer_idx) for emb in form_embeds]

        # Neurons that fire for ALL examples of this form
        consistent = set.intersection(*all_active) if all_active else set()

        # Neurons that fire for ANY example of this form
        any_active = set.union(*all_active) if all_active else set()

        form_neurons[form_name] = {
            'consistent': consistent,
            'any': any_active,
            'n_consistent': len(consistent),
            'n_any': len(any_active),
        }

    return form_neurons


def compute_form_specificity(form_neurons: Dict) -> Dict:
    """
    Which neurons are SPECIFIC to each form (fire for one form, not others)?
    """
    all_forms = list(form_neurons.keys())

    specificity = {}

    for form in all_forms:
        form_specific = form_neurons[form]['consistent'].copy()

        # Remove neurons that are consistent in OTHER forms
        for other_form in all_forms:
            if other_form != form:
                form_specific -= form_neurons[other_form]['consistent']

        specificity[form] = {
            'specific_neurons': form_specific,
            'n_specific': len(form_specific),
            'n_consistent': form_neurons[form]['n_consistent'],
            'specificity_ratio': len(form_specific) / max(1, form_neurons[form]['n_consistent']),
        }

    return specificity


def run_test():
    print("=" * 70)
    print("  CLOSED FORM -> NEURON ACTIVATION MAPPING")
    print("=" * 70)
    print()
    print("  Question: Do inputs with different closed forms")
    print("            activate different neurons?")
    print()
    print("  Loading GPT-2...")

    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    print("  Generating embeddings for different forms...")
    embeddings = generate_form_embeddings(model, tokenizer)

    print()
    print(f"  Forms: {list(embeddings.keys())}")
    print(f"  Examples per form: {len(list(embeddings.values())[0])}")

    for layer_idx in [0, 5, 11]:
        print()
        print("-" * 70)
        print(f"  LAYER {layer_idx + 1}")
        print("-" * 70)

        form_neurons = analyze_form_neuron_mapping(model, embeddings, layer_idx)

        print()
        print("  Neurons consistently active per form:")
        for form, data in form_neurons.items():
            print(f"    {form:12}: {data['n_consistent']:4} consistent, {data['n_any']:4} any")

        specificity = compute_form_specificity(form_neurons)

        print()
        print("  Form-SPECIFIC neurons (fire for this form, not others):")
        for form, data in specificity.items():
            print(f"    {form:12}: {data['n_specific']:4} specific (of {data['n_consistent']} consistent)")

        # Check overlap between forms
        print()
        print("  Overlap matrix (consistent neurons shared between forms):")
        forms = list(form_neurons.keys())
        print(f"    {'':12}", end="")
        for f in forms:
            print(f"{f[:8]:>9}", end="")
        print()

        for f1 in forms:
            print(f"    {f1:12}", end="")
            for f2 in forms:
                overlap = len(form_neurons[f1]['consistent'] & form_neurons[f2]['consistent'])
                print(f"{overlap:9}", end="")
            print()

    print()
    print("=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print()
    print("  If forms have SPECIFIC neurons -> form detection can route computation")
    print("  If forms share neurons -> routing won't help much")
    print()

    return {
        'embeddings': embeddings,
        'form_neurons': form_neurons,
        'specificity': specificity
    }


if __name__ == "__main__":
    results = run_test()
