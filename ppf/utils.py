"""
PPF Utility Functions

Pretty printing and helper functions for PPF results.
"""

import numpy as np

from .types import FormType, PPFResult
from .residual_layer import PPFStackResult
from .hierarchical import HierarchicalResult


def print_ppf_result(result: PPFResult, data: np.ndarray) -> None:
    """Pretty print PPF analysis results"""
    print("=" * 60)
    print("Promising Partial Form (PPF) Analysis")
    print("=" * 60)
    print(f"Data points: {len(data)}")
    print(f"Partial forms detected: {len(result.partial_forms)}")
    print(f"Validated forms: {len(result.validated_forms)}")
    print(f"Structure score: {result.structure_score:.2%}")
    print(f"Noise level (std): {result.noise_level:.4f}")
    print()

    if result.validated_forms:
        print("Validated Forms:")
        print("-" * 40)
        for i, pf in enumerate(result.validated_forms):
            print(f"  [{i+1}] {pf.fit.form_type.value}")
            print(f"      Range: [{pf.start_idx}:{pf.end_idx}] ({pf.end_idx - pf.start_idx} points)")
            print(f"      R²: {pf.fit.r_squared:.3f}")
            print(f"      Confidence: {pf.confidence:.3f}")
            print(f"      Residual std: {pf.fit.residual_std:.4f}")
            print(f"      Noise-like residuals: {pf.fit.is_noise_like}")
            print()

    if result.partial_forms and len(result.partial_forms) > len(result.validated_forms):
        print("Unvalidated Forms (failed extrapolation test):")
        print("-" * 40)
        for pf in result.partial_forms:
            if pf not in result.validated_forms:
                print(f"  - {pf.fit.form_type.value} [{pf.start_idx}:{pf.end_idx}]")
                print(f"    R²: {pf.fit.r_squared:.3f} (looked good but didn't extrapolate)")


def print_stack_result(result: PPFStackResult, data: np.ndarray) -> None:
    """Pretty print PPF stack analysis results"""
    print("=" * 65)
    print("PPF + Residual Storage Analysis")
    print("=" * 65)
    print(f"Data points:        {len(data)}")
    print(f"Iterations:         {result.iterations}")
    print(f"Forms found:        {len(result.form_stack)}")
    print(f"Total parameters:   {result.total_params}")
    print(f"Stopped because:    {result.stopped_reason}")
    print()
    print(f"Original size:      {result.original_size:,} bytes")
    print(f"Compressed size:    {result.compressed_size:,} bytes")
    print(f"Compression ratio:  {result.compression_ratio:.3f}")
    print(f"Space savings:      {result.space_savings:.1%}")
    print()

    if result.form_stack:
        print("Form Stack (in extraction order):")
        print("-" * 45)
        for i, layer in enumerate(result.form_stack):
            print(f"  [{i+1}] {layer.form_type.value}")
            print(f"      Params: {layer.params}")
            print(f"      R²: {layer.r_squared:.4f}")
            print(f"      Residual entropy: {layer.residual_entropy:.4f}")
            print(f"      Compression gain: {layer.compression_gain:.1%}")
            print()

    # Verify reconstruction
    x = np.arange(len(data), dtype=float)
    reconstructed = result.reconstruct(x)
    reconstruction_error = np.max(np.abs(data - reconstructed))
    print(f"Reconstruction error (max): {reconstruction_error:.2e}")

    # Final residual stats
    print()
    print("Final Residuals:")
    print("-" * 45)
    print(f"  Mean:   {np.mean(result.final_residuals):.6f}")
    print(f"  Std:    {np.std(result.final_residuals):.6f}")
    print(f"  Min:    {np.min(result.final_residuals):.6f}")
    print(f"  Max:    {np.max(result.final_residuals):.6f}")


def print_hierarchical_result(result: HierarchicalResult, data: np.ndarray) -> None:
    """Pretty print hierarchical analysis results"""
    print("=" * 70)
    print("HIERARCHICAL PATTERN DETECTION RESULTS")
    print("=" * 70)
    print(f"Data points: {len(data)}")
    print(f"Levels detected: {len(result.levels)}")
    print()

    for level in result.levels:
        print(f"Level {level.level}: {level.description}")
        print("-" * 50)

        if level.window_fits:
            print(f"  Windows successfully fit: {len(level.window_fits)}")
            print(f"  Coverage: {level.coverage:.1%}")
            print(f"  Dominant form: {level.dominant_form.value if level.dominant_form else 'N/A'}")

            r_squareds = [wf.r_squared for wf in level.window_fits]
            print(f"  R² range: {min(r_squareds):.2f} - {max(r_squareds):.2f}")
            print(f"  R² mean: {np.mean(r_squareds):.2f}")

        if level.parameter_evolutions:
            print(f"  Parameter evolutions:")
            for name, evo in level.parameter_evolutions.items():
                print(f"    {name}:")
                print(f"      Range: {evo.values.min():.3f} to {evo.values.max():.3f}")
                print(f"      Std: {np.std(evo.values):.3f}")

                if evo.has_structure:
                    print(f"      Variance explained: {evo.variance_explained:.1%}")
                    print(f"      Forms found:")
                    for form in evo.ppf_result.form_stack:
                        print(f"        - {form.form_type.value} (R²={form.r_squared:.2f})")
                        if form.form_type == FormType.SINE:
                            freq = form.params[1]
                            period_windows = 2 * np.pi / freq if freq > 0 else float('inf')
                            print(f"          Period: {period_windows:.1f} windows")
                else:
                    print(f"      No meta-pattern detected")

        print()
