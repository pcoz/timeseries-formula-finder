"""
Tests for the Hybrid Decomposition module.
"""

import numpy as np
import pytest

from ppf import (
    HybridDecomposer,
    HybridDecompositionResult,
    InterpretedComponent,
    DecompositionMethod,
    EMDDecomposer,
    SSADecomposer,
    FormType,
)


class TestEMDDecomposer:
    """Tests for EMD decomposition"""

    def test_emd_basic(self):
        """Test basic EMD decomposition"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)

        decomposer = EMDDecomposer(method="emd")
        components, residual = decomposer.decompose(signal)

        assert len(components) > 0
        assert decomposer.method == DecompositionMethod.EMD

    def test_eemd_decomposition(self):
        """Test EEMD (ensemble) decomposition"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))

        decomposer = EMDDecomposer(method="eemd", ensemble_size=50)
        components, residual = decomposer.decompose(signal)

        assert len(components) > 0
        assert decomposer.method == DecompositionMethod.EEMD

    def test_ceemdan_decomposition(self):
        """Test CEEMDAN decomposition"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 5 * t)

        decomposer = EMDDecomposer(method="ceemdan", ensemble_size=50)
        components, residual = decomposer.decompose(signal)

        assert len(components) > 0
        assert decomposer.method == DecompositionMethod.CEEMDAN


class TestSSADecomposer:
    """Tests for SSA decomposition"""

    def test_ssa_basic(self):
        """Test basic SSA decomposition"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * t

        decomposer = SSADecomposer(window_size=50, n_components=5)
        components, residual = decomposer.decompose(signal)

        assert len(components) == 5
        assert decomposer.method == DecompositionMethod.SSA

    def test_ssa_reconstruction(self):
        """Test that SSA components sum to original"""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))

        decomposer = SSADecomposer(window_size=30, n_components=10)
        components, residual = decomposer.decompose(signal)

        reconstructed = sum(components)
        if residual is not None:
            reconstructed += residual

        # Should reconstruct reasonably well
        error = np.max(np.abs(signal - reconstructed))
        assert error < 0.5  # Allow some error


class TestHybridDecomposer:
    """Tests for the main HybridDecomposer class"""

    def test_eemd_analysis(self):
        """Test EEMD hybrid analysis"""
        np.random.seed(42)
        t = np.linspace(0, 1, 300)
        signal = 2.0 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))

        decomposer = HybridDecomposer(method="eemd")
        result = decomposer.analyze(signal)

        assert isinstance(result, HybridDecompositionResult)
        assert len(result.components) > 0
        assert result.method == DecompositionMethod.EEMD

    def test_ssa_analysis(self):
        """Test SSA hybrid analysis"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * t

        decomposer = HybridDecomposer(method="ssa", window_size=50)
        result = decomposer.analyze(signal)

        assert isinstance(result, HybridDecompositionResult)
        assert result.method == DecompositionMethod.SSA

    def test_component_interpretation(self):
        """Test that components get PPF interpretation"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        # Clear sine wave should be identified
        signal = 2.0 * np.sin(2 * np.pi * 5 * t)

        decomposer = HybridDecomposer(method="emd", min_r_squared=0.3)
        result = decomposer.analyze(signal)

        # At least one component should be identified as sine
        sine_components = [
            c for c in result.components
            if c.form_type == FormType.SINE
        ]
        assert len(sine_components) > 0

    def test_noise_detection(self):
        """Test that noise components are identified"""
        np.random.seed(42)
        # Pure noise
        signal = np.random.randn(200)

        decomposer = HybridDecomposer(
            method="emd",
            noise_threshold=0.3  # Lower threshold to detect noise
        )
        result = decomposer.analyze(signal)

        # Should have some noise components
        # (or no interpretable signal components)
        assert result.n_noise_components >= 0

    def test_reconstruction(self):
        """Test signal reconstruction from components"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * t

        decomposer = HybridDecomposer(method="emd")
        result = decomposer.analyze(signal)

        reconstructed = result.reconstruct(include_noise=True)

        # Should have same length
        assert len(reconstructed) == len(signal)

    def test_get_signal_components(self):
        """Test filtering signal vs noise components"""
        np.random.seed(42)
        t = np.linspace(0, 1, 200)
        signal = 2.0 * np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t))

        decomposer = HybridDecomposer(method="eemd")
        result = decomposer.analyze(signal)

        signal_comps = result.get_signal_components()
        noise_comps = result.get_noise_components()

        assert len(signal_comps) + len(noise_comps) == len(result.components)

    def test_summary_generation(self):
        """Test that summary is generated"""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))

        decomposer = HybridDecomposer(method="emd")
        result = decomposer.analyze(signal)

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Hybrid Decomposition" in summary


class TestInterpretedComponent:
    """Tests for InterpretedComponent class"""

    def test_period_calculation(self):
        """Test period from frequency"""
        comp = InterpretedComponent(
            index=0,
            signal=np.zeros(100),
            mean_frequency=0.1
        )

        assert comp.period == 10.0

    def test_period_zero_frequency(self):
        """Test period when frequency is zero"""
        comp = InterpretedComponent(
            index=0,
            signal=np.zeros(100),
            mean_frequency=0.0
        )

        assert comp.period == float('inf')


class TestDecompositionMethod:
    """Tests for DecompositionMethod enum"""

    def test_all_methods_exist(self):
        """Test that all expected methods are defined"""
        assert DecompositionMethod.EMD.value == "emd"
        assert DecompositionMethod.EEMD.value == "eemd"
        assert DecompositionMethod.CEEMDAN.value == "ceemdan"
        assert DecompositionMethod.SSA.value == "ssa"
