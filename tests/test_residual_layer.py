"""
Tests for the PPF Residual Layer module.
"""

import numpy as np
import pytest

from ppf import (
    PPFResidualLayer,
    PPFStackResult,
    EntropyMethod,
    FormType,
    measure_entropy,
    measure_entropy_gzip,
    measure_entropy_spectral,
)


class TestEntropyMeasurement:
    """Tests for entropy measurement functions"""

    def test_gzip_entropy_high_for_random(self):
        """Test that random data has high gzip entropy"""
        np.random.seed(42)
        random_data = np.random.normal(0, 1, 100)
        
        entropy = measure_entropy_gzip(random_data)
        
        assert 0.5 < entropy <= 1.0  # High entropy

    def test_spectral_entropy_low_for_sine(self):
        """Test that sine wave has low spectral entropy"""
        x = np.arange(100)
        sine_data = np.sin(0.1 * x)
        
        entropy = measure_entropy_spectral(sine_data)
        
        assert entropy < 0.3  # Low entropy (peaked spectrum)

    def test_spectral_entropy_high_for_noise(self):
        """Test that white noise has high spectral entropy"""
        np.random.seed(42)
        # Use more samples for more reliable spectral estimate
        noise_data = np.random.normal(0, 1, 500)

        entropy = measure_entropy_spectral(noise_data)

        # For white noise, spectral flatness should be relatively high
        # (higher than structured signals like sine waves)
        assert entropy > 0.3  # Higher than structured data (~0.01-0.1)

    def test_measure_entropy_dispatch(self):
        """Test that measure_entropy dispatches correctly"""
        data = np.random.normal(0, 1, 100)
        
        gzip_result = measure_entropy(data, EntropyMethod.GZIP)
        spectral_result = measure_entropy(data, EntropyMethod.SPECTRAL)
        
        # Both should return values in [0, 1]
        assert 0 <= gzip_result <= 1
        assert 0 <= spectral_result <= 1


class TestPPFResidualLayer:
    """Tests for PPFResidualLayer class"""

    def test_init_validation(self):
        """Test that constructor validates parameters"""
        with pytest.raises(ValueError):
            PPFResidualLayer(noise_threshold=1.5)
        
        with pytest.raises(ValueError):
            PPFResidualLayer(min_points=2)

    def test_analyze_sine_wave(self, synthetic_sine_data):
        """Test analyzing a sine wave with linear trend"""
        _, data = synthetic_sine_data
        
        layer = PPFResidualLayer(
            entropy_method=EntropyMethod.SPECTRAL,
            noise_threshold=0.5,
            min_r_squared=0.3
        )
        result = layer.analyze(data)
        
        assert isinstance(result, PPFStackResult)
        assert len(result.form_stack) > 0
        assert result.iterations > 0

    def test_analyze_pure_noise(self):
        """Test that pure noise results in no forms"""
        np.random.seed(42)
        noise_data = np.random.normal(0, 1, 100)
        
        layer = PPFResidualLayer(
            entropy_method=EntropyMethod.SPECTRAL,
            noise_threshold=0.3,
            min_r_squared=0.5
        )
        result = layer.analyze(noise_data)
        
        # Should find few or no forms in pure noise
        assert len(result.form_stack) <= 1

    def test_reconstruction_accuracy(self, synthetic_sine_data):
        """Test that reconstruction matches original data"""
        _, data = synthetic_sine_data
        
        layer = PPFResidualLayer(
            entropy_method=EntropyMethod.SPECTRAL,
            min_r_squared=0.3
        )
        result = layer.analyze(data)
        
        x = np.arange(len(data), dtype=float)
        reconstructed = result.reconstruct(x)
        
        # Reconstruction should be very close to original
        max_error = np.max(np.abs(data - reconstructed))
        assert max_error < 1e-10


class TestPPFStackResult:
    """Tests for PPFStackResult class"""

    def test_compression_properties(self):
        """Test compression ratio calculations"""
        result = PPFStackResult()
        result.original_size = 1000
        result.compressed_size = 500
        
        assert result.compression_ratio == 0.5
        assert result.space_savings == 0.5

    def test_serialization_roundtrip(self, synthetic_sine_data):
        """Test that serialization and deserialization work"""
        _, data = synthetic_sine_data
        
        layer = PPFResidualLayer(entropy_method=EntropyMethod.SPECTRAL)
        result = layer.analyze(data)
        
        # Serialize
        compressed = result.to_compressed_bytes()
        
        # Deserialize
        restored = PPFStackResult.from_compressed_bytes(compressed)
        
        # Check restoration
        assert len(restored.form_stack) == len(result.form_stack)
        np.testing.assert_array_almost_equal(
            restored.final_residuals,
            result.final_residuals
        )
