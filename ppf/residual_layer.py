"""
PPF + Residual Storage Layer

Augments PPF (Promising Partial Form) detection with explicit residual storage.
This layer sits on top of PPF and provides:

1. Recursive form extraction - keeps finding forms until residuals are noise
2. Entropy-based noise detection - via gzip compression or spectral flatness
3. Compression-aware stopping - stops when adding forms doesn't improve ratio
4. Dual output - form stack (analysis) + compressed representation (storage)

The key insight: residuals that don't compress well are noise (high entropy).
Residuals that compress well have structure (low entropy) - keep extracting.
"""

import numpy as np
import gzip
import struct
from dataclasses import dataclass, field
from typing import List

from .types import FormType, EntropyMethod, FormLayer
from .detector import evaluate_form, fit_form


@dataclass
class PPFStackResult:
    """
    Result of PPF + Residual analysis.
    Provides both analysis view (form stack) and compression view.
    """
    # Analysis view
    form_stack: List[FormLayer] = field(default_factory=list)
    final_residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    original_size: int = 0
    compressed_size: int = 0
    iterations: int = 0
    stopped_reason: str = ""

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved (< 1.0 means compression worked)"""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size

    @property
    def space_savings(self) -> float:
        """Percentage of space saved"""
        return 1.0 - self.compression_ratio

    @property
    def total_params(self) -> int:
        """Total number of parameters across all forms"""
        return sum(len(layer.params) for layer in self.form_stack)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Reconstruct original data from forms + residuals.

        original ≈ form1(x) + form2(x) + ... + final_residuals
        """
        result = np.zeros_like(x, dtype=float)

        for layer in self.form_stack:
            result += layer.evaluate(x)

        if len(self.final_residuals) == len(x):
            result += self.final_residuals

        return result

    def to_compressed_bytes(self) -> bytes:
        """
        Serialize to compressed bytes for storage.
        Format: [n_forms][form1_type][form1_params]...[residuals]
        """
        parts = []

        # Number of forms
        parts.append(struct.pack('I', len(self.form_stack)))

        # Each form: type (1 byte) + n_params (1 byte) + params (float64 each)
        for layer in self.form_stack:
            form_type_id = list(FormType).index(layer.form_type)
            parts.append(struct.pack('BB', form_type_id, len(layer.params)))
            for p in layer.params:
                parts.append(struct.pack('d', p))

        # Residuals: length + gzipped float64 array
        residual_bytes = self.final_residuals.astype(np.float64).tobytes()
        compressed_residuals = gzip.compress(residual_bytes)
        parts.append(struct.pack('I', len(self.final_residuals)))
        parts.append(struct.pack('I', len(compressed_residuals)))
        parts.append(compressed_residuals)

        return b''.join(parts)

    @classmethod
    def from_compressed_bytes(cls, data: bytes) -> 'PPFStackResult':
        """Deserialize from compressed bytes"""
        offset = 0

        # Number of forms
        n_forms = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # Read forms
        form_stack = []
        for _ in range(n_forms):
            form_type_id, n_params = struct.unpack('BB', data[offset:offset+2])
            offset += 2

            params = []
            for _ in range(n_params):
                p = struct.unpack('d', data[offset:offset+8])[0]
                params.append(p)
                offset += 8

            form_stack.append(FormLayer(
                form_type=list(FormType)[form_type_id],
                params=np.array(params),
                r_squared=0.0,
                residual_entropy=0.0,
                compression_gain=0.0
            ))

        # Read residuals
        n_residuals = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4
        compressed_len = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        compressed_residuals = data[offset:offset+compressed_len]
        residual_bytes = gzip.decompress(compressed_residuals)
        final_residuals = np.frombuffer(residual_bytes, dtype=np.float64)

        result = cls()
        result.form_stack = form_stack
        result.final_residuals = final_residuals
        return result


def measure_entropy_gzip(data: np.ndarray) -> float:
    """
    Measure entropy using gzip compression ratio.

    Returns value in [0, 1]:
        ~1.0 = high entropy (noise, incompressible)
        ~0.0 = low entropy (structured, compressible)
    """
    if len(data) == 0:
        return 1.0

    raw_bytes = data.astype(np.float64).tobytes()
    compressed = gzip.compress(raw_bytes, compresslevel=6)

    ratio = len(compressed) / len(raw_bytes)
    return min(ratio, 1.0)


def measure_entropy_spectral(data: np.ndarray) -> float:
    """
    Measure entropy using spectral flatness (Wiener entropy).

    GPU-friendly alternative to gzip.

    Returns value in [0, 1]:
        ~1.0 = flat spectrum (white noise)
        ~0.0 = peaked spectrum (periodic structure)
    """
    if len(data) < 4:
        return 1.0

    centered = data - np.mean(data)

    if np.std(centered) < 1e-10:
        return 0.0  # Constant = no entropy

    fft = np.fft.rfft(centered)
    power = np.abs(fft) ** 2

    # Avoid log(0)
    power = power + 1e-10

    log_power = np.log(power)
    geometric_mean = np.exp(np.mean(log_power))
    arithmetic_mean = np.mean(power)

    flatness = geometric_mean / arithmetic_mean
    return float(flatness)


def measure_entropy(data: np.ndarray, method: EntropyMethod) -> float:
    """Measure entropy using specified method"""
    if method == EntropyMethod.GZIP:
        return measure_entropy_gzip(data)
    elif method == EntropyMethod.SPECTRAL:
        return measure_entropy_spectral(data)
    else:
        raise ValueError(f"Unknown entropy method: {method}")


def estimate_compressed_size(form_stack: List[FormLayer], residuals: np.ndarray) -> int:
    """
    Estimate total compressed size of form stack + residuals.
    """
    # Form parameters: 8 bytes per float64 param + 2 bytes header per form
    form_size = sum(2 + len(layer.params) * 8 for layer in form_stack)

    # Residuals: estimate compressed size
    if len(residuals) == 0:
        residual_size = 0
    else:
        raw = residuals.astype(np.float64).tobytes()
        compressed = gzip.compress(raw, compresslevel=6)
        residual_size = len(compressed) + 8  # +8 for length headers

    return form_size + residual_size


class PPFResidualLayer:
    """
    PPF + Residual Storage Layer

    Recursively extracts forms from data until residuals are noise,
    using compression ratio as the guide.

    Args:
        entropy_method: How to measure entropy (GZIP or SPECTRAL)
        noise_threshold: Entropy above this = noise, stop extracting
        min_compression_gain: Minimum improvement to justify adding a form
        max_iterations: Maximum forms to stack
        min_r_squared: Minimum R² for a form to be accepted
        min_points: Minimum data points to attempt form detection

    Raises:
        ValueError: If parameters are invalid
    """

    def __init__(
        self,
        entropy_method: EntropyMethod = EntropyMethod.GZIP,
        noise_threshold: float = 0.85,
        min_compression_gain: float = 0.05,
        max_iterations: int = 5,
        min_r_squared: float = 0.5,
        min_points: int = 20,
    ):
        if not 0 <= noise_threshold <= 1:
            raise ValueError("noise_threshold must be between 0 and 1")
        if not 0 <= min_compression_gain <= 1:
            raise ValueError("min_compression_gain must be between 0 and 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0 <= min_r_squared <= 1:
            raise ValueError("min_r_squared must be between 0 and 1")
        if min_points < 4:
            raise ValueError("min_points must be at least 4")

        self.entropy_method = entropy_method
        self.noise_threshold = noise_threshold
        self.min_compression_gain = min_compression_gain
        self.max_iterations = max_iterations
        self.min_r_squared = min_r_squared
        self.min_points = min_points

    def analyze(self, data: np.ndarray) -> PPFStackResult:
        """
        Analyze data, extracting forms until residuals are noise
        or compression stops improving.

        Args:
            data: Input data array

        Returns:
            PPFStackResult with form stack and final residuals
        """
        result = PPFStackResult()
        result.original_size = len(data) * 8  # float64

        if len(data) < self.min_points:
            result.final_residuals = data.copy()
            result.compressed_size = estimate_compressed_size([], data)
            result.stopped_reason = "insufficient_data"
            return result

        current_data = data.copy()
        x = np.arange(len(data), dtype=float)

        best_compressed_size = estimate_compressed_size([], current_data)

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1

            # Try ALL form types and pick the one with best variance reduction
            best_fit = None
            best_variance_reduction = 0.0
            best_new_residuals = None

            current_variance = np.var(current_data)

            for form_type in FormType:
                fit = fit_form(x, current_data, form_type)

                if fit is None or fit.r_squared < self.min_r_squared:
                    continue

                trial_residuals = current_data - evaluate_form(fit.form_type, fit.params, x)
                residual_variance = np.var(trial_residuals)

                variance_reduction = (current_variance - residual_variance) / current_variance

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_fit = fit
                    best_new_residuals = trial_residuals

            # No form explains enough variance
            if best_fit is None or best_variance_reduction < self.min_compression_gain:
                result.stopped_reason = f"no_form_reduces_variance (best reduction={best_variance_reduction:.3f})"
                break

            # Accept this form
            residual_entropy = measure_entropy(best_new_residuals, self.entropy_method)

            new_layer = FormLayer(
                form_type=best_fit.form_type,
                params=best_fit.params,
                r_squared=best_fit.r_squared,
                residual_entropy=residual_entropy,
                compression_gain=best_variance_reduction
            )
            result.form_stack.append(new_layer)

            current_data = best_new_residuals
            best_compressed_size = estimate_compressed_size(result.form_stack, current_data)

            # Check if residuals are noise-like (stop condition)
            if residual_entropy > self.noise_threshold:
                result.stopped_reason = f"residuals_are_noise (entropy={residual_entropy:.3f})"
                break

        else:
            result.stopped_reason = "max_iterations_reached"

        result.final_residuals = current_data
        result.compressed_size = estimate_compressed_size(result.form_stack, current_data)

        return result
