"""
Hybrid Decomposition Layer

Combines signal decomposition methods (EMD, SSA) with PPF interpretation.

Architecture:
1. Decomposition Layer - Separates signal into components (EMD/SSA)
2. Interpretation Layer - Names components using PPF form fitting
3. Reconstruction Layer - Rebuilds signal from interpreted components

Key Insight: EMD/SSA are good at SEPARATION, PPF is good at INTERPRETATION.
Together they provide both clean decomposition AND meaningful form names.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
from enum import Enum
from abc import ABC, abstractmethod

from .types import FormType, FitResult
from .detector import fit_form, find_best_form, evaluate_form
from .residual_layer import measure_entropy, EntropyMethod


class DecompositionMethod(Enum):
    """Available decomposition methods"""
    EMD = "emd"           # Empirical Mode Decomposition
    EEMD = "eemd"         # Ensemble EMD (more robust)
    CEEMDAN = "ceemdan"   # Complete EEMD with Adaptive Noise
    SSA = "ssa"           # Singular Spectrum Analysis


@dataclass
class InterpretedComponent:
    """A signal component with PPF interpretation"""
    index: int                      # Component index (0 = highest frequency)
    signal: np.ndarray              # The component signal

    # PPF interpretation
    form_type: Optional[FormType] = None
    form_params: Optional[np.ndarray] = None
    r_squared: float = 0.0

    # Characteristics
    mean_frequency: float = 0.0     # Estimated dominant frequency
    amplitude: float = 0.0          # RMS amplitude
    variance_contribution: float = 0.0  # Fraction of total variance
    entropy: float = 0.0            # Spectral entropy (noise-like?)

    # Interpretation
    interpretation: str = ""        # Human-readable description
    is_noise: bool = False          # Classified as noise component?

    @property
    def period(self) -> float:
        """Estimated period (inverse of frequency)"""
        if self.mean_frequency > 0:
            return 1.0 / self.mean_frequency
        return float('inf')

    def evaluate_form(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted form at given x values"""
        if self.form_type is None or self.form_params is None:
            return np.zeros_like(x)
        return evaluate_form(self.form_type, self.form_params, x)


@dataclass
class HybridDecompositionResult:
    """Result of hybrid decomposition + interpretation"""
    method: DecompositionMethod
    components: List[InterpretedComponent] = field(default_factory=list)
    residual: Optional[np.ndarray] = None

    # Summary statistics
    total_variance: float = 0.0
    explained_variance: float = 0.0
    n_signal_components: int = 0
    n_noise_components: int = 0

    @property
    def variance_explained_ratio(self) -> float:
        """Fraction of variance explained by signal components"""
        if self.total_variance == 0:
            return 0.0
        return self.explained_variance / self.total_variance

    def reconstruct(self, include_noise: bool = True) -> np.ndarray:
        """Reconstruct signal from components"""
        if not self.components:
            return np.array([])

        result = np.zeros_like(self.components[0].signal)
        for comp in self.components:
            if include_noise or not comp.is_noise:
                result += comp.signal
        return result

    def get_signal_components(self) -> List[InterpretedComponent]:
        """Get only signal (non-noise) components"""
        return [c for c in self.components if not c.is_noise]

    def get_noise_components(self) -> List[InterpretedComponent]:
        """Get only noise components"""
        return [c for c in self.components if c.is_noise]

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"Hybrid Decomposition ({self.method.value})",
            "=" * 50,
            f"Components: {len(self.components)} total",
            f"  Signal: {self.n_signal_components}",
            f"  Noise: {self.n_noise_components}",
            f"Variance explained: {self.variance_explained_ratio:.1%}",
            "",
            "Components:",
            "-" * 40,
        ]

        for comp in self.components:
            marker = "[NOISE]" if comp.is_noise else "[SIGNAL]"
            form_str = comp.form_type.value if comp.form_type else "unknown"
            lines.append(f"  {comp.index}: {marker} {form_str}")
            lines.append(f"      {comp.interpretation}")
            if comp.form_type == FormType.SINE:
                lines.append(f"      Period: {comp.period:.2f} samples")
            lines.append(f"      Variance: {comp.variance_contribution:.1%}")
            lines.append("")

        return "\n".join(lines)


def estimate_frequency(signal: np.ndarray) -> float:
    """Estimate dominant frequency using zero-crossings and FFT"""
    if len(signal) < 4:
        return 0.0

    # Method 1: Zero crossings
    centered = signal - np.mean(signal)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(centered))) > 0)
    freq_zc = zero_crossings / (2 * len(signal))

    # Method 2: FFT peak
    fft = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered))
    magnitudes = np.abs(fft[1:])  # Skip DC

    if len(magnitudes) > 0:
        peak_idx = np.argmax(magnitudes) + 1
        freq_fft = freqs[peak_idx]
    else:
        freq_fft = 0.0

    # Use FFT if it gives reasonable result, otherwise zero-crossings
    return freq_fft if freq_fft > 0 else freq_zc


def interpret_component(
    signal: np.ndarray,
    index: int,
    total_variance: float,
    noise_threshold: float = 0.5,
    min_r_squared: float = 0.3
) -> InterpretedComponent:
    """
    Interpret a single component using PPF form fitting.

    Args:
        signal: The component signal
        index: Component index
        total_variance: Total variance of original signal
        noise_threshold: Entropy above this = noise
        min_r_squared: Minimum R² for form acceptance

    Returns:
        InterpretedComponent with form interpretation
    """
    comp = InterpretedComponent(
        index=index,
        signal=signal.copy()
    )

    # Basic characteristics
    comp.amplitude = np.sqrt(np.mean(signal**2))  # RMS
    comp.mean_frequency = estimate_frequency(signal)
    comp.variance_contribution = np.var(signal) / total_variance if total_variance > 0 else 0.0
    comp.entropy = measure_entropy(signal, EntropyMethod.SPECTRAL)

    # Is this noise?
    comp.is_noise = comp.entropy > noise_threshold

    # Try to fit a form
    x = np.arange(len(signal), dtype=float)
    best_fit = find_best_form(x, signal, min_r_squared=min_r_squared)

    if best_fit:
        comp.form_type = best_fit.form_type
        comp.form_params = best_fit.params.copy()
        comp.r_squared = best_fit.r_squared

    # Generate interpretation
    comp.interpretation = _generate_interpretation(comp)

    return comp


def _generate_interpretation(comp: InterpretedComponent) -> str:
    """Generate human-readable interpretation of component"""
    if comp.is_noise:
        return f"Noise component (entropy={comp.entropy:.2f})"

    if comp.form_type is None:
        return "Unidentified structure"

    if comp.form_type == FormType.SINE:
        period = comp.period
        if period < 10:
            return f"High-frequency oscillation (period={period:.1f})"
        elif period < 100:
            return f"Medium-frequency cycle (period={period:.1f})"
        else:
            return f"Low-frequency cycle (period={period:.1f})"

    elif comp.form_type == FormType.LINEAR:
        slope = comp.form_params[0] if comp.form_params is not None else 0
        if abs(slope) < 0.001:
            return "Near-constant baseline"
        elif slope > 0:
            return f"Upward trend (slope={slope:.4f})"
        else:
            return f"Downward trend (slope={slope:.4f})"

    elif comp.form_type == FormType.QUADRATIC:
        return "Quadratic trend component"

    elif comp.form_type == FormType.EXPONENTIAL:
        rate = comp.form_params[1] if comp.form_params is not None and len(comp.form_params) > 1 else 0
        if rate > 0:
            return f"Exponential growth (rate={rate:.4f})"
        else:
            return f"Exponential decay (rate={rate:.4f})"

    elif comp.form_type == FormType.CONSTANT:
        return "Constant offset/baseline"

    return f"{comp.form_type.value} form"


class BaseDecomposer(ABC):
    """Abstract base class for signal decomposers"""

    @abstractmethod
    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Decompose signal into components.

        Returns:
            Tuple of (list of components, optional residual)
        """
        pass

    @property
    @abstractmethod
    def method(self) -> DecompositionMethod:
        """Return the decomposition method"""
        pass


class EMDDecomposer(BaseDecomposer):
    """
    Empirical Mode Decomposition wrapper.

    Decomposes signal into Intrinsic Mode Functions (IMFs).
    """

    def __init__(
        self,
        method: Literal["emd", "eemd", "ceemdan"] = "eemd",
        max_imfs: Optional[int] = None,
        noise_width: float = 0.05,
        ensemble_size: int = 100,
    ):
        """
        Args:
            method: EMD variant to use
            max_imfs: Maximum number of IMFs to extract
            noise_width: Noise amplitude for EEMD/CEEMDAN
            ensemble_size: Number of ensemble trials for EEMD/CEEMDAN
        """
        self._method_name = method
        self.max_imfs = max_imfs
        self.noise_width = noise_width
        self.ensemble_size = ensemble_size

        # Import PyEMD
        try:
            from PyEMD import EMD, EEMD, CEEMDAN
            self._EMD = EMD
            self._EEMD = EEMD
            self._CEEMDAN = CEEMDAN
        except ImportError:
            raise ImportError(
                "PyEMD is required for EMD decomposition. "
                "Install with: pip install EMD-signal"
            )

    @property
    def method(self) -> DecompositionMethod:
        if self._method_name == "emd":
            return DecompositionMethod.EMD
        elif self._method_name == "eemd":
            return DecompositionMethod.EEMD
        else:
            return DecompositionMethod.CEEMDAN

    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Decompose signal using EMD variant"""
        if self._method_name == "emd":
            decomposer = self._EMD()
        elif self._method_name == "eemd":
            decomposer = self._EEMD(
                noise_width=self.noise_width,
                trials=self.ensemble_size
            )
        else:  # ceemdan
            decomposer = self._CEEMDAN(
                epsilon=self.noise_width,
                trials=self.ensemble_size
            )

        # Perform decomposition
        if self.max_imfs:
            imfs = decomposer(signal, max_imf=self.max_imfs)
        else:
            imfs = decomposer(signal)

        # Last IMF is often the residual/trend
        components = list(imfs[:-1]) if len(imfs) > 1 else list(imfs)
        residual = imfs[-1] if len(imfs) > 1 else None

        return components, residual


class SSADecomposer(BaseDecomposer):
    """
    Singular Spectrum Analysis decomposition.

    Uses trajectory matrix and SVD to decompose signal.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        n_components: Optional[int] = None,
        grouping: Optional[List[List[int]]] = None,
    ):
        """
        Args:
            window_size: Embedding window size (default: len(signal)//2)
            n_components: Number of components to extract
            grouping: Optional grouping of singular values into components
        """
        self.window_size = window_size
        self.n_components = n_components
        self.grouping = grouping

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.SSA

    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Decompose signal using SSA"""
        n = len(signal)

        # Determine window size
        L = self.window_size if self.window_size else n // 2
        L = min(L, n - 1)
        K = n - L + 1

        # Build trajectory matrix
        X = np.zeros((L, K))
        for i in range(L):
            X[i, :] = signal[i:i + K]

        # SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        # Determine number of components
        n_comp = self.n_components if self.n_components else min(len(s), 10)
        n_comp = min(n_comp, len(s))

        # Reconstruct components
        components = []
        for i in range(n_comp):
            # Rank-1 matrix
            Xi = s[i] * np.outer(U[:, i], Vt[i, :])

            # Diagonal averaging to get component signal
            comp = self._diagonal_average(Xi, n)
            components.append(comp)

        # Residual is sum of remaining components
        if n_comp < len(s):
            residual = signal - sum(components)
        else:
            residual = None

        return components, residual

    def _diagonal_average(self, X: np.ndarray, n: int) -> np.ndarray:
        """Convert trajectory matrix back to time series via diagonal averaging"""
        L, K = X.shape
        result = np.zeros(n)
        counts = np.zeros(n)

        for i in range(L):
            for j in range(K):
                idx = i + j
                result[idx] += X[i, j]
                counts[idx] += 1

        return result / counts


class HybridDecomposer:
    """
    Hybrid signal decomposition combining EMD/SSA with PPF interpretation.

    This is the main entry point for hybrid analysis.

    Example:
        decomposer = HybridDecomposer(method="eemd")
        result = decomposer.analyze(signal)

        for comp in result.get_signal_components():
            print(f"{comp.form_type}: {comp.interpretation}")
    """

    def __init__(
        self,
        method: Literal["emd", "eemd", "ceemdan", "ssa"] = "eemd",
        noise_threshold: float = 0.5,
        min_r_squared: float = 0.3,
        min_variance_contribution: float = 0.01,
        **decomposer_kwargs
    ):
        """
        Args:
            method: Decomposition method to use
            noise_threshold: Entropy above this = noise component
            min_r_squared: Minimum R² for form fitting
            min_variance_contribution: Ignore components below this variance fraction
            **decomposer_kwargs: Additional arguments for decomposer
        """
        self.noise_threshold = noise_threshold
        self.min_r_squared = min_r_squared
        self.min_variance_contribution = min_variance_contribution

        # Create decomposer
        if method in ("emd", "eemd", "ceemdan"):
            self.decomposer = EMDDecomposer(method=method, **decomposer_kwargs)
        elif method == "ssa":
            self.decomposer = SSADecomposer(**decomposer_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze(self, signal: np.ndarray) -> HybridDecompositionResult:
        """
        Perform hybrid decomposition and interpretation.

        Args:
            signal: Input signal to analyze

        Returns:
            HybridDecompositionResult with interpreted components
        """
        # Step 1: Decompose
        raw_components, residual = self.decomposer.decompose(signal)

        # Step 2: Calculate total variance
        total_variance = np.var(signal)

        # Step 3: Interpret each component
        interpreted = []
        for i, comp_signal in enumerate(raw_components):
            # Skip tiny components
            var_contribution = np.var(comp_signal) / total_variance if total_variance > 0 else 0
            if var_contribution < self.min_variance_contribution:
                continue

            interp = interpret_component(
                comp_signal,
                index=i,
                total_variance=total_variance,
                noise_threshold=self.noise_threshold,
                min_r_squared=self.min_r_squared
            )
            interpreted.append(interp)

        # Step 4: Handle residual
        if residual is not None and len(residual) > 0:
            var_contribution = np.var(residual) / total_variance if total_variance > 0 else 0
            if var_contribution >= self.min_variance_contribution:
                residual_interp = interpret_component(
                    residual,
                    index=len(interpreted),
                    total_variance=total_variance,
                    noise_threshold=self.noise_threshold,
                    min_r_squared=self.min_r_squared
                )
                # Residual often contains trend
                if residual_interp.form_type in (FormType.LINEAR, FormType.CONSTANT, FormType.QUADRATIC):
                    residual_interp.interpretation = "Trend/baseline: " + residual_interp.interpretation
                interpreted.append(residual_interp)

        # Step 5: Build result
        result = HybridDecompositionResult(
            method=self.decomposer.method,
            components=interpreted,
            residual=residual,
            total_variance=total_variance,
        )

        # Calculate summary stats
        signal_components = [c for c in interpreted if not c.is_noise]
        noise_components = [c for c in interpreted if c.is_noise]

        result.n_signal_components = len(signal_components)
        result.n_noise_components = len(noise_components)
        result.explained_variance = sum(
            c.variance_contribution * total_variance for c in signal_components
        )

        return result


def print_hybrid_result(result: HybridDecompositionResult) -> None:
    """Pretty print hybrid decomposition result"""
    print("=" * 70)
    print(f"HYBRID DECOMPOSITION RESULT ({result.method.value.upper()})")
    print("=" * 70)
    print(f"Total components: {len(result.components)}")
    print(f"Signal components: {result.n_signal_components}")
    print(f"Noise components: {result.n_noise_components}")
    print(f"Variance explained by signal: {result.variance_explained_ratio:.1%}")
    print()

    print("Components (ordered by frequency, high to low):")
    print("-" * 60)

    for comp in result.components:
        status = "[NOISE]" if comp.is_noise else "[SIGNAL]"
        form = comp.form_type.value if comp.form_type else "?"

        print(f"\n  Component {comp.index}: {status}")
        print(f"    Form: {form} (R²={comp.r_squared:.2f})")
        print(f"    Interpretation: {comp.interpretation}")
        print(f"    Frequency: {comp.mean_frequency:.4f} (period={comp.period:.1f})")
        print(f"    Amplitude (RMS): {comp.amplitude:.4f}")
        print(f"    Variance contribution: {comp.variance_contribution:.1%}")
        print(f"    Entropy: {comp.entropy:.3f}")

        if comp.form_type == FormType.SINE and comp.form_params is not None:
            print(f"    Sine params: amp={comp.form_params[0]:.3f}, freq={comp.form_params[1]:.4f}")

    print()
