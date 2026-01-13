"""
Hierarchical Form Detection

Extends PPF + Residual Layer to detect nested patterns:
- Level 1: Find forms in data windows
- Level 2: Find forms in how Level 1 parameters evolve
- Level N: Recursively find patterns in patterns

Key insight: "Irregular" data often has hierarchical structure where
form parameters themselves follow forms at longer timescales.

Example: Sunspot data
- Level 1: ~11 year solar cycle (sine)
- Level 2: ~100 year Gleissberg cycle (modulates amplitude)
- Level 3: Potentially longer cycles...
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .types import FormType, EntropyMethod, WindowFit
from .detector import fit_form, evaluate_form
from .residual_layer import PPFResidualLayer, PPFStackResult


@dataclass
class ParameterEvolution:
    """How a form parameter evolves across windows"""
    param_name: str
    positions: np.ndarray      # Window center positions
    values: np.ndarray         # Parameter values at each position
    ppf_result: Optional[PPFStackResult] = None  # Form analysis of this evolution

    @property
    def has_structure(self) -> bool:
        """Does the parameter evolution have detectable structure?"""
        if self.ppf_result is None:
            return False
        return len(self.ppf_result.form_stack) > 0

    @property
    def variance_explained(self) -> float:
        """How much variance is explained by detected forms"""
        if self.ppf_result is None or len(self.values) == 0:
            return 0.0
        residual_var = np.var(self.ppf_result.final_residuals)
        total_var = np.var(self.values)
        if total_var == 0:
            return 1.0
        return 1.0 - (residual_var / total_var)


@dataclass
class HierarchyLevel:
    """One level in the hierarchical decomposition"""
    level: int
    description: str
    window_fits: List[WindowFit] = field(default_factory=list)
    parameter_evolutions: Dict[str, ParameterEvolution] = field(default_factory=dict)
    dominant_form: Optional[FormType] = None
    coverage: float = 0.0  # Fraction of windows successfully fit


@dataclass
class HierarchicalResult:
    """Complete hierarchical decomposition result"""
    levels: List[HierarchyLevel] = field(default_factory=list)
    original_data_length: int = 0

    def summary(self) -> str:
        """Generate text summary of hierarchy"""
        lines = []
        lines.append("Hierarchical Structure Detected:")
        lines.append("=" * 50)

        for level in self.levels:
            lines.append(f"\nLevel {level.level}: {level.description}")
            lines.append("-" * 40)

            if level.window_fits:
                lines.append(f"  Windows analyzed: {len(level.window_fits)}")
                lines.append(f"  Coverage: {level.coverage:.1%}")
                lines.append(f"  Dominant form: {level.dominant_form.value if level.dominant_form else 'mixed'}")

            if level.parameter_evolutions:
                lines.append(f"  Parameter evolutions analyzed:")
                for name, evo in level.parameter_evolutions.items():
                    if evo.has_structure:
                        lines.append(f"    - {name}: {evo.variance_explained:.1%} variance explained")
                        for form in evo.ppf_result.form_stack:
                            if form.form_type == FormType.SINE:
                                freq = form.params[1]
                                period = 2 * np.pi / freq if freq > 0 else float('inf')
                                lines.append(f"        -> {form.form_type.value} (period={period:.1f} windows)")
                            else:
                                lines.append(f"        -> {form.form_type.value}")
                    else:
                        lines.append(f"    - {name}: no structure detected")

        return "\n".join(lines)


class HierarchicalDetector:
    """
    Hierarchical Pattern Detector

    Finds nested patterns by:
    1. Fitting forms to windows of data
    2. Extracting parameter sequences
    3. Finding forms in parameter evolution
    4. Recursively continuing up the hierarchy

    Args:
        window_size: Size of windows for form fitting (auto-detected if None)
        window_overlap: Overlap between windows (0.0 = none, 0.5 = 50%)
        min_r_squared: Minimum R² for accepting a form
        preferred_form: If set, only try this form type
        max_levels: Maximum hierarchy levels to detect
        min_windows_for_meta: Minimum windows needed to analyze parameter evolution
        entropy_method: Method for measuring entropy
        noise_threshold: Entropy above this = noise
        min_variance_gain: Minimum variance improvement to accept a form

    Raises:
        ValueError: If parameters are invalid
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        window_overlap: float = 0.0,
        min_r_squared: float = 0.3,
        preferred_form: Optional[FormType] = None,
        max_levels: int = 3,
        min_windows_for_meta: int = 8,
        entropy_method: EntropyMethod = EntropyMethod.SPECTRAL,
        noise_threshold: float = 0.5,
        min_variance_gain: float = 0.05,
    ):
        if window_size is not None and window_size < 4:
            raise ValueError("window_size must be at least 4 or None")
        if not 0 <= window_overlap < 1:
            raise ValueError("window_overlap must be between 0 and 1 (exclusive)")
        if not 0 <= min_r_squared <= 1:
            raise ValueError("min_r_squared must be between 0 and 1")
        if max_levels < 1:
            raise ValueError("max_levels must be at least 1")
        if min_windows_for_meta < 4:
            raise ValueError("min_windows_for_meta must be at least 4")
        if not 0 <= noise_threshold <= 1:
            raise ValueError("noise_threshold must be between 0 and 1")
        if not 0 <= min_variance_gain <= 1:
            raise ValueError("min_variance_gain must be between 0 and 1")

        self.window_size = window_size
        self.window_overlap = window_overlap
        self.min_r_squared = min_r_squared
        self.preferred_form = preferred_form
        self.max_levels = max_levels
        self.min_windows_for_meta = min_windows_for_meta
        self.entropy_method = entropy_method
        self.noise_threshold = noise_threshold
        self.min_variance_gain = min_variance_gain

        # PPF layer for parameter evolution analysis
        self.ppf_layer = PPFResidualLayer(
            entropy_method=entropy_method,
            noise_threshold=noise_threshold,
            min_compression_gain=min_variance_gain,
            min_r_squared=0.15,  # Lower threshold for meta-patterns
            max_iterations=4
        )

    def _estimate_window_size(self, data: np.ndarray) -> int:
        """Estimate good window size using FFT to find dominant period"""
        if len(data) < 20:
            return len(data) // 2

        centered = data - np.mean(data)
        fft = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(len(centered))

        magnitudes = np.abs(fft[1:])
        if len(magnitudes) == 0:
            return len(data) // 4

        peak_idx = np.argmax(magnitudes) + 1
        peak_freq = freqs[peak_idx]

        if peak_freq > 0:
            period = int(1.0 / peak_freq)
            window = min(max(period, 20), len(data) // 3)
            return window

        return len(data) // 4

    def _fit_window(
        self,
        data: np.ndarray,
        start: int,
        end: int
    ) -> Optional[WindowFit]:
        """Fit best form to a data window"""
        window = data[start:end]
        x = np.arange(len(window), dtype=float)

        if self.preferred_form:
            fit = fit_form(x, window, self.preferred_form)
            if fit and fit.r_squared >= self.min_r_squared:
                return WindowFit(
                    window_start=start,
                    window_end=end,
                    center_position=(start + end) / 2.0,
                    form_type=fit.form_type,
                    params=fit.params.copy(),
                    r_squared=fit.r_squared
                )
        else:
            best_fit = None
            best_r_squared = self.min_r_squared

            for form_type in FormType:
                fit = fit_form(x, window, form_type)
                if fit and fit.r_squared > best_r_squared:
                    best_fit = fit
                    best_r_squared = fit.r_squared

            if best_fit:
                return WindowFit(
                    window_start=start,
                    window_end=end,
                    center_position=(start + end) / 2.0,
                    form_type=best_fit.form_type,
                    params=best_fit.params.copy(),
                    r_squared=best_fit.r_squared
                )

        return None

    def _extract_parameter_evolutions(
        self,
        window_fits: List[WindowFit],
        form_type: FormType
    ) -> Dict[str, ParameterEvolution]:
        """Extract how each parameter evolves across windows"""

        matching = [wf for wf in window_fits if wf.form_type == form_type]
        if len(matching) < self.min_windows_for_meta:
            return {}

        if form_type == FormType.SINE:
            param_names = ['amplitude', 'frequency', 'phase', 'offset']
        elif form_type == FormType.LINEAR:
            param_names = ['slope', 'intercept']
        elif form_type == FormType.QUADRATIC:
            param_names = ['a', 'b', 'c']
        elif form_type == FormType.EXPONENTIAL:
            param_names = ['amplitude', 'rate', 'offset']
        elif form_type == FormType.CONSTANT:
            param_names = ['value']
        else:
            param_names = [f'param_{i}' for i in range(len(matching[0].params))]

        evolutions = {}
        positions = np.array([wf.center_position for wf in matching])

        for i, name in enumerate(param_names):
            if i >= len(matching[0].params):
                break

            values = np.array([wf.params[i] for wf in matching])

            # Special handling: amplitude should be absolute
            if name == 'amplitude':
                values = np.abs(values)

            # Special handling: phase needs unwrapping
            if name == 'phase':
                values = np.unwrap(values)

            evolutions[name] = ParameterEvolution(
                param_name=name,
                positions=positions,
                values=values
            )

        return evolutions

    def _analyze_parameter_evolution(
        self,
        evolution: ParameterEvolution
    ) -> ParameterEvolution:
        """Apply PPF to find forms in parameter evolution"""
        if len(evolution.values) < self.min_windows_for_meta:
            return evolution

        result = self.ppf_layer.analyze(evolution.values)
        evolution.ppf_result = result

        return evolution

    def analyze(self, data: np.ndarray) -> HierarchicalResult:
        """
        Perform hierarchical pattern detection.

        Args:
            data: Input data array

        Returns:
            HierarchicalResult with all detected levels
        """
        result = HierarchicalResult(original_data_length=len(data))

        window_size = self.window_size
        if window_size is None:
            window_size = self._estimate_window_size(data)

        step = int(window_size * (1 - self.window_overlap))
        step = max(step, 1)

        # Level 1: Fit forms to windows
        level1 = HierarchyLevel(
            level=1,
            description=f"Base forms (window={window_size})"
        )

        for start in range(0, len(data) - window_size + 1, step):
            end = start + window_size
            window_fit = self._fit_window(data, start, end)
            if window_fit:
                level1.window_fits.append(window_fit)

        if level1.window_fits:
            level1.coverage = len(level1.window_fits) / ((len(data) - window_size) // step + 1)

            form_counts = {}
            for wf in level1.window_fits:
                form_counts[wf.form_type] = form_counts.get(wf.form_type, 0) + 1
            level1.dominant_form = max(form_counts, key=form_counts.get)

        result.levels.append(level1)

        # Level 2+: Analyze parameter evolutions
        current_level = level1

        for level_num in range(2, self.max_levels + 1):
            if not current_level.window_fits:
                break

            if current_level.dominant_form is None:
                break

            evolutions = self._extract_parameter_evolutions(
                current_level.window_fits,
                current_level.dominant_form
            )

            if not evolutions:
                break

            new_level = HierarchyLevel(
                level=level_num,
                description=f"Parameter evolution of {current_level.dominant_form.value}"
            )

            any_structure = False
            for name, evo in evolutions.items():
                analyzed = self._analyze_parameter_evolution(evo)
                new_level.parameter_evolutions[name] = analyzed
                if analyzed.has_structure:
                    any_structure = True

            if not any_structure:
                result.levels.append(new_level)
                break

            result.levels.append(new_level)
            break  # For now, stop at parameter evolution analysis

        return result
