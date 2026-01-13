"""
PPF Shared Types

Central location for types used across the PPF package.
Prevents circular imports and duplicate definitions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class FormType(Enum):
    """Types of mathematical forms that can be detected"""
    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SINE = "sine"
    EXPONENTIAL = "exponential"


class EntropyMethod(Enum):
    """Methods for measuring entropy/randomness in data"""
    GZIP = "gzip"           # Accurate, CPU-bound
    SPECTRAL = "spectral"   # GPU-ready, good for periodic structures


@dataclass
class FitResult:
    """Result of fitting a form to data"""
    form_type: FormType
    params: np.ndarray
    r_squared: float
    residuals: np.ndarray
    residual_std: float
    is_noise_like: bool  # Do residuals look like white noise?


@dataclass
class PartialForm:
    """A detected partial form with extrapolation capability"""
    start_idx: int
    end_idx: int
    fit: FitResult
    confidence: float

    def extrapolate(self, n_points: int) -> np.ndarray:
        """Extrapolate the form beyond its detected region"""
        # Imported here to avoid circular import
        from .detector import evaluate_form
        form_length = self.end_idx - self.start_idx
        indices = np.arange(form_length, form_length + n_points)
        return evaluate_form(self.fit.form_type, self.fit.params, indices)


@dataclass
class PPFResult:
    """Result of PPF analysis"""
    partial_forms: List[PartialForm] = field(default_factory=list)
    validated_forms: List[PartialForm] = field(default_factory=list)
    structure_score: float = 0.0
    noise_level: float = 0.0


@dataclass
class FormLayer:
    """A single layer in the form stack"""
    form_type: FormType
    params: np.ndarray
    r_squared: float
    residual_entropy: float  # Entropy of residuals after this form
    compression_gain: float  # How much this form improved compression

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate this form at given x values"""
        from .detector import evaluate_form
        return evaluate_form(self.form_type, self.params, x)

    def param_bytes(self) -> int:
        """Size of parameters in bytes"""
        return len(self.params) * 8  # float64


@dataclass
class WindowFit:
    """Result of fitting a form to a single window"""
    window_start: int
    window_end: int
    center_position: float  # Center of window (for plotting parameter evolution)
    form_type: FormType
    params: np.ndarray
    r_squared: float

    def get_param(self, name: str) -> Optional[float]:
        """Extract named parameter based on form type"""
        if self.form_type == FormType.SINE:
            param_map = {'amplitude': 0, 'frequency': 1, 'phase': 2, 'offset': 3}
        elif self.form_type == FormType.LINEAR:
            param_map = {'slope': 0, 'intercept': 1}
        elif self.form_type == FormType.QUADRATIC:
            param_map = {'a': 0, 'b': 1, 'c': 2}
        elif self.form_type == FormType.EXPONENTIAL:
            param_map = {'amplitude': 0, 'rate': 1, 'offset': 2}
        elif self.form_type == FormType.CONSTANT:
            param_map = {'value': 0}
        else:
            param_map = {}

        if name in param_map:
            return self.params[param_map[name]]
        return None
