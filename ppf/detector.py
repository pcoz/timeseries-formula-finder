"""
Promising Partial Form (PPF) Detector

Detects mathematical forms in noisy data by:
1. Fitting candidate forms using least-squares (not exact matching)
2. Analyzing residuals to determine if they look like noise
3. Extrapolating promising forms past noise gaps
4. Validating by checking if extrapolation matches subsequent data
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.optimize import curve_fit

from .types import FormType, FitResult, PartialForm, PPFResult


# Form evaluation functions
def constant_func(x, c):
    return np.full_like(x, c, dtype=float)


def linear_func(x, a, b):
    return a * x + b


def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c


def sine_func(x, amp, freq, phase, offset):
    return amp * np.sin(freq * x + phase) + offset


def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c


FORM_FUNCS = {
    FormType.CONSTANT: (constant_func, 1, [0.0]),
    FormType.LINEAR: (linear_func, 2, [1.0, 0.0]),
    FormType.QUADRATIC: (quadratic_func, 3, [0.0, 1.0, 0.0]),
    FormType.SINE: (sine_func, 4, [1.0, 0.1, 0.0, 0.0]),
    FormType.EXPONENTIAL: (exponential_func, 3, [1.0, 0.01, 0.0]),
}


def evaluate_form(form_type: FormType, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a form at given x values"""
    func, _, _ = FORM_FUNCS[form_type]
    return func(x, *params)


def fit_form(x: np.ndarray, y: np.ndarray, form_type: FormType) -> Optional[FitResult]:
    """
    Fit a form to data using least-squares regression.

    Args:
        x: Independent variable values
        y: Dependent variable values
        form_type: Type of form to fit

    Returns:
        FitResult if fitting succeeds, None otherwise
    """
    if len(x) < 3:
        return None

    func, n_params, p0 = FORM_FUNCS[form_type]

    try:
        # Initial guess based on data characteristics
        if form_type == FormType.CONSTANT:
            p0 = [np.mean(y)]
        elif form_type == FormType.LINEAR:
            p0 = [np.polyfit(x, y, 1)[0], np.polyfit(x, y, 1)[1]]
        elif form_type == FormType.QUADRATIC:
            coeffs = np.polyfit(x, y, 2)
            p0 = [coeffs[0], coeffs[1], coeffs[2]]
        elif form_type == FormType.SINE:
            # Use FFT to estimate dominant frequency
            # First detrend to remove linear components that confuse FFT
            from scipy import signal as scipy_signal
            y_detrended = scipy_signal.detrend(y, type='linear')

            amp = (np.max(y_detrended) - np.min(y_detrended)) / 2
            offset = np.mean(y)

            # FFT-based frequency estimation on detrended data
            n = len(y_detrended)
            if n > 4:
                fft = np.fft.rfft(y_detrended)
                freqs = np.fft.rfftfreq(n, d=1.0)

                # Find peak frequency (skip DC component at index 0)
                magnitudes = np.abs(fft[1:])
                if len(magnitudes) > 0:
                    peak_idx = np.argmax(magnitudes) + 1
                    freq = 2 * np.pi * freqs[peak_idx]
                else:
                    freq = 2 * np.pi / n
            else:
                freq = 2 * np.pi / max(n, 10)

            phase = 0.0
            p0 = [amp, freq, phase, offset]
        elif form_type == FormType.EXPONENTIAL:
            p0 = [y[0] if y[0] != 0 else 1.0, 0.01, 0.0]

        # Fit with bounds to prevent extreme values
        if form_type == FormType.SINE:
            bounds = ([0, 0, -np.pi, -np.inf], [np.inf, np.pi, np.pi, np.inf])
            params, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)
        elif form_type == FormType.EXPONENTIAL:
            bounds = ([-np.inf, -1, -np.inf], [np.inf, 1, np.inf])
            params, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)
        else:
            params, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)

        # Calculate fit quality
        y_pred = func(x, *params)
        residuals = y - y_pred

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        residual_std = np.std(residuals)
        is_noise_like = check_residuals_are_noise(residuals)

        return FitResult(
            form_type=form_type,
            params=params,
            r_squared=r_squared,
            residuals=residuals,
            residual_std=residual_std,
            is_noise_like=is_noise_like
        )

    except Exception:
        return None


def check_residuals_are_noise(residuals: np.ndarray, significance: float = 0.05) -> bool:
    """
    Test if residuals look like white noise.

    Uses autocorrelation and runs test.

    Args:
        residuals: Array of residual values
        significance: Significance level for statistical tests

    Returns:
        True if residuals appear to be white noise
    """
    if len(residuals) < 10:
        return True  # Not enough data to tell

    # Test 1: Autocorrelation at lag 1 should be near 0
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        if abs(autocorr) > 0.3:
            return False

    # Test 2: Runs test for randomness
    signs = np.sign(residuals - np.mean(residuals))
    runs = 1 + np.sum(signs[:-1] != signs[1:])
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)

    if n_pos > 0 and n_neg > 0:
        expected_runs = 1 + 2 * n_pos * n_neg / (n_pos + n_neg)
        std_runs = np.sqrt(2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg) /
                          ((n_pos + n_neg)**2 * (n_pos + n_neg - 1)))
        if std_runs > 0:
            z_score = abs(runs - expected_runs) / std_runs
            if z_score > 2.5:
                return False

    return True


def find_best_form(x: np.ndarray, y: np.ndarray, min_r_squared: float = 0.7) -> Optional[FitResult]:
    """
    Try all form types and return the best fit.

    Uses model selection criteria (prefer simpler models if fit is similar).

    Args:
        x: Independent variable values
        y: Dependent variable values
        min_r_squared: Minimum R² threshold for acceptance

    Returns:
        Best FitResult if any form meets threshold, None otherwise
    """
    results = []

    for form_type in FormType:
        fit = fit_form(x, y, form_type)
        if fit and fit.r_squared >= min_r_squared:
            results.append(fit)

    if not results:
        return None

    # Sort by R² but penalize complex models slightly
    def score(fit):
        n_params = len(fit.params)
        complexity_penalty = 0.02 * n_params
        return fit.r_squared - complexity_penalty

    results.sort(key=score, reverse=True)
    return results[0]


class PPFDetector:
    """
    Promising Partial Form Detector

    Detects forms in noisy data by fitting (not exact matching),
    then validates by extrapolation.

    Args:
        min_window: Minimum window size for form detection
        min_r_squared: Minimum R² for accepting a form
        extrapolation_window: Points to extrapolate for validation
        validation_threshold: Correlation threshold for validation

    Raises:
        ValueError: If parameters are invalid
    """

    def __init__(
        self,
        min_window: int = 20,
        min_r_squared: float = 0.7,
        extrapolation_window: int = 10,
        validation_threshold: float = 0.6,
    ):
        if min_window < 4:
            raise ValueError("min_window must be at least 4")
        if not 0 <= min_r_squared <= 1:
            raise ValueError("min_r_squared must be between 0 and 1")
        if extrapolation_window < 1:
            raise ValueError("extrapolation_window must be at least 1")
        if not 0 <= validation_threshold <= 1:
            raise ValueError("validation_threshold must be between 0 and 1")

        self.min_window = min_window
        self.min_r_squared = min_r_squared
        self.extrapolation_window = extrapolation_window
        self.validation_threshold = validation_threshold

    def detect_partial_forms(self, data: np.ndarray) -> List[PartialForm]:
        """
        Scan data for promising partial forms using sliding window.

        Args:
            data: Input data array

        Returns:
            List of detected PartialForm objects
        """
        partial_forms = []
        n = len(data)
        pos = 0

        while pos < n - self.min_window:
            best_fit = None
            best_length = 0

            for length in range(self.min_window, min(n - pos, self.min_window * 4)):
                window = data[pos:pos + length]
                x = np.arange(len(window))

                fit = find_best_form(x, window, self.min_r_squared)

                if fit and fit.is_noise_like:
                    best_fit = fit
                    best_length = length
                else:
                    break

            if best_fit and best_length >= self.min_window:
                confidence = best_fit.r_squared * (1.0 if best_fit.is_noise_like else 0.7)

                partial_forms.append(PartialForm(
                    start_idx=pos,
                    end_idx=pos + best_length,
                    fit=best_fit,
                    confidence=confidence
                ))

                pos += best_length
            else:
                pos += 1

        return partial_forms

    def validate_by_extrapolation(
        self,
        data: np.ndarray,
        partial_form: PartialForm,
        skip_window: int = 5,
        check_window: int = 10
    ) -> Tuple[bool, float]:
        """
        Validate a partial form by extrapolating and checking match.

        Args:
            data: Full data array
            partial_form: Form to validate
            skip_window: Points to skip past form end
            check_window: Points to check after skip

        Returns:
            Tuple of (is_valid, correlation_score)
        """
        end = partial_form.end_idx
        check_start = end + skip_window
        check_end = check_start + check_window

        if check_end > len(data):
            return False, 0.0

        actual = data[check_start:check_end]

        form_length = partial_form.end_idx - partial_form.start_idx
        extrap_start = form_length + skip_window
        extrap_end = extrap_start + check_window
        extrap_indices = np.arange(extrap_start, extrap_end)
        predicted = evaluate_form(
            partial_form.fit.form_type,
            partial_form.fit.params,
            extrap_indices
        )

        if np.std(actual) > 0 and np.std(predicted) > 0:
            correlation = np.corrcoef(actual, predicted)[0, 1]
        else:
            correlation = 1.0 if np.allclose(actual, predicted, rtol=0.1) else 0.0

        is_valid = correlation > self.validation_threshold
        return is_valid, correlation

    def analyze(self, data: np.ndarray) -> PPFResult:
        """
        Full PPF analysis: detect, validate, and score.

        Args:
            data: Input data array

        Returns:
            PPFResult with detected and validated forms
        """
        partial_forms = self.detect_partial_forms(data)

        validated_forms = []
        for pf in partial_forms:
            is_valid, corr = self.validate_by_extrapolation(data, pf)
            if is_valid:
                pf.confidence *= (0.5 + 0.5 * corr)
                validated_forms.append(pf)

        if len(data) == 0:
            structure_score = 0.0
            noise_level = 0.0
        else:
            covered = sum(pf.end_idx - pf.start_idx for pf in validated_forms)
            structure_score = covered / len(data)

            all_residuals = []
            for pf in validated_forms:
                all_residuals.extend(pf.fit.residuals)
            noise_level = np.std(all_residuals) if all_residuals else np.std(data)

        return PPFResult(
            partial_forms=partial_forms,
            validated_forms=validated_forms,
            structure_score=structure_score,
            noise_level=noise_level
        )
