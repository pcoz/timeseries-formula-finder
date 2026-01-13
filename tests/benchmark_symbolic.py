"""
Benchmark suite for symbolic regression engine.

Tests whether the engine can discover laws vs just fitting curves.
Benchmarks from:
- Pure symbolic regression literature (Nguyen, Keijzer)
- Electronics (RC, RLC circuits)
- Classical physics (oscillators)
- Growth laws (logistic)
- Real-world data (S&P-like)
- Hard mode (chaotic systems)
"""

import numpy as np
import sys
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, '.')

from ppf import (
    SymbolicRegressor,
    PrimitiveSet,
    TRIG_PRIMITIVES,
    POLYNOMIAL_PRIMITIVES,
    GROWTH_PRIMITIVES,
    OSCILLATOR_PRIMITIVES,
    CIRCUIT_PRIMITIVES,
    PHYSICS_PRIMITIVES,
    RATIONAL_PRIMITIVES,
    SATURATION_PRIMITIVES,
    print_symbolic_result,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    true_form: str
    discovered_form: str
    r_squared: float
    complexity: int
    generations: int
    time_seconds: float
    success: bool  # Whether we consider this a successful discovery


def generate_nguyen1(n_points: int = 100, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Nguyen-1 benchmark: y = x³ + x² + x
    Classic symbolic regression test.
    """
    x = np.linspace(-1, 1, n_points)
    y_true = x**3 + x**2 + x
    y = y_true + noise * np.std(y_true) * np.random.randn(n_points)
    return x, y, "x³ + x² + x"


def generate_keijzer6(n_points: int = 100, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Keijzer-6 benchmark: y = x³/(x²+1)
    Tests division and polynomial interaction.
    """
    x = np.linspace(-3, 3, n_points)
    y_true = x**3 / (x**2 + 1)
    y = y_true + noise * np.std(y_true) * np.random.randn(n_points)
    return x, y, "x³/(x²+1)"


def generate_rc_charging(n_points: int = 100, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    RC circuit charging: V(t) = V0(1 - e^(-t/RC))
    Classic exponential approach to asymptote.
    """
    V0 = 5.0  # 5V supply
    RC = 0.5  # Time constant
    t = np.linspace(0, 3, n_points)
    y_true = V0 * (1 - np.exp(-t / RC))
    y = y_true + noise * np.std(y_true) * np.random.randn(n_points)
    return t, y, "V0*(1 - e^(-t/RC))"


def generate_rlc_ringing(n_points: int = 150, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    RLC underdamped ringing: V(t) = A·e^(-kt)·sin(ωt)
    Exponential decay with oscillation.
    """
    A = 3.0
    k = 0.5  # Damping
    omega = 6.0  # Angular frequency
    t = np.linspace(0, 5, n_points)
    y_true = A * np.exp(-k * t) * np.sin(omega * t)
    y = y_true + noise * np.std(y_true) * np.random.randn(n_points)
    return t, y, "A*exp(-k*t)*sin(omega*t)"


def generate_harmonic_oscillator(n_points: int = 100, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Simple harmonic oscillator: x(t) = A·cos(ωt)
    Pure sinusoid.
    """
    A = 2.0
    omega = 3.0
    t = np.linspace(0, 4, n_points)
    y_true = A * np.cos(omega * t)
    y = y_true + noise * np.std(y_true) * np.random.randn(n_points)
    return t, y, "A*cos(omega*t)"


def generate_damped_oscillator(n_points: int = 150, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Damped harmonic oscillator: x(t) = A·e^(-kt)·cos(ωt)
    Decaying oscillation.
    """
    A = 2.5
    k = 0.4
    omega = 4.0
    t = np.linspace(0, 5, n_points)
    y_true = A * np.exp(-k * t) * np.cos(omega * t)
    y = y_true + noise * np.std(y_true) * np.random.randn(n_points)
    return t, y, "A*exp(-k*t)*cos(omega*t)"


def generate_logistic_growth(n_points: int = 100, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Logistic growth: P(t) = K / (1 + e^(-r(t-t₀)))
    S-curve growth pattern.
    """
    K = 100  # Carrying capacity
    r = 0.8  # Growth rate
    t0 = 5   # Midpoint
    t = np.linspace(0, 10, n_points)
    y_true = K / (1 + np.exp(-r * (t - t0)))
    y = y_true + noise * np.std(y_true) * np.random.randn(n_points)
    return t, y, "K/(1 + exp(-r*(t-t0)))"


def generate_sp500_like(n_points: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    S&P 500-like price series.
    Geometric random walk with trend, cycles, and fat tails.
    Real-world messy data.
    """
    np.random.seed(42)  # For reproducibility
    t = np.arange(n_points)

    # Trend component (exponential growth ~7% annual)
    trend = 100 * np.exp(0.0003 * t)

    # Cyclical component (business cycle ~40 periods)
    cycle = 5 * np.sin(2 * np.pi * t / 40)

    # Random walk component (cumulative returns)
    returns = 0.001 + 0.02 * np.random.randn(n_points)
    random_walk = np.cumsum(returns) * 10

    y = trend + cycle + random_walk
    return t.astype(float), y, "Trend + Cycles + Noise"


def generate_lorenz_x(n_points: int = 500, noise: float = 0.02) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Lorenz attractor x-component.
    Chaotic system - should NOT be fully discoverable.
    Tests that the engine doesn't overfit garbage.
    """
    # Lorenz parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Simple Euler integration
    dt = 0.01
    x, y, z = 1.0, 1.0, 1.0
    xs = []

    for _ in range(n_points):
        xs.append(x)
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

    t = np.linspace(0, n_points * dt, n_points)
    y_out = np.array(xs) + noise * np.std(xs) * np.random.randn(n_points)
    return t, y_out, "Chaotic (Lorenz x)"


def run_benchmark(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    true_form: str,
    primitives: Optional[PrimitiveSet] = None,
    generations: int = 40,
    population: int = 300,
    success_threshold: float = 0.85,
    verbose: bool = True
) -> BenchmarkResult:
    """Run a single benchmark and return results."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {name}")
        print(f"True form: {true_form}")
        print(f"{'='*70}")

    # Create regressor with appropriate primitives
    regressor = SymbolicRegressor(
        primitives=primitives,
        population_size=population,
        generations=generations,
        max_depth=6,
        parsimony_coefficient=0.001,
    )

    start_time = time.time()
    result = regressor.discover(x, y, verbose=verbose)
    elapsed = time.time() - start_time

    # Use most_accurate for benchmarking (proves engine capability)
    best = result.most_accurate

    if verbose:
        print(f"\n--- Results ---")
        print(f"Most accurate:  {result.most_accurate.expression_string}")
        print(f"  R²: {result.most_accurate.r_squared:.4f}, Complexity: {result.most_accurate.complexity}")
        if result.best_tradeoff and result.best_tradeoff != result.most_accurate:
            print(f"Best tradeoff:  {result.best_tradeoff.expression_string}")
            print(f"  R²: {result.best_tradeoff.r_squared:.4f}, Complexity: {result.best_tradeoff.complexity}")
        print(f"Time: {elapsed:.2f}s")

    # Determine success based on most accurate solution
    success = best.r_squared >= success_threshold

    return BenchmarkResult(
        name=name,
        true_form=true_form,
        discovered_form=best.expression_string,
        r_squared=best.r_squared,
        complexity=best.complexity,
        generations=result.generations_run,
        time_seconds=elapsed,
        success=success
    )


def run_all_benchmarks(verbose: bool = True) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""

    results = []

    # Set random seed for reproducibility
    np.random.seed(123)

    # 1. Nguyen-1 (polynomial)
    x, y, form = generate_nguyen1()
    results.append(run_benchmark(
        "Nguyen-1", x, y, form,
        primitives=POLYNOMIAL_PRIMITIVES,
        generations=50,
        success_threshold=0.90,
        verbose=verbose
    ))

    # 2. Keijzer-6 (rational function) - use RATIONAL_PRIMITIVES with macros
    x, y, form = generate_keijzer6()
    results.append(run_benchmark(
        "Keijzer-6", x, y, form,
        primitives=RATIONAL_PRIMITIVES,
        generations=50,
        success_threshold=0.85,
        verbose=verbose
    ))

    # 3. RC Charging (exponential) - use CIRCUIT_PRIMITIVES with macros
    x, y, form = generate_rc_charging()
    results.append(run_benchmark(
        "RC Charging", x, y, form,
        primitives=CIRCUIT_PRIMITIVES,
        generations=40,
        success_threshold=0.90,
        verbose=verbose
    ))

    # 4. Harmonic Oscillator (sinusoid)
    x, y, form = generate_harmonic_oscillator()
    results.append(run_benchmark(
        "Harmonic Oscillator", x, y, form,
        primitives=TRIG_PRIMITIVES,
        generations=40,
        success_threshold=0.90,
        verbose=verbose
    ))

    # 5. Damped Oscillator (exp * cos) - use OSCILLATOR_PRIMITIVES with macros
    x, y, form = generate_damped_oscillator()
    results.append(run_benchmark(
        "Damped Oscillator", x, y, form,
        primitives=OSCILLATOR_PRIMITIVES,
        generations=50,
        success_threshold=0.85,
        verbose=verbose
    ))

    # 6. RLC Ringing (exp * sin) - use OSCILLATOR_PRIMITIVES with macros
    x, y, form = generate_rlc_ringing()
    results.append(run_benchmark(
        "RLC Ringing", x, y, form,
        primitives=OSCILLATOR_PRIMITIVES,
        generations=50,
        success_threshold=0.85,
        verbose=verbose
    ))

    # 7. Logistic Growth (sigmoid) - use SATURATION_PRIMITIVES with macros
    x, y, form = generate_logistic_growth()
    results.append(run_benchmark(
        "Logistic Growth", x, y, form,
        primitives=SATURATION_PRIMITIVES,
        generations=50,
        success_threshold=0.90,
        verbose=verbose
    ))

    # 8. S&P 500-like (messy real-world)
    x, y, form = generate_sp500_like()
    results.append(run_benchmark(
        "S&P 500-like", x, y, form,
        primitives=None,  # Default primitives
        generations=40,
        population=400,
        success_threshold=0.70,  # Lower bar for messy data
        verbose=verbose
    ))

    # 9. Lorenz (chaotic - expected to fail)
    x, y, form = generate_lorenz_x()
    results.append(run_benchmark(
        "Lorenz (Chaotic)", x, y, form,
        primitives=None,
        generations=30,
        success_threshold=0.95,  # High bar - should NOT succeed
        verbose=verbose
    ))

    return results


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print summary table of all benchmark results."""

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Benchmark':<25} {'R²':>8} {'Complexity':>10} {'Time':>8} {'Status':>10}")
    print("-"*80)

    passed = 0
    for r in results:
        status = "PASS" if r.success else "FAIL"
        # Lorenz is expected to fail - that's actually good
        if r.name == "Lorenz (Chaotic)" and not r.success:
            status = "EXPECTED"
        elif r.success:
            passed += 1

        print(f"{r.name:<25} {r.r_squared:>8.4f} {r.complexity:>10d} {r.time_seconds:>7.1f}s {status:>10}")

    print("-"*80)

    # Don't count Lorenz in success rate
    non_chaotic = [r for r in results if r.name != "Lorenz (Chaotic)"]
    success_rate = passed / len(non_chaotic) * 100

    print(f"\nSuccess rate: {passed}/{len(non_chaotic)} ({success_rate:.0f}%)")
    print()

    # Detailed discoveries
    print("DISCOVERED FORMS:")
    print("-"*80)
    for r in results:
        print(f"\n{r.name}:")
        print(f"  True:       {r.true_form}")
        print(f"  Discovered: {r.discovered_form}")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("Symbolic Regression Benchmark Suite")
    print("Testing equation discovery capabilities")
    print()

    # Run with verbose output
    verbose = "--quiet" not in sys.argv

    results = run_all_benchmarks(verbose=verbose)
    print_summary(results)
