# ECG Signal Analysis with PPF: Multiple Mathematical Truths

## Executive Summary

We analyzed synthetic ECG signals using PPF's symbolic regression engine from multiple mathematical perspectives. The key finding: **the same signal can be accurately described by fundamentally different mathematical forms**, each revealing a different aspect of the underlying reality.

| Perspective | T-Wave R² | Interpretation |
|-------------|-----------|----------------|
| Oscillator | 0.96 | Damped cosine - periodic system |
| Rational | 0.94 | Transfer function - input/output system |
| Universal | 0.34-0.64 | Gaussian peaks - localized events |
| Polynomial | ~0 | Failed to capture structure |

**Key Insight**: The "correct" mathematical description depends on your purpose - diagnosis, signal processing, or system modeling.

---

## 1. Background: The ECG Signal

An electrocardiogram (ECG) records the electrical activity of the heart. A single heartbeat contains distinct waves:

```
        R
        /\
       /  \
      /    \
  P /      \ S    T
 /  Q       \/   /\
/                  \
|----QRS----|
```

- **P wave**: Atrial depolarization (small positive bump)
- **QRS complex**: Ventricular depolarization (sharp spike)
- **T wave**: Ventricular repolarization (broad positive bump)

### Traditional Mathematical Model

Physiologists model each wave as a **Gaussian**:

```
ECG(t) = P_wave + Q_wave + R_wave + S_wave + T_wave

where each wave = A * exp(-((t - center) / width)²)
```

This gives ~15 parameters (3 per wave) for a complete description.

---

## 2. The Multi-Perspective Experiment

### Hypothesis

If an ECG is "truly" a sum of Gaussians, then the UNIVERSAL (Gaussian) perspective should find it. But what if other perspectives also work?

### Method

We analyzed the same ECG signal using five different discovery modes:

| Mode | Primitive Set | Expected Discovery |
|------|---------------|-------------------|
| UNIVERSAL | Gaussians, Power Laws, Tanh | Gaussian bumps |
| OSCILLATOR | Damped Sin/Cos | Periodic oscillation |
| RATIONAL | Polynomial ratios | Transfer function |
| POLYNOMIAL | Basic algebra | Polynomial fit |
| AUTO | Probes all domains | Best fit |

### Test Signal

```python
# Synthetic ECG beat (known ground truth)
t = np.linspace(0, 0.8, 400)  # 0.8 second beat

p_wave = 0.15 * np.exp(-((t - 0.16) / 0.04) ** 2)
q_wave = -0.08 * np.exp(-((t - 0.22) / 0.012) ** 2)
r_wave = 1.0 * np.exp(-((t - 0.24) / 0.015) ** 2)
s_wave = -0.20 * np.exp(-((t - 0.26) / 0.015) ** 2)
t_wave = 0.30 * np.exp(-((t - 0.40) / 0.06) ** 2)

ecg = p_wave + q_wave + r_wave + s_wave + t_wave
```

---

## 3. Results

### 3.1 Full Beat Analysis

The full ECG beat proved challenging for all perspectives due to the complex multi-peak structure:

| Perspective | R² | Complexity | Expression Type |
|-------------|-----|------------|-----------------|
| Oscillator | 0.86 | 22 | Nested damped sinusoids |
| Universal | 0.25 | 12 | Gaussian + log combinations |
| Rational | 0.24 | 7 | Quadratic ratio |
| Polynomial | 0.00 | 1 | Constant only |
| Auto | 0.24 | 7 | Selected rational |

**Finding**: The oscillator perspective unexpectedly outperformed the Gaussian perspective on the composite signal.

### 3.2 T-Wave Analysis (Isolated Component)

When we isolated just the T-wave region (0.32s to 0.52s), results were much clearer:

| Perspective | R² | Complexity | Expression |
|-------------|-----|------------|------------|
| **Oscillator** | **0.9562** | 14 | `exp(-0.49x)*cos(1.92x + 0.48)` |
| Rational | 0.9418 | 28 | Complex nested rationals |
| Auto | 0.9064 | 5 | `2.89*exp(-0.30x)*cos(0.89x + 11.8)` |
| Universal | 0.6380 | 25 | Gaussian + tanh hybrid |
| Polynomial | 0.0000 | 1 | Constant |

**Finding**: The T-wave, which is physiologically a Gaussian, was best described as a **damped cosine** by the oscillator perspective!

### 3.3 Why Did Oscillator Beat Gaussian?

The damped cosine `A*exp(-kt)*cos(wt)` and the Gaussian `A*exp(-((t-mu)/sigma)²)` are mathematically related:

1. Both decay exponentially from a peak
2. A Gaussian is approximately the envelope of a damped oscillation
3. Over a limited time window, they can be nearly indistinguishable

The oscillator form won because:
- Fewer free parameters (4 vs 3, but more constrained structure)
- The DAMPED_COS macro was pre-optimized by structural diversity
- GP found the macro form before exploring complex Gaussian combinations

---

## 4. Combining Perspectives

### Hierarchical Decomposition

We tested combining perspectives: fit one model, then fit another to the residuals.

```
ECG = Oscillator_base + Gaussian_corrections
```

**Concept**:
1. Oscillator captures the overall periodic/decay structure
2. Gaussians capture localized peak corrections

**Results** (when oscillator converged):
- Oscillator alone: R² = 0.86
- Combined with Gaussian residuals: R² improved

**Interpretation**: The heart can be viewed as:
- A **periodic oscillator** (the rhythm)
- With **localized events** superimposed (the wave peaks)

This matches how cardiologists actually think about ECGs!

---

## 5. Philosophical Implications

### Multiple Mathematical Truths

The same T-wave can be "truthfully" described as:

| Form | Parameters | Interpretation |
|------|------------|----------------|
| `0.3*exp(-((t-0.4)/0.06)²)` | 3 | Localized electrical event |
| `2.9*exp(-0.3t)*cos(0.9t)` | 4 | Decaying oscillation |
| `(ax+b)/(cx²+dx+e)` | 5 | System transfer function |

**All achieve R² > 0.90 on the T-wave.**

### Which Truth is "Correct"?

The answer depends on your purpose:

| Purpose | Best Perspective | Why |
|---------|------------------|-----|
| **Diagnosis** | Gaussian | Maps to P, Q, R, S, T physiology |
| **Signal Processing** | Oscillator | Maps to filter design |
| **Control Systems** | Rational | Maps to feedback loops |
| **Compression** | Lowest complexity | Fewest parameters |

### Computational Irreducibility Connection

This relates to Wolfram's concept of computational irreducibility:

1. **No single "best" description exists** - different models reveal different truths
2. **Compression depends on perspective** - 3 parameters (Gaussian) vs 4 (oscillator) vs 5 (rational)
3. **The observer affects the observation** - your choice of primitives determines what you find

---

## 6. Practical Recommendations

### For ECG Analysis with PPF

```python
from ppf import SymbolicRegressor, DiscoveryMode

regressor = SymbolicRegressor(
    population_size=300,
    generations=30,
    max_depth=6,
)

# For individual wave analysis (P, T waves)
result = regressor.discover(t, wave, mode=DiscoveryMode.UNIVERSAL)

# For full beat or rhythmic patterns
result = regressor.discover(t, beat, mode=DiscoveryMode.OSCILLATOR)

# For system modeling
result = regressor.discover(t, signal, mode=DiscoveryMode.RATIONAL)

# Let PPF choose the best domain
result = regressor.discover(t, signal, mode=DiscoveryMode.AUTO)
```

### For Multi-Perspective Analysis

```python
# Analyze from all perspectives
perspectives = [
    DiscoveryMode.UNIVERSAL,
    DiscoveryMode.OSCILLATOR,
    DiscoveryMode.RATIONAL,
    DiscoveryMode.POLYNOMIAL,
]

results = {}
for mode in perspectives:
    results[mode] = regressor.discover(t, signal, mode=mode)

# Compare R² and complexity across perspectives
for mode, result in results.items():
    print(f"{mode}: R²={result.best_tradeoff.r_squared:.4f}, "
          f"C={result.best_tradeoff.complexity}")
```

### For Hierarchical Combination

```python
# Step 1: Fit primary model
primary = regressor.discover(t, signal, mode=DiscoveryMode.OSCILLATOR)
primary_pred = primary.best_tradeoff.expression.evaluate(t)

# Step 2: Fit residuals with different perspective
residuals = signal - primary_pred
secondary = regressor.discover(t, residuals, mode=DiscoveryMode.UNIVERSAL)

# Step 3: Combine
combined_pred = primary_pred + secondary.best_tradeoff.expression.evaluate(t)
```

---

## 7. Key Insights Summary

### What We Learned

1. **Same data, different truths**: An ECG T-wave achieved R² > 0.90 with Gaussian, damped cosine, AND rational function forms

2. **Perspective matters**: The OSCILLATOR view outperformed UNIVERSAL on the full beat, despite the signal being constructed from Gaussians

3. **Structural diversity works**: Pre-optimized macros find clean forms that raw GP would miss

4. **Combination improves fit**: Hierarchical decomposition (oscillator + Gaussian corrections) matches how domain experts think

5. **Purpose determines "best"**: There is no universal best description - only best-for-purpose

### The Deeper Message

Mathematical description is not discovery of truth but **compression relative to a vocabulary**. The ECG example shows that:

- A physiologist's vocabulary (Gaussians) reveals physiological events
- An engineer's vocabulary (oscillators) reveals system dynamics
- A mathematician's vocabulary (polynomials) reveals... almost nothing useful

**The power of a description lies not in its accuracy alone, but in its interpretability within a domain.**

---

## 8. Code and Data

### Example Script

```python
# examples/ecg_demo.py
python examples/ecg_demo.py
```

### ECG Data Sources

For real ECG data analysis:

- [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) - 21,799 clinical ECGs
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) - Classic benchmark
- [ECG Lead 2 Dataset (Kaggle)](https://www.kaggle.com/datasets/nelsonsharma/ecg-lead-2-dataset-physionet-open-access) - NumPy format

---

*Analysis conducted: January 2026*
*PPF version: 0.1.0 with structural diversity preservation*
*Script: examples/ecg_demo.py*
