# Promising Partial Form (PPF)

## A Law-Aware Architecture for Time-Series Intelligence

**Author**: Edward Chalk
**Contact**: edward@fleetingswallow.com
**Version**: 1.0 (January 2026)

---

## Abstract

Promising Partial Form (PPF) is a hybrid symbolic-statistical system for discovering, validating, and deploying governing mathematical forms from time-series data. Unlike black-box machine learning, PPF produces explicit, interpretable equations that describe oscillation, decay, saturation, feedback, and transients in physical, biological, and engineered systems. PPF decomposes signals into layers of Promising Partial Forms, iteratively removing discovered structure and analyzing residuals until only noise remains. This enables compact, explainable, and edge-deployable models suitable for any 1D time-series or signal analysis task - from real-time monitoring to scientific discovery to feature extraction for downstream AI systems.

---

## 1. Introduction

### 1.1 The Interpretability Gap

Modern AI systems excel at prediction but struggle with three fundamental challenges:

1. **Interpretability**: Neural networks are black boxes - we cannot explain *why* they make predictions
2. **Generalization**: Models fail outside their training distribution
3. **Deployment**: Edge devices cannot run large neural networks

Classical scientific modeling addresses these issues through explicit mathematical laws, but:
- Requires domain expertise to hypothesize forms
- Does not scale to unknown or complex systems
- Cannot handle noise or incomplete data

### 1.2 The PPF Solution

PPF occupies the middle ground: it **automatically discovers mathematical laws** from raw time-series data while preserving:

- **Physical meaning**: Equations map to real phenomena (decay rates, frequencies, saturation limits)
- **Deployability**: Models fit in 50-100 bytes
- **Reliability**: Parsimony pressure and residual analysis prevent overfitting

---

## 2. Core Philosophy: Promising Partial Forms

### 2.1 Definition

A **Promising Partial Form** is a mathematical expression that:
1. Explains a **meaningful portion** of a signal
2. Has **high explanatory power** (R² above threshold)
3. Has **low complexity** (parsimony)
4. Maps to a **recognizable mathematical family**

### 2.2 Layered Decomposition

Complex signals are modeled as sums of partial forms:

```
y(t) ≈ f₁(t) + f₂(t) + ... + fₖ(t) + ε(t)
```

Where:
- `fᵢ(t)` = Discovered partial form (oscillation, decay, trend, etc.)
- `ε(t)` = Noise-like residual
- `K` = Number of layers (determined automatically)

### 2.3 The Discovery Loop

PPF operates through iterative refinement:

```
┌─────────────────────────────────────────────────────────┐
│  Input Signal y(t)                                      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  1. DISCOVER: Find best form f(t) via symbolic search  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  2. VALIDATE: Check R² > threshold, form is meaningful │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  3. SUBTRACT: Compute residual r(t) = y(t) - f(t)      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  4. TEST: Is residual noise-like? (entropy, structure) │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        [YES: Done]              [NO: y(t) ← r(t), GOTO 1]
```

### 2.4 Why "Promising"?

Not every fitted function is a true law. A form is "promising" when:

1. **Statistical significance**: R² exceeds noise floor
2. **Structural match**: Form belongs to known physical family
3. **Residual independence**: Subtracting the form leaves no systematic pattern
4. **Parsimony**: Simpler forms preferred over complex equivalents

---

## 3. Information-Theoretic Foundation

### 3.1 PPF as Scientific Compression

Richard Feynman described science as discovering the rules by which nature behaves. In information-theoretic terms, this means finding the shortest description that explains data. PPF implements this principle computationally: it searches for compact mathematical forms that best compress time-series data into explicit equations.

This positions PPF as an algorithmic implementation of scientific discovery:
1. **Decompose** phenomena into components
2. **Propose** candidate mathematical laws
3. **Evaluate** explanatory power (R²)
4. **Prefer** simpler explanations (parsimony)
5. **Analyze** residuals for missing structure

This mirrors the epistemology of physics itself.

### 3.2 Equations vs. Weights

Neural networks perform statistical interpolation - they learn mappings from inputs to outputs but do not produce explicit representations of governing processes. They compress data into weight space, but that compression is **opaque**.

PPF compresses data into **equations**. The compression is explicit, interpretable, and physically meaningful.

| Aspect | Neural Network | PPF |
|--------|----------------|-----|
| Compression target | Weight matrices | Mathematical equations |
| Interpretability | Opaque | Transparent |
| Representation | Statistical | Symbolic |
| Knowledge form | Implicit patterns | Explicit laws |

### 3.3 The Layered Information Architecture

PPF enables a powerful information-processing stack:

```
Raw Signal → PPF (discover laws) → ML (extract meaning) → Decisions
     │              │                      │                  │
  High entropy   Low entropy         Semantic level      Action level
  (measurements) (equations)         (classification)    (response)
```

PPF converts high-entropy raw data into low-entropy, structured representations (equations with parameters). Downstream ML systems can then reason over these compact, meaningful variables rather than raw measurements.

### 3.4 Benefits of This Architecture

This layered approach enables AI systems that are:

- **More data-efficient**: Laws generalize from fewer examples than statistical patterns
- **More interpretable**: Decisions trace back to explicit mathematical relationships
- **More robust**: Physical laws transfer across conditions better than fitted curves
- **Edge-deployable**: Equations require bytes, not megabytes

It restores the separation between **physics** (what patterns exist) and **semantics** (what they mean).

---

## 4. Architecture

PPF consists of four integrated layers:

### 4.1 Signal Decomposition

Before symbolic search, PPF optionally decomposes signals using:

| Method | Purpose | Best For |
|--------|---------|----------|
| **EMD** (Empirical Mode Decomposition) | Separate oscillatory modes | Non-stationary signals |
| **SSA** (Singular Spectrum Analysis) | Extract trends and periodicities | Noisy periodic data |
| **Direct** | Skip decomposition | Clean signals |

Decomposition helps isolate components that map cleanly to single forms.

### 4.2 Symbolic Regression Engine

The core discovery mechanism uses genetic programming (GP):

```python
Population of expression trees
     │
     ▼
┌────────────────┐
│ EVALUATE       │ ← Fitness = R² - λ × complexity
│ FITNESS        │
└────────────────┘
     │
     ▼
┌────────────────┐
│ SELECT         │ ← Tournament selection
│ PARENTS        │
└────────────────┘
     │
     ▼
┌────────────────┐
│ CROSSOVER      │ ← Subtree exchange
│ & MUTATE       │
└────────────────┘
     │
     ▼
┌────────────────┐
│ OPTIMIZE       │ ← L-BFGS-B on constants
│ CONSTANTS      │
└────────────────┘
     │
     └──→ Repeat for N generations
```

Key features:
- **Parsimony pressure**: λ coefficient penalizes complexity
- **Pareto front**: Track accuracy vs. complexity tradeoffs
- **Constant optimization**: Fine-tune parameters after structure evolves

### 4.3 Domain Macros

Pure GP struggles with multiplicative compositions across function families (e.g., `exp(-kt) × sin(ωt)`). PPF introduces **macros** - pre-composed templates:

| Macro | Formula | Physical Meaning |
|-------|---------|------------------|
| `DAMPED_SIN` | `a·exp(-k·t)·sin(ω·t + φ)` | Decaying oscillation |
| `DAMPED_COS` | `a·exp(-k·t)·cos(ω·t + φ)` | Decaying oscillation |
| `RC_CHARGE` | `a·(1 - exp(-k·t)) + c` | Capacitor charging |
| `EXP_DECAY` | `a·exp(-k·t) + c` | Exponential decay |
| `SIGMOID` | `a / (1 + exp(-k·(x-x₀)))` | Saturation curve |
| `LOGISTIC` | `a / (1 + b·exp(-k·x))` | Population growth |
| `HILL` | `a·xⁿ / (kⁿ + xⁿ)` | Enzyme kinetics |
| `RATIO` | `(a·x + b) / (c·x + d)` | Rational function |
| `POWER_LAW` | `a·x^b + c` | Scaling law |
| `GAUSSIAN` | `a·exp(-((x-μ)/σ)²)` | Peak/distribution |
| `TANH_STEP` | `a·tanh(k·(x-x₀)) + c` | Smooth step |

Macros enable:
- Faster convergence to complex forms
- Domain-appropriate search spaces
- Interpretable parameter extraction

### 4.4 Residual Analysis

After discovering a form, PPF analyzes residuals to determine if structure remains:

| Method | Measures | Threshold |
|--------|----------|-----------|
| **Spectral entropy** | Frequency concentration | > 0.9 = noise |
| **Autocorrelation** | Temporal structure | < 0.1 = noise |
| **Compressibility** | gzip ratio | > 0.95 = noise |

If residuals show structure, PPF recurses to find additional forms.

---

## 5. Discovery Modes

PPF supports multiple search strategies:

| Mode | Primitives | Use Case |
|------|------------|----------|
| `AUTO` | Probes all domains | Unknown signal type |
| `OSCILLATOR` | sin, cos, damped forms | Vibrations, waves |
| `CIRCUIT` | exp, RC charge | Electronics |
| `GROWTH` | sigmoid, logistic, Hill | Biology, adoption |
| `RATIONAL` | ratio, rational2 | Feedback systems |
| `POLYNOMIAL` | add, mul, powers | Algebraic relationships |
| `UNIVERSAL` | power law, Gaussian, tanh | General patterns |
| `DISCOVER` | All primitives, no macros | Novel form discovery |
| `IDENTIFY` | All macros | Known form families |

### AUTO Mode

The recommended default. AUTO mode:
1. Probes each domain with quick GP runs
2. Ranks domains by best R² achieved
3. Runs full search in top domain
4. Analyzes residuals for additional structure

---

## 6. Reliability Mechanisms

PPF includes multiple safeguards against overfitting and spurious discovery:

### 6.1 Parsimony Pressure

Fitness includes complexity penalty:
```
score = R² - λ × complexity
```

This favors simpler explanations over complex curve-fitting.

### 6.2 Pareto Front

PPF maintains the full accuracy-complexity tradeoff, allowing users to choose:
- **Most accurate**: Best R², potentially complex
- **Most parsimonious**: Simplest form above threshold
- **Best tradeoff**: Knee of Pareto curve

### 6.3 Residual Testing

A discovered form is only accepted if:
1. Residuals are statistically noise-like
2. No systematic patterns remain
3. Entropy exceeds threshold

### 6.4 Domain Validation

Forms are validated against expected domain characteristics:
- Oscillator forms should have reasonable frequencies
- Decay rates should be positive
- Saturation limits should be finite

---

## 7. Export and Deployment

### 7.1 Export Targets

PPF exports discovered forms to production-ready code:

| Target | Use Case | Size |
|--------|----------|------|
| **Python** | Scripting, prototyping | ~300 bytes |
| **C99** | Embedded, microcontrollers | ~200 bytes |
| **JSON** | Storage, transmission, audit | ~1 KB |

### 7.2 Safety Wrappers

Exported code includes protection against numerical hazards:
- `safe_div`: Division by zero → epsilon
- `safe_log`: log(0) → log(epsilon)
- `clamp_exp`: Overflow prevention

### 7.3 Edge Deployment

A typical PPF model:
- **Parameters**: 4-8 floats (16-32 bytes)
- **Code**: ~50 bytes (formula evaluation)
- **Inference**: ~10 FLOPs

Compare to neural networks:
- **Parameters**: 1000s of weights (4-100 KB)
- **Inference**: 1000s of MACs

---

## 8. Feature Extraction for Downstream AI

PPF serves as a **physics-informed feature extractor** for machine learning:

### 8.1 Interpretable Features

From each discovered form, PPF extracts:

| Feature | Meaning |
|---------|---------|
| `dominant_family` | Form type (oscillation, growth, etc.) |
| `r2` | Explanatory power |
| `complexity` | Expression size |
| `amplitude` | Signal magnitude |
| `frequency` / `omega` | Oscillation rate |
| `damping_k` | Decay rate |
| `phase` | Phase offset |
| `K` / `L` | Saturation limit |
| `mu` / `sigma` | Peak location/width |

### 8.2 ML Integration

```python
# Extract PPF features from each sample
features = [extract_features(ppf_result) for result in results]

# Convert to feature matrix
X, names = feature_matrix(features)

# Train downstream classifier
classifier.fit(X, labels)
```

Benefits:
- Features have **physical meaning** (not arbitrary neural activations)
- Feature importance reveals **which physics matters**
- Models generalize better to new conditions

---

## 9. Applications

### 9.1 Demonstrated Applications

| Domain | Application | PPF Contribution |
|--------|-------------|------------------|
| **IoT** | Temperature prediction | 4-parameter daily cycle model |
| **Biomedical** | ECG analysis | T-wave morphology features |
| **Industrial** | Vibration monitoring | Fault signature classification |
| **Physics** | Law discovery | Kepler's law from orbital data |

### 9.2 Potential Applications

- **Finance**: Trend/cycle decomposition
- **Control systems**: Transfer function identification
- **Chemistry**: Reaction kinetics modeling
- **Climate**: Seasonal pattern extraction
- **Audio**: Envelope and modulation analysis

---

## 10. Comparison with Alternatives

| Aspect | PPF | Neural Networks | Traditional Curve Fitting |
|--------|-----|-----------------|---------------------------|
| **Interpretability** | Full | None | Full |
| **Automation** | Full | Full | Manual |
| **Model size** | ~50 bytes | ~50 KB | ~50 bytes |
| **Noise handling** | Built-in | Built-in | Limited |
| **Novel discovery** | Yes | No | No |
| **Domain guidance** | Optional | None | Required |

PPF uniquely combines:
- Automatic discovery (like neural networks)
- Interpretable output (like classical modeling)
- Compact deployment (like hand-crafted formulas)

---

## 11. Extensibility

### 11.1 New Macro Families

Add domain-specific templates:
```python
class MacroOp(Enum):
    MY_MACRO = "my_macro"  # New pattern

def _eval_my_macro(x, params):
    a, b, c = params
    return a * my_function(b * x + c)
```

### 11.2 Multi-Variable Models

Export layer supports N-variable expressions if discovered.

### 11.3 Differential Equations

Future work: discover ODEs from time-series data.

### 11.4 Neural-Symbolic Integration

PPF forms can serve as:
- Feature extractors for neural networks
- Interpretable components in hybrid models
- Initialization for physics-informed neural networks (PINNs)

---

## 12. Conclusion

PPF provides a general-purpose framework for **1D time-series and signal analysis** that automatically discovers the mathematical structure underlying data. By combining symbolic regression with domain macros, residual analysis, and edge-ready export, PPF bridges the gap between black-box prediction and interpretable science.

Key contributions:
1. **Promising Partial Form** concept for layered decomposition
2. **Domain macros** for efficient discovery of complex forms
3. **Export layer** for edge deployment
4. **Feature extraction** for downstream ML integration

PPF applies to any domain where data follows mathematical patterns: real-time sensor monitoring, scientific discovery, signal characterization, and as interpretable feature extractors for machine learning. When data has underlying mathematical structure, PPF attempts to discover it. When it doesn't succeed (or the data lacks structure), PPF's residual analysis reveals this, providing valuable diagnostic information.

---

## References

1. Koza, J. R. (1992). *Genetic Programming*. MIT Press.
2. Schmidt, M., & Lipson, H. (2009). Distilling free-form natural laws from experimental data. *Science*, 324(5923), 81-85.
3. Cranmer, M. (2023). PySR: High-performance symbolic regression in Python and Julia.
4. Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A*.
5. Udrescu, S.-M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, 6(16), eaay2631.

---

## Appendix A: Mathematical Formulations

### A.1 Fitness Function

```
fitness(expr) = R²(expr, data) - λ × complexity(expr)
```

Where:
- `R² = 1 - SS_res / SS_tot`
- `complexity = number of nodes in expression tree`
- `λ = parsimony coefficient (default: 0.001)`

### A.2 Pareto Dominance

Expression A dominates B if:
- `R²(A) ≥ R²(B)` AND `complexity(A) ≤ complexity(B)`
- At least one inequality is strict

### A.3 Residual Entropy

Spectral entropy:
```
H = -Σ pᵢ log(pᵢ)
```
Where `pᵢ` is normalized power at frequency `i`.

High entropy (> 0.9) indicates noise-like residual.

---

## Appendix B: Macro Parameter Reference

| Macro | Parameters | Formula |
|-------|------------|---------|
| `DAMPED_SIN` | a, k, ω, φ | `a·exp(-k·t)·sin(ω·t + φ)` |
| `DAMPED_COS` | a, k, ω, φ | `a·exp(-k·t)·cos(ω·t + φ)` |
| `RC_CHARGE` | a, k, c | `a·(1 - exp(-k·t)) + c` |
| `EXP_DECAY` | a, k, c | `a·exp(-k·t) + c` |
| `SIGMOID` | a, k, x₀ | `a / (1 + exp(-k·(x - x₀)))` |
| `LOGISTIC` | a, b, k | `a / (1 + b·exp(-k·x))` |
| `HILL` | a, K, n | `a·xⁿ / (Kⁿ + xⁿ)` |
| `RATIO` | a, b, c, d | `(a·x + b) / (c·x + d)` |
| `RATIONAL2` | a, b, c, d, e, f | `(a·x² + b·x + c) / (d·x² + e·x + f)` |
| `POWER_LAW` | a, b, c | `a·x^b + c` |
| `GAUSSIAN` | a, μ, σ | `a·exp(-((x-μ)/σ)²)` |
| `TANH_STEP` | a, k, x₀, c | `a·tanh(k·(x - x₀)) + c` |
