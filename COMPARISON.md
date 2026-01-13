# PPF vs. Other Symbolic Regression Approaches

This document compares PPF to other popular symbolic regression tools and explains when each approach is most appropriate.

## Overview of Alternatives

### 1. PySR (Python Symbolic Regression)

**What it is**: High-performance symbolic regression using Julia backend with Python interface. State-of-the-art in academic benchmarks.

**GitHub**: https://github.com/MilesCranmer/PySR

| Aspect | PySR | PPF |
|--------|------|-----|
| **Backend** | Julia (SymbolicRegression.jl) | Pure Python/NumPy |
| **Speed** | Very fast (multi-threaded Julia) | Moderate |
| **Search** | Multi-population GP | Single-population GP with macros |
| **Complexity control** | Parsimony pressure | Parsimony + Pareto front |
| **Export** | LaTeX, SymPy, NumPy | Python, C, JSON, feature vectors |
| **Dependencies** | Requires Julia runtime | NumPy/SciPy only |
| **Learning curve** | Moderate (Julia install) | Low |

**When to use PySR**:
- Maximum raw search performance needed
- Academic research requiring state-of-the-art
- Large datasets where speed matters
- Julia ecosystem integration

**When to use PPF instead**:
- Need C export for embedded/edge deployment
- Want to avoid Julia dependency
- Feature extraction for downstream ML is important
- Domain-specific macros match your problem

---

### 2. gplearn

**What it is**: Scikit-learn compatible genetic programming library for symbolic regression.

**GitHub**: https://github.com/trevorstephens/gplearn

| Aspect | gplearn | PPF |
|--------|---------|-----|
| **Integration** | sklearn API (fit/predict/transform) | Custom API |
| **Operators** | Basic (+, -, *, /, sin, cos, etc.) | Basic + domain macros |
| **Export** | None built-in | Python, C, JSON |
| **Parsimony** | Bloat control | Parsimony + Pareto |
| **Focus** | sklearn pipelines | Form discovery + deployment |

**When to use gplearn**:
- Already using sklearn pipelines
- Need transformer for feature engineering
- Simple expressions sufficient
- Familiar sklearn API preferred

**When to use PPF instead**:
- Need export to C/embedded
- Complex multiplicative forms (damped oscillations)
- Want interpretable feature extraction
- Need JSON serialization for deployment

---

### 3. Eureqa (DataRobot)

**What it is**: Commercial symbolic regression tool, now part of DataRobot. Pioneered modern symbolic regression.

| Aspect | Eureqa/DataRobot | PPF |
|--------|------------------|-----|
| **Cost** | Commercial license | Free (MIT) |
| **Interface** | GUI + API | Python API |
| **Performance** | Highly optimized | Good |
| **Support** | Enterprise support | Community |
| **Export** | Various formats | Python, C, JSON |

**When to use Eureqa/DataRobot**:
- Enterprise environment with budget
- Need commercial support and SLAs
- Non-technical users (GUI)
- Integration with DataRobot platform

**When to use PPF instead**:
- Open source requirement
- Need to modify/extend the code
- Cost-sensitive project
- Edge deployment focus

---

### 4. SymPy (rsolve/recurrence)

**What it is**: Python library for symbolic mathematics. Can solve recurrences and simplify expressions.

| Aspect | SymPy | PPF |
|--------|-------|-----|
| **Approach** | Algebraic solving | Statistical fitting |
| **Input** | Exact symbolic sequences | Noisy numerical data |
| **Output** | Exact closed forms | Approximate formulas |
| **Noise tolerance** | None | Built-in |

**When to use SymPy**:
- Exact mathematical sequences (Fibonacci, etc.)
- No noise in data
- Need symbolic manipulation
- Algebraic simplification needed

**When to use PPF instead**:
- Real sensor data with noise
- Approximate fits acceptable
- Don't know the exact form family
- Need export/deployment features

---

### 5. Neural Networks / Deep Learning

| Aspect | Neural Networks | PPF |
|--------|-----------------|-----|
| **Model size** | 10KB - 100MB | 50 bytes |
| **Interpretability** | Black box | Human-readable formula |
| **Inference cost** | 1000s of MACs | 10 FLOPs |
| **Flexibility** | Any function | Structured forms |
| **Data requirements** | Large datasets | Can work with small data |
| **Extrapolation** | Poor beyond training data | Good (correct form) |

**When to use Neural Networks**:
- Very complex patterns without structure
- Large training datasets available
- Interpretability not required
- Compute resources available at inference

**When to use PPF instead**:
- Underlying physics suggests simple form
- Need interpretability for domain experts
- Edge deployment with severe constraints
- Extrapolation beyond training data needed
- Small datasets

---

### 6. AI Feynman

**What it is**: Neural network-guided symbolic regression specifically designed for physics equations. Uses neural nets to detect simplifying properties (symmetries, separability, compositionality) then applies symbolic regression on simplified subproblems.

**Paper**: Udrescu & Tegmark, "AI Feynman: A physics-inspired method for symbolic regression" (Science, 2020)

**GitHub**: https://github.com/SJ001/AI-Feynman

| Aspect | AI Feynman | PPF |
|--------|------------|-----|
| **Approach** | NN-guided decomposition + brute-force search | Genetic programming with macros |
| **Strength** | Physics equations with symmetries | General time-series patterns |
| **Preprocessing** | Neural net identifies structure | Optional EMD/SSA decomposition |
| **Search method** | Brute-force + polynomial fitting | Evolutionary search |
| **Compositionality** | Explicit decomposition | Domain macros |
| **Dependencies** | PyTorch, sklearn, sympy | NumPy/SciPy only |
| **Export** | SymPy expressions | Python, C, JSON |

**When to use AI Feynman**:
- Known physics equations with symmetries
- Data from physical simulations
- Multi-variable relationships with separability
- Academic physics research
- Equations from Feynman's lectures

**When to use PPF instead**:
- Real-world sensor data with noise
- 1D time-series analysis
- Edge deployment requirements
- Domain-specific patterns (oscillations, decay, saturation)
- Need C export or feature extraction
- Lightweight dependencies preferred

**Key difference**: AI Feynman excels at rediscovering known physics equations by exploiting their mathematical structure (symmetries, dimensional analysis). PPF focuses on practical signal analysis with deployment-ready output. AI Feynman asks "what physics equation generated this?"; PPF asks "what mathematical form describes this signal?"

---

## Detailed Comparison Matrix

| Feature | PPF | PySR | gplearn | Eureqa | SymPy | AI Feynman |
|---------|-----|------|---------|--------|-------|------------|
| Open source | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Pure Python | ✓ | ✗ (Julia) | ✓ | ✗ | ✓ | ✗ (PyTorch) |
| Domain macros | ✓ | ✗ | ✗ | ✗ | N/A | ✗ |
| C export | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| JSON export | ✓ | ✗ | ✗ | ? | ✗ | ✗ |
| Feature extraction | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Noise tolerance | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Pareto front | ✓ | ✓ | ✗ | ✓ | N/A | ✗ |
| sklearn compatible | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Multi-threaded | ✗ | ✓ | ✗ | ✓ | N/A | ✓ |
| GUI | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Physics-aware | ✗ | ✗ | ✗ | ✗ | N/A | ✓ |

---

## When PPF is the Right Choice

### Best Use Cases for PPF

1. **Edge AI Deployment**
   - Need to deploy to microcontrollers (ESP32, STM32, Arduino)
   - Model size must be < 1KB
   - Cannot run TensorFlow Lite
   - C export is essential

2. **Interpretable Feature Engineering**
   - Want to extract parameters (frequency, damping, etc.) for downstream ML
   - Domain experts need to understand the model
   - Regulatory requirements for explainability

3. **Signals with Known Physics**
   - Vibration analysis (damped oscillations)
   - Biological rhythms (circadian, cardiac)
   - Growth/saturation phenomena
   - RC circuit behavior

4. **Small Data Situations**
   - Limited training examples
   - Domain knowledge can guide search
   - Neural networks would overfit

5. **Dependency-Light Environments**
   - Cannot install Julia
   - Minimal dependencies preferred
   - Air-gapped systems

### When to Choose Alternatives

1. **Choose PySR when**:
   - Raw performance is critical
   - Large datasets
   - Academic benchmarking
   - Julia ecosystem integration

2. **Choose gplearn when**:
   - Already using sklearn
   - Need sklearn pipeline integration
   - Simple expressions sufficient

3. **Choose Neural Networks when**:
   - Very complex non-structured patterns
   - Large training data available
   - Interpretability not required
   - Compute resources available

4. **Choose SymPy when**:
   - Exact mathematical sequences
   - No noise in data
   - Need symbolic manipulation

---

## Performance Comparison

### Benchmark: Damped Oscillator

Data: `y = 2.5 * exp(-0.3 * t) * sin(4.0 * t + 0.5) + noise`

| Tool | R² | Expression Found | Time |
|------|-----|------------------|------|
| PPF | 0.985 | `2.5*exp(-0.3*x)*sin(4.0*x+0.5)` | 2.1s |
| PySR | 0.983 | `2.49*exp(-0.31*x)*sin(4.01*x+0.49)` | 0.8s |
| gplearn | 0.71 | Complex nested expression | 5.2s |

**Analysis**: PySR is faster, but PPF matches accuracy and finds the exact macro form. gplearn struggles with multiplicative composition.

### Benchmark: Polynomial (Nguyen-1)

Data: `y = x³ + x² + x`

| Tool | R² | Expression Found | Time |
|------|-----|------------------|------|
| PPF | 0.9998 | `x³ + x² + x` | 1.8s |
| PySR | 0.9999 | `x³ + x² + x` | 0.5s |
| gplearn | 0.9997 | `x³ + x² + x` | 2.1s |

**Analysis**: All tools handle polynomials well. PySR is fastest.

### Benchmark: Export Size

Exporting `2.5*exp(-0.3*t)*sin(4.0*t+0.5)`:

| Format | PPF | Notes |
|--------|-----|-------|
| Python function | 312 bytes | Includes safety wrappers |
| C function | 289 bytes | With clamp_exp |
| JSON bundle | 1.2 KB | Full metadata |
| TensorFlow Lite equivalent | ~50 KB | For comparison |

---

## Conclusion

PPF occupies a specific niche in the symbolic regression ecosystem:

**PPF excels at**:
- Discover → Deploy workflows
- Edge/embedded deployment
- Physics-informed discovery (macros)
- Feature extraction for downstream ML
- Low-dependency environments

**PPF is not optimal for**:
- Maximum raw search performance (use PySR)
- sklearn pipeline integration (use gplearn)
- Exact symbolic sequences (use SymPy)
- Very complex unstructured patterns (use NNs)

The key differentiator is the **export layer** and **domain macros** - PPF is designed for the complete workflow from data to deployed edge model, not just finding equations.
