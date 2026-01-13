# Contributing to PPF

Thank you for your interest in contributing to PPF! This document provides guidelines for contributing.

## Getting Started

### Prerequisites

- Python 3.8+
- numpy
- scipy
- pytest (for testing)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/pcoz/timeseries-formula-finder.git
cd ppf

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy pytest pytest-cov

# Run tests to verify setup
pytest tests/ -v
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Create a new issue with:
   - Clear title describing the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Minimal code example

### Suggesting Features

1. Open an issue with `[Feature Request]` prefix
2. Describe the use case
3. Explain why existing functionality doesn't suffice
4. If possible, sketch an API design

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest tests/ -v`
6. Commit with clear message: `git commit -m "Add feature X"`
7. Push to your fork: `git push origin feature/my-feature`
8. Open a Pull Request

## Code Guidelines

### Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for public functions

```python
def export_python(
    expr: ExprNode,
    *,
    fn_name: str = "ppf_model",
    signature: Tuple[str, ...] = ("t",),
    safe: bool = True
) -> str:
    """
    Generate standalone Python code for an expression.

    Args:
        expr: Expression tree to export
        fn_name: Name for the generated function
        signature: Variable names in function signature
        safe: Whether to emit safety wrappers

    Returns:
        Python code string

    Example:
        >>> code = export_python(expr, fn_name="predict")
        >>> exec(code)
        >>> result = predict(1.5)
    """
```

### Testing

- Add tests for new functionality
- Tests go in `tests/test_<module>.py`
- Use descriptive test names
- Include docstrings explaining what's tested

```python
def test_export_python_with_macro():
    """Verify that macro expressions export correctly to Python."""
    expr = ExprNode(
        NodeType.MACRO,
        macro_op=MacroOp.DAMPED_SIN,
        macro_params=[1.0, 0.5, 2.0, 0.0]
    )

    code = export_python(expr)

    # Verify it runs
    exec(code, globals())
    result = ppf_model(1.0)
    assert np.isfinite(result)
```

### Documentation

- Update docstrings for API changes
- Update relevant .md files
- Add examples for new features

## Areas of Interest

We're particularly interested in contributions in these areas:

### New Macro Templates

Add domain-specific macros for new applications:

```python
# In symbolic_types.py
class MacroOp(Enum):
    # ...existing macros...
    NEW_MACRO = "new_macro"  # Description

MACRO_PARAM_COUNT[MacroOp.NEW_MACRO] = 3  # Number of parameters

# In symbolic_types.py - add evaluation function
def _eval_new_macro(x: np.ndarray, params: List[float]) -> np.ndarray:
    a, b, c = params
    return a * some_function(b * x) + c

MACRO_FUNCS[MacroOp.NEW_MACRO] = _eval_new_macro
```

### New Export Targets

Add exporters for new languages/platforms:

- Rust
- WebAssembly
- ONNX
- Julia
- MATLAB

Follow the pattern in `ppf/export/python_export.py`.

### Performance Optimization

The GP engine could be faster:

- Vectorized fitness evaluation
- Cython/Numba acceleration
- Parallel population evaluation

### Integration

Help integrate PPF with other tools:

- sklearn Pipeline/Transformer interface
- AutoML frameworks
- Jupyter widgets

## Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting)
3. New features need tests and documentation
4. Breaking changes need discussion in issue first

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Credit others' work appropriately

## Questions?

- Open an issue for questions
- Email: edward@fleetingswallow.com

Thank you for contributing!
