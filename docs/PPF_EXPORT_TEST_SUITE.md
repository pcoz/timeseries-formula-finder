# PPF Export Layer - Test Suite Design

**Version**: 1.0
**Date**: January 2026
**Related**: PPF_EXPORT_LAYER_TSD.md

---

## 1. Test Strategy Overview

### 1.1 Test Pyramid

```
                    ┌─────────────┐
                    │ Integration │  ← Edge simulation, end-to-end
                   ─┴─────────────┴─
                  ┌─────────────────┐
                  │  Golden Tests   │  ← Known benchmarks, regression
                 ─┴─────────────────┴─
                ┌───────────────────────┐
                │     Unit Tests        │  ← Individual functions
               ─┴───────────────────────┴─
```

### 1.2 Test Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Unit Tests | ~60 | Individual function correctness |
| Round-Trip Tests | ~15 | JSON export/import fidelity |
| Golden Tests | ~12 | Regression against known expressions |
| Safety Tests | ~20 | Edge cases (div/0, log(-1), exp(∞)) |
| Integration Tests | ~5 | End-to-end workflows |
| Feature Tests | ~15 | Feature extraction accuracy |

**Total: ~127 tests**

### 1.3 Test Infrastructure

```python
# conftest.py - Shared fixtures

import pytest
import numpy as np
from ppf import ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp

@pytest.fixture
def simple_expr():
    """x + 1"""
    return ExprNode.binary(
        BinaryOp.ADD,
        ExprNode.variable("x"),
        ExprNode.constant(1.0)
    )

@pytest.fixture
def damped_sin_expr():
    """3.004 * exp(-0.5 * t) * sin(6.0 * t + -1.579)"""
    return ExprNode.macro(
        MacroOp.DAMPED_SIN,
        [3.004, 0.5, 6.0, -1.579]  # a, k, w, phi
    )

@pytest.fixture
def gaussian_expr():
    """2.5 * exp(-((x - 3.0) / 0.8)^2)"""
    return ExprNode.macro(
        MacroOp.GAUSSIAN,
        [2.5, 3.0, 0.8]  # a, mu, sigma
    )

@pytest.fixture
def rational_expr():
    """(2x + 1) / (x + 3)"""
    return ExprNode.macro(
        MacroOp.RATIO,
        [2.0, 1.0, 1.0, 3.0]  # a, b, c, d
    )

@pytest.fixture
def nested_expr():
    """sin(exp(-x) + cos(x))"""
    # Complex nested expression for stress testing
    ...

@pytest.fixture
def test_inputs():
    """Standard test input arrays"""
    return {
        "linspace": np.linspace(0, 10, 100),
        "negative": np.linspace(-5, 5, 100),
        "small": np.linspace(0, 0.1, 50),
        "large": np.linspace(0, 1000, 100),
        "edge": np.array([0, 1e-15, 1e-10, 1, 10, 100, 1e10]),
    }
```

---

## 2. Unit Tests

### 2.1 Python Export Tests

**File**: `tests/test_export_python.py`

```python
class TestExportPythonBasic:
    """Basic Python export functionality"""

    def test_constant_only(self):
        """Export expression that is just a constant"""
        expr = ExprNode.constant(42.0)
        code = export_python(expr)
        assert "def ppf_model" in code
        assert "return 42.0" in code
        assert "import math" in code

    def test_variable_only(self):
        """Export expression that is just a variable"""
        expr = ExprNode.variable("t")
        code = export_python(expr, signature=("t",))
        assert "def ppf_model(t):" in code
        assert "return t" in code

    def test_custom_function_name(self):
        """Custom function name is used"""
        expr = ExprNode.constant(1.0)
        code = export_python(expr, fn_name="my_model")
        assert "def my_model" in code
        assert "ppf_model" not in code

    def test_multi_variable_signature(self):
        """Multiple variables in signature"""
        expr = ExprNode.binary(BinaryOp.ADD,
            ExprNode.variable("x"),
            ExprNode.variable("y"))
        code = export_python(expr, signature=("x", "y"))
        assert "def ppf_model(x, y):" in code


class TestExportPythonOperators:
    """Test all operator exports"""

    @pytest.mark.parametrize("op,symbol", [
        (BinaryOp.ADD, "+"),
        (BinaryOp.SUB, "-"),
        (BinaryOp.MUL, "*"),
        (BinaryOp.DIV, "/"),
    ])
    def test_binary_operators(self, op, symbol):
        """Binary operators export correctly"""
        expr = ExprNode.binary(op,
            ExprNode.variable("x"),
            ExprNode.constant(2.0))
        code = export_python(expr, safe=False)
        assert symbol in code

    def test_pow_operator(self):
        """Power operator uses math.pow or **"""
        expr = ExprNode.binary(BinaryOp.POW,
            ExprNode.variable("x"),
            ExprNode.constant(2.0))
        code = export_python(expr)
        assert "**" in code or "pow" in code

    @pytest.mark.parametrize("op,func", [
        (UnaryOp.SIN, "math.sin"),
        (UnaryOp.COS, "math.cos"),
        (UnaryOp.EXP, "math.exp"),
        (UnaryOp.LOG, "math.log"),
        (UnaryOp.SQRT, "math.sqrt"),
        (UnaryOp.ABS, "abs"),
        (UnaryOp.NEG, "-"),
    ])
    def test_unary_operators(self, op, func):
        """Unary operators export correctly"""
        expr = ExprNode.unary(op, ExprNode.variable("x"))
        code = export_python(expr, safe=False)
        assert func in code or (func == "-" and "- x" in code)


class TestExportPythonSafety:
    """Safety wrapper tests"""

    def test_safe_div_emitted(self):
        """Safe division helper is emitted when safe=True"""
        expr = ExprNode.binary(BinaryOp.DIV,
            ExprNode.variable("x"),
            ExprNode.variable("y"))
        code = export_python(expr, signature=("x", "y"), safe=True)
        assert "def safe_div" in code
        assert "safe_div(x, y)" in code

    def test_safe_log_emitted(self):
        """Safe log helper is emitted when safe=True"""
        expr = ExprNode.unary(UnaryOp.LOG, ExprNode.variable("x"))
        code = export_python(expr, safe=True)
        assert "def safe_log" in code
        assert "safe_log(x)" in code

    def test_clamp_exp_emitted(self):
        """Clamp exp helper is emitted when safe=True"""
        expr = ExprNode.unary(UnaryOp.EXP, ExprNode.variable("x"))
        code = export_python(expr, safe=True)
        assert "def clamp_exp" in code
        assert "clamp_exp(x)" in code

    def test_no_safety_when_disabled(self):
        """No safety helpers when safe=False"""
        expr = ExprNode.binary(BinaryOp.DIV,
            ExprNode.variable("x"),
            ExprNode.constant(2.0))
        code = export_python(expr, safe=False)
        assert "safe_div" not in code
        assert "x / 2.0" in code


class TestExportPythonMacros:
    """Macro expansion tests"""

    def test_damped_sin_expansion(self, damped_sin_expr):
        """DAMPED_SIN macro expands correctly"""
        code = export_python(damped_sin_expr)
        assert "math.exp" in code
        assert "math.sin" in code
        # Check structure: a * exp(-k*t) * sin(w*t + phi)
        assert "*" in code

    def test_gaussian_expansion(self, gaussian_expr):
        """GAUSSIAN macro expands correctly"""
        code = export_python(gaussian_expr)
        assert "math.exp" in code
        # Should have squared term

    def test_all_macros_export(self):
        """Every defined macro can be exported"""
        for macro_op in MacroOp:
            param_count = MACRO_PARAM_COUNT[macro_op]
            params = [1.0] * param_count
            expr = ExprNode.macro(macro_op, params)
            code = export_python(expr)
            assert "def ppf_model" in code


class TestExportPythonDeterminism:
    """Deterministic output tests"""

    def test_same_expr_same_output(self, damped_sin_expr):
        """Same expression produces identical output"""
        code1 = export_python(damped_sin_expr)
        code2 = export_python(damped_sin_expr)
        assert code1 == code2

    def test_determinism_across_runs(self, nested_expr):
        """Complex expression produces identical output across multiple calls"""
        outputs = [export_python(nested_expr) for _ in range(10)]
        assert all(o == outputs[0] for o in outputs)


class TestExportPythonExecution:
    """Test that exported code actually runs correctly"""

    def test_exported_matches_evaluate(self, damped_sin_expr, test_inputs):
        """Exported Python function matches ExprNode.evaluate()"""
        code = export_python(damped_sin_expr, signature=("t",))

        # Execute the exported code
        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        exported_fn = local_ns["ppf_model"]

        # Compare outputs
        t = test_inputs["linspace"]
        expected = damped_sin_expr.evaluate(t)

        for i, t_val in enumerate(t):
            result = exported_fn(t_val)
            assert abs(result - expected[i]) < 1e-10

    def test_no_numpy_dependency(self, simple_expr):
        """Exported code runs without numpy"""
        code = export_python(simple_expr)

        # Execute in namespace without numpy
        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        # Should work with plain Python float
        result = fn(5.0)
        assert isinstance(result, float)
```

---

### 2.2 C Export Tests

**File**: `tests/test_export_c.py`

```python
class TestExportCBasic:
    """Basic C export functionality"""

    def test_includes_math_h(self):
        """Output includes math.h"""
        expr = ExprNode.constant(1.0)
        code = export_c(expr)
        assert "#include <math.h>" in code

    def test_function_signature(self):
        """Correct function signature generated"""
        expr = ExprNode.variable("t")
        code = export_c(expr, signature=("double t",))
        assert "double ppf_model(double t)" in code

    def test_static_inline(self):
        """Function is static inline"""
        expr = ExprNode.constant(1.0)
        code = export_c(expr)
        assert "static inline" in code

    def test_custom_function_name(self):
        """Custom function name is used"""
        expr = ExprNode.constant(1.0)
        code = export_c(expr, fn_name="sensor_model")
        assert "sensor_model" in code


class TestExportCFloat:
    """Float vs double tests"""

    def test_double_by_default(self):
        """Uses double math by default"""
        expr = ExprNode.unary(UnaryOp.SIN, ExprNode.variable("t"))
        code = export_c(expr)
        assert "double" in code
        assert "sin(" in code
        assert "sinf(" not in code

    def test_float_when_requested(self):
        """Uses float math when use_float=True"""
        expr = ExprNode.unary(UnaryOp.SIN, ExprNode.variable("t"))
        code = export_c(expr, use_float=True)
        assert "float" in code
        assert "sinf(" in code

    @pytest.mark.parametrize("op,double_fn,float_fn", [
        (UnaryOp.SIN, "sin", "sinf"),
        (UnaryOp.COS, "cos", "cosf"),
        (UnaryOp.EXP, "exp", "expf"),
        (UnaryOp.LOG, "log", "logf"),
        (UnaryOp.SQRT, "sqrt", "sqrtf"),
    ])
    def test_float_function_variants(self, op, double_fn, float_fn):
        """Correct float function variants used"""
        expr = ExprNode.unary(op, ExprNode.variable("x"))
        double_code = export_c(expr, use_float=False)
        float_code = export_c(expr, use_float=True)
        assert double_fn + "(" in double_code
        assert float_fn + "(" in float_code


class TestExportCSafety:
    """C safety wrapper tests"""

    def test_safe_div_emitted(self):
        """Safe division helper emitted"""
        expr = ExprNode.binary(BinaryOp.DIV,
            ExprNode.variable("a"),
            ExprNode.variable("b"))
        code = export_c(expr, signature=("double a", "double b"), safe=True)
        assert "static inline double safe_div" in code
        assert "fabs(b)" in code

    def test_clamp_exp_arg_emitted(self):
        """Exp argument clamping emitted"""
        expr = ExprNode.unary(UnaryOp.EXP, ExprNode.variable("x"))
        code = export_c(expr, safe=True)
        assert "clamp_exp_arg" in code
        assert "-60" in code and "60" in code


class TestExportCMacroStyle:
    """Macro style tests"""

    def test_inline_by_default(self, damped_sin_expr):
        """Macros inlined by default"""
        code = export_c(damped_sin_expr, macro_style="inline")
        # Should not have damped_sin function
        assert "static inline double damped_sin(" not in code
        # Should have expanded form
        assert "exp(" in code
        assert "sin(" in code

    def test_helpers_when_requested(self, damped_sin_expr):
        """Helper functions emitted when requested"""
        code = export_c(damped_sin_expr, macro_style="helpers")
        assert "static inline double damped_sin(" in code

    def test_all_macros_have_helpers(self):
        """Every macro can generate a helper function"""
        for macro_op in MacroOp:
            param_count = MACRO_PARAM_COUNT[macro_op]
            params = [1.0] * param_count
            expr = ExprNode.macro(macro_op, params)
            code = export_c(expr, macro_style="helpers")
            # Should have a helper function
            assert "static inline" in code


class TestExportCNumericFormatting:
    """Numeric formatting tests"""

    def test_integer_constants_have_decimal(self):
        """Integer constants formatted as doubles"""
        expr = ExprNode.constant(2)
        code = export_c(expr)
        assert "2.0" in code or "2." in code

    def test_precision_preserved(self):
        """High precision constants preserved"""
        expr = ExprNode.constant(3.141592653589793)
        code = export_c(expr)
        assert "3.14159265358" in code  # At least 12 sig figs

    def test_negative_constants(self):
        """Negative constants formatted correctly"""
        expr = ExprNode.constant(-1.579)
        code = export_c(expr)
        assert "-1.579" in code
        assert "+-" not in code  # No +- artifact


class TestExportCCompilable:
    """Tests that exported C code compiles (requires gcc)"""

    @pytest.mark.skipif(not shutil.which("gcc"), reason="gcc not available")
    def test_compiles_c99(self, damped_sin_expr, tmp_path):
        """Exported code compiles as C99"""
        code = export_c(damped_sin_expr)

        # Write to file
        c_file = tmp_path / "model.c"
        c_file.write_text(code + "\nint main() { return 0; }\n")

        # Compile
        result = subprocess.run(
            ["gcc", "-std=c99", "-c", "-Wall", "-Werror", str(c_file)],
            capture_output=True
        )
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"
```

---

### 2.3 JSON Export Tests

**File**: `tests/test_export_json.py`

```python
class TestExportJSONSchema:
    """JSON schema tests"""

    def test_schema_version_present(self, damped_sin_expr):
        """Schema version is included"""
        bundle = export_json(make_result(damped_sin_expr))
        assert bundle["schema"] == "ppf.export.model.v1"

    def test_required_fields_present(self, damped_sin_expr):
        """All required fields present"""
        bundle = export_json(make_result(damped_sin_expr))
        assert "expression" in bundle
        assert "metrics" in bundle
        assert "expression" in bundle
        assert "string" in bundle["expression"]

    def test_timestamp_format(self, damped_sin_expr):
        """Timestamp is ISO 8601 format"""
        bundle = export_json(make_result(damped_sin_expr))
        # Should parse without error
        datetime.fromisoformat(bundle["created_utc"].replace("Z", "+00:00"))


class TestExportJSONExpression:
    """Expression serialization tests"""

    def test_string_representation(self, damped_sin_expr):
        """Expression string is included"""
        bundle = export_json(make_result(damped_sin_expr))
        assert "string" in bundle["expression"]
        assert len(bundle["expression"]["string"]) > 0

    def test_tree_included_by_default(self, damped_sin_expr):
        """Expression tree included by default"""
        bundle = export_json(make_result(damped_sin_expr))
        assert "tree" in bundle["expression"]
        assert bundle["expression"]["tree"] is not None

    def test_tree_excluded_when_disabled(self, damped_sin_expr):
        """Tree excluded when include_expr_tree=False"""
        bundle = export_json(make_result(damped_sin_expr), include_expr_tree=False)
        assert bundle["expression"].get("tree") is None


class TestExportJSONTreeSerialization:
    """Tree serialization tests"""

    def test_const_node(self):
        """Constant node serializes correctly"""
        expr = ExprNode.constant(3.14)
        bundle = export_json(make_result(expr))
        tree = bundle["expression"]["tree"]
        assert tree["type"] == "CONST"
        assert tree["value"] == 3.14

    def test_var_node(self):
        """Variable node serializes correctly"""
        expr = ExprNode.variable("t")
        bundle = export_json(make_result(expr))
        tree = bundle["expression"]["tree"]
        assert tree["type"] == "VAR"
        assert tree["name"] == "t"

    def test_unary_node(self):
        """Unary op node serializes correctly"""
        expr = ExprNode.unary(UnaryOp.SIN, ExprNode.variable("x"))
        bundle = export_json(make_result(expr))
        tree = bundle["expression"]["tree"]
        assert tree["type"] == "UNARY_OP"
        assert tree["op"] == "SIN"
        assert "child" in tree

    def test_binary_node(self):
        """Binary op node serializes correctly"""
        expr = ExprNode.binary(BinaryOp.ADD,
            ExprNode.variable("x"),
            ExprNode.constant(1.0))
        bundle = export_json(make_result(expr))
        tree = bundle["expression"]["tree"]
        assert tree["type"] == "BINARY_OP"
        assert tree["op"] == "ADD"
        assert "children" in tree
        assert len(tree["children"]) == 2

    def test_macro_node(self, damped_sin_expr):
        """Macro node serializes as MACRO_CALL"""
        bundle = export_json(make_result(damped_sin_expr))
        tree = bundle["expression"]["tree"]
        assert tree["type"] == "MACRO_CALL"
        assert tree["name"] == "DAMPED_SIN"
        assert "args" in tree
        assert len(tree["args"]) == 4  # a, k, w, phi (t is implicit)


class TestExportJSONMetrics:
    """Metrics serialization tests"""

    def test_r2_included(self, damped_sin_expr):
        """R² metric included"""
        bundle = export_json(make_result(damped_sin_expr, r2=0.98))
        assert bundle["metrics"]["r2"] == 0.98

    def test_complexity_included(self, damped_sin_expr):
        """Complexity metric included"""
        bundle = export_json(make_result(damped_sin_expr))
        assert "complexity" in bundle["metrics"]
        assert isinstance(bundle["metrics"]["complexity"], int)


class TestExportJSONMetadata:
    """Metadata tests"""

    def test_seed_included(self, damped_sin_expr):
        """Random seed included in metadata"""
        bundle = export_json(make_result(damped_sin_expr, seed=42))
        assert bundle["metadata"]["seed"] == 42

    def test_mode_included(self, damped_sin_expr):
        """Discovery mode included"""
        bundle = export_json(make_result(damped_sin_expr, mode="OSCILLATOR"))
        assert bundle["metadata"]["mode"] == "OSCILLATOR"


class TestExportJSONSerializable:
    """JSON serialization tests"""

    def test_json_dumps_succeeds(self, damped_sin_expr):
        """Bundle can be serialized to JSON string"""
        bundle = export_json(make_result(damped_sin_expr))
        json_str = json.dumps(bundle)
        assert isinstance(json_str, str)

    def test_no_numpy_types(self, damped_sin_expr):
        """No numpy types in bundle (they don't serialize)"""
        bundle = export_json(make_result(damped_sin_expr))

        def check_no_numpy(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    check_no_numpy(v)
            elif isinstance(obj, list):
                for v in obj:
                    check_no_numpy(v)
            else:
                assert not isinstance(obj, np.ndarray)
                assert not isinstance(obj, np.floating)
                assert not isinstance(obj, np.integer)

        check_no_numpy(bundle)
```

---

### 2.4 JSON Load Tests

**File**: `tests/test_load_json.py`

```python
class TestLoadJSONBasic:
    """Basic load functionality"""

    def test_load_returns_expr_and_metadata(self, damped_sin_expr):
        """Load returns expression and metadata tuple"""
        bundle = export_json(make_result(damped_sin_expr))
        expr, metadata = load_json(bundle)
        assert isinstance(expr, ExprNode)
        assert isinstance(metadata, dict)

    def test_loaded_expr_evaluates(self, damped_sin_expr, test_inputs):
        """Loaded expression can be evaluated"""
        bundle = export_json(make_result(damped_sin_expr))
        expr, _ = load_json(bundle)
        t = test_inputs["linspace"]
        result = expr.evaluate(t)
        assert len(result) == len(t)


class TestLoadJSONRoundTrip:
    """Round-trip tests"""

    def test_constant_round_trip(self):
        """Constant expression survives round-trip"""
        original = ExprNode.constant(3.14159)
        bundle = export_json(make_result(original))
        loaded, _ = load_json(bundle)
        assert loaded.evaluate(np.array([0])) == original.evaluate(np.array([0]))

    def test_variable_round_trip(self):
        """Variable expression survives round-trip"""
        original = ExprNode.variable("t")
        bundle = export_json(make_result(original))
        loaded, _ = load_json(bundle)
        t = np.array([1, 2, 3])
        np.testing.assert_array_equal(loaded.evaluate(t), original.evaluate(t))

    def test_complex_expr_round_trip(self, nested_expr, test_inputs):
        """Complex nested expression survives round-trip"""
        bundle = export_json(make_result(nested_expr))
        loaded, _ = load_json(bundle)
        t = test_inputs["linspace"]
        np.testing.assert_array_almost_equal(
            loaded.evaluate(t),
            nested_expr.evaluate(t),
            decimal=10
        )

    def test_macro_round_trip(self, damped_sin_expr, test_inputs):
        """Macro expression survives round-trip without expansion"""
        bundle = export_json(make_result(damped_sin_expr))
        loaded, _ = load_json(bundle)

        # Should still be a macro, not expanded
        assert loaded.node_type == NodeType.MACRO
        assert loaded.macro_op == MacroOp.DAMPED_SIN

        # Values should match
        t = test_inputs["linspace"]
        np.testing.assert_array_almost_equal(
            loaded.evaluate(t),
            damped_sin_expr.evaluate(t),
            decimal=10
        )

    @pytest.mark.parametrize("macro_op", list(MacroOp))
    def test_all_macros_round_trip(self, macro_op, test_inputs):
        """Every macro type survives round-trip"""
        param_count = MACRO_PARAM_COUNT[macro_op]
        params = [float(i + 1) for i in range(param_count)]
        original = ExprNode.macro(macro_op, params)

        bundle = export_json(make_result(original))
        loaded, _ = load_json(bundle)

        t = test_inputs["small"]  # Use small range for stability
        np.testing.assert_array_almost_equal(
            loaded.evaluate(t),
            original.evaluate(t),
            decimal=8
        )


class TestLoadJSONValidation:
    """Validation tests"""

    def test_invalid_schema_raises(self):
        """Unknown schema version raises error"""
        bundle = {"schema": "ppf.export.model.v99", "expression": {}}
        with pytest.raises(SchemaVersionError):
            load_json(bundle)

    def test_unknown_node_type_raises(self):
        """Unknown node type raises error"""
        bundle = {
            "schema": "ppf.export.model.v1",
            "expression": {
                "string": "x",
                "tree": {"type": "UNKNOWN_TYPE"}
            },
            "metrics": {"r2": 0.9, "complexity": 1}
        }
        with pytest.raises(UnsupportedNodeError):
            load_json(bundle)

    def test_unknown_operator_raises(self):
        """Unknown operator raises error"""
        bundle = {
            "schema": "ppf.export.model.v1",
            "expression": {
                "string": "x",
                "tree": {"type": "UNARY_OP", "op": "UNKNOWN_OP", "child": {"type": "VAR", "name": "x"}}
            },
            "metrics": {"r2": 0.9, "complexity": 1}
        }
        with pytest.raises(UnsupportedOperatorError):
            load_json(bundle)


class TestLoadJSONMetadata:
    """Metadata loading tests"""

    def test_metrics_loaded(self, damped_sin_expr):
        """Metrics available in metadata"""
        bundle = export_json(make_result(damped_sin_expr, r2=0.98, complexity=5))
        _, metadata = load_json(bundle)
        assert metadata["metrics"]["r2"] == 0.98
        assert metadata["metrics"]["complexity"] == 5

    def test_seed_loaded(self, damped_sin_expr):
        """Seed available in metadata"""
        bundle = export_json(make_result(damped_sin_expr, seed=42))
        _, metadata = load_json(bundle)
        assert metadata["metadata"]["seed"] == 42
```

---

## 3. Safety Tests

**File**: `tests/test_safety.py`

```python
class TestPythonSafety:
    """Python safety behavior tests"""

    def test_division_by_zero(self):
        """Division by zero doesn't crash"""
        expr = ExprNode.binary(BinaryOp.DIV,
            ExprNode.constant(1.0),
            ExprNode.variable("x"))
        code = export_python(expr, safe=True)

        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        # Should not raise
        result = fn(0.0)
        assert np.isfinite(result)

    def test_division_by_small_number(self):
        """Division by very small number is clamped"""
        expr = ExprNode.binary(BinaryOp.DIV,
            ExprNode.constant(1.0),
            ExprNode.variable("x"))
        code = export_python(expr, safe=True)

        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        result = fn(1e-15)
        assert np.isfinite(result)
        assert abs(result) < 1e15  # Not astronomical

    def test_log_of_zero(self):
        """Log of zero doesn't crash"""
        expr = ExprNode.unary(UnaryOp.LOG, ExprNode.variable("x"))
        code = export_python(expr, safe=True)

        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        result = fn(0.0)
        assert np.isfinite(result)

    def test_log_of_negative(self):
        """Log of negative doesn't crash"""
        expr = ExprNode.unary(UnaryOp.LOG, ExprNode.variable("x"))
        code = export_python(expr, safe=True)

        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        result = fn(-1.0)
        assert np.isfinite(result)

    def test_exp_overflow(self):
        """Exp of large number doesn't overflow"""
        expr = ExprNode.unary(UnaryOp.EXP, ExprNode.variable("x"))
        code = export_python(expr, safe=True)

        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        result = fn(1000.0)
        assert np.isfinite(result)

    def test_exp_underflow(self):
        """Exp of large negative number doesn't underflow to exactly 0"""
        expr = ExprNode.unary(UnaryOp.EXP, ExprNode.variable("x"))
        code = export_python(expr, safe=True)

        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        result = fn(-1000.0)
        assert np.isfinite(result)


class TestCSafety:
    """C safety behavior tests (execution requires compilation)"""

    @pytest.mark.skipif(not shutil.which("gcc"), reason="gcc not available")
    def test_c_division_by_zero(self, tmp_path):
        """C code handles division by zero"""
        expr = ExprNode.binary(BinaryOp.DIV,
            ExprNode.constant(1.0),
            ExprNode.variable("x"))
        code = export_c(expr, safe=True)

        # Add test harness
        test_code = code + """
        #include <stdio.h>
        int main() {
            double result = ppf_model(0.0);
            printf("%f\\n", result);
            return (result == result) ? 0 : 1;  // Check not NaN
        }
        """

        c_file = tmp_path / "test.c"
        c_file.write_text(test_code)
        exe_file = tmp_path / "test"

        subprocess.run(["gcc", "-std=c99", "-o", str(exe_file), str(c_file), "-lm"], check=True)
        result = subprocess.run([str(exe_file)], capture_output=True)
        assert result.returncode == 0
```

---

## 4. Golden Tests

**File**: `tests/test_golden.py`

```python
GOLDEN_CASES = [
    {
        "name": "damped_oscillator",
        "expr": lambda: ExprNode.macro(MacroOp.DAMPED_SIN, [3.004, 0.5015, 6.008, -1.579]),
        "test_points": np.linspace(0, 5, 50),
        "expected_r2": 0.9999,  # Against itself
    },
    {
        "name": "rc_charge",
        "expr": lambda: ExprNode.macro(MacroOp.RC_CHARGE, [5.0, 2.0, 0.0]),
        "test_points": np.linspace(0, 3, 50),
        "expected_r2": 0.9999,
    },
    {
        "name": "gaussian",
        "expr": lambda: ExprNode.macro(MacroOp.GAUSSIAN, [2.5, 3.0, 0.8]),
        "test_points": np.linspace(0, 6, 50),
        "expected_r2": 0.9999,
    },
    {
        "name": "logistic",
        "expr": lambda: ExprNode.macro(MacroOp.LOGISTIC, [100.0, 1.0, 0.8]),
        "test_points": np.linspace(0, 10, 50),
        "expected_r2": 0.9999,
    },
    {
        "name": "power_law",
        "expr": lambda: ExprNode.macro(MacroOp.POWER_LAW, [2.0, 1.5, 0.5]),
        "test_points": np.linspace(0.1, 5, 50),
        "expected_r2": 0.9999,
    },
    {
        "name": "rational",
        "expr": lambda: ExprNode.macro(MacroOp.RATIO, [2.0, 1.0, 1.0, 3.0]),
        "test_points": np.linspace(0, 10, 50),
        "expected_r2": 0.9999,
    },
]


class TestGoldenPython:
    """Golden tests for Python export"""

    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c["name"])
    def test_golden_python(self, case):
        """Exported Python matches original expression"""
        expr = case["expr"]()
        t = case["test_points"]

        # Get reference values
        expected = expr.evaluate(t)

        # Export and execute
        code = export_python(expr, signature=("t",))
        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        # Compare
        actual = np.array([fn(ti) for ti in t])
        r2 = 1 - np.sum((actual - expected)**2) / np.sum((expected - np.mean(expected))**2)

        assert r2 >= case["expected_r2"], f"R² = {r2:.6f}, expected >= {case['expected_r2']}"


class TestGoldenC:
    """Golden tests for C export"""

    @pytest.mark.skipif(not shutil.which("gcc"), reason="gcc not available")
    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c["name"])
    def test_golden_c(self, case, tmp_path):
        """Exported C matches original expression"""
        expr = case["expr"]()
        t = case["test_points"]

        # Get reference values
        expected = expr.evaluate(t)

        # Export C code
        c_code = export_c(expr, signature=("double t",))

        # Create test harness
        test_points_str = ", ".join(f"{ti}" for ti in t)
        harness = f"""
        {c_code}
        #include <stdio.h>
        int main() {{
            double t[] = {{{test_points_str}}};
            int n = {len(t)};
            for (int i = 0; i < n; i++) {{
                printf("%.15f\\n", ppf_model(t[i]));
            }}
            return 0;
        }}
        """

        # Compile and run
        c_file = tmp_path / "test.c"
        c_file.write_text(harness)
        exe_file = tmp_path / "test"

        subprocess.run(["gcc", "-std=c99", "-o", str(exe_file), str(c_file), "-lm"], check=True)
        result = subprocess.run([str(exe_file)], capture_output=True, text=True)

        # Parse output
        actual = np.array([float(line) for line in result.stdout.strip().split("\n")])

        # Compare
        r2 = 1 - np.sum((actual - expected)**2) / np.sum((expected - np.mean(expected))**2)
        assert r2 >= case["expected_r2"], f"R² = {r2:.6f}, expected >= {case['expected_r2']}"


class TestGoldenRoundTrip:
    """Golden tests for JSON round-trip"""

    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c["name"])
    def test_golden_roundtrip(self, case):
        """JSON round-trip preserves expression"""
        expr = case["expr"]()
        t = case["test_points"]

        # Get reference values
        expected = expr.evaluate(t)

        # Round-trip through JSON
        bundle = export_json(make_result(expr))
        loaded, _ = load_json(bundle)

        # Compare
        actual = loaded.evaluate(t)
        np.testing.assert_array_almost_equal(actual, expected, decimal=10)
```

---

## 5. Feature Extraction Tests

**File**: `tests/test_features.py`

```python
class TestExtractFeaturesCommon:
    """Common feature extraction tests"""

    def test_r2_extracted(self, damped_sin_expr):
        """R² is extracted"""
        result = make_result(damped_sin_expr, r2=0.95)
        features = extract_features(result)
        assert features["r2"] == 0.95

    def test_complexity_extracted(self, damped_sin_expr):
        """Complexity is extracted"""
        result = make_result(damped_sin_expr, complexity=5)
        features = extract_features(result)
        assert features["complexity"] == 5

    def test_mode_extracted(self, damped_sin_expr):
        """Mode is extracted"""
        result = make_result(damped_sin_expr, mode="OSCILLATOR")
        features = extract_features(result)
        assert features["mode_chosen"] == "OSCILLATOR"


class TestExtractFeaturesOscillator:
    """Oscillator-specific feature extraction"""

    def test_damped_sin_features(self):
        """DAMPED_SIN extracts amplitude, damping, omega, phase"""
        expr = ExprNode.macro(MacroOp.DAMPED_SIN, [3.0, 0.5, 6.0, -1.5])
        result = make_result(expr, mode="OSCILLATOR")
        features = extract_features(result)

        assert features["dominant_family"] == "oscillation"
        assert abs(features["amplitude"] - 3.0) < 0.01
        assert abs(features["damping_k"] - 0.5) < 0.01
        assert abs(features["omega"] - 6.0) < 0.01
        assert abs(features["phase"] - (-1.5)) < 0.01


class TestExtractFeaturesGaussian:
    """Gaussian-specific feature extraction"""

    def test_gaussian_features(self):
        """GAUSSIAN extracts amplitude, mu, sigma"""
        expr = ExprNode.macro(MacroOp.GAUSSIAN, [2.5, 3.0, 0.8])
        result = make_result(expr, mode="UNIVERSAL")
        features = extract_features(result)

        assert features["dominant_family"] == "peaks"
        assert abs(features["amplitude"] - 2.5) < 0.01
        assert abs(features["mu"] - 3.0) < 0.01
        assert abs(features["sigma"] - 0.8) < 0.01


class TestExtractFeaturesLogistic:
    """Logistic-specific feature extraction"""

    def test_logistic_features(self):
        """LOGISTIC extracts K, r"""
        expr = ExprNode.macro(MacroOp.LOGISTIC, [100.0, 1.0, 0.8])
        result = make_result(expr, mode="GROWTH")
        features = extract_features(result)

        assert features["dominant_family"] in ["growth", "saturation"]
        assert abs(features["K"] - 100.0) < 0.1


class TestExtractFeaturesStability:
    """Feature stability tests"""

    def test_same_expr_same_features(self, damped_sin_expr):
        """Same expression produces same features"""
        result = make_result(damped_sin_expr)
        f1 = extract_features(result)
        f2 = extract_features(result)
        assert f1 == f2

    def test_keys_are_strings(self, damped_sin_expr):
        """All feature keys are strings"""
        result = make_result(damped_sin_expr)
        features = extract_features(result)
        assert all(isinstance(k, str) for k in features.keys())

    def test_values_are_numeric_or_string(self, damped_sin_expr):
        """All values are numeric or string"""
        result = make_result(damped_sin_expr)
        features = extract_features(result)
        for v in features.values():
            assert isinstance(v, (int, float, str))


class TestFeatureVector:
    """Feature vector tests"""

    def test_returns_array_and_names(self, damped_sin_expr):
        """Returns both array and names"""
        result = make_result(damped_sin_expr)
        features = extract_features(result)
        vec, names = feature_vector(features)

        assert isinstance(vec, np.ndarray)
        assert isinstance(names, list)
        assert len(vec) == len(names)

    def test_order_is_consistent(self, damped_sin_expr):
        """Feature order is consistent across calls"""
        result = make_result(damped_sin_expr)
        features = extract_features(result)

        _, names1 = feature_vector(features)
        _, names2 = feature_vector(features)

        assert names1 == names2

    def test_edge_min_schema(self, damped_sin_expr):
        """edge_min schema produces smaller vector"""
        result = make_result(damped_sin_expr)
        features = extract_features(result)

        vec_min, _ = feature_vector(features, schema="ppf.features.v1.edge_min")
        vec_full, _ = feature_vector(features, schema="ppf.features.v1.full")

        assert len(vec_min) < len(vec_full)
```

---

## 6. Integration Tests

**File**: `tests/test_integration.py`

```python
class TestEndToEndWorkflow:
    """Full discover → export → deploy workflow"""

    def test_discover_export_python(self):
        """Full workflow: data → discover → export Python"""
        # Generate synthetic data
        t = np.linspace(0, 5, 100)
        y = 3.0 * np.exp(-0.5 * t) * np.sin(6.0 * t - 1.5) + 0.1 * np.random.randn(100)

        # Discover
        regressor = SymbolicRegressor(generations=20)
        result = regressor.discover(t, y, mode=DiscoveryMode.OSCILLATOR)

        # Export
        code = export_python(result.best_tradeoff.expression)

        # Execute
        local_ns = {}
        exec(code, {"math": __import__("math")}, local_ns)
        fn = local_ns["ppf_model"]

        # Verify
        predictions = np.array([fn(ti) for ti in t])
        r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
        assert r2 > 0.8  # Should fit reasonably well

    def test_discover_export_json_load(self):
        """Full workflow: discover → JSON → load → evaluate"""
        # Generate data
        t = np.linspace(0, 5, 100)
        y = 2.5 * np.exp(-((t - 2.5) / 0.8)**2) + 0.1 * np.random.randn(100)

        # Discover
        regressor = SymbolicRegressor(generations=20)
        result = regressor.discover(t, y, mode=DiscoveryMode.UNIVERSAL)

        # Export to JSON
        bundle = export_json(result.best_tradeoff)

        # Serialize and deserialize (simulate storage/transmission)
        json_str = json.dumps(bundle)
        loaded_bundle = json.loads(json_str)

        # Load back
        expr, metadata = load_json(loaded_bundle)

        # Verify
        predictions = expr.evaluate(t)
        r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
        assert r2 > 0.8

    def test_feature_extraction_for_ml(self):
        """Extract features for downstream ML"""
        # Multiple discovery runs
        datasets = [
            (np.linspace(0, 5, 100), 3.0 * np.exp(-0.5 * t) * np.sin(6.0 * t)),  # Oscillator
            (np.linspace(0, 5, 100), 100 / (1 + np.exp(-0.8 * (t - 2.5)))),      # Logistic
            (np.linspace(0, 5, 100), 2.5 * np.exp(-((t - 2.5) / 0.8)**2)),       # Gaussian
        ]

        feature_rows = []
        for t, y in datasets:
            regressor = SymbolicRegressor(generations=15)
            result = regressor.discover(t, y + 0.1 * np.random.randn(len(t)), mode=DiscoveryMode.AUTO)
            features = extract_features(result)
            vec, names = feature_vector(features)
            feature_rows.append(vec)

        # Stack into feature matrix
        X = np.vstack(feature_rows)
        assert X.shape[0] == 3  # 3 samples
        assert X.shape[1] == len(names)  # Consistent feature count
```

---

## 7. Test Utilities

**File**: `tests/helpers.py`

```python
def make_result(expr, r2=0.95, complexity=None, mode="AUTO", seed=42):
    """Create a mock SymbolicFitResult for testing"""
    if complexity is None:
        complexity = expr.size()

    return SymbolicFitResult(
        expression=expr,
        expression_string=expr.to_string(),
        r_squared=r2,
        rmse=np.sqrt(1 - r2),
        complexity=complexity,
        mode=mode,
        seed=seed,
    )


def expressions_equal(expr1, expr2, test_points=None):
    """Check if two expressions are functionally equal"""
    if test_points is None:
        test_points = np.linspace(0, 10, 100)

    v1 = expr1.evaluate(test_points)
    v2 = expr2.evaluate(test_points)

    return np.allclose(v1, v2, rtol=1e-10, atol=1e-10)
```

---

## 8. CI Configuration

**File**: `.github/workflows/test-export.yml`

```yaml
name: Export Layer Tests

on:
  push:
    paths:
      - 'ppf/export/**'
      - 'ppf/features/**'
      - 'tests/test_export_*.py'
      - 'tests/test_features.py'
      - 'tests/test_golden.py'
  pull_request:
    paths:
      - 'ppf/export/**'
      - 'ppf/features/**'

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.10', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov

      - name: Run unit tests
        run: pytest tests/test_export_*.py tests/test_features.py -v --cov=ppf/export --cov=ppf/features

      - name: Run golden tests
        run: pytest tests/test_golden.py -v

      - name: Run integration tests
        run: pytest tests/test_integration.py -v

  c-compilation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install GCC
        run: sudo apt-get install -y gcc

      - name: Install package
        run: pip install -e .

      - name: Run C compilation tests
        run: pytest tests/test_export_c.py tests/test_golden.py -v -k "compile or golden_c"
```

---

## 9. Test Coverage Requirements

### Minimum Coverage Targets

| Module | Line Coverage | Branch Coverage |
|--------|---------------|-----------------|
| `ppf/export/python_export.py` | 95% | 90% |
| `ppf/export/c_export.py` | 95% | 90% |
| `ppf/export/json_export.py` | 95% | 85% |
| `ppf/export/load.py` | 95% | 90% |
| `ppf/features/extract.py` | 90% | 85% |
| `ppf/features/vectorize.py` | 95% | 90% |

### Critical Paths (100% Coverage Required)

- All operator exports (every `UnaryOp`, `BinaryOp`)
- All macro exports (every `MacroOp`)
- Safety wrapper emission
- JSON tree serialization/deserialization
- Schema validation

---

## 10. Test Data Files

### Directory Structure

```
tests/
├── golden/
│   ├── damped_oscillator.json    # Reference JSON bundle
│   ├── damped_oscillator.py      # Reference Python output
│   ├── damped_oscillator.c       # Reference C output
│   ├── gaussian.json
│   ├── gaussian.py
│   ├── gaussian.c
│   ├── logistic.json
│   ├── logistic.py
│   ├── logistic.c
│   └── ...
├── fixtures/
│   ├── valid_bundles/            # Valid JSON bundles for load testing
│   ├── invalid_bundles/          # Invalid JSON for error handling tests
│   └── edge_cases/               # Edge case expressions
```

---

*End of Test Suite Design*
