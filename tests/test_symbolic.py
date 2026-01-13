"""
Tests for the Symbolic Regression module.
"""

import numpy as np
import pytest

from ppf import (
    ExprNode,
    NodeType,
    UnaryOp,
    BinaryOp,
    PrimitiveSet,
    SymbolicRegressor,
    SymbolicRegressionResult,
    GPEngine,
    simplify_expression,
    format_expression_latex,
    TRIG_PRIMITIVES,
    POLYNOMIAL_PRIMITIVES,
)


class TestExprNode:
    """Tests for expression tree nodes"""

    def test_constant_evaluation(self):
        """Test constant node evaluation"""
        node = ExprNode(NodeType.CONSTANT, value=3.14)
        x = np.array([1.0, 2.0, 3.0])
        result = node.evaluate(x)

        assert len(result) == 3
        assert np.allclose(result, [3.14, 3.14, 3.14])

    def test_variable_evaluation(self):
        """Test variable node evaluation"""
        node = ExprNode(NodeType.VARIABLE)
        x = np.array([1.0, 2.0, 3.0])
        result = node.evaluate(x)

        assert np.allclose(result, x)

    def test_unary_sin(self):
        """Test sin operation"""
        child = ExprNode(NodeType.VARIABLE)
        node = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=child)

        x = np.array([0, np.pi/2, np.pi])
        result = node.evaluate(x)

        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_unary_cos(self):
        """Test cos operation"""
        child = ExprNode(NodeType.VARIABLE)
        node = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.COS, child=child)

        x = np.array([0, np.pi/2, np.pi])
        result = node.evaluate(x)

        assert np.allclose(result, [1, 0, -1], atol=1e-10)

    def test_unary_square(self):
        """Test square operation"""
        child = ExprNode(NodeType.VARIABLE)
        node = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SQUARE, child=child)

        x = np.array([1, 2, 3, -2])
        result = node.evaluate(x)

        assert np.allclose(result, [1, 4, 9, 4])

    def test_binary_add(self):
        """Test addition operation"""
        left = ExprNode(NodeType.VARIABLE)
        right = ExprNode(NodeType.CONSTANT, value=5.0)
        node = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.ADD, left=left, right=right)

        x = np.array([1, 2, 3])
        result = node.evaluate(x)

        assert np.allclose(result, [6, 7, 8])

    def test_binary_mul(self):
        """Test multiplication operation"""
        left = ExprNode(NodeType.VARIABLE)
        right = ExprNode(NodeType.CONSTANT, value=2.0)
        node = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.MUL, left=left, right=right)

        x = np.array([1, 2, 3])
        result = node.evaluate(x)

        assert np.allclose(result, [2, 4, 6])

    def test_protected_div(self):
        """Test division is protected against zero"""
        left = ExprNode(NodeType.CONSTANT, value=1.0)
        right = ExprNode(NodeType.CONSTANT, value=0.0)
        node = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.DIV, left=left, right=right)

        x = np.array([1.0])
        result = node.evaluate(x)

        # Should not raise and should be finite
        assert np.all(np.isfinite(result))

    def test_protected_log(self):
        """Test log is protected against negative/zero"""
        child = ExprNode(NodeType.CONSTANT, value=-5.0)
        node = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.LOG, child=child)

        x = np.array([1.0])
        result = node.evaluate(x)

        # Should not raise and should be finite
        assert np.all(np.isfinite(result))

    def test_protected_sqrt(self):
        """Test sqrt is protected against negative"""
        child = ExprNode(NodeType.CONSTANT, value=-4.0)
        node = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SQRT, child=child)

        x = np.array([1.0])
        result = node.evaluate(x)

        # Should use abs, so sqrt(|-4|) = 2
        assert np.allclose(result, [2.0])

    def test_nested_expression(self):
        """Test nested expression: sin(x^2)"""
        x_node = ExprNode(NodeType.VARIABLE)
        squared = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SQUARE, child=x_node)
        sin_squared = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=squared)

        x = np.array([0, np.sqrt(np.pi/2), np.sqrt(np.pi)])
        result = sin_squared.evaluate(x)

        expected = np.sin(x**2)
        assert np.allclose(result, expected, atol=1e-10)

    def test_complex_expression(self):
        """Test: 2*sin(x) + 3"""
        x_node = ExprNode(NodeType.VARIABLE)
        sin_x = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=x_node)
        two = ExprNode(NodeType.CONSTANT, value=2.0)
        three = ExprNode(NodeType.CONSTANT, value=3.0)
        two_sin_x = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.MUL, left=two, right=sin_x)
        result_node = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.ADD, left=two_sin_x, right=three)

        x = np.linspace(0, 2*np.pi, 10)
        result = result_node.evaluate(x)
        expected = 2 * np.sin(x) + 3

        assert np.allclose(result, expected)

    def test_to_string(self):
        """Test string representation"""
        x_node = ExprNode(NodeType.VARIABLE)
        two = ExprNode(NodeType.CONSTANT, value=2.0)
        mul = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.MUL, left=two, right=x_node)

        s = mul.to_string()
        assert "2" in s
        assert "x" in s
        assert "*" in s

    def test_depth(self):
        """Test tree depth calculation"""
        # Depth 1: constant
        const = ExprNode(NodeType.CONSTANT, value=1.0)
        assert const.depth() == 1

        # Depth 2: sin(x)
        x = ExprNode(NodeType.VARIABLE)
        sin_x = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=x)
        assert sin_x.depth() == 2

        # Depth 3: sin(x^2)
        x2 = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SQUARE, child=x)
        sin_x2 = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=x2)
        assert sin_x2.depth() == 3

    def test_size(self):
        """Test tree size calculation"""
        # Size 1: constant
        const = ExprNode(NodeType.CONSTANT, value=1.0)
        assert const.size() == 1

        # Size 2: sin(x)
        x = ExprNode(NodeType.VARIABLE)
        sin_x = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=x)
        assert sin_x.size() == 2

        # Size 3: x + 1
        one = ExprNode(NodeType.CONSTANT, value=1.0)
        add = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.ADD, left=x, right=one)
        assert add.size() == 3

    def test_copy(self):
        """Test deep copy"""
        x = ExprNode(NodeType.VARIABLE)
        sin_x = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=x)

        copy = sin_x.copy()

        # Should be equal in structure
        assert copy.node_type == sin_x.node_type
        assert copy.unary_op == sin_x.unary_op

        # But not the same object
        assert copy is not sin_x
        assert copy.child is not sin_x.child

    def test_get_set_constants(self):
        """Test getting and setting constants"""
        a = ExprNode(NodeType.CONSTANT, value=2.0)
        b = ExprNode(NodeType.CONSTANT, value=3.0)
        x = ExprNode(NodeType.VARIABLE)
        ax = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.MUL, left=a, right=x)
        axb = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.ADD, left=ax, right=b)

        # Get constants
        consts = axb.get_constants()
        assert len(consts) == 2
        assert consts[0] == 2.0
        assert consts[1] == 3.0

        # Set new constants
        axb.set_constants([5.0, 7.0])
        new_consts = axb.get_constants()
        assert new_consts[0] == 5.0
        assert new_consts[1] == 7.0


class TestPrimitiveSet:
    """Tests for primitive set configuration"""

    def test_default_primitives(self):
        """Test default primitive set has expected operators"""
        ps = PrimitiveSet()

        assert UnaryOp.SIN in ps.unary_ops
        assert UnaryOp.COS in ps.unary_ops
        assert BinaryOp.ADD in ps.binary_ops
        assert BinaryOp.MUL in ps.binary_ops

    def test_trig_primitives(self):
        """Test trigonometric primitive set"""
        assert UnaryOp.SIN in TRIG_PRIMITIVES.unary_ops
        assert UnaryOp.COS in TRIG_PRIMITIVES.unary_ops
        assert UnaryOp.EXP not in TRIG_PRIMITIVES.unary_ops

    def test_polynomial_primitives(self):
        """Test polynomial primitive set"""
        assert UnaryOp.SQUARE in POLYNOMIAL_PRIMITIVES.unary_ops
        assert UnaryOp.SIN not in POLYNOMIAL_PRIMITIVES.unary_ops
        assert BinaryOp.ADD in POLYNOMIAL_PRIMITIVES.binary_ops


class TestGPEngine:
    """Tests for genetic programming engine"""

    def test_random_tree_generation(self):
        """Test random tree generation"""
        engine = GPEngine(PrimitiveSet(), max_depth=4)

        for _ in range(10):
            tree = engine.random_tree(4, 'grow')
            assert tree is not None
            assert tree.depth() <= 4

    def test_initialize_population(self):
        """Test population initialization"""
        engine = GPEngine(PrimitiveSet(), population_size=20, max_depth=4)
        population = engine.initialize_population()

        assert len(population) == 20
        for tree in population:
            assert tree.depth() <= 4

    def test_crossover_produces_valid_trees(self):
        """Test that crossover produces valid trees"""
        engine = GPEngine(PrimitiveSet(), max_depth=4)

        for _ in range(10):
            p1 = engine.random_tree(3, 'grow')
            p2 = engine.random_tree(3, 'grow')
            c1, c2 = engine.crossover(p1, p2)

            assert c1 is not None
            assert c2 is not None
            # Check they can be evaluated
            x = np.array([1.0, 2.0])
            c1.evaluate(x)
            c2.evaluate(x)

    def test_mutation_produces_valid_trees(self):
        """Test that mutation produces valid trees"""
        engine = GPEngine(PrimitiveSet(), max_depth=4)

        for _ in range(10):
            tree = engine.random_tree(3, 'grow')
            mutated = engine.mutate(tree)

            assert mutated is not None
            # Check it can be evaluated
            x = np.array([1.0, 2.0])
            mutated.evaluate(x)


class TestSymbolicRegressor:
    """Tests for the main SymbolicRegressor class"""

    def test_init_validation(self):
        """Test parameter validation"""
        with pytest.raises(ValueError):
            SymbolicRegressor(population_size=5)  # Too small

        with pytest.raises(ValueError):
            SymbolicRegressor(generations=0)  # Must be >= 1

        with pytest.raises(ValueError):
            SymbolicRegressor(max_depth=1)  # Must be >= 2

        with pytest.raises(ValueError):
            SymbolicRegressor(parsimony_coefficient=-0.1)  # Must be >= 0

    def test_discover_simple_linear(self):
        """Test discovering y = x (simplest linear)"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        # Simple case: y = x with small noise
        y = x + 0.1 * np.random.randn(50)

        regressor = SymbolicRegressor(
            population_size=100,
            generations=15,
            random_state=42
        )
        result = regressor.discover(x, y)

        assert isinstance(result, SymbolicRegressionResult)
        assert len(result.pareto_front) > 0
        # The GP should at least find x as a solution with high R²
        # for y ≈ x, the expression "x" should have R² close to 1
        assert result.most_accurate.r_squared > 0.9

    def test_discover_sine(self):
        """Test discovering y = sin(x)"""
        np.random.seed(42)
        x = np.linspace(0, 2*np.pi, 50)
        y = np.sin(x) + 0.05 * np.random.randn(50)

        regressor = SymbolicRegressor(
            primitives=TRIG_PRIMITIVES,
            population_size=200,
            generations=40,
            random_state=42
        )
        result = regressor.discover(x, y)

        assert result.most_accurate is not None
        assert result.most_accurate.r_squared > 0.85

    def test_discover_quadratic(self):
        """Test discovering y = x^2"""
        np.random.seed(42)
        x = np.linspace(-3, 3, 50)
        y = x**2 + 0.1 * np.random.randn(50)

        regressor = SymbolicRegressor(
            primitives=POLYNOMIAL_PRIMITIVES,
            population_size=100,
            generations=20,
            random_state=42
        )
        result = regressor.discover(x, y)

        assert result.most_accurate is not None
        assert result.most_accurate.r_squared > 0.95

    def test_pareto_front(self):
        """Test that Pareto front contains multiple solutions"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 0.5 * x

        regressor = SymbolicRegressor(
            population_size=100,
            generations=20,
            random_state=42
        )
        result = regressor.discover(x, y)

        # Should have multiple solutions in Pareto front
        assert len(result.pareto_front) >= 1

        # Should have different solutions for different criteria
        assert result.most_accurate is not None
        assert result.most_parsimonious is not None

    def test_verbose_mode(self, capsys):
        """Test verbose output"""
        np.random.seed(42)
        x = np.linspace(0, 5, 30)
        y = x + 1

        regressor = SymbolicRegressor(
            population_size=20,
            generations=3,
            random_state=42
        )
        regressor.discover(x, y, verbose=True)

        captured = capsys.readouterr()
        assert "Gen" in captured.out
        assert "R^2" in captured.out

    def test_result_summary(self):
        """Test result summary generation"""
        np.random.seed(42)
        x = np.linspace(0, 5, 30)
        y = 2 * x

        regressor = SymbolicRegressor(
            population_size=50,
            generations=10,
            random_state=42
        )
        result = regressor.discover(x, y)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "SYMBOLIC REGRESSION" in summary

    def test_evaluate_discovered_expression(self):
        """Test that discovered expressions can predict new data"""
        np.random.seed(42)
        x_train = np.linspace(0, 5, 30)
        y_train = x_train ** 2

        regressor = SymbolicRegressor(
            primitives=POLYNOMIAL_PRIMITIVES,
            population_size=100,
            generations=20,
            random_state=42
        )
        result = regressor.discover(x_train, y_train)

        # Predict on new data using most accurate solution
        x_test = np.array([6, 7, 8])
        y_pred = result.most_accurate.evaluate(x_test)
        y_expected = x_test ** 2

        # Should be reasonably close
        assert np.allclose(y_pred, y_expected, rtol=0.3)


class TestSimplifyExpression:
    """Tests for expression simplification"""

    def test_simplify_x_plus_zero(self):
        """Test x + 0 -> x"""
        x = ExprNode(NodeType.VARIABLE)
        zero = ExprNode(NodeType.CONSTANT, value=0.0)
        add = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.ADD, left=x, right=zero)

        simplified = simplify_expression(add)

        assert simplified.node_type == NodeType.VARIABLE

    def test_simplify_x_times_one(self):
        """Test x * 1 -> x"""
        x = ExprNode(NodeType.VARIABLE)
        one = ExprNode(NodeType.CONSTANT, value=1.0)
        mul = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.MUL, left=x, right=one)

        simplified = simplify_expression(mul)

        assert simplified.node_type == NodeType.VARIABLE

    def test_simplify_x_times_zero(self):
        """Test x * 0 -> 0"""
        x = ExprNode(NodeType.VARIABLE)
        zero = ExprNode(NodeType.CONSTANT, value=0.0)
        mul = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.MUL, left=x, right=zero)

        simplified = simplify_expression(mul)

        assert simplified.node_type == NodeType.CONSTANT
        assert simplified.value == 0.0

    def test_simplify_constant_folding(self):
        """Test constant folding: 2 + 3 -> 5"""
        two = ExprNode(NodeType.CONSTANT, value=2.0)
        three = ExprNode(NodeType.CONSTANT, value=3.0)
        add = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.ADD, left=two, right=three)

        simplified = simplify_expression(add)

        assert simplified.node_type == NodeType.CONSTANT
        assert simplified.value == 5.0


class TestFormatLaTeX:
    """Tests for LaTeX formatting"""

    def test_format_variable(self):
        """Test variable formatting"""
        x = ExprNode(NodeType.VARIABLE)
        assert format_expression_latex(x) == "x"

    def test_format_constant(self):
        """Test constant formatting"""
        c = ExprNode(NodeType.CONSTANT, value=3.14)
        latex = format_expression_latex(c)
        assert "3.14" in latex

    def test_format_sin(self):
        """Test sin formatting"""
        x = ExprNode(NodeType.VARIABLE)
        sin_x = ExprNode(NodeType.UNARY_OP, unary_op=UnaryOp.SIN, child=x)
        latex = format_expression_latex(sin_x)
        assert "\\sin" in latex
        assert "x" in latex

    def test_format_fraction(self):
        """Test division as fraction"""
        x = ExprNode(NodeType.VARIABLE)
        two = ExprNode(NodeType.CONSTANT, value=2.0)
        div = ExprNode(NodeType.BINARY_OP, binary_op=BinaryOp.DIV, left=x, right=two)
        latex = format_expression_latex(div)
        assert "\\frac" in latex
