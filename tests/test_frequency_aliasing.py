"""DAMPED_SIN/COS frequencies must be reported in the Nyquist band of the grid."""
import numpy as np

from ppf.symbolic import fold_aliased_frequencies, optimize_constants
from ppf.symbolic_types import ExprNode, NodeType, MacroOp


def _macro(op, params):
    return ExprNode(node_type=NodeType.MACRO, macro_op=op, macro_params=list(params))


def test_fold_preserves_samples_and_reports_principal_frequency():
    x = np.arange(3, 40, dtype=float)              # daily grid, non-zero origin
    for op in (MacroOp.DAMPED_COS, MacroOp.DAMPED_SIN):
        t = _macro(op, [2.0, 0.1, 2 * np.pi - 0.05, 1.3])   # alias of w = 0.05
        before = t.evaluate(x)
        fold_aliased_frequencies(t, x)
        a, k, w, phi = t.macro_params
        assert 0 <= w <= np.pi, w
        assert abs(w - 0.05) < 1e-9
        assert np.allclose(t.evaluate(x), before, atol=1e-9)


def test_fold_respects_grid_spacing_and_nonuniform_noop():
    x = np.arange(0, 10, 0.5)                      # dx = 0.5 -> Nyquist pi/0.5
    t = _macro(MacroOp.DAMPED_COS, [1.0, 0.0, 1.0 + 4 * np.pi, 0.2])
    before = t.evaluate(x)
    fold_aliased_frequencies(t, x)
    assert abs(t.macro_params[2] - 1.0) < 1e-9 and np.allclose(t.evaluate(x), before)
    xn = np.array([0.0, 1.0, 3.0, 7.0])
    t = _macro(MacroOp.DAMPED_COS, [1.0, 0.0, 7.0, 0.0])
    fold_aliased_frequencies(t, xn)
    assert t.macro_params[2] == 7.0


def test_optimize_constants_returns_folded_weekly_fit():
    x = np.arange(1, 50, dtype=float)
    y = 0.3 * np.cos(2 * np.pi / 7 * x + 0.4)
    t = _macro(MacroOp.DAMPED_COS, [0.3, 0.0, 2 * np.pi / 7 + 2 * np.pi, 0.4])  # aliased start
    t, mse = optimize_constants(t, x, y)
    assert mse < 1e-6
    assert abs(2 * np.pi / t.macro_params[2] - 7.0) < 1e-3   # reported period is 7 days
