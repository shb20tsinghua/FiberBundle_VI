"""Unit tests for the fixed-point bundle path-following solver ``FP_VI``.

The suite is self-contained (no conftest.py helpers) and covers

* validation of ``iter_init`` and of the ``operator.value_Jaco`` interface;
* the bundle Jacobian against the differential of the bundle section:
  ``d tilde{tau}_i = pi_i o sum_j J_{ij}(pi) (I - 1 pi_j^T) dtheta_j``;
* the singularity-avoidance update of the barrier mass;
* convergence on a problem with a known solution for both correctors;
* the ``gap_tol`` and ``maxnit`` stopping rules;
* exactness of the Broyden Jacobian of ``DerivFreeWrapper`` on linear operators.

The tests use the same private helper state (``_reset``/``_setup``) as
``solve`` so that the internal stages can be exercised directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from builtin_model import DerivFreeWrapper  # noqa: E402
from solver import FP_VI  # noqa: E402


# --------------------------------------------------------------------- models
class _LinearOperator:
    """Operator ``F(x) = A x.ravel() + b`` with the constant Jacobian ``A``."""

    def __init__(self, A, b):
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)

    def value(self, x):
        shape = np.shape(x)
        return (self.A @ np.ravel(x) + self.b).reshape(shape)

    def value_Jaco(self, x):
        shape = np.shape(x)
        return self.value(x), self.A.reshape(shape + shape)


class _BrokenOperator:
    """Operator whose ``value_Jaco`` violates the documented interface."""

    def __init__(self, mode):
        self.mode = mode

    def value_Jaco(self, x):
        shape = np.shape(x)
        if self.mode == "one-value":
            return np.zeros(shape)
        if self.mode == "three-values":
            return np.zeros(shape), np.zeros(shape + shape), None
        if self.mode == "wrong-value-shape":
            return np.zeros(x.size), np.zeros(shape + shape)
        if self.mode == "wrong-jacobian-shape":
            return np.zeros(shape), np.zeros((x.size, x.size))
        raise RuntimeError("the operator rejected the starting point")


# -------------------------------------------------------------------- helpers
@pytest.fixture(autouse=True)
def _seed_global_numpy_rng():
    """The singularity-avoidance step draws from the global NumPy RNG."""
    np.random.seed(0)


def _make_linear_problem(m, n, seed=0):
    """Return ``(operator, p)`` for ``F(x) = A (x - p)``; ``p`` is the solution."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n), size=m)
    A = np.zeros((m * n, m * n))
    for i in range(m):
        factor = rng.standard_normal((n, n))
        A[i * n:(i + 1) * n, i * n:(i + 1) * n] = factor @ factor.T + np.eye(n)
    return _LinearOperator(A, -A @ p.ravel()), p


def _configured_solver(operator, iter_init, gap_tol=1e-8,
                       efficient_corrector=True):
    """Initialise ``FP_VI`` exactly as ``solve`` does, without iterating."""
    solver = FP_VI(operator)
    solver._reset(gap_tol=gap_tol, efficient_corrector=efficient_corrector,
                  maxnit=10000, verbose=False, print_preci=6)
    solver._setup(np.asarray(iter_init, dtype=float))
    return solver


def _tau_tilde(operator, theta):
    """Bundle section ``tilde{tau}_i = pi_i o (F_i - (pi_i^T F_i) 1)``."""
    exponential = np.exp(theta - theta.max(axis=-1, keepdims=True))
    pi = exponential / exponential.sum(axis=-1, keepdims=True)
    values = operator.value(pi)
    return pi * (values - np.sum(pi * values, axis=-1, keepdims=True))


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "iter_init",
    [
        pytest.param(np.array([[0.4, 0.5]]), id="row-not-summing-to-one"),
        pytest.param(np.array([[0.0, 1.0]]), id="zero-entry"),
        pytest.param(np.array([[-0.1, 1.1]]), id="negative-entry"),
        pytest.param(np.array([[np.nan, 1.0]]), id="nan-entry"),
    ],
)
def test_iter_init_must_be_in_the_relative_interior(iter_init):
    """A start outside the relative interior raises a ValueError."""
    operator, _ = _make_linear_problem(1, 2)

    with pytest.raises(ValueError, match="relative interior"):
        FP_VI(operator).solve(iter_init, gap_tol=1e-8)


def test_value_jaco_interface_failure():
    """``value_Jaco`` interface failure raises a ValueError."""
    start = np.array([[0.5, 0.5]])
    for mode in ("one-value", "three-values", "wrong-value-shape",
                 "wrong-jacobian-shape", "raises"):
        with pytest.raises(ValueError, match="value_Jaco"):
            FP_VI(_BrokenOperator(mode)).solve(start, gap_tol=1e-8)


# ------------------------------------------------------- bundle Jacobian test
@pytest.mark.parametrize(("m", "n"), [(1, 2), (2, 3), (3, 4)])
def test_bundle_jacobian_matches_the_bundle_section_differential(m, n):
    """``d tilde{tau}_i = pi_i o sum_j J_{ij}(pi) (I - 1 pi_j^T) dtheta_j``.

    ``J`` is the bundle Jacobian returned by ``_evaluate_state``.
    """
    operator, p = _make_linear_problem(m, n, seed=0)
    rng = np.random.default_rng(7)
    pi = p * np.exp(0.05 * rng.standard_normal((m, n)))
    pi /= pi.sum(axis=-1, keepdims=True)
    solver = _configured_solver(operator, pi)

    *_, bundle_jacobian = solver._evaluate_state(pi)

    dtheta = rng.standard_normal((m, n))
    dtheta_tilde = dtheta - np.sum(pi * dtheta, axis=-1, keepdims=True)
    predicted = pi * (bundle_jacobian @ dtheta_tilde.ravel()).reshape(m, n)

    theta = np.log(pi)
    step = 1e-6
    tau_plus = _tau_tilde(operator, theta + step * dtheta)
    tau_minus = _tau_tilde(operator, theta - step * dtheta)
    actual = (tau_plus - tau_minus) / (2.0 * step)

    assert_allclose(predicted, actual, rtol=1e-7, atol=1e-8)


# ------------------------------------------------------- singularity avoidance
@pytest.mark.parametrize("m", [1, 2, 3])
def test_avoid_singularity_keeps_barrier_mass_above_the_gap(m):
    """After ``_avoid_singularity`` the new ``tau_isum`` exceeds ``tau_check_isum``."""
    operator, p = _make_linear_problem(m, 2, seed=0)
    solver = _configured_solver(operator, p)
    tau_isum = np.full(m, 1000.0)
    tau_check_isum = np.linspace(0.2, 0.6, m)

    tau_isum_new = solver._avoid_singularity(tau_isum, tau_check_isum)

    assert tau_isum_new.shape == tau_check_isum.shape
    assert np.all(tau_isum_new > tau_check_isum)


# --------------------------------------------------------------- convergence
def test_solver_returns_known_solution_with_both_correctors():
    """Both correctors return the known solution of a linear problem."""
    operator, p = _make_linear_problem(2, 3, seed=0)
    for efficient_corrector in (True, False):
        solution, summary = FP_VI(operator).solve(
            p, gap_tol=1e-8, efficient_corrector=efficient_corrector)
        assert solution is not None
        assert np.all(solution > 0)
        assert_allclose(solution.sum(axis=1), np.ones(2), rtol=0, atol=1e-12)
        assert_allclose(solution, p, atol=1e-6)
        assert summary[1] <= 1e-8


# ---------------------------------------------------------------- tolerances
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gap_tol_is_respected():
    """A loose and a tight gap tolerance are both honoured."""
    operator, _ = _make_linear_problem(1, 3)
    start = np.full((1, 3), 1.0 / 3.0)
    nits = []
    for gap_tol in (1e-3, 1e-6):
        solution, summary = FP_VI(operator).solve(start, gap_tol=gap_tol)
        assert solution is not None
        assert summary[1] <= gap_tol
        values = operator.value(solution)
        reduced = values - values.min(axis=-1, keepdims=True)
        gaps = np.sum(solution * reduced, axis=-1)
        assert gaps.max() <= gap_tol
        nits.append(summary[0])

    assert nits[0] <= nits[1]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("maxnit", [1, 5])
def test_maxnit_is_respected(maxnit, capsys):
    """The iteration limit stops the solver and is reported in ``summary[0]``."""
    operator, p = _make_linear_problem(1, 2, seed=0)

    solution, summary = FP_VI(operator).solve(p, gap_tol=1e-8, maxnit=maxnit)

    assert solution is None
    assert summary[0] == maxnit - 1
    assert "Maximum Iteration Number Exceeded!" in capsys.readouterr().out


# ------------------------------------------------------ DerivFreeWrapper exactness
@pytest.mark.parametrize("shape", [(1, 2), (2, 3)])
def test_deriv_free_wrapper_is_exact_for_linear_operators(shape):
    """Broyden updates keep ``B`` equal to the exact Jacobian of a linear operator."""
    rng = np.random.default_rng(4)
    dim = int(np.prod(shape))
    A = rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)
    model = _LinearOperator(A, b)
    wrapper = DerivFreeWrapper(model)

    points = [np.full(shape, 0.25)]
    points.extend(0.3 * rng.standard_normal(shape) for _ in range(5))
    for step, x in enumerate(points):
        values, jacobian = wrapper.value_Jaco(x)
        assert values.shape == shape
        assert jacobian.shape == shape + shape
        assert_allclose(jacobian.reshape(dim, dim), A, rtol=1e-8, atol=1e-8)
        assert wrapper.n_updates == step
