"""Predictor-corrector path-following solver for ``VI(F, Delta_n^m)``.

``FP_VI`` follows paths on the fixed-point bundle.
The paper's notation is kept:

* ``pi``      -- primal point, an ``(m, n)`` array with each row in ``Delta_n``;
* ``tau``     -- barrier parameter, an ``(m, n)`` array (``tau_isum`` is its
  row-sum vector);
* ``F``       -- operator value, an ``(m, n)`` array;
* ``G``       -- fixed-point bundle residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

from .find_root_numpy import find_root_isum

FloatArray = NDArray[np.float64]

# Iteration-trace layout (also printed by ``solve`` when ``verbose=True``).
_RECORD_COLUMNS = ("nit", "gap", "barrier_sum", "G_norm", "d_sign", "ratio",
                   "accept", "tang_stepln", "avoid_n")
_FLOAT_COLUMNS = frozenset(("gap", "barrier_sum", "G_norm", "ratio",
                            "tang_stepln"))
_SUMMARY_COLUMNS = ("nit", "gap", "barrier_sum", "G_norm", "avoid_n")

# Input-validation messages.
_VALIDATION_INTERIOR = (
    "iter_init must be a non-empty 2-D array in the relative interior of the "
    "product of simplices: every entry must be strictly positive and every "
    "row must sum to 1")
_VALIDATION_OPERATOR = (
    "operator.value_Jaco must accept iter_init and return (F, Jacobian) with "
    "F.shape == iter_init.shape and "
    "Jacobian.shape == iter_init.shape + iter_init.shape")

# Default scheduling constants of the path-following scheme.
_INITIAL_BARRIER_SUM = 1e3
_INITIAL_STEP_LENGTH = 1e-1
_MAX_STEP_LENGTH = 1.0
_MIN_STEP_LENGTH = 1e-5
_MAX_PREDICTOR_FRACTION = 0.1
_ACCEPTANCE_RATIO = 1.0
_RATIO_FAST = 0.1
_RATIO_INTERMEDIATE = 0.5
_RATIO_SLOW = 0.9
_STEP_GROWTH_FAST = 2.0
_STEP_GROWTH = 1.5
_STEP_GROWTH_INTERMEDIATE = 1 + 1e-8
_STEP_REDUCTION = 0.5
_AVOIDANCE_PERIOD = 300
_TAU_FLOOR = 1e-9
_SINGULARITY_INFLATION = 1 + 2e-1
_AVOIDANCE_TOLERANCE = 0.1


@dataclass
class _Record:
    """One row of the iteration trace."""

    nit: int = 0
    gap: float = 0
    barrier_sum: float = 0
    G_norm: float = 0
    d_sign: int = 0
    ratio: float = 0
    accept: bool = False
    tang_stepln: float = 0
    avoid_n: int = 0


def _format_header(print_precision: int) -> str:
    """Format the iteration-trace header."""
    width = 6 + print_precision
    cells = [f"{column:^{width if column in _FLOAT_COLUMNS else 7}}"
             for column in _RECORD_COLUMNS]
    return "|" + "|".join(cells) + "|\n"


def _format_row(record: _Record, print_precision: int) -> str:
    """Format one iteration-trace row."""
    cells = [f"{getattr(record, column):.{print_precision}e}"
             if column in _FLOAT_COLUMNS else f"{getattr(record, column):^7d}"
             for column in _RECORD_COLUMNS]
    return "|" + "|".join(cells) + "|\n"


class FP_VI:
    """Solve ``VI(F, Delta_n^m)`` by path-following on the fixed-point bundle.

    Parameters
    ----------
    operator
        Object exposing ``value_Jaco(pi) -> (F, Jacobian)``.  ``F`` has shape
        ``(m, n)`` and the Jacobian shape ``(m, n, m, n)``.
    """

    def __init__(self, operator) -> None:
        self._operator = operator
        self._record = _Record()

    # ------------------------------------------------------------------ API
    def solve(self, iter_init, gap_tol, efficient_corrector=True, maxnit=10000,
              verbose=False, print_preci=6):
        """Run the path-following iteration.

        Parameters
        ----------
        iter_init : (m, n) array
            Starting point, each row in the relative interior of ``Delta_n``.
        gap_tol : float
            Stop once ``gap_i(pi) = pi_i^T F_i(pi) - min_a F_{i,a}(pi)`` is at
            most this value for *every* block ``i``.
        efficient_corrector : bool
            ``True`` selects the corrector derived from the barrier problem
            (the faster default); ``False`` selects the corrector derived from
            the perturbed KKT conditions.
        maxnit : int
            Maximum number of predictor-corrector iterations.
        verbose : bool
            ``True`` prints the iteration trace to stdout: a header followed by
            one row per predictor-corrector iteration.  ``False`` (the default)
            prints nothing.
        print_preci : int
            Number of decimal places in the printed trace.

        Returns
        -------
        (solution, info)
            ``solution`` is the final iterate (or ``None`` if the iteration
            limit was hit); ``info`` is
            ``[nit, gap, barrier_sum, G_norm, avoid_n]``.

        Raises
        ------
        ValueError
            If ``iter_init`` is not in the relative interior of the product of
            simplices, or if ``operator.value_Jaco(iter_init)`` does not return
            an operator value and Jacobian with the required shapes.
        """
        iter_init = self._validate_input(iter_init)
        self._reset(gap_tol, efficient_corrector, maxnit, verbose, print_preci)
        self._setup(iter_init)

        if self._verbose:
            print(_format_header(self._print_precision), end="", flush=True)
        solution = self._follow_path(iter_init)
        return solution, [getattr(self._record, column) for column in _SUMMARY_COLUMNS]

    def _validate_input(self, iter_init) -> FloatArray:
        """Validate the starting point and the operator interface.

        ``iter_init`` must be a non-empty 2-D array in the relative interior of
        the product of simplices: every entry is strictly positive and every
        row sums to 1.  The operator is queried once here and must return a
        value with the shape of ``iter_init`` and a Jacobian with shape
        ``iter_init.shape + iter_init.shape``.
        """
        try:
            iter_init = np.asarray(iter_init, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(_VALIDATION_INTERIOR) from error
        if (iter_init.ndim != 2 or iter_init.size == 0
                or not np.all(iter_init > 0)
                or not np.allclose(iter_init.sum(axis=1), 1.0,
                                   rtol=1e-12, atol=1e-12)):
            raise ValueError(_VALIDATION_INTERIOR)
        try:
            value, jacobian = self._operator.value_Jaco(iter_init)
        except Exception as error:  # the operator rejected the starting point
            raise ValueError(_VALIDATION_OPERATOR) from error
        if (np.shape(value) != iter_init.shape
                or np.shape(jacobian) != iter_init.shape + iter_init.shape):
            raise ValueError(_VALIDATION_OPERATOR)
        return iter_init

    def _setup(self, iter_init) -> None:
        """Initialise the domain-dependent state for a run started at ``iter_init``."""
        self._m, self._n = iter_init.shape
        self._mn = iter_init.shape[0] * iter_init.shape[1]
        self._tau_floor = min(_TAU_FLOOR, self._gap_tol / self._mn)
        self._identity = np.eye(self._n)[None, :, :]
        self._diagonal = np.diag_indices(self._mn)
        self._ones = np.ones(self._m)
        self._index = np.arange(self._mn)

    # ------------------------------------------------------- path-following
    def _reset(self, gap_tol, efficient_corrector, maxnit, verbose,
               print_preci) -> None:
        """Store the run configuration and clear the previous trace."""
        self._maxnit = maxnit
        self._gap_tol = gap_tol
        self._efficient_corrector = efficient_corrector
        self._verbose = verbose
        self._print_precision = print_preci
        self._record = _Record()

    def _follow_path(self, pi):
        """Follow the bundle path until every block satisfies ``gap_tol``."""
        avoidance_count, ratio_accept = 0, _ACCEPTANCE_RATIO
        self._tangent_step = _INITIAL_STEP_LENGTH

        pi_acc, tau_isum_acc, tau_check_isum_acc = (
            pi, _INITIAL_BARRIER_SUM * self._ones, 1.0 * self._ones)
        for nit in range(self._maxnit):
            (pi_new, tau_isum_new, tau_check_isum_new, direction_sign, ratio,
             residual_after) = self._predictor_corrector(pi_acc, tau_isum_acc)

            if (accept := not np.isnan(ratio) and ratio < ratio_accept):
                pi_acc, tau_isum_acc, tau_check_isum_acc = (
                    pi_new, tau_isum_new, tau_check_isum_new)
                if ratio < _RATIO_FAST:
                    self._tangent_step *= _STEP_GROWTH_FAST
                elif ratio < _RATIO_INTERMEDIATE:
                    self._tangent_step *= _STEP_GROWTH
                elif ratio < _RATIO_SLOW:
                    self._tangent_step *= _STEP_GROWTH_INTERMEDIATE
                else:
                    self._tangent_step *= _STEP_REDUCTION
            else:
                self._tangent_step *= _STEP_REDUCTION
            if (self._tangent_step < _MIN_STEP_LENGTH
                    or (nit + 1) % _AVOIDANCE_PERIOD == 0):
                tau_isum_acc = self._avoid_singularity(tau_isum_acc,
                                                       tau_check_isum_acc)
                avoidance_count += 1
                self._tangent_step = _INITIAL_STEP_LENGTH

            self._record.nit = nit
            self._record.gap = tau_check_isum_acc.max()
            self._record.barrier_sum = tau_isum_acc.sum()
            self._record.G_norm = residual_after
            self._record.d_sign = int(direction_sign)
            self._record.ratio = ratio
            self._record.accept = accept
            self._record.tang_stepln = self._tangent_step
            self._record.avoid_n = avoidance_count
            if self._verbose:
                print(_format_row(self._record, self._print_precision), end="", flush=True)
            if (tau_check_isum_acc <= self._gap_tol).all():
                return pi_acc
        print("Maximum Iteration Number Exceeded!")
        return None

    def _avoid_singularity(self, tau_isum, tau_check_isum):
        """Redistribute the barrier mass to move away from a singular Jacobian."""
        tolerance = _AVOIDANCE_TOLERANCE
        weights = np.random.dirichlet(self._ones)
        tau_sum_max = _SINGULARITY_INFLATION * tau_isum.sum()
        return (weights * ((1 - tolerance) * tau_sum_max - tau_check_isum.sum())
                + tau_check_isum + tolerance / self._m * tau_sum_max)

    # ------------------------------------------------------- linear algebra
    def _evaluate_state(self, pi):
        """Evaluate ``F``, the gaps and the bundle Jacobian at ``pi``.

        The operator is queried on every call: it may be stateful (for example
        the derivative-free Broyden wrapper, which refreshes its secant Jacobian
        every few evaluations), so results must not be cached.  The Jacobian is
        returned as a fresh copy because callers add the barrier term to it in
        place.
        """
        values, jacobian = self._operator.value_Jaco(pi)
        reduced = values - values.min(axis=-1, keepdims=True)
        gaps = np.vecdot(pi, reduced, axis=-1)
        projector = self._identity - pi[:, None, :]
        bundle_jacobian = np.multiply(jacobian, pi, order="C").reshape(self._mn, self._mn)
        bundle_jacobian.reshape(-1)[:: self._mn + 1] += np.matvec(projector, values).ravel()
        bundle_jacobian_rows = bundle_jacobian.reshape(self._m, self._n, self._mn)
        bundle_jacobian_rows -= np.matmul(pi[:, None, :], bundle_jacobian_rows)
        bundle_jacobian_cols = bundle_jacobian_rows.reshape(self._mn, self._m, self._n)
        bundle_jacobian_cols -= np.einsum("rj,jb->rjb", bundle_jacobian_cols.sum(axis=2), pi)
        return values, reduced, gaps, projector, bundle_jacobian

    def _compute_corrector(self, pi, tau):
        """One regularised Newton corrector step for a fixed barrier ``tau``."""
        values, reduced, gaps, projector, bundle_jacobian = self._evaluate_state(pi)
        tau_isum = tau.sum(axis=-1)
        residual = np.matvec(projector.swapaxes(1, 2), pi * values - tau).ravel()
        residual_norm = np.sqrt(residual.dot(residual))
        bundle_jacobian[self._diagonal] += np.repeat(tau_isum, self._n)
        if self._efficient_corrector:
            if self._m > 1:
                root = find_root_isum(reduced, tau, tau_isum)[:, None]
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    root = brentq(lambda v: (tau[0] / (reduced[0] + v)).sum() - 1,
                                  0, tau_isum[0])
            root = reduced + root
            normal_matrix = (bundle_jacobian.T * (pi / root).ravel()[None, :]).dot(bundle_jacobian)
            right_hand_side = bundle_jacobian.T.dot((pi - tau / root).flatten())
        else:
            barrier_residual = np.matvec(projector, values - tau / pi).ravel()
            right_hand_side = bundle_jacobian.T.dot(barrier_residual)
            normal_matrix = bundle_jacobian.T.dot(bundle_jacobian)
        normal_matrix[self._diagonal] += min(1, residual_norm / self._mn)
        corrector = np.linalg.solve(normal_matrix, right_hand_side).reshape(self._m, self._n)
        corrector -= np.vecdot(pi, corrector)[:, None]
        return corrector, residual_norm, gaps

    def _compute_predictor(self, pi, tau_isum, enforce_decrease=False):
        """One tangent predictor step for the current barrier mass ``tau_isum``."""
        values, reduced, gaps, projector, bundle_jacobian = self._evaluate_state(pi)
        reduced_sum = reduced + (tau_isum - gaps)[:, None]
        tau = pi * reduced_sum
        bundle_jacobian[self._diagonal] += np.repeat(tau_isum, self._n)
        right_hand_side = np.matvec(projector, reduced_sum).flatten()
        if enforce_decrease:
            direction_sign = 1
            tangent = np.linalg.solve(bundle_jacobian, right_hand_side)
        else:
            lu, pivots = lu_factor(bundle_jacobian, check_finite=False)
            direction_sign = np.prod(np.sign(np.diagonal(lu))) * (-1) ** np.count_nonzero(
                pivots != self._index)
            tangent = lu_solve((lu, pivots), right_hand_side, check_finite=False)
        return tangent, tau, direction_sign

    def _predictor_corrector(self, pi, tau_isum):
        """One predictor step followed by one regularised Newton corrector step."""
        tangent, tau, direction_sign = self._compute_predictor(pi, tau_isum)
        tangent_norm = np.sqrt(1 + tangent.dot(tangent))
        eta = min(_MAX_PREDICTOR_FRACTION,
                  min(_MAX_STEP_LENGTH, self._tangent_step) / tangent_norm)
        self._tangent_step = eta * tangent_norm

        tau_new = (tau * (1 - eta * direction_sign)).clip(min=self._tau_floor)
        pi_pred = pi * np.exp(-tangent.reshape((self._m, self._n)) * eta * direction_sign)
        pi_pred /= pi_pred.sum(axis=-1, keepdims=True)

        corrector, residual_before, _ = self._compute_corrector(pi_pred, tau_new)
        pi_corrected = pi_pred * np.exp(-corrector)
        pi_corrected /= pi_corrected.sum(axis=-1, keepdims=True)

        values_c, _, gaps_c, projector_c, _ = self._evaluate_state(pi_corrected)
        residual_after = np.matvec(projector_c.swapaxes(1, 2),
                                   pi_corrected * values_c - tau_new).ravel()
        residual_after_norm = np.sqrt(residual_after.dot(residual_after))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = residual_after_norm / residual_before

        return (pi_corrected, tau_new.sum(axis=-1), gaps_c, direction_sign, ratio,
                residual_after_norm)
