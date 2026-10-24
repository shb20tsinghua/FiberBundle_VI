"""Predictor-corrector path-following solver for ``VI(F, Delta_n^m)``.

``VI_simplex_product`` treats the product of ``m`` simplices as a composite
fixed-point bundle and follows the same predictor-corrector scheme as
``vi_numpy``.  The paper's notation is kept:

* ``pi``      -- primal point, an ``(m, n)`` array with each row in ``Delta_n``;
* ``tau``     -- barrier parameter, an ``(m, n)`` array (``tau_isum`` is its
  row-sum vector);
* ``F``       -- operator value, an ``(m, n)`` array;
* ``G``       -- fixed-point bundle residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from .find_root_numpy import find_root_isum

FloatArray = NDArray[np.float64]

# Iteration-trace layout (also written to ``record_file`` by ``solve``).
_RECORD_COLUMNS = ("nit", "gap", "tau_sum", "G_norm", "d_sign", "subnit",
                   "tang_stepln", "avoid_n")
_FLOAT_COLUMNS = frozenset(("gap", "tau_sum", "G_norm", "tang_stepln"))
_SUMMARY_COLUMNS = ("nit", "gap", "tau_sum", "G_norm", "avoid_n")

# Default scheduling constants of the path-following scheme.
_INITIAL_BARRIER_SUM = 1e3
_INITIAL_STEP_LENGTH = 1e-1
_STEP_GROWTH = 1.5
_STEP_REDUCTION = 0.5
_DIRECTION_FLIP_DAMPING = 0.9
_MAX_PREDICTOR_FRACTION = 0.1
_SINGULARITY_INFLATION = 1 + 2e-2
_TAU_FLOOR = 1e-9
_SUBNIT_LOW_FRACTION = 0.3
_SUBNIT_HIGH_FRACTION = 0.6
_STEP_GROWTH_INTERMEDIATE = 1 + 1e-8
_AVOIDANCE_TOLERANCE = 0.1


@dataclass
class _Record:
    """One row of the iteration trace."""

    nit: int = 0
    gap: float = 0
    tau_sum: float = 0
    G_norm: float = 0
    d_sign: int = 0
    subnit: int = 0
    tang_stepln: float = 0
    avoid_n: int = 0


def _format_header(print_precision: int) -> str:
    width = 6 + print_precision
    cells = [f"{column:^{width if column in _FLOAT_COLUMNS else 7}}"
             for column in _RECORD_COLUMNS]
    return "|" + "|".join(cells) + "|\n"


def _format_row(record: _Record, print_precision: int) -> str:
    cells = [f"{getattr(record, column):.{print_precision}e}"
             if column in _FLOAT_COLUMNS else f"{getattr(record, column):^7d}"
             for column in _RECORD_COLUMNS]
    return "|" + "|".join(cells) + "|\n"


class VI_simplex_product:
    """Solve ``VI(F, Delta_n^m)`` by path following on the fixed-point bundle.

    Parameters
    ----------
    operator
        Object exposing ``value_Jaco(pi) -> (F, Jacobian)``.  ``F`` has shape
        ``(m, n)`` and the Jacobian shape ``(m, n, m, n)``.
    """

    def __init__(self, operator) -> None:
        self._operator = operator
        # All remaining state is (re)initialised by :meth:`solve`.
        self._record = _Record()
        self._record_io: TextIO | None = None
        self._cache_valid = False
        self._cache_point: FloatArray | None = None
        self._cache_entries = None

    # ------------------------------------------------------------------ API
    def solve(self, iter_init, gap_tol, efficient_corrector=True, maxnit=50000,
              maxsubnit=10, verbose=0, record_file=None, print_preci=6):
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
        maxsubnit : int
            Maximum number of corrector subiterations per predictor step.
        verbose : int
            ``0`` silent, ``1`` one trace line per completed corrector, ``2``
            one trace line per corrector subiteration.  Only effective when
            ``record_file`` is given.
        record_file : str or None
            Destination of the iteration trace; ``None`` disables it.
        print_preci : int
            Number of decimal places in the trace.

        Returns
        -------
        (solution, summary)
            ``solution`` is the final iterate (or ``None`` if the iteration
            limit was hit); ``summary`` is
            ``[nit, gap, tau_sum, G_norm, avoid_n]``.
        """
        self._reset(gap_tol, efficient_corrector, maxnit, maxsubnit, verbose,
                    record_file, print_preci)
        self._m, self._n = iter_init.shape
        self._mn = self._m * self._n
        self._tau_floor = min(_TAU_FLOOR, self._gap_tol / self._mn)
        self._identity = np.eye(self._n)[None, :, :]
        self._diagonal = np.diag_indices(self._mn)
        self._ones = np.ones(self._m)
        self._index = np.arange(self._mn)

        if self._verbose >= 1:
            self._record_io = open(record_file, "w")
            self._record_io.write(_format_header(self._print_precision))
        try:
            solution = self._follow_path(iter_init)
        except UserWarning as error:
            print(error)
            solution = None
        finally:
            if self._record_io is not None:
                self._record_io.close()
        return solution, [getattr(self._record, column) for column in _SUMMARY_COLUMNS]

    # ------------------------------------------------------- path following
    def _reset(self, gap_tol, efficient_corrector, maxnit, maxsubnit, verbose,
               record_file, print_preci) -> None:
        self._nit = 0
        self._maxnit = maxnit
        self._maxsubnit = maxsubnit
        self._gap_tol = gap_tol
        self._efficient_corrector = efficient_corrector
        self._verbose = verbose if record_file else 0
        self._record_file = record_file
        self._print_precision = print_preci
        self._record = _Record()
        self._record_io = None
        self._cache_valid = False
        self._cache_point = None
        self._cache_entries = None

    def _follow_path(self, pi):
        subnit_low = int(self._maxsubnit * _SUBNIT_LOW_FRACTION)
        subnit_high = int(self._maxsubnit * _SUBNIT_HIGH_FRACTION)
        self._tangent_step = _INITIAL_STEP_LENGTH
        avoidance_count = 0
        corrected, subnit, pi, tau_isum, gap, direction = self._predictor_corrector(
            pi, _INITIAL_BARRIER_SUM * self._ones)
        while True:
            if (gap <= self._gap_tol).all():
                return pi
            if corrected:
                if subnit < subnit_low:
                    adjustment = _STEP_GROWTH
                elif subnit < subnit_high:
                    adjustment = _STEP_GROWTH_INTERMEDIATE
                else:
                    adjustment = _STEP_REDUCTION
            else:
                tau_isum = self._avoid_singularity(tau_isum, gap)
                adjustment = _STEP_REDUCTION
                avoidance_count += 1
            self._record.subnit = subnit
            self._record.tang_stepln = self._tangent_step
            self._record.avoid_n = avoidance_count
            if self._verbose >= 1:
                self._write_record()
            corrected, subnit, pi, tau_isum, gap, next_direction = self._predictor_corrector(
                pi, tau_isum)
            if next_direction != direction:
                adjustment = _DIRECTION_FLIP_DAMPING
            direction = next_direction
            self._tangent_step *= adjustment

    def _avoid_singularity(self, tau_isum, gap):
        """Redistribute the barrier mass and move along the fiber to avoid the singularity."""
        weights = np.random.dirichlet(self._ones)
        total = _SINGULARITY_INFLATION * tau_isum.sum()
        return (weights * ((1 - _AVOIDANCE_TOLERANCE) * total - gap.sum())
                + gap + _AVOIDANCE_TOLERANCE / self._m * total)

    # ------------------------------------------------------- linear algebra
    def _evaluate_state(self, pi):
        """Evaluate ``F``, the gaps and the projected bundle Jacobian at ``pi``.

        The result is cached for repeated calls with the *same* array object
        (the predictor and the last corrector subiteration ask for identical
        data).  The Jacobian is returned as a fresh copy because callers add
        the barrier term to it in place.
        """
        if not (self._cache_valid and pi is self._cache_point):
            values, jacobian = self._operator.value_Jaco(pi)
            reduced = values - values.min(axis=-1, keepdims=True)
            gaps = np.vecdot(pi, reduced, axis=-1)
            projector = self._identity - pi[:, None, :]
            bundle_jacobian = np.multiply(jacobian, pi, order="C").reshape(self._mn, self._mn)
            bundle_jacobian.reshape(-1)[::self._mn + 1] += np.matvec(projector, values).ravel()
            bundle_jacobian_rows = bundle_jacobian.reshape(self._m, self._n, self._mn)
            bundle_jacobian_rows -= np.matmul(pi[:, None, :], bundle_jacobian_rows)
            bundle_jacobian_cols = bundle_jacobian_rows.reshape(self._mn, self._m, self._n)
            bundle_jacobian_cols -= np.einsum("rj,jb->rjb", bundle_jacobian_cols.sum(axis=2), pi)
            self._cache_valid = True
            self._cache_point = pi
            self._cache_entries = (values, reduced, gaps, projector, bundle_jacobian)
        values, reduced, gaps, projector, bundle_jacobian = self._cache_entries
        return values, reduced, gaps, projector, bundle_jacobian.copy()

    def _compute_corrector(self, pi, tau, corrector_tol):
        """One regularised Newton corrector step for fixed ``tau``."""
        values, reduced, gaps, projector, bundle_jacobian = self._evaluate_state(pi)
        tau_isum = tau.sum(axis=-1)
        residual = np.matvec(projector.swapaxes(1, 2), pi * values - tau).ravel()
        residual_norm = np.sqrt(residual.dot(residual))
        if residual_norm <= corrector_tol:
            return None, residual_norm, gaps
        bundle_jacobian[self._diagonal] += np.repeat(tau_isum, self._n)
        if self._efficient_corrector:
            root = reduced + find_root_isum(reduced, tau, tau_isum)[:, None]
            normal_matrix = (bundle_jacobian.T * (pi / root).ravel()[None, :]).dot(bundle_jacobian)
            right_hand_side = bundle_jacobian.T.dot((pi - tau / root).flatten())
        else:
            barrier_residual = np.matvec(projector, values - tau / pi).ravel()
            right_hand_side = bundle_jacobian.T.dot(
                barrier_residual / max(1, np.sqrt(barrier_residual.dot(barrier_residual))))
            normal_matrix = bundle_jacobian.T.dot(bundle_jacobian)
        normal_matrix[self._diagonal] += min(1, residual_norm / self._mn)
        step = np.linalg.solve(normal_matrix, right_hand_side).reshape(self._m, self._n)
        step -= np.vecdot(pi, step)[:, None]
        return step, residual_norm, gaps

    def _predictor_corrector(self, pi, tau_isum, enforce_decrease=False):
        """One predictor step followed by up to ``maxsubnit`` corrector steps."""
        values, reduced, gaps, projector, bundle_jacobian = self._evaluate_state(pi)
        cache_entries = self._cache_entries
        reduced_sum = reduced + (tau_isum - gaps)[:, None]
        tau = pi * reduced_sum

        bundle_jacobian[self._diagonal] += np.repeat(tau_isum, self._n)
        right_hand_side = np.matvec(projector, reduced_sum).flatten()
        if enforce_decrease:
            direction = 1
            tangent = np.linalg.solve(bundle_jacobian, right_hand_side)
        else:
            lu, pivots = lu_factor(bundle_jacobian, check_finite=False)
            direction = np.prod(np.sign(np.diagonal(lu))) * (-1) ** np.count_nonzero(
                pivots != self._index)
            tangent = lu_solve((lu, pivots), right_hand_side, check_finite=False)

        tangent_norm = np.sqrt(1 + tangent.dot(tangent))
        eta = min(_MAX_PREDICTOR_FRACTION, self._tangent_step / tangent_norm)
        self._tangent_step = eta * tangent_norm

        tau = (tau * (1 - eta * direction)).clip(min=self._tau_floor)
        tau_isum_new = tau.sum(axis=-1)
        self._record.tau_sum = tau_isum_new.max()
        pi_next = pi * np.exp(-tangent.reshape((self._m, self._n)) * eta * direction)
        pi_next /= pi_next.sum(axis=-1, keepdims=True)

        corrector_tol = min(_TAU_FLOOR, tau.min())
        corrected = False
        for subnit in range(self._maxsubnit):
            if self._nit > self._maxnit:
                raise UserWarning("Maximum Iteration Number Exceeded!")
            step, residual_norm, gaps = self._compute_corrector(pi_next, tau, corrector_tol)
            self._record.nit = self._nit
            self._record.gap = gaps.max()
            self._record.G_norm = residual_norm
            self._record.d_sign = int(direction)
            if self._verbose >= 2:
                self._write_record()
            corrected = step is None
            if corrected:
                break
            pi_next = pi_next * np.exp(-step)
            pi_next /= pi_next.sum(axis=-1, keepdims=True)
            if np.isnan(pi_next).any():
                break
            self._nit += 1
        if not corrected:
            self._cache_valid = True
            self._cache_point = pi
            self._cache_entries = cache_entries
            pi_next, gaps, tau_isum_new = pi, cache_entries[2], tau_isum
        return corrected, subnit, pi_next, tau_isum_new, gaps, direction

    # ------------------------------------------------------------- output
    def _write_record(self) -> None:
        self._record_io.write(_format_row(self._record, self._print_precision))
        self._record_io.flush()
