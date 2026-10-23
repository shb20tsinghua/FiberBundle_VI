"""Predictor-corrector path-following solver for ``VI(F, Delta_n)``.

``VI_simplex`` follows the fixed-point bundle path of a variational inequality
on the probability simplex.  The paper's notation is kept:

* ``sigma``  -- primal point in ``Delta_n`` (never mutated in place);
* ``mu``     -- barrier parameter, one component per coordinate;
* ``F``      -- operator value ``F(sigma)``;
* ``G``      -- fixed-point bundle residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]

# Iteration-trace layout (also written to ``record_file`` by ``solve``).
_RECORD_COLUMNS = ("nit", "gap", "mu_sum", "G_norm", "d_sign", "subnit",
                   "tang_stepln", "avoid_n")
_FLOAT_COLUMNS = frozenset(("gap", "mu_sum", "G_norm", "tang_stepln"))
_SUMMARY_COLUMNS = ("nit", "gap", "mu_sum", "G_norm", "avoid_n")

# Default scheduling constants of the path-following scheme.
_INITIAL_BARRIER_SUM = 1e3
_INITIAL_STEP_LENGTH = 1e-1
_STEP_GROWTH = 1.5
_STEP_REDUCTION = 0.5
_DIRECTION_FLIP_DAMPING = 0.9
_MAX_PREDICTOR_FRACTION = 0.1
_SINGULARITY_INFLATION = 1 + 2e-2
_MU_FLOOR = 1e-9
_SUBNIT_LOW_FRACTION = 0.3
_SUBNIT_HIGH_FRACTION = 0.6
_STEP_GROWTH_INTERMEDIATE = 1 + 1e-8


@dataclass
class _Record:
    """One row of the iteration trace."""

    nit: int = 0
    gap: float = 0
    mu_sum: float = 0
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


class VI_simplex:
    """Solve ``VI(F, Delta_n)`` by path following on the fixed-point bundle.

    Parameters
    ----------
    operator
        Object exposing ``value_Jaco(sigma) -> (F, Jacobian)``.  ``F`` has
        shape ``(n,)`` and the Jacobian shape ``(n, n)``.
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
        iter_init : (n,) array
            Starting point in the relative interior of ``Delta_n``.
        gap_tol : float
            Stop once ``gap(sigma) = sigma^T F(sigma) - min_a F_a(sigma)`` is
            at most this value.
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
            ``[nit, gap, mu_sum, G_norm, avoid_n]``.
        """
        self._reset(gap_tol, efficient_corrector, maxnit, maxsubnit, verbose,
                    record_file, print_preci)
        self._dimension = len(iter_init)
        self._mu_floor = min(_MU_FLOOR, self._gap_tol / self._dimension)
        self._identity = np.eye(self._dimension)
        self._diagonal = np.diag_indices(self._dimension)
        self._index = np.arange(self._dimension)

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

    def _follow_path(self, sigma):
        subnit_low = int(self._maxsubnit * _SUBNIT_LOW_FRACTION)
        subnit_high = int(self._maxsubnit * _SUBNIT_HIGH_FRACTION)
        self._tangent_step = _INITIAL_STEP_LENGTH
        avoidance_count = 0
        corrected, subnit, sigma, mu_sum, gap, direction = self._predictor_corrector(
            sigma, _INITIAL_BARRIER_SUM)
        while True:
            if gap <= self._gap_tol:
                return sigma
            if corrected:
                if subnit < subnit_low:
                    adjustment = _STEP_GROWTH
                elif subnit < subnit_high:
                    adjustment = _STEP_GROWTH_INTERMEDIATE
                else:
                    adjustment = _STEP_REDUCTION
            else:
                mu_sum = self._avoid_singularity(mu_sum, gap)
                adjustment = _STEP_REDUCTION
                avoidance_count += 1
            self._record.subnit = subnit
            self._record.tang_stepln = self._tangent_step
            self._record.avoid_n = avoidance_count
            if self._verbose >= 1:
                self._write_record()
            corrected, subnit, sigma, mu_sum, gap, next_direction = self._predictor_corrector(
                sigma, mu_sum)
            if next_direction != direction:
                adjustment = _DIRECTION_FLIP_DAMPING
            direction = next_direction
            self._tangent_step *= adjustment

    def _avoid_singularity(self, mu_sum, gap):
        """Move along the fiber to avoid the singularity."""
        return _SINGULARITY_INFLATION * mu_sum

    # ------------------------------------------------------- linear algebra
    def _evaluate_state(self, sigma):
        """Evaluate ``F``, the gap and the projected bundle Jacobian at ``sigma``.

        The result is cached for repeated calls with the *same* array object
        (the predictor and the last corrector subiteration ask for identical
        data).  The Jacobian is returned as a fresh copy because callers add
        the barrier term to it in place.
        """
        if not (self._cache_valid and sigma is self._cache_point):
            values, jacobian = self._operator.value_Jaco(sigma)
            reduced = values - values.min()
            gap = sigma.dot(reduced)
            projector = self._identity - sigma[None, :]
            bundle_jacobian = np.multiply(jacobian, sigma[None, :], order="C")
            bundle_jacobian[self._diagonal] += projector.dot(values)
            bundle_jacobian -= sigma.dot(bundle_jacobian)
            bundle_jacobian -= np.outer(bundle_jacobian.sum(axis=1), sigma)
            self._cache_valid = True
            self._cache_point = sigma
            self._cache_entries = (values, reduced, gap, projector, bundle_jacobian)
        values, reduced, gap, projector, bundle_jacobian = self._cache_entries
        return values, reduced, gap, projector, bundle_jacobian.copy()

    def _compute_corrector(self, sigma, mu):
        """One regularised Newton corrector step for fixed ``mu``."""
        values, reduced, gap, projector, bundle_jacobian = self._evaluate_state(sigma)
        residual = projector.T.dot(sigma * values - mu)
        residual_norm = np.sqrt(residual.dot(residual))
        if residual_norm <= self._mu_floor:
            return None, residual_norm, gap
        mu_sum = mu.sum()
        bundle_jacobian[self._diagonal] += mu_sum
        if self._efficient_corrector:
            # The bracketing root finder evaluates the barrier equation at both
            # ends of [0, mu_sum]; at v = 0 a component with reduced = 0 gives
            # a benign division by zero.
            with np.errstate(divide="ignore", invalid="ignore"):
                root = reduced + brentq(lambda v: (mu / (reduced + v)).sum() - 1,
                                        0, mu_sum)
            normal_matrix = (bundle_jacobian.T * (sigma / root)[None, :]).dot(bundle_jacobian)
            right_hand_side = bundle_jacobian.T.dot(sigma - mu / root)
        else:
            barrier_residual = projector.dot(values - mu / sigma)
            right_hand_side = bundle_jacobian.T.dot(
                barrier_residual / max(1, np.sqrt(barrier_residual.dot(barrier_residual))))
            normal_matrix = bundle_jacobian.T.dot(bundle_jacobian)
        normal_matrix[self._diagonal] += min(1, residual_norm / self._dimension)
        step = np.linalg.solve(normal_matrix, right_hand_side)
        step -= sigma.dot(step)
        return step, residual_norm, gap

    def _predictor_corrector(self, sigma, mu_sum, enforce_decrease=False):
        """One predictor step followed by up to ``maxsubnit`` corrector steps."""
        values, reduced, gap, projector, bundle_jacobian = self._evaluate_state(sigma)
        cache_entries = self._cache_entries
        reduced_sum = reduced + mu_sum - gap
        mu = sigma * reduced_sum

        bundle_jacobian[self._diagonal] += mu_sum
        right_hand_side = projector.dot(reduced_sum)
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

        mu = (mu * (1 - eta * direction)).clip(min=self._mu_floor)
        mu_sum_new = mu.sum()
        self._record.mu_sum = mu_sum_new
        sigma_next = sigma * np.exp(-tangent * (eta * direction))
        sigma_next /= sigma_next.sum()

        corrected = False
        for subnit in range(self._maxsubnit):
            if self._nit > self._maxnit:
                raise UserWarning("Maximum Iteration Number Exceeded!")
            step, residual_norm, gap = self._compute_corrector(sigma_next, mu)
            self._record.nit = self._nit
            self._record.gap = gap
            self._record.G_norm = residual_norm
            self._record.d_sign = int(direction)
            if self._verbose >= 2:
                self._write_record()
            corrected = residual_norm <= self._mu_floor
            if corrected:
                break
            sigma_next = sigma_next * np.exp(-step)
            sigma_next /= sigma_next.sum()
            if np.isnan(sigma_next).any():
                break
            self._nit += 1
        if not corrected:
            self._cache_valid = True
            self._cache_point = sigma
            self._cache_entries = cache_entries
            sigma_next, gap, mu_sum_new = sigma, cache_entries[2], mu_sum
        return corrected, subnit, sigma_next, mu_sum_new, gap, direction

    # ------------------------------------------------------------- output
    def _write_record(self) -> None:
        self._record_io.write(_format_row(self._record, self._print_precision))
        self._record_io.flush()
