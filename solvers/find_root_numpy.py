"""Bit-identical reimplementation of ``scipy.optimize.elementwise.find_root``.

The efficient (barrier-problem) corrector of ``game_numpy`` needs the scalar
``v`` that solves ``sum_i tau[j, i] / (r_difference[j, i] + v) = 1`` for every
simplex block ``j``.  This module reproduces the SciPy routine exactly (same
algorithm, operations and tolerances), so the roots are bit-identical. It is
used only because it makes the solver about twice as fast as calling
``scipy.optimize.elementwise.find_root``.  See the SciPy documentation of
``scipy.optimize.elementwise.find_root`` for the algorithm and tolerances.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_FLOAT_INFO = np.finfo(float)
_FATOL = _FLOAT_INFO.tiny
_XATOL = 4 * _FLOAT_INFO.tiny
_XRTOL = 4 * _FLOAT_INFO.eps
_MAX_ITERATIONS = int(np.log2(_FLOAT_INFO.max) - np.log2(_FLOAT_INFO.smallest_normal))
_INITIAL_PARAMETER = 0.5


def _barrier_equation(v: FloatArray, r_difference: FloatArray, tau: FloatArray) -> FloatArray:
    """Return ``f_j(v) = sum_i tau[j, i] / (r_difference[j, i] + v) - 1``."""
    return (tau / (r_difference + v[:, None])).sum(axis=-1) - 1


def _bracket_update(x1, f1, x2, f2):
    """SciPy's convergence check for the active brackets."""
    abs_f1 = np.abs(f1)
    abs_f2 = np.abs(f2)
    best = np.where(abs_f1 < abs_f2, x1, x2)
    stop = np.minimum(abs_f1, abs_f2) <= _FATOL

    same_sign = (np.sign(f1) == np.sign(f2)) & ~stop
    if same_sign.any():
        best[same_sign] = np.nan
        stop |= same_sign

    invalid = (~(np.isfinite(x1) & np.isfinite(x2)) | (np.isnan(f1) & np.isnan(f2))) & ~stop
    if invalid.any():
        best[invalid] = np.nan
        stop |= invalid

    width = np.abs(x2 - x1)
    tolerance = np.abs(best) * _XRTOL + _XATOL
    stop |= width < tolerance
    return stop, best, width, tolerance


def find_root_isum(r_difference: FloatArray, tau: FloatArray, tau_isum: FloatArray) -> FloatArray:
    """Solve ``sum_i tau[j, i] / (r_difference[j, i] + v) = 1`` for every row ``j``.

    ``r_difference`` and ``tau`` have shape ``(m, n)``; ``tau_isum`` is the
    ``(m,)`` right end of the bracket.  Returns the ``(m,)`` roots.  The result
    is bit-identical to ``scipy.optimize.elementwise.find_root`` used on the
    same scalar equation.
    """
    # The equation is evaluated at both bracket ends, where it can be infinite;
    # this is expected and handled by the bracketing logic.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _find_root_isum(r_difference, tau, tau_isum)


def _find_root_isum(r_difference: FloatArray, tau: FloatArray, tau_isum: FloatArray) -> FloatArray:
    """Implementation of :func:`find_root_isum` (see there for the contract)."""
    count = r_difference.shape[0]
    active = np.arange(count)
    r, t = r_difference, tau

    x1 = np.zeros(count)
    x2 = tau_isum.copy()
    f1 = _barrier_equation(x1, r, t)
    f2 = _barrier_equation(x2, r, t)
    x3 = np.empty(count)
    f3 = np.empty(count)
    roots = np.full(count, np.nan)
    parameter = _INITIAL_PARAMETER
    iteration = 0

    stop, best, width, tolerance = _bracket_update(x1, f1, x2, f2)
    if stop.any():
        roots[active[stop]] = best[stop]
        keep = ~stop
        active, r, t = active[keep], r[keep], t[keep]
        x1, f1, x2, f2 = x1[keep], f1[keep], x2[keep], f2[keep]
        width, tolerance = width[keep], tolerance[keep]

    while active.size and iteration < _MAX_ITERATIONS:
        x = x1 + parameter * (x2 - x1)
        f = _barrier_equation(x, r, t)

        left = np.sign(f) == np.sign(f1)
        x3 = np.where(left, x1, x2)
        f3 = np.where(left, f1, f2)
        right = ~left
        x2 = np.where(right, x1, x2)
        f2 = np.where(right, f1, f2)
        x1 = x
        f1 = f
        iteration += 1

        stop, best, width, tolerance = _bracket_update(x1, f1, x2, f2)
        if stop.any():
            roots[active[stop]] = best[stop]
            keep = ~stop
            active, r, t = active[keep], r[keep], t[keep]
            x1, f1, x2, f2 = x1[keep], f1[keep], x2[keep], f2[keep]
            x3, f3, width, tolerance = x3[keep], f3[keep], width[keep], tolerance[keep]
        if not active.size:
            break

        # Trial parameter, with a bisection fallback.
        ratio = (x1 - x2) / (x3 - x2)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = (f1 - f2) / (f3 - f2)
        interpolate = ((1 - np.sqrt(1 - ratio)) < phi) & (phi < np.sqrt(ratio))
        parameter = np.full(active.size, 0.5)
        if interpolate.any():
            alpha = (x3 - x1) / (x2 - x1)
            f1_i, f2_i, f3_i, alpha_i = f1[interpolate], f2[interpolate], f3[interpolate], alpha[interpolate]
            parameter[interpolate] = (
                f1_i / (f1_i - f2_i) * f3_i / (f3_i - f2_i)
                - alpha_i * f1_i / (f3_i - f1_i) * f2_i / (f2_i - f3_i)
            )
        parameter = np.clip(parameter, 0.5 * tolerance / width, 1 - 0.5 * tolerance / width)

    if active.size:
        roots[active] = best
    return roots
