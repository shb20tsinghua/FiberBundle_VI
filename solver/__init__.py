"""Path-following solvers for variational inequalities over simplices.

``FP_VI`` solves ``VI(F, Delta_n^m)`` by a single-level predictor-corrector
path-following iteration; ``VI(F, Delta_n)`` is the special case ``m = 1``.
The operator interface is ``value_Jaco(x) -> (F, JF)``.  Pass a path as
``record_file`` to ``solve`` to record the iteration trace.
"""
from .fpvi_numpy import FP_VI

__all__ = ["FP_VI"]
