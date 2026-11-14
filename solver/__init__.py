"""Path-following solvers for variational inequalities over simplices.

``FP_VI`` solves ``VI(F, Delta_n^m)`` by a single-level predictor-corrector
path-following iteration.  The operator interface is
``value_Jaco(x) -> (F, JF)``.  Pass ``verbose=True`` to ``solve`` to print the
iteration trace.
"""
from .fpvi_numpy import FP_VI

__all__ = ["FP_VI"]
