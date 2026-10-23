"""Path-following solvers for variational inequalities over simplices.

``VI_simplex`` solves ``VI(F, Delta_n)``; ``VI_simplex_product`` solves
``VI(F, Delta_n^m)``.  Both use the predictor-corrector fixed-point bundle
method and share the ``value_Jaco(x) -> (F, JF)`` operator interface.
"""
from .game_numpy import VI_simplex_product
from .vi_numpy import VI_simplex

__all__ = ["VI_simplex", "VI_simplex_product"]
