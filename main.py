"""Experiment driver and command-line interface for the solver.

This module contains

* the driver that builds one instance of ``VI(F, Delta_n^m)`` with a built-in
  random operator and solves it with the path-following solver;
* the result-line formatting used by the data files;
* a small command-line interface for running one instance.

``F`` is either the neural-network operator :class:`builtin_model.nn_F` or the
low-rank payoff-tensor operator :class:`builtin_model.payoff_F`.  Both are
solved by ``FP_VI``; the simplex case ``m = 1`` is a special case of the
product domain ``Delta_n^m``.

Running ``python main.py`` without arguments solves the demo instance used in
the README: a payoff-tensor game with ``m = n = 16`` players/actions, CP rank
``R = 256`` and seed ``0``, solved with the barrier corrector.  Add ``--verbose``
to print the per-iteration trace to stdout.
"""
from __future__ import annotations

import argparse
import math
import numbers
import os
import sys
from typing import Sequence

_BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_DEFAULT_BLAS_THREADS = 1


def _parse_thread_setting(value: str) -> int | None:
    """Return an explicit BLAS thread count, or ``None`` for the library defaults."""
    if value.lower() in ("auto", "none", "default"):
        return None
    try:
        threads = int(value)
    except ValueError as error:
        raise ValueError(f"invalid thread count {value!r}") from error
    if threads < 1:
        raise ValueError(f"the thread count must be positive, got {threads}")
    return threads


def _apply_blas_threads(threads: int | None) -> None:
    """Set the BLAS/OpenMP environment variables before NumPy is imported."""
    if threads is None:
        return
    for variable in _BLAS_THREAD_VARIABLES:
        os.environ[variable] = str(threads)


def _thread_count_argument(value: str) -> int | None:
    """argparse wrapper around :func:`_parse_thread_setting`."""
    try:
        return _parse_thread_setting(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _threads_from_command_line(arguments: Sequence[str]) -> int | None:
    """Extract ``--threads`` from the raw command line (needed before import)."""
    for position, token in enumerate(arguments):
        if token == "--threads":
            if position + 1 < len(arguments):
                return _parse_thread_setting(arguments[position + 1])
            return _DEFAULT_BLAS_THREADS
        if token.startswith("--threads="):
            return _parse_thread_setting(token.split("=", 1)[1])
    return _DEFAULT_BLAS_THREADS


def _positive_int(value: str) -> int:
    """argparse type for a positive integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _positive_float(value: str) -> float:
    """argparse type for a positive finite float."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def _parse_layer_list(value: str) -> list[int]:
    """Parse the bracketed list form of ``--hidden-layers``."""
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise argparse.ArgumentTypeError(
            'hidden layers must be a bracketed list, e.g. --hidden-layers "[8, 4]"')
    inner = text[1:-1].strip()
    if not inner:
        return []
    parts = [part.strip() for part in inner.split(",")]
    if any(not part for part in parts):
        raise argparse.ArgumentTypeError("invalid hidden-layer list: empty entry")
    try:
        widths = [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid hidden-layer width in {value!r}") from error
    if any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError("hidden-layer widths must be positive integers")
    return widths


def _validated_hidden_layers(hidden_layers) -> list[int]:
    """Return ``hidden_layers`` as a list of positive ints, rejecting scalars.

    The list may be empty, which means a network with no hidden layer.
    """
    if isinstance(hidden_layers, (numbers.Integral, str)):
        raise TypeError("hidden_layers must be a list of positive integers, not a scalar")
    try:
        widths = list(hidden_layers)
    except TypeError as error:
        raise TypeError("hidden_layers must be a list of positive integers") from error
    if any(not isinstance(width, numbers.Integral) or int(width) < 1 for width in widths):
        raise ValueError("each hidden-layer width must be a positive integer")
    return [int(width) for width in widths]


if __name__ == "__main__":
    # This must run before NumPy is imported; argparse validates/reports errors later.
    try:
        _apply_blas_threads(_threads_from_command_line(sys.argv[1:]))
    except ValueError:
        pass

import numpy as np

from builtin_model import nn_F, payoff_F, DerivFreeWrapper
from solver import FP_VI


# ------------------------------------------------------------ experiment setup
PRINT_PRECISION = 6
GAP_TOLERANCE = 1e-5
MAX_ITERATIONS = 10_000

# Names of the built-in operators accepted by the command line.
MODEL_NN = "nn_F"
MODEL_PAYOFF = "payoff_F"
MODELS = (MODEL_NN, MODEL_PAYOFF)


# ------------------------------------------------------------- run instances
def run_instance(seed: int, dimension: tuple[int, int], model: str = MODEL_PAYOFF,
                 hidden_layers: Sequence[int] | None = None, rank: int | None = None,
                 verbose: bool = False, efficient_corrector: bool = True,
                 gap_tol: float = GAP_TOLERANCE,
                 derivative_free: bool = False) -> str:
    """Solve one seeded instance of ``VI(F, Delta_n^m)`` and return its result line.

    ``model`` selects the built-in operator:

    * ``"nn_F"`` builds ``nn_F(seed, [m*n, *hidden_layers, m*n], 0)``; the
      hidden layers default to ``[m*n]``;
    * ``"payoff_F"`` builds ``payoff_F(seed, m, n, rank)``; the CP rank
      defaults to ``m*n``.

    ``hidden_layers`` and ``rank`` configure different models, so passing the
    option of the other model is an error.

    With ``derivative_free=True`` the operator is wrapped in
    :class:`builtin_model.DerivFreeWrapper`, so the solver only receives
    function values plus a Broyden Jacobian estimate.  With ``verbose=True``
    the solver prints the per-iteration trace to stdout; otherwise stdout stays
    silent.

    Parameters
    ----------
    seed : int
        Seed of the operator parameters and of the starting point.
    dimension : tuple of int
        ``(m, n)``: number of players and of actions per player.
    model : {"nn_F", "payoff_F"}
        Which built-in operator to use.
    hidden_layers : sequence of int, optional
        Hidden-layer widths of the neural network (``nn_F`` only; default
        ``[m*n]``).
    rank : int, optional
        CP rank of the payoff tensor (``payoff_F`` only; default ``m*n``).
    verbose : bool
        ``True`` prints the per-iteration trace to stdout; ``False`` (the
        default) keeps stdout silent.
    efficient_corrector : bool
        ``True`` uses the barrier corrector, ``False`` the KKT corrector.
    gap_tol : float
        Required gap tolerance.
    derivative_free : bool
        Replace the analytic Jacobian by the Broyden approximation.
    """
    m, n = dimension
    if model == MODEL_NN:
        if rank is not None:
            raise ValueError("rank is only used by the payoff_F model")
        if hidden_layers is None:
            hidden_layers = [m * n]
        else:
            hidden_layers = _validated_hidden_layers(hidden_layers)
        operator = nn_F(seed, [m * n, *hidden_layers, m * n], 0)
    elif model == MODEL_PAYOFF:
        if hidden_layers is not None:
            raise ValueError("hidden_layers is not used by the payoff_F model")
        if rank is None:
            rank = m * n
        if not isinstance(rank, numbers.Integral) or int(rank) < 1:
            raise ValueError("rank must be a positive integer")
        operator = payoff_F(seed, m, n, int(rank))
    else:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    if derivative_free:
        operator = DerivFreeWrapper(operator)

    np.random.seed(seed)
    initial_point = np.random.dirichlet(np.ones(n), size=(m,))
    solution, summary = FP_VI(operator).solve(
        initial_point, gap_tol=gap_tol, efficient_corrector=efficient_corrector,
        maxnit=MAX_ITERATIONS, verbose=verbose, print_preci=PRINT_PRECISION)
    formatted = "|".join(
        f"{item:^7d}" if type(item) is int else f"{item:.{PRINT_PRECISION}e}"
        for item in summary
    )
    return f"|{seed:^5d}|{repr(solution is not None):7}|{formatted}|\n"


# ------------------------------------------------------------------ CLI
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve one instance of VI(F, Delta_n^m) with the unified "
                    "single-level solver (m = 1 is the simplex case).")
    parser.add_argument("--dimension", type=_positive_int, nargs=2, required=True,
                        metavar=("M", "N"), help="dimensions m n")
    parser.add_argument("--model", choices=MODELS, required=True,
                        help="operator model: nn_F (neural network) or payoff_F (payoff tensor)")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the instance (default 0)")
    parser.add_argument("--gap-tol", type=_positive_float, default=GAP_TOLERANCE,
                        help="required gap tolerance (default 1e-5)")
    parser.add_argument("--hidden-layers", type=_parse_layer_list, metavar="LIST",
                        help="bracketed list of hidden-layer widths, e.g. '[8, 4]' or '[]' "
                             "(nn_F only; default [m*n])")
    parser.add_argument("--rank", type=_positive_int,
                        help="CP rank of the payoff-tensor model (payoff_F only; default m*n)")
    parser.add_argument("--kkt", action="store_true",
                        help="use the KKT corrector instead of the barrier corrector")
    parser.add_argument("--derivative-free", action="store_true",
                        help="use the Broyden rank-one Jacobian wrapper (DerivFreeWrapper)")
    parser.add_argument("--threads", type=_thread_count_argument, default=_DEFAULT_BLAS_THREADS,
                        metavar="N|auto",
                        help="number of BLAS/OpenMP threads (default 1); 'auto' keeps the "
                             "library defaults")
    parser.add_argument("--verbose", action="store_true",
                        help="print the per-iteration trace to stdout (redirect stdout "
                             "to save it, e.g. > data/record.log)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        # Demo instance used in the README: 16-player, 16-action payoff game
        # with CP rank R = 256, solved with the barrier corrector.
        print(run_instance(0, (16, 16), model=MODEL_PAYOFF, rank=256))
        return

    args = _parse_args(arguments)
    if args.model == MODEL_NN and args.rank is not None:
        raise SystemExit("--rank is only used with --model payoff_F")
    if args.model == MODEL_PAYOFF and args.hidden_layers is not None:
        raise SystemExit("--hidden-layers is only used with --model nn_F")

    m, n = args.dimension
    line = run_instance(
        args.seed, (m, n), model=args.model, hidden_layers=args.hidden_layers,
        rank=args.rank, verbose=args.verbose, efficient_corrector=not args.kkt,
        gap_tol=args.gap_tol, derivative_free=args.derivative_free)
    print(line)


if __name__ == "__main__":
    main()
