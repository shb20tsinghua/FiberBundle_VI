"""Experiment driver for the fixed-point bundle solvers.

This module contains

* the two random operator models used in the papers
  (:class:`NeuralNetworkOperator`, :class:`PayoffTensorOperator`);
* driver that solve a instance and format its result line;
* a small command-line interface for running one instance.

Running ``python main.py`` without arguments solves the demo instance used in
the README.

"""
from __future__ import annotations

import argparse
import numbers
import os
import sys
from typing import Iterable, Sequence

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
from numpy.typing import NDArray

from solvers import VI_simplex, VI_simplex_product

np.seterr(divide="ignore", invalid="ignore", over="ignore")

FloatArray = NDArray[np.float64]

# --------------------------------------------------------------------- paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RECORD_FILE = os.path.join(DATA_DIR, "record.log")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------ experiment setup
PRINT_PRECISION = 6
GAP_TOLERANCE = 1e-5
MAX_ITERATIONS = 50_000
MAX_SUBITERATIONS = 10

# Rescaled Xavier normal initialisation: standard deviation sqrt(2/(fan_in+fan_out)).
_WEIGHT_INIT_SCALE = lambda fan_in, fan_out: 1 / np.sqrt((fan_in + fan_out) / 2)
_BIAS_INIT_SCALE = 1e-2

ACTIVATIONS = (np.tanh, lambda x: 1 / (1 + np.exp(-x)), lambda x: x)
ACTIVATION_GRADIENTS = (lambda h: 1 - h ** 2, lambda h: h * (1 - h), lambda h: np.ones_like(h))


class NeuralNetworkOperator:
    """Tanh/sigmoid/linear fully connected network used as a random operator ``F``.

    Parameters
    ----------
    seed : int
        Seed of the random parameter draw.
    layer_widths : sequence of int
        Layer widths ``[dim, *hidden, dim]``.
    activation : int
        ``0`` tanh, ``1`` sigmoid, ``2`` linear.

    Notes
    -----
    Weights use the Xavier normal distribution, which keeps the tanh
    activations out of saturation and therefore keeps ``F`` complex.  Biases
    are drawn from ``N(0, 1e-2^2)``.  The model can be used on ``Delta_n``
    (``dim = n``) and on ``Delta_n^m`` (``dim = m*n``).
    """

    def __init__(self, seed: int, layer_widths: Sequence[int], activation: int) -> None:
        np.random.seed(seed)
        widths = list(layer_widths)
        self._weights = [
            np.random.normal(loc=0.0, scale=_WEIGHT_INIT_SCALE(widths[i], widths[i + 1]),
                             size=(widths[i + 1], widths[i]))
            for i in range(len(widths) - 1)
        ]
        self._biases = [
            np.random.normal(loc=0.0, scale=_BIAS_INIT_SCALE, size=widths[i + 1])
            for i in range(len(widths) - 1)
        ]
        self._activation = ACTIVATIONS[activation]
        self._activation_gradient = ACTIVATION_GRADIENTS[activation]

    def value_Jaco(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``F(x)`` and its Jacobian, both with the shape of ``x``."""
        hidden = x.ravel()
        forward_cache = [hidden]
        for weight, bias in zip(self._weights, self._biases):
            hidden = self._activation(np.dot(weight, hidden) + bias)
            forward_cache.append(hidden)
        jacobian = self._activation_gradient(forward_cache[-1])[:, None] * self._weights[-1]
        for i in reversed(range(len(self._weights) - 1)):
            jacobian *= self._activation_gradient(forward_cache[i + 1])[None, :]
            jacobian = np.dot(jacobian, self._weights[i])
        return hidden.reshape(x.shape), jacobian.reshape(x.shape + x.shape)


class PayoffTensorOperator:
    """Low-rank CP payoff operator of a finite normal-form game.

    For ``m`` players and ``n`` actions the payoff function is

        f_i(pi) = sum_{r=1}^{R} prod_{j=1}^{m} (U_{r,i,j} pi_j),

    and the operator is ``F_i(pi) = grad_{pi_i} f_i(pi)``.  The tensor entries
    are drawn from ``N(1, 1/m)``; ``R = m*n`` is the rank used in the experiment.
    This model is defined on the product domain ``Delta_n^m`` only.
    """

    def __init__(self, seed: int, players: int, actions: int, rank: int) -> None:
        np.random.seed(seed)
        self._payoff = np.random.normal(loc=1.0, scale=1 / np.sqrt(players),
                                        size=(rank, players, actions, players))
        self._players = players
        self._actions = actions
        self._payoff_t = self._payoff.transpose(0, 1, 3, 2)
        self._diagonal = self._payoff.diagonal(axis1=1, axis2=3).transpose(0, 2, 1)
        self._diagonal_t = np.ascontiguousarray(self._diagonal.transpose(1, 2, 0))
        self._payoff_permuted = np.ascontiguousarray(self._payoff.transpose(3, 0, 1, 2))
        self._row_index = np.arange(players)
        self._scaled = np.empty((players, rank, players))
        self._weight = np.empty((rank, players * actions))
        self._weight_blocks = self._weight.reshape(rank, players, actions)
        self._jacobian = np.empty((players, actions, players * actions))
        # Bound the temporary of the broadcast division below (~2 M doubles).
        self._block_size = max(1, min(players, (1 << 21) // (8 * rank * players)))

    def value_Jaco(self, pi: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``F(pi)`` (shape ``(m, n)``) and its Jacobian (shape ``(m, n, m, n)``)."""
        players, actions = self._players, self._actions
        contracted = np.matmul(self._payoff_t, pi[..., None])[..., 0]
        product = np.prod(contracted, axis=1)
        diagonal = contracted.diagonal(axis1=1, axis2=2)
        values = np.einsum("ri,rip->ip", product / diagonal, self._diagonal)

        scaled = self._scaled
        for start in range(0, players, self._block_size):
            stop = min(start + self._block_size, players)
            np.divide(product[:, start:stop].T[..., None],
                      np.einsum("ri,rji->irj", diagonal[:, start:stop],
                                contracted[:, :, start:stop]),
                      out=scaled[start:stop])
        scaled[self._row_index, :, self._row_index] = 0

        weight = self._weight
        blocks = self._weight_blocks
        jacobian = self._jacobian
        permuted = self._payoff_permuted
        diagonal_t = self._diagonal_t
        for i in range(players):
            np.einsum("rjq,rj->rjq", permuted[i], scaled[i], out=blocks)
            np.matmul(diagonal_t[i], weight, out=jacobian[i])
        return values, jacobian.reshape(players, actions, players, actions)


# ------------------------------------------------------------------ formatting
def format_result(seed: int, result, summary: Iterable) -> str:
    """Format one result line in the layout used by the data files."""
    formatted = "|".join(
        f"{item:^7d}" if type(item) is int else f"{item:.{PRINT_PRECISION}e}"
        for item in summary
    )
    return f"|{seed:^5d}|{repr(result is not None):7}|{formatted}|\n"


# ------------------------------------------------------------- single instances
def run_vi_instance(seed: int, n: int, hidden_layers: Sequence[int] | None = None,
                    verbose: int = 0, efficient_corrector: bool = True) -> str:
    """Solve one seeded ``VI(F, Delta_n)`` instance and return its result line.

    ``hidden_layers`` defaults to ``[n]`` (i.e. ``[m*n]`` with ``m = 1``).
    """
    if hidden_layers is None:
        hidden_layers = [n]
    else:
        hidden_layers = _validated_hidden_layers(hidden_layers)
    operator = NeuralNetworkOperator(seed, [n, *hidden_layers, n], 0)

    np.random.seed(seed)
    initial_point = np.random.dirichlet(np.ones(n))
    solution, summary = VI_simplex(operator).solve(
        initial_point, gap_tol=GAP_TOLERANCE, efficient_corrector=efficient_corrector,
        maxnit=MAX_ITERATIONS, maxsubnit=MAX_SUBITERATIONS, verbose=verbose,
        record_file=RECORD_FILE, print_preci=PRINT_PRECISION)
    return format_result(seed, solution, summary)


def run_game_instance(seed: int, dimension: tuple[int, int],
                      hidden_layers: Sequence[int] | None = None, verbose: int = 0,
                      payoff_game: bool = True, efficient_corrector: bool = True,
                      rank: int | None = None) -> str:
    """Solve one seeded ``VI(F, Delta_n^m)`` instance and return its result line.

    The payoff-tensor model is configured by ``rank`` (the CP rank ``R``, which
    defaults to ``m*n``); the neural-network model is configured by
    ``hidden_layers`` (which defaults to ``[m*n]``).  The two are different
    quantities, so passing the option of the other model is an error.
    """
    m, n = dimension
    if payoff_game:
        if hidden_layers is not None:
            raise ValueError("hidden_layers is not used by the payoff-tensor model")
        if rank is None:
            rank = m * n
        if not isinstance(rank, numbers.Integral) or int(rank) < 1:
            raise ValueError("rank must be a positive integer")
        operator = PayoffTensorOperator(seed, m, n, int(rank))
    else:
        if rank is not None:
            raise ValueError("rank is only used by the payoff-tensor model")
        if hidden_layers is None:
            hidden_layers = [m * n]
        else:
            hidden_layers = _validated_hidden_layers(hidden_layers)
        operator = NeuralNetworkOperator(seed, [m * n, *hidden_layers, m * n], 0)

    np.random.seed(seed)
    initial_point = np.random.dirichlet(np.ones(n), size=(m,))
    solution, summary = VI_simplex_product(operator).solve(
        initial_point, gap_tol=GAP_TOLERANCE, efficient_corrector=efficient_corrector,
        maxnit=MAX_ITERATIONS, maxsubnit=MAX_SUBITERATIONS, verbose=verbose,
        record_file=RECORD_FILE, print_preci=PRINT_PRECISION)
    return format_result(seed, solution, summary)


# ------------------------------------------------------------------ CLI
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve one instance: payoff_F always uses game_numpy; nn_F uses "
                    "vi_numpy when m = 1 and game_numpy when m > 1.")
    parser.add_argument("--dimension", type=int, nargs=2, required=True, metavar=("M", "N"),
                        help="dimensions m n")
    parser.add_argument("--model", choices=("nn_F", "payoff_F"), required=True,
                        help="operator model: nn_F (neural network) or payoff_F (payoff tensor)")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the instance (default 0)")
    parser.add_argument("--hidden-layers", type=_parse_layer_list, metavar="LIST",
                        help="bracketed list of hidden-layer widths, e.g. '[8, 4]' or '[]' (default [m*n])")
    parser.add_argument("--rank", type=int,
                        help="CP rank of the payoff-tensor model (default: m*n)")
    parser.add_argument("--kkt", action="store_true",
                        help="use the KKT corrector instead of the barrier corrector")
    parser.add_argument("--threads", type=_thread_count_argument, default=_DEFAULT_BLAS_THREADS,
                        metavar="N|auto",
                        help="number of BLAS/OpenMP threads (default 1); 'auto' keeps the library defaults")
    parser.add_argument("--verbose", type=int, default=0, choices=(0, 1, 2),
                        help="iteration-trace verbosity (default 0)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        # Demo instance used in the README: 16-player, 16-action payoff game, R = 256.
        print(run_game_instance(0, (16, 16), rank=256))
        return

    args = _parse_args(arguments)
    efficient_corrector = not args.kkt
    m, n = args.dimension
    if m < 1 or n < 1:
        raise SystemExit("--dimension expects positive integers m n")

    if args.model == "payoff_F":
        if args.hidden_layers is not None:
            raise SystemExit("--hidden-layers is only used with --model nn_F")
        line = run_game_instance(args.seed, (m, n), verbose=args.verbose,
                                 payoff_game=True, efficient_corrector=efficient_corrector,
                                 rank=args.rank)
    elif m == 1:
        if args.rank is not None:
            raise SystemExit("--rank is only used with --model payoff_F")
        line = run_vi_instance(args.seed, n, args.hidden_layers, verbose=args.verbose,
                               efficient_corrector=efficient_corrector)
    else:
        if args.rank is not None:
            raise SystemExit("--rank is only used with --model payoff_F")
        line = run_game_instance(args.seed, (m, n), hidden_layers=args.hidden_layers,
                                 verbose=args.verbose, payoff_game=False,
                                 efficient_corrector=efficient_corrector)
    print(line)


if __name__ == "__main__":
    main()
