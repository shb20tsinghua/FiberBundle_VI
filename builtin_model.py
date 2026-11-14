"""Random operator models and a derivative-free Jacobian wrapper.

The module provides the two random operators used in the experiments:

* :class:`nn_F` -- a fully connected neural network with tanh, sigmoid or linear
  activation, used as a general real-analytic operator ``F``;
* :class:`payoff_F` -- the gradient operator of a finite normal-form game whose
  payoff tensor has low CP (canonical polyadic) rank.

Both operators expose ``value(x)`` and ``value_Jaco(x)``.  The latter returns
``(F(x), JF(x))``, where ``F`` has shape ``x.shape`` and ``JF`` has shape
``x.shape + x.shape``.
The solver needs only ``value_Jaco``; ``value`` is needed by
:class:`DerivFreeWrapper`.

:class:`DerivFreeWrapper` turns either model into a derivative-free operator: it
evaluates only ``value`` and maintains a Broyden rank-one approximation of the
Jacobian to provide a ``value_Jaco`` interface.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# The neural network uses the rescaled Xavier normal initialisation: weights
# have standard deviation sqrt(2/(fan_in+fan_out)), which keeps tanh units out of
# saturation; biases are drawn from N(0, 1e-2**2).
_WEIGHT_INIT_SCALE = lambda fan_in, fan_out: 1 / np.sqrt((fan_in + fan_out) / 2)
_BIAS_INIT_SCALE = 1e-2

# Activation functions and their derivatives, in the order of the ``activation``
# parameter: 0 = tanh, 1 = sigmoid, 2 = linear.
ACTIVATIONS = (
    np.tanh,
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: x,
)
ACTIVATION_GRADIENTS = (
    lambda h: 1 - h ** 2,
    lambda h: h * (1 - h),
    lambda h: np.ones_like(h),
)

# Defaults of the derivative-free wrapper.
_DEFAULT_REL_STEP = 1e-6
_DEFAULT_REFRESH_EVERY = 200
# A Broyden update is skipped when ||s||^2 is below this threshold (the previous
# Jacobian approximation is then reused for the same point).
_MIN_STEP_NORM_SQUARED = 1e-14


class nn_F:
    """Fully connected neural network used as a random operator ``F``.

    Parameters
    ----------
    seed : int
        Seed of the random parameter draw.
    layer_widths : sequence of int
        Layer widths ``[dim, *hidden_layers, dim]`` of the network.
    activation : int
        ``0`` tanh, ``1`` sigmoid, ``2`` linear.

    Notes
    -----
    Weights use the rescaled Xavier normal distribution, with standard deviation
    ``1 / sqrt((fan_in + fan_out) / 2)``, so the tanh activations stay out of
    saturation and the resulting operator is highly nonlinear.  Biases are
    drawn from ``N(0, 1e-2**2)``.  The model represents a general real-analytic
    operator and can be used on ``Delta_n`` (``dim = n``) as well as on
    ``Delta_n^m`` (``dim = m*n``).

    Attributes
    ----------
    ws : list of ndarray
        Weight matrices ``W_l`` of shape ``(width[l+1], width[l])``.
    bs : list of ndarray
        Bias vectors ``b_l`` of shape ``(width[l+1],)``.
    activation, grad_activation : callable
        Activation ``sigma`` and its derivative ``sigma'``.
    """

    def __init__(self, seed: int, layer_widths: Sequence[int], activation: int) -> None:
        np.random.seed(seed)
        widths = list(layer_widths)
        self.ws = [
            np.random.normal(loc=0.0, scale=_WEIGHT_INIT_SCALE(widths[i], widths[i + 1]),
                             size=(widths[i + 1], widths[i]))
            for i in range(len(widths) - 1)
        ]
        self.bs = [
            np.random.normal(loc=0.0, scale=_BIAS_INIT_SCALE, size=widths[i + 1])
            for i in range(len(widths) - 1)
        ]
        self.activation = ACTIVATIONS[activation]
        self.grad_activation = ACTIVATION_GRADIENTS[activation]

    def value(self, x: FloatArray) -> FloatArray:
        """Evaluate ``F(x)``; the result has the shape of ``x``."""
        shape = x.shape
        hidden = x.ravel()
        for weight, bias in zip(self.ws, self.bs):
            hidden = self.activation(np.dot(weight, hidden) + bias)
        return hidden.reshape(shape)

    def value_Jaco(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``F(x)`` and its Jacobian, both with the shape of ``x``.

        The Jacobian has shape ``x.shape + x.shape``; it is obtained by
        backpropagation through the cached forward pass.
        """
        shape = x.shape
        hidden = x.ravel()
        forward_cache = [hidden]
        for weight, bias in zip(self.ws, self.bs):
            hidden = self.activation(np.dot(weight, hidden) + bias)
            forward_cache.append(hidden)
        jacobian = self.grad_activation(forward_cache[-1])[:, None] * self.ws[-1]
        for i in reversed(range(len(self.ws) - 1)):
            jacobian *= self.grad_activation(forward_cache[i + 1])[None, :]
            jacobian = np.dot(jacobian, self.ws[i])
        return hidden.reshape(shape), jacobian.reshape(shape + shape)


class payoff_F:
    """Gradient operator of a finite normal-form game with a low-rank CP payoff.

    For ``m`` players with ``n`` actions each, the payoff function of player
    ``i`` is

        f_i(pi) = sum_{r=1}^{R} prod_{j=1}^{m} (U_{r,i,j} pi_j),

    with ``pi_j`` in ``Delta_n``, and the operator is the gradient of the payoff,
    ``F_i(pi) = grad_{pi_i} f_i(pi)``.  The tensor
    ``U`` has shape ``(R, m, n, m)`` and is represented in CP form, so the
    storage is ``O(R m n)`` instead of the ``O(n^m)`` of a dense tensor.

    Parameters
    ----------
    seed : int
        Seed of the random payoff draw.
    m : int
        Number of players.
    n : int
        Number of actions per player.
    r : int
        CP rank ``R`` of the payoff tensor; ``R = m*n`` is the setting used in
        the experiments.

    Notes
    -----
    The tensor entries are drawn from ``N(1, 1/m)``.  The operator is defined on
    the product domain ``Delta_n^m``; at ``m = 1`` it reduces to an operator on
    ``Delta_n``.  The model precomputes the layouts

    * ``payoff`` -- ``U`` in its natural ``(R, m, n, m)`` layout,
    * ``payoff_T`` -- ``U`` transposed to contract with ``pi``,
    * ``U_ii`` / ``U_ii_T`` -- the diagonal players ``j = i`` of ``U`` and its
      transposed layout, used by the Jacobian,
    * ``pf`` -- ``U`` permuted to the layout of the Jacobian product,

    and the temporaries reused by :meth:`value_Jaco`.
    """

    def __init__(self, seed: int, m: int, n: int, r: int) -> None:
        np.random.seed(seed)
        self.payoff = np.random.normal(loc=1.0, scale=1 / np.sqrt(m), size=(r, m, n, m))
        self.m, self.n = m, n
        self.payoff_T = self.payoff.transpose(0, 1, 3, 2)
        self.U_ii = self.payoff.diagonal(axis1=1, axis2=3).transpose(0, 2, 1)
        self.U_ii_T = np.ascontiguousarray(self.U_ii.transpose(1, 2, 0))
        self.pf = np.ascontiguousarray(self.payoff.transpose(3, 0, 1, 2))
        self._idx_m = np.arange(m)
        self._cT = np.empty((m, r, m))
        self._W = np.empty((r, m * n))
        self._W3 = self._W.reshape(r, m, n)
        self._J = np.empty((m, n, m * n))
        # Bound the temporary of the broadcast division below (~2 M doubles).
        self._cT_block = max(1, min(m, (1 << 21) // (8 * r * m)))

    def value(self, pi: FloatArray) -> FloatArray:
        """Evaluate ``F(pi)``; the result has the shape of ``pi``."""
        shape = pi.shape
        pi = np.atleast_2d(pi)
        contracted = np.matmul(self.payoff_T, pi[..., None])[..., 0]
        product = np.prod(contracted, axis=1)
        diagonal = contracted.diagonal(axis1=1, axis2=2)
        values = np.einsum("ri,rip->ip", product / diagonal, self.U_ii)
        return values.reshape(shape)

    def value_Jaco(self, pi: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``F(pi)`` and its Jacobian, both with the shape of ``pi``.

        The Jacobian has shape ``pi.shape + pi.shape``, i.e. ``(m, n, m, n)``;
        ``JF[i, :, j, :]`` is the derivative of block ``F_i`` with respect to
        block ``pi_j``.
        """
        m = self.m
        shape = pi.shape
        pi = np.atleast_2d(pi)
        contracted = np.matmul(self.payoff_T, pi[..., None])[..., 0]
        product = np.prod(contracted, axis=1)
        diagonal = contracted.diagonal(axis1=1, axis2=2)
        values = np.einsum("ri,rip->ip", product / diagonal, self.U_ii)

        scaled = self._cT
        for start in range(0, m, self._cT_block):
            stop = min(start + self._cT_block, m)
            # scaled[i, r, j] = prod_r / (U_{r,i,j} pi_j) for j != i; the
            # diagonal j = i is overwritten with zero below.
            np.divide(product[:, start:stop].T[..., None],
                      np.einsum("ri,rji->irj", diagonal[:, start:stop],
                                contracted[:, :, start:stop]),
                      out=scaled[start:stop])
        scaled[self._idx_m, :, self._idx_m] = 0

        weight, blocks = self._W, self._W3
        jacobian, permuted, diagonal_t = self._J, self.pf, self.U_ii_T
        for i in range(m):
            np.einsum("rjq,rj->rjq", permuted[i], scaled[i], out=blocks)
            np.matmul(diagonal_t[i], weight, out=jacobian[i])
        return values.reshape(shape), jacobian.reshape(shape + shape)


class DerivFreeWrapper:
    """Broyden rank-one wrapper that provides ``value_Jaco`` without derivatives.

    The wrapper evaluates only ``model.value(x)`` and maintains an approximate
    Jacobian ``B``.  The first call builds ``B`` by one-sided finite
    differences with relative step ``rel_step``; every following call performs
    the rank-one secant (Broyden) update

        B  <-  B + (y - B s) s^T / (s^T s),
        s = x - x_prev,   y = F - F_prev.

    Parameters
    ----------
    model : object
        Any operator exposing ``value(x)``.
    rel_step : float
        Relative finite-difference step used to initialise ``B``; the absolute
        step for coordinate ``k`` is ``rel_step * max(1, |x_k|)``.
    refresh_every : int
        Re-initialise ``B`` by finite differences after this many Broyden
        updates; prevents drift.  ``0`` disables re-initialisation.

    Notes
    -----
    It turns ``value(x) -> F`` into ``value_Jaco(x) -> (F, JF)`` for the solver.
    The first call performs ``dim`` extra operator evaluations to build the
    finite-difference matrix; every subsequent call performs one operator
    evaluation plus an ``O(dim^2)`` Broyden update.  Consecutive calls at the
    same point reuse ``B`` unchanged (the secant update is skipped when
    ``||s||^2 <= 1e-14``).
    """

    def __init__(self, model, rel_step: float = _DEFAULT_REL_STEP,
                 refresh_every: int = _DEFAULT_REFRESH_EVERY) -> None:
        if not hasattr(model, "value"):
            raise TypeError("model must expose a value(x) method")
        self.model = model
        self.rel_step = rel_step
        self.refresh_every = refresh_every
        self.B = None
        self.x_prev = None
        self.F_prev = None
        self.n_updates = 0

    def _init_B(self, x: FloatArray, F: FloatArray) -> None:
        """Build ``B`` by one-sided finite differences around ``x``."""
        x_flat = x.ravel()
        F_flat = F.ravel()
        dim = x_flat.size
        B = np.empty((dim, dim))
        for k in range(dim):
            h = self.rel_step * max(1.0, abs(x_flat[k]))
            if h == 0.0:
                h = self.rel_step
            x_perturbed = x_flat.copy()
            x_perturbed[k] += h
            B[:, k] = (self.model.value(x_perturbed.reshape(x.shape)).ravel() - F_flat) / h
        self.B = B
        self.x_prev = x_flat.copy()
        self.F_prev = F_flat.copy()
        self.n_updates = 0

    def value_Jaco(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``F(x)`` and the current Broyden approximation of ``JF(x)``."""
        x = np.asarray(x)
        shape = x.shape
        x_flat = x.ravel()
        F = self.model.value(x)
        F_flat = F.ravel()

        if self.B is None:
            self._init_B(x, F)
        else:
            s = x_flat - self.x_prev
            s_norm2 = s @ s
            if s_norm2 > _MIN_STEP_NORM_SQUARED:      # skip no-op (same point)
                y = F_flat - self.F_prev
                self.B += np.outer(y - self.B @ s, s) / s_norm2
                self.n_updates += 1
                if self.refresh_every and self.n_updates >= self.refresh_every:
                    self._init_B(x, F)                # re-anchor B
                else:
                    self.x_prev = x_flat.copy()
                    self.F_prev = F_flat.copy()

        return F, self.B.reshape(shape + shape)


__all__ = ["nn_F", "payoff_F", "DerivFreeWrapper"]
