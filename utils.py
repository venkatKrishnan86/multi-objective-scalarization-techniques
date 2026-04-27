"""Shared utilities for multi-objective RL agents.

Classes
-------
ReplayBuffer
    Fixed-capacity circular experience replay buffer for off-policy learning.

Functions
---------
compute_utopian(env) -> np.ndarray
    Load or compute and cache the utopian point for an environment.
scalarize_q(q_vec, weight, p, utopian) -> torch.Tensor
    Negated Lp-norm scalarization of vector Q-values,
    shape (batch, reward_dim, n_actions) → (batch, n_actions).
scalarize_vec(vec, weight, p, utopian) -> np.ndarray
    Negated Lp-norm scalarization of reward/return vectors,
    shape (..., reward_dim) → (...,).  Pure NumPy for tabular agents.
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer.

    Stores ``(state, action, reward, next_state, done)`` transitions and
    supports uniform random sampling for off-policy learning.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.  Oldest entries are evicted
        automatically when the buffer is full.

    Methods
    -------
    push(state, action, reward, next_state, done)
        Append a single transition.
    sample(batch_size)
        Draw a random batch without replacement.
    __len__()
        Current number of stored transitions.

    Example
    -------
    >>> buf = ReplayBuffer(capacity=10_000)
    >>> buf.push(obs, action, reward_vec, next_obs, done)
    >>> batch = buf.sample(64)
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append a single ``(s, a, r, s', done)`` transition.

        Parameters
        ----------
        state : np.ndarray, shape (obs_dim,)
        action : int
        reward : np.ndarray, shape (reward_dim,)
            Raw multi-objective reward vector from the environment.
        next_state : np.ndarray, shape (obs_dim,)
        done : bool
        """
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        """Return a uniformly random batch of transitions.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        list of ``(state, action, reward, next_state, done)`` tuples
        """
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


def compute_utopian(env) -> np.ndarray:
    """Load or compute and cache the utopian (ideal) point for an environment.

    The utopian point z is the component-wise maximum over the Pareto front:

        z_i = max_{f in PF} f_i

    On the first call the point is computed from ``env.unwrapped.pareto_front``
    and saved to ``outputs/<env_name>/utopian.npy``.  Subsequent calls load
    the cached file directly.

    Parameters
    ----------
    env : gymnasium.Env
        A MO-Gymnasium environment that exposes ``pareto_front(gamma)``.

    Returns
    -------
    np.ndarray, shape (reward_dim,)
    """
    env_name = env.spec.id
    path = Path("outputs") / env_name / "utopian.npy"

    if path.exists():
        return np.load(path)

    pareto_front = np.array(env.unwrapped.pareto_front(gamma=1.0))
    utopian = np.max(pareto_front, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, utopian)
    return utopian


def scalarize_q(
    q_vec: torch.Tensor,
    weight: torch.Tensor,
    p: float,
    utopian: torch.Tensor,
) -> torch.Tensor:
    """Lp-norm scalarization of vector Q-values (PyTorch).

    Returns the positive weighted Lp-distance to the utopian point.
    Minimise the output (``argmin``) to select the action closest to utopian.

    Parameters
    ----------
    q_vec : torch.Tensor, shape (batch, reward_dim, n_actions)
    weight : torch.Tensor, shape (1, reward_dim, 1)
    p : float
        Lp-norm exponent.  Use ``float("inf")`` for Chebyshev.
    utopian : torch.Tensor, shape (1, reward_dim, 1)

    Returns
    -------
    torch.Tensor, shape (batch, n_actions)
        d(a) = ( Σ_i  w_i · |Q_i(a) − z_i|^p )^(1/p)
    """
    diff = (q_vec - utopian).abs()
    if p == float("inf"):
        return (weight * diff).max(dim=1).values
    return (weight * diff.pow(p)).sum(dim=1).pow(1.0 / p)


def scalarize_vec(
    vec: np.ndarray,
    weight: np.ndarray,
    p: float,
    utopian: np.ndarray,
) -> np.ndarray:
    """Lp-norm scalarization of reward/return vectors (NumPy).

    Returns the positive weighted Lp-distance to the utopian point.
    Minimise the output (``argmin``) to select the action or episode closest
    to utopian.  Accepts batched or single vectors.

    Parameters
    ----------
    vec : np.ndarray, shape (..., reward_dim)
    weight : np.ndarray, shape (reward_dim,)
    p : float
        Lp-norm exponent.  Use ``float("inf")`` for Chebyshev.
    utopian : np.ndarray, shape (reward_dim,)

    Returns
    -------
    np.ndarray, shape (...)
        d = ( Σ_i  w_i · |v_i − z_i|^p )^(1/p)
    """
    diff = np.abs(vec - utopian)
    if p == float("inf"):
        return (weight * diff).max(axis=-1)
    return ((weight * diff ** p).sum(axis=-1)) ** (1.0 / p)
