"""
DQN Agent for Multi-Objective Deep Sea Treasure (MO-Gymnasium).

This module provides two classes:

- ``ReplayBuffer`` — fixed-capacity circular experience replay buffer.
- ``Agent``        — multi-objective DQN agent that trains a
                     ``MultiObjectiveDQN`` network using Lp-norm scalarization
                     with weight vector w > 0 (strictly positive orthant).

The scalarization is applied to the vector Q-values to produce a scalar
signal used for action selection and TD learning.  No scalarization is baked
into the network; it is applied externally so that the same network weights
can be queried under different (w, p) settings.

Classes
-------
ReplayBuffer
    Circular experience replay buffer backed by a collections.deque.
Agent
    Multi-objective DQN agent with epsilon-greedy exploration, a frozen
    target network, and Lp-norm scalarization.

Usage
-----
    import mo_gymnasium as mo_gym
    import numpy as np
    from agent import Agent

    env = mo_gym.make('deep-sea-treasure-v0')
    w   = np.array([0.5, 0.5])   # weight vector in positive orthant
    agent = Agent(env=env, weight=w, p=2.0)

    for _ in range(10_000):
        agent.simulate()
        agent.learn(batch_size=64)
"""

import copy
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dqn import MultiObjectiveDQN


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
        Maximum number of transitions to store. Oldest entries are evicted
        automatically when the buffer is full.

    Methods
    -------
    push(state, action, reward, next_state, done)
        Append a single transition.
    sample(batch_size)
        Draw a random batch of transitions without replacement.
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
        """Append a single transition to the buffer.

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
        """Return a random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        list of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """    Multi-objective DQN agent with utopian-point Lp-norm scalarization.

    Maintains an online ``MultiObjectiveDQN`` network and a periodically-
    synced frozen target network.  Action selection and TD targets are both
    computed from the Lp-norm scalarization of the vector Q-values using the
    supplied weight vector ``w > 0`` and the environment's utopian point ``z``:

        s(a) = ( Σ_i  w_i · |Q_i(a) − z_i|^p )^(1/p)

    The utopian point is computed from the environment's Pareto front (via
    ``get_pareto_front``) on the first run and cached to
    ``outputs/<env_name>/utopian.npy`` for subsequent runs.

    Parameters
    ----------
    env : gymnasium.Env
        MO-Gymnasium environment (e.g. ``deep-sea-treasure-v0``).
    weight : np.ndarray, shape (reward_dim,)
        Scalarization weight vector.  Must satisfy ``w_i > 0`` for all i
        (strictly positive orthant).
    p : float
        Lp-norm exponent.  ``p=1`` → linear, ``p=inf`` → Chebyshev.
    hidden_dims : list[int]
        Hidden layer sizes for the Q-network.  Default: ``[64, 64]``.
    lr : float
        Adam learning rate.  Default: ``1e-3``.
    gamma : float
        Discount factor.  Default: ``0.99``.
    epsilon : float
        Initial epsilon for epsilon-greedy exploration.  Default: ``1.0``.
    epsilon_min : float
        Minimum epsilon after decay.  Default: ``0.05``.
    epsilon_decay : float
        Multiplicative decay applied to epsilon after each ``simulate()``
        call.  Default: ``0.995``.
    buffer_capacity : int
        Maximum replay buffer size.  Default: ``10_000``.
    warmup_frac : float
        Fraction of ``buffer_capacity`` that must be filled before
        ``learn()`` starts updating the network.  Default: ``0.1``.
    target_update_freq : int
        Number of ``simulate()`` steps between target network syncs.
        Default: ``100``.
    tau : float
        Polyak averaging coefficient for target network updates.
        ``θ_target ← (1 - τ) θ_target + τ θ_online``.
        ``tau=1.0`` recovers a hard copy; smaller values (e.g. ``0.005``)
        give slow, stable tracking.  Default: ``0.005``.
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).  Default: ``"cpu"``.

    Methods
    -------
    get_pareto_front(gamma) -> np.ndarray
        Return the environment's true Pareto front as an array.
    simulate() -> tuple
        Take one environment step; store transition; decay epsilon.
    remember(state, action, reward, next_state, done)
        Manually push a transition into the replay buffer.
    learn(batch_size, n_epochs) -> float | None
        Sample from the buffer and update the network.  Returns mean loss
        or ``None`` if the warmup threshold is not yet reached.

    Example
    -------
    >>> agent = Agent(env=env, weight=np.array([0.5, 0.5]), p=2.0)
    >>> for _ in range(50_000):
    ...     agent.simulate()
    ...     loss = agent.learn(batch_size=64)
    """

    def __init__(
        self,
        env,
        weight: np.ndarray,
        p: float,
        hidden_dims: list[int] = [64, 64],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        warmup_frac: float = 0.1,
        target_update_freq: int = 100,
        tau: float = 0.005,
        device: str = "cpu",
    ) -> None:
        assert np.all(weight > 0), "All weight components must be strictly positive (positive orthant cone)."

        self.env = env
        self.p = p
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.warmup_size = int(warmup_frac * buffer_capacity)
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.device = torch.device(device)
        self.steps = 0

        obs_dim    = env.observation_space.shape[0]
        n_actions  = env.action_space.n
        reward_dim = env.unwrapped.reward_space.shape[0]

        self.n_actions  = n_actions
        self.reward_dim = reward_dim

        # weight tensor for scalarization: shape (1, reward_dim, 1)
        self.weight = torch.tensor(weight, dtype=torch.float32, device=self.device)
        self.weight = self.weight.view(1, reward_dim, 1)

        # networks
        self.online_net = MultiObjectiveDQN(
            obs_dim=obs_dim,
            reward_dim=reward_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.target_net = copy.deepcopy(self.online_net)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(buffer_capacity)

        # utopian point z — shape (1, reward_dim, 1) for broadcasting
        utopian_np = self._compute_utopian()
        self.utopian = torch.tensor(utopian_np, dtype=torch.float32, device=self.device)
        self.utopian = self.utopian.view(1, reward_dim, 1)

        obs, _ = env.reset()
        self.state = np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Pareto front & utopian point
    # ------------------------------------------------------------------

    def get_pareto_front(self, gamma: float = 1.0) -> np.ndarray:
        """Return the environment's true Pareto front.

        Parameters
        ----------
        gamma : float
            Discount factor used by the environment's Pareto-front solver.
            Use ``1.0`` for undiscounted returns.  Default: ``1.0``.

        Returns
        -------
        np.ndarray, shape (n_points, reward_dim)
            Each row is one Pareto-optimal reward vector.
        """
        return np.array(self.env.unwrapped.pareto_front(gamma=gamma))

    def _compute_utopian(self) -> np.ndarray:
        """Load or compute and cache the utopian (ideal) point.

        The utopian point z is defined as the component-wise maximum over
        the Pareto front:  z_i = max_{f in PF} f_i.

        On the first call the point is computed from ``get_pareto_front``
        and saved to ``outputs/<env_name>/utopian.npy``.  Subsequent calls
        load the cached file directly.

        Returns
        -------
        np.ndarray, shape (reward_dim,)
        """
        env_name = self.env.spec.id
        path = Path("outputs") / env_name / "utopian.npy"

        if path.exists():
            return np.load(path)

        pareto_front = self.get_pareto_front(gamma=1.0)
        utopian = np.max(pareto_front, axis=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, utopian)
        return utopian

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a single observation array to a float32 device tensor.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)

        Returns
        -------
        torch.Tensor, shape (1, obs_dim)
        """
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _scalarize(self, q_vec: torch.Tensor) -> torch.Tensor:
        """Apply utopian-point Lp-norm scalarization to vector Q-values.

        Parameters
        ----------
        q_vec : torch.Tensor, shape (batch, reward_dim, n_actions)

        Returns
        -------
        torch.Tensor, shape (batch, n_actions)
            Scalarized Q-values.  For finite p:
                s(a) = ( Σ_i  w_i · |Q_i(a) − z_i|^p )^(1/p)
            For p = inf:
                s(a) = max_i  w_i · |Q_i(a) − z_i|
            where z is the utopian point (broadcast shape (1, reward_dim, 1)).
        """
        assert len(q_vec.shape) == 3, "Q-vector must be a 3D tensor"
        assert q_vec.shape[1] == self.reward_dim, "Q-vector reward dimension must match the input reward dimension"
        assert q_vec.shape[2] == self.n_actions, "Q-vector action dimension must match the input action dimension"

        diff = (q_vec - self.utopian).abs()
        if self.p == float("inf"):
            return (self.weight * diff).max(dim=1).values
        return (self.weight * diff.pow(self.p)).sum(dim=1).pow(1.0 / self.p)

    def _scalarize_reward(self, reward: np.ndarray) -> float:
        """Scalarize a single reward vector to a scalar float.

        Applies the same utopian-point Lp-norm formula as ``_scalarize``:
            s = ( Σ_i  w_i · |r_i − z_i|^p )^(1/p)

        Parameters
        ----------
        reward : np.ndarray, shape (reward_dim,)

        Returns
        -------
        float
        """
        r = torch.tensor(reward, dtype=torch.float32, device=self.device)
        r = r.view(1, self.reward_dim, 1)
        diff = (r - self.utopian).abs()
        if self.p == float("inf"):
            return (self.weight * diff).max().item()
        return (self.weight * diff.pow(self.p)).sum().pow(1.0 / self.p).item()

    def _select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray, shape (obs_dim,)

        Returns
        -------
        int
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_vec = self.online_net(self._obs_to_tensor(state))
            q_scalar = self._scalarize(q_vec)
        return int(q_scalar.argmax(dim=1).item())

    def _update_target(self) -> None:
        """Polyak-average online network weights into the target network.

        Applies the soft update:
            θ_target ← (1 - τ) · θ_target + τ · θ_online

        where ``τ = self.tau``.  Setting ``tau=1.0`` recovers a hard copy.
        """
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau).add_(online_param.data * self.tau)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self) -> tuple:
        """Take one environment step and store the transition.

        Selects an action via epsilon-greedy, steps the environment, stores
        the transition in the replay buffer, resets the environment on
        episode termination, decays epsilon, and syncs the target network
        every ``target_update_freq`` steps.

        Returns
        -------
        tuple : (state, action, reward, next_state, done)
            ``state`` and ``next_state`` are ``np.ndarray``;
            ``reward`` is the raw multi-objective ``np.ndarray``;
            ``done`` is ``bool``.
        """
        state  = self.state
        action = self._select_action(state)

        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done       = terminated or truncated
        next_state = np.array(next_obs, dtype=np.float32)
        reward     = np.array(reward,   dtype=np.float32)

        self.remember(state, action, reward, next_state, done)

        self.state = next_state
        if done:
            obs, _ = self.env.reset()
            self.state = np.array(obs, dtype=np.float32)

        self.steps += 1

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if self.steps % self.target_update_freq == 0:
            self._update_target()

        return state, action, reward, next_state, done

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer.

        Parameters
        ----------
        state : np.ndarray, shape (obs_dim,)
        action : int
        reward : np.ndarray, shape (reward_dim,)
        next_state : np.ndarray, shape (obs_dim,)
        done : bool
        """
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self, batch_size: int, n_epochs: int = 1) -> float | None:
        """Sample from the replay buffer and update the online network.

        Learning is skipped and ``None`` is returned until the replay buffer
        reaches the warmup threshold (``warmup_frac * buffer_capacity``
        transitions).

        Parameters
        ----------
        batch_size : int
            Number of transitions sampled per epoch.
        n_epochs : int
            Number of gradient update steps per call.  Default: ``1``.

        Returns
        -------
        float or None
            Mean TD loss over all epochs, or ``None`` if still in warmup.
        """
        if len(self.buffer) < self.warmup_size:
            return None

        total_loss = 0.0
        for _ in range(n_epochs):
            batch = self.buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_t      = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
            next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            actions_t     = torch.tensor(actions,               dtype=torch.long,    device=self.device)
            dones_t       = torch.tensor(dones,                 dtype=torch.float32, device=self.device)

            # scalarize reward vectors → shape (batch,)
            rewards_scalar = torch.tensor(
                [self._scalarize_reward(r) for r in rewards],
                dtype=torch.float32,
                device=self.device,
            )

            # Q(s, a) for each objective, then scalarize → (batch, n_actions)
            q_vec_online  = self.online_net(states_t)
            q_scalar_all  = self._scalarize(q_vec_online)         # (batch, n_actions)
            q_current     = q_scalar_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_vec_next   = self.target_net(next_states_t)
                q_scalar_next = self._scalarize(q_vec_next)       # (batch, n_actions)
                q_next_max    = q_scalar_next.max(dim=1).values

            td_target = rewards_scalar + self.gamma * q_next_max * (1.0 - dones_t)

            loss = self.loss_fn(q_current, td_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_epochs
