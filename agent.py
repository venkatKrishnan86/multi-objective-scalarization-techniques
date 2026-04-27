"""
Multi-Objective RL agents for Deep Sea Treasure (MO-Gymnasium).

All agents share a common ``BaseAgent`` abstract class that handles:
  - utopian-point computation and caching
  - NumPy Lp-norm scalarization (``scalarize``)
  - Pareto-front retrieval (``get_pareto_front``)

Concrete implementations
------------------------
DQNAgent
    Off-policy multi-objective DQN with a vector Q-network
    ``(batch, reward_dim, n_actions)``, experience replay, and Polyak-averaged
    target network.  Scalarization enters only at action selection (not in the
    TD loss), so the network learns objective-specific cumulative values.

TabularAgent
    Exact tabular Q-table ``Q[s, reward_dim, n_actions]`` updated by
    first-visit Monte Carlo returns over full episodes.  No function
    approximation — guaranteed to converge on small state spaces.

CEMAgent
    Cross-Entropy Method direct policy search.  Maintains a tabular softmax
    policy ``π(a|s) = softmax(logits[s])``, generates batches of rollouts,
    selects elite episodes by scalarized cumulative return, and updates logits
    via MLE on elite action frequencies.

Supporting classes
------------------
ReplayBuffer
    Fixed-capacity circular buffer for DQNAgent experience replay
    (defined in ``utils.py``).

Usage
-----
    import mo_gymnasium as mo_gym
    import numpy as np
    from agent import DQNAgent, TabularAgent, CEMAgent

    env = mo_gym.make('deep-sea-treasure-v0')
    w   = np.array([0.5, 0.5])

    agent = TabularAgent(env=env, weight=w, p=1.0)
    for _ in range(2_000):
        agent.simulate()
        agent.learn()
"""

import copy
import math
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from dqn import MultiObjectiveDQN
from utils import ReplayBuffer, compute_utopian, scalarize_q, scalarize_vec


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base class for all MORL agents.

    Provides shared initialisation, utopian-point computation, NumPy
    scalarization, and the Pareto-front accessor.  Concrete subclasses must
    implement ``simulate()`` and ``learn()``.

    Parameters
    ----------
    env : gymnasium.Env
        MO-Gymnasium environment.
    weight : np.ndarray, shape (reward_dim,)
        Scalarization weight vector.  All components must be strictly positive.
    p : float
        Lp-norm exponent.  ``p=1`` → linear, ``p=inf`` → Chebyshev.
    gamma : float
        Discount factor.  Default: ``0.99``.
    epsilon : float
        Initial epsilon for epsilon-greedy exploration.  Default: ``1.0``.
    epsilon_min : float
        Minimum epsilon after decay.  Default: ``0.05``.
    epsilon_decay : float
        Multiplicative decay applied to epsilon per step.  Default: ``0.9995``.
    """

    def __init__(
        self,
        env,
        weight: np.ndarray,
        p: float,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ) -> None:
        assert np.all(weight > 0), (
            "All weight components must be strictly positive (positive orthant cone)."
        )
        self.env           = env
        self.p             = p
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.weight        = weight.astype(np.float64)

        self.n_actions  = env.action_space.n
        self.reward_dim = env.unwrapped.reward_space.shape[0]

        # Utopian point z: component-wise max over the Pareto front
        self.utopian = compute_utopian(env).astype(np.float64)

    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------

    def scalarize(self, vec: np.ndarray) -> np.ndarray:
        """Lp-norm scalarization of reward/return vectors (NumPy).

        Returns the positive weighted Lp-distance to the utopian point.
        Use ``argmin`` on the result to select the closest action or episode.

        Parameters
        ----------
        vec : np.ndarray, shape (..., reward_dim)

        Returns
        -------
        np.ndarray, shape (...)
            d = ( Σ_i  w_i · |v_i − z_i|^p )^(1/p)
        """
        return scalarize_vec(vec, self.weight, self.p, self.utopian)

    def get_pareto_front(self, gamma: float = 1.0) -> np.ndarray:
        """Return the environment's true Pareto front.

        Parameters
        ----------
        gamma : float
            Discount factor used by the environment's solver.  Default: ``1.0``.

        Returns
        -------
        np.ndarray, shape (n_points, reward_dim)
        """
        return np.array(self.env.unwrapped.pareto_front(gamma=gamma))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def simulate(self) -> tuple:
        """Run a full episode and return ``(total_reward_vec, episode_length)``."""

    @abstractmethod
    def learn(self, batch_size: int = None) -> float:
        """Update the agent from collected experience and return a loss/metric."""


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """Multi-objective DQN with vector Q-network and experience replay.

    Maintains an online ``MultiObjectiveDQN`` (output shape
    ``(batch, reward_dim, n_actions)``) and a Polyak-averaged target network.
    Scalarization enters **only** at action selection — the TD loss operates on
    raw vector Q-values so each objective's value function is learned
    independently:

        Q_i(s,a) ← r_i + γ · Q_i(s', a*)
        a* = argmin_a  f_w( Q(s', ·) )

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
        Q-network hidden layer sizes.  Default: ``[64, 64]``.
    lr : float
        Adam learning rate.  Default: ``1e-3``.
    gamma : float
        Discount factor.  Default: ``0.99``.
    epsilon : float
        Initial epsilon for epsilon-greedy exploration.  Default: ``1.0``.
    epsilon_min : float
        Minimum epsilon after decay.  Default: ``0.05``.
    epsilon_decay : float
        Multiplicative decay applied to epsilon after each environment step.
        Default: ``0.99975`` (decays to ~0.08 over 10,000 steps).
    buffer_capacity : int
        Replay buffer size.  Default: ``2_000``.
    warmup_frac : float
        Fraction of ``buffer_capacity`` that must be filled before
        ``learn()`` starts updating the network.  Default: ``0.2``.
    target_update_freq : int
        Steps between Polyak target updates.  Default: ``100``.
    tau : float
        Polyak averaging coefficient for target network updates.
        ``θ_target ← (1 - τ) θ_target + τ θ_online``.
        ``tau=1.0`` recovers a hard copy; smaller values (e.g. ``0.005``)
        give slow, stable tracking.  Default: ``0.005``.
    device : str
        Torch device string.  Default: ``"cpu"``.

        Example
    -------
    >>> agent = DQNAgent(env=env, weight=np.array([0.5, 0.5]), p=2.0)
    >>> for _ in range(10_000):
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
        epsilon_decay: float = 0.99975,
        buffer_capacity: int = 2_000,
        warmup_frac: float = 0.2,
        target_update_freq: int = 100,
        tau: float = 0.005,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(env, weight, p, gamma, epsilon, epsilon_min, epsilon_decay)
        self.warmup_size       = int(warmup_frac * buffer_capacity)
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.device = torch.device(device)
        self._steps = 0

        obs_dim = env.observation_space.shape[0]

        # PyTorch tensors for scalarization — shape (1, reward_dim, 1)
        self.weight_t  = torch.tensor(
            self.weight, dtype=torch.float32, device=self.device
        ).view(1, self.reward_dim, 1)
        self.utopian_t = torch.tensor(
            self.utopian, dtype=torch.float32, device=self.device
        ).view(1, self.reward_dim, 1)

        self.online_net = MultiObjectiveDQN(
            obs_dim=obs_dim,
            reward_dim=self.reward_dim,
            n_actions=self.n_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.target_net = copy.deepcopy(self.online_net)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(buffer_capacity)
        self._state    = None

    # ------------------------------------------------------------------
    # DQN-specific helpers
    # ------------------------------------------------------------------

    def _scalarize_q(self, q_vec: torch.Tensor) -> torch.Tensor:
        """Negated Lp-norm scalarization of vector Q-values (PyTorch).

        Parameters
        ----------
        q_vec : torch.Tensor, shape (batch, reward_dim, n_actions)

        Returns
        -------
        torch.Tensor, shape (batch, n_actions)
            Scalarized Q-values.  The output is negated so that maximizing
            the scalarized Q corresponds to minimizing distance to the
            utopian point.  For finite p:
                s(a) = −( Σ_i  w_i · |Q_i(a) − z_i|^p )^(1/p)
            For p = inf:
                s(a) = −max_i  w_i · |Q_i(a) − z_i|
            where z is the utopian point (broadcast shape (1, reward_dim, 1)).
        """
        assert len(q_vec.shape) == 3, "Q-vector must be a 3D tensor"
        assert q_vec.shape[1] == self.reward_dim, "Q-vector reward dimension must match the input reward dimension"
        assert q_vec.shape[2] == self.n_actions, "Q-vector action dimension must match the input action dimension"
        return scalarize_q(q_vec, self.weight_t, self.p, self.utopian_t)

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert observation array to a float32 device tensor, shape (1, obs_dim)."""
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray, shape (obs_dim,)

        Returns
        -------
        action : int
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_vec    = self.online_net(self._obs_to_tensor(state))
            q_scalar = self._scalarize_q(q_vec)
        return int(q_scalar.argmin(dim=1).item())

    def _update_target(self) -> None:
        """
            Polyak-average online network weights into the target network.

            Applies the soft update:
                θ_target ← (1 - τ) · θ_target + τ · θ_online

            where ``τ = self.tau``.  Setting ``tau=1.0`` recovers a hard copy.
        """
        for t_p, o_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_p.data.mul_(1.0 - self.tau).add_(o_p.data * self.tau)

    def _step(self) -> tuple:
        """Execute one environment step from the current state.

        Selects an action via epsilon-greedy, steps the environment, stores
        the transition in the replay buffer, decays epsilon, and triggers a
        Polyak target-network update every ``target_update_freq`` steps.

        Returns
        -------
        tuple : (state, action, reward, next_state, done)
            ``state`` and ``next_state`` are ``np.ndarray``;
            ``reward`` is the raw multi-objective ``np.ndarray``;
            ``done`` is ``bool``.
        """
        state  = self._state
        action = self._select_action(state)

        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done       = terminated or truncated
        next_state = np.array(next_obs, dtype=np.float32)
        reward     = np.array(reward,   dtype=np.float32)

        self.remember(state, action, reward, next_state, done)
        self._state  = next_state
        self._steps += 1

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if self._steps % self.target_update_freq == 0:
            self._update_target()
            self._steps = 0

        return state, action, reward, next_state, done

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def simulate(self) -> tuple:
        """Run a full episode, storing transitions in the replay buffer.

        Returns
        -------
        tuple : (total_reward, episode_length)
            ``total_reward`` (np.ndarray, shape (reward_dim,)) — sum of raw
            multi-objective reward vectors accumulated over the episode.
            ``episode_length`` (int) — number of steps taken.
        """
        obs, _      = self.env.reset()
        self._state = np.array(obs, dtype=np.float32)

        total_reward   = np.zeros(self.reward_dim, dtype=np.float32)
        episode_length = 0
        done           = False

        while not done:
            _, _, reward, _, done = self._step()
            total_reward   += reward
            episode_length += 1

        return total_reward, episode_length

    def learn(self, batch_size: int = 64, n_epochs: int = 5) -> float | None:
        """Sample from the replay buffer and update the online network.

        Skips learning until the warmup threshold is reached.  The TD target
        uses raw vector rewards so each objective is learned independently:

            Q_i(s,a) ← r_i + γ · Q_i(s', a*)
            a* = argmin_a  _scalarize_q(Q(s', ·))

        Parameters
        ----------
        batch_size : int
        n_epochs : int
            Gradient updates per call.  Default: ``5``.

        Returns
        -------
        float or None
            Mean TD loss (averaged over objectives and epochs), or ``None``
            if still in warmup.

        Notes
        -----
        Rewards are kept as raw vectors throughout the Bellman update so that
        each objective's Q-value is trained consistently on its own cumulative
        reward.  ``_scalarize`` is used **only** for action selection (choosing
        ``a*`` in the next state), keeping the TD equation self-consistent:

            Q_i(s,a) ← r_i + γ · Q_i(s', a*)
            a* = argmin_a  _scalarize(Q(s', a))
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
            rewards_t     = torch.tensor(np.array(rewards),     dtype=torch.float32, device=self.device)

            q_vec_online = self.online_net(states_t)
            action_idx   = actions_t.view(-1, 1, 1).expand(-1, self.reward_dim, 1)
            q_current    = q_vec_online.gather(2, action_idx).squeeze(2)

            with torch.no_grad():
                q_vec_next = self.target_net(next_states_t)
                a_next     = self._scalarize_q(q_vec_next).argmin(dim=1)
                a_next_idx = a_next.view(-1, 1, 1).expand(-1, self.reward_dim, 1)
                q_next_vec = q_vec_next.gather(2, a_next_idx).squeeze(2)

            td_target = rewards_t + self.gamma * q_next_vec * (1.0 - dones_t.unsqueeze(1))

            loss = self.loss_fn(q_current, td_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_epochs


# ---------------------------------------------------------------------------
# Tabular Agent
# ---------------------------------------------------------------------------

class TabularAgent(BaseAgent):
    """Tabular multi-objective Q-learning agent (first-visit Monte Carlo).

    Stores ``Q[s, reward_dim, n_actions]`` — the expected cumulative vector
    reward for each state-action pair.  After each full episode the Q-table is
    updated via first-visit Monte Carlo returns:

        G_t = r_t + γ · G_{t+1}                    (vector)
        Q[s, :, a] ← Q[s, :, a] + α · (G_t − Q[s, :, a])   (first visit only)

    Scalarization is applied only at action selection:

        a* = argmin_a  scalarize( Q[s, :, ·].T )

    Parameters
    ----------
    env : gymnasium.Env
        Observation must be ``[row, col]`` (Deep Sea Treasure grid position).
    weight : np.ndarray, shape (reward_dim,)
    p : float
    alpha : float
        MC step-size.  Default: ``0.1``.
    gamma : float
        Default: ``0.99``.
    epsilon : float
        Default: ``1.0``.
    epsilon_min : float
        Default: ``0.05``.
    epsilon_decay : float
        Per-step decay.  Default: ``0.9995``.
    """

    def __init__(
        self,
        env,
        weight: np.ndarray,
        p: float,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        **kwargs,
    ) -> None:
        super().__init__(env, weight, p, gamma, epsilon, epsilon_min, epsilon_decay)

        self.alpha = alpha

        obs_high    = env.observation_space.high
        self.n_cols = int(obs_high[1]) + 1
        n_rows      = int(obs_high[0]) + 1
        self.n_states = n_rows * self.n_cols

        self.Q = np.zeros(
            (self.n_states, self.reward_dim, self.n_actions), dtype=np.float64
        )
        self._trajectory: list[tuple] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_idx(self, obs: np.ndarray) -> int:
        """Map grid observation ``[row, col]`` to a flat state index."""
        return int(obs[0]) * self.n_cols + int(obs[1])

    def _select_action(self, state_idx: int) -> int:
        """Epsilon-greedy action using ``self.scalarize`` on Q-rows."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        scores = self.scalarize(self.Q[state_idx].T)   # (n_actions,)
        return int(np.argmin(scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self) -> tuple:
        """Run a full episode and store the trajectory for ``learn()``.

        Returns
        -------
        tuple : (total_reward, episode_length)
        """
        self._trajectory = []
        obs, _ = self.env.reset()
        obs    = np.array(obs, dtype=np.float32)

        total_reward   = np.zeros(self.reward_dim, dtype=np.float64)
        episode_length = 0
        done           = False

        while not done:
            s_idx  = self._state_to_idx(obs)
            action = self._select_action(s_idx)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done   = terminated or truncated
            reward = np.array(reward, dtype=np.float64)
            obs    = np.array(next_obs, dtype=np.float32)

            self._trajectory.append((s_idx, action, reward))
            total_reward   += reward
            episode_length += 1
            self.epsilon    = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return total_reward.astype(np.float32), episode_length

    def learn(self, batch_size: int = None) -> float:
        """First-visit MC update on the stored episode trajectory.

        Returns
        -------
        float
            Mean absolute update magnitude, or ``0.0`` if trajectory is empty.
        """
        if not self._trajectory:
            return 0.0

        T      = len(self._trajectory)
        G_vecs = np.zeros((T, self.reward_dim), dtype=np.float64)
        G      = np.zeros(self.reward_dim, dtype=np.float64)
        for t in reversed(range(T)):
            G = self._trajectory[t][2] + self.gamma * G
            G_vecs[t] = G

        visited: set[tuple[int, int]] = set()
        total_delta = 0.0
        n_updates   = 0

        for t in range(T):
            s_idx, action, _ = self._trajectory[t]
            key = (s_idx, action)
            if key in visited:
                continue
            visited.add(key)
            delta = G_vecs[t] - self.Q[s_idx, :, action]
            self.Q[s_idx, :, action] += self.alpha * delta
            total_delta += float(np.abs(delta).mean())
            n_updates   += 1

        self._trajectory = []
        return total_delta / max(n_updates, 1)


# ---------------------------------------------------------------------------
# CEM Agent
# ---------------------------------------------------------------------------

class CEMAgent(BaseAgent):
    """Cross-Entropy Method agent for tabular MORL.

    Maintains a tabular softmax policy ``π(a|s) = softmax(logits[s])``.
    Each ``learn()`` call generates ``n_rollouts`` full episodes, scores them
    by scalarized cumulative return ``f_w(G_0_vec)``, selects the top elite
    fraction, and updates ``logits`` toward the MLE action frequencies:

        logits[s, a] ← (1 − lr) · logits[s, a]
                      + lr · log( count(a | s in elite) + smooth )

    Parameters
    ----------
    env : gymnasium.Env
        Observation must be ``[row, col]``.
    weight : np.ndarray, shape (reward_dim,)
    p : float
    gamma : float
        Default: ``0.99``.
    n_rollouts : int
        Episodes per ``learn()`` call.  Default: ``50``.
    elite_frac : float
        Top fraction kept as elite.  Default: ``0.2``.
    lr : float
        Logit blend rate.  Default: ``0.3``.
    smooth : float
        Laplace smoothing for action counts.  Default: ``1e-3``.
    epsilon : float
        Initial exploration rate for ``simulate()``.  Default: ``0.1``.
    epsilon_min : float
        Default: ``0.05``.
    epsilon_decay : float
        Per-step decay in ``simulate()``.  Default: ``0.9999``.
    """

    def __init__(
        self,
        env,
        weight: np.ndarray,
        p: float,
        gamma: float = 0.99,
        n_rollouts: int = 50,
        elite_frac: float = 0.2,
        lr: float = 0.3,
        smooth: float = 1e-3,
        epsilon: float = 0.1,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9999,
        **kwargs,
    ) -> None:
        super().__init__(env, weight, p, gamma, epsilon, epsilon_min, epsilon_decay)

        self.n_rollouts = n_rollouts
        self.lr         = lr
        self.smooth     = smooth
        self.n_elite    = max(1, math.ceil(n_rollouts * elite_frac))

        obs_high    = env.observation_space.high
        self.n_cols = int(obs_high[1]) + 1
        n_rows      = int(obs_high[0]) + 1
        self.n_states = n_rows * self.n_cols

        self.logits = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_idx(self, obs: np.ndarray) -> int:
        """Map grid observation ``[row, col]`` to a flat state index."""
        return int(obs[0]) * self.n_cols + int(obs[1])

    def _policy_probs(self, state_idx: int) -> np.ndarray:
        """Numerically stable softmax over ``logits[state_idx]``."""
        lg  = self.logits[state_idx]
        lg  = lg - lg.max()
        exp = np.exp(lg)
        return exp / exp.sum()

    def _select_action(self, state_idx: int) -> int:
        """Epsilon-greedy action via current softmax policy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        probs = self._policy_probs(state_idx)
        if self.epsilon == 0.0:
            return int(np.argmax(probs))
        return int(np.random.choice(self.n_actions, p=probs))

    def _run_episode(self) -> tuple:
        """Run one stochastic episode (no epsilon) for ``learn()`` rollouts.

        Returns
        -------
        tuple : (trajectory, G_0_vec)
            trajectory — list of ``(state_idx, action, reward_vec)``
            G_0_vec    — np.ndarray, shape (reward_dim,)
        """
        obs, _ = self.env.reset()
        obs    = np.array(obs, dtype=np.float32)

        trajectory   = []
        total_reward = np.zeros(self.reward_dim, dtype=np.float64)
        done         = False

        while not done:
            s_idx  = self._state_to_idx(obs)
            probs  = self._policy_probs(s_idx)
            action = int(np.random.choice(self.n_actions, p=probs))

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done   = terminated or truncated
            reward = np.array(reward, dtype=np.float64)
            obs    = np.array(next_obs, dtype=np.float32)

            trajectory.append((s_idx, action, reward))
            total_reward += reward

        # Discounted cumulative vector return from step 0
        G = np.zeros(self.reward_dim, dtype=np.float64)
        for _, _, r in reversed(trajectory):
            G = r + self.gamma * G

        return trajectory, G

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self) -> tuple:
        """Run one episode with the current policy for logging/epsilon decay.

        Returns
        -------
        tuple : (total_reward, episode_length)
        """
        obs, _ = self.env.reset()
        obs    = np.array(obs, dtype=np.float32)

        total_reward   = np.zeros(self.reward_dim, dtype=np.float64)
        episode_length = 0
        done           = False

        while not done:
            s_idx  = self._state_to_idx(obs)
            action = self._select_action(s_idx)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done   = terminated or truncated
            reward = np.array(reward, dtype=np.float64)
            obs    = np.array(next_obs, dtype=np.float32)

            total_reward   += reward
            episode_length += 1
            self.epsilon    = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return total_reward.astype(np.float32), episode_length

    def learn(self, batch_size: int = None) -> float:
        """Run one CEM iteration and update the policy.

        Generates ``n_rollouts`` stochastic episodes, selects the top
        ``n_elite`` by scalarized return, and updates logits via MLE.

        Returns
        -------
        float
            Mean scalarized return of the elite rollouts.
        """
        trajectories = []
        G_vecs       = []

        for _ in range(self.n_rollouts):
            traj, G = self._run_episode()
            trajectories.append(traj)
            G_vecs.append(G)

        returns_arr = np.array(G_vecs)                                     # (n_rollouts, reward_dim)
        scores      = self.scalarize(returns_arr)                          # (n_rollouts,)
        elite_idx   = np.argsort(scores)[:self.n_elite]

        counts = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        for idx in elite_idx:
            for s_idx, action, _ in trajectories[idx]:
                counts[s_idx, action] += 1.0

        for s in np.where(counts.sum(axis=1) > 0)[0]:
            new_logits      = np.log(counts[s] + self.smooth)
            self.logits[s]  = (1.0 - self.lr) * self.logits[s] + self.lr * new_logits

        return float(scores[elite_idx].mean())
