"""
DQN Networks for Deep Sea Treasure (MO-Gymnasium).

This module defines two Q-networks used by DQN agents:

- ``DQN``               — standard single-objective network; output shape
                          ``(batch_size, n_actions)``
- ``MultiObjectiveDQN`` — multi-objective network; output shape
                          ``(batch_size, reward_dim, n_actions)``

Environment specifics
---------------------
- Observation space : Box(0, 11, (2,), int32)  — (row, col) grid position
- Action space      : Discrete(4)              — up, down, left, right
- Reward space      : 2-D vector               — (time_penalty, treasure_value)

Classes
-------
DQN
    Single-objective MLP Q-network. Use after scalarizing the reward externally.
MultiObjectiveDQN
    Multi-objective MLP Q-network. Outputs one Q-value per (objective, action)
    pair without any scalarization inside the network.

Usage
-----
    from dqn import DQN, MultiObjectiveDQN
    import torch

    obs = torch.zeros(8, 2)                                    # batch of 8

    net_so = DQN(obs_dim=2, n_actions=4, hidden_dims=[64, 64])
    q_so   = net_so(obs)                                       # shape [8, 4]

    net_mo = MultiObjectiveDQN(obs_dim=2, reward_dim=2, n_actions=4)
    q_mo   = net_mo(obs)                                       # shape [8, 2, 4]
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Single-objective MLP Q-network.

    Maps a flat observation vector to Q-values for each discrete action.
    The reward must be scalarized externally before training.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the (flattened) observation. For Deep Sea Treasure
        this is 2 (row, col). Default: 2.
    n_actions : int
        Number of discrete actions. For Deep Sea Treasure this is 4.
        Default: 4.
    hidden_dims : list[int]
        Sizes of the hidden layers in order. A ReLU activation is inserted
        between every pair of consecutive layers. Default: [64, 64].

    Input
    -----
    x : torch.Tensor, shape (batch_size, obs_dim), dtype float32
        Batch of observations. Observations should be normalised to [0, 1]
        before being passed in (divide by the grid maximum, e.g. 10).

    Output
    ------
    torch.Tensor, shape (batch_size, n_actions), dtype float32
        Raw (un-normalised) Q-values for each action.

    Example
    -------
    >>> net = DQN(obs_dim=2, n_actions=4, hidden_dims=[64, 64])
    >>> obs = torch.zeros(8, 2)        # batch of 8 observations
    >>> q   = net(obs)                 # shape: [8, 4]
    """

    def __init__(
        self,
        obs_dim: int = 2,
        n_actions: int = 4,
        hidden_dims: list[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        layer_sizes = [obs_dim] + hidden_dims
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of observations.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, obs_dim), dtype float32

        Returns
        -------
        torch.Tensor, shape (batch_size, n_actions), dtype float32
        """
        return self.net(x)


class MultiObjectiveDQN(nn.Module):
    """Multi-objective MLP Q-network.

    Maps a flat observation vector to per-objective Q-values for every
    discrete action. The output tensor Q has shape
    ``(batch_size, reward_dim, n_actions)``, where ``Q[:, n, a]`` is the
    expected return for objective ``n`` when taking action ``a``.

    No scalarization is applied inside this network. Scalarization is
    performed externally using the weight vector w > 0 (strictly positive
    orthant) and the chosen Lp-norm.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the (flattened) observation. For Deep Sea Treasure
        this is 2 (row, col). Default: 2.
    reward_dim : int
        Number of reward objectives. For Deep Sea Treasure this is 2
        (time_penalty, treasure_value). Default: 2.
    n_actions : int
        Number of discrete actions. For Deep Sea Treasure this is 4.
        Default: 4.
    hidden_dims : list[int]
        Sizes of the hidden layers in order. A ReLU activation is inserted
        between every pair of consecutive layers. Default: [64, 64].

    Input
    -----
    x : torch.Tensor, shape (batch_size, obs_dim), dtype float32
        Batch of observations. Observations should be normalised to [0, 1]
        before being passed in (divide by the grid maximum, e.g. 10).

    Output
    ------
    torch.Tensor, shape (batch_size, reward_dim, n_actions), dtype float32
        Raw (un-normalised) Q-values for every (objective, action) pair.

    Example
    -------
    >>> net = MultiObjectiveDQN(obs_dim=2, reward_dim=2, n_actions=4)
    >>> obs = torch.zeros(8, 2)        # batch of 8 observations
    >>> q   = net(obs)                 # shape: [8, 2, 4]
    >>> q[:, 0, :]                     # time-penalty  Q-values, shape [8, 4]
    >>> q[:, 1, :]                     # treasure-val  Q-values, shape [8, 4]
    """

    def __init__(
        self,
        obs_dim: int = 2,
        reward_dim: int = 2,
        n_actions: int = 4,
        hidden_dims: list[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.reward_dim = reward_dim
        self.n_actions = n_actions

        layer_sizes = [obs_dim] + hidden_dims
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], reward_dim * n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-objective Q-values for a batch of observations.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, obs_dim), dtype float32

        Returns
        -------
        torch.Tensor, shape (batch_size, reward_dim, n_actions), dtype float32
        """
        batch_size = x.shape[0]
        flat = self.net(x)
        return flat.view(batch_size, self.reward_dim, self.n_actions)
