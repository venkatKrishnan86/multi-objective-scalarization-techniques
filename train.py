"""
Training script for Lp-norm scalarization sweep on Deep Sea Treasure.

Sweeps over environments, p values, and weight vectors.  For each
(env, p, w) combination a fresh ``Agent`` is trained for ``n_episodes``
full episodes.  Per-episode reward vectors and episode lengths are saved to:

    outputs/<env_name>/p_<p>/w_<w0>_<w1>/rewards.npy
    outputs/<env_name>/p_<p>/w_<w0>_<w1>/lengths.npy

Functions
---------
train_agent(env_name, weight, p, n_episodes, batch_size, agent_kwargs)
    Train a single agent and persist results.
main()
    Parse CLI arguments and run the full sweep.

Usage
-----
    # full sweep (both envs, all p, 9 weight vectors, 500 episodes each)
    python train.py

    # quick test
    python train.py --env deep-sea-treasure-v0 --p 2.0 --n_episodes 50

    # multiple p values
    python train.py --env deep-sea-treasure-v0 --p 1.0 2.0 inf --n_episodes 200
"""

import argparse
from pathlib import Path

import numpy as np
import mo_gymnasium as mo_gym

from agent import DQNAgent, TabularAgent, CEMAgent

def train_agent(
    env_name: str,
    weight: np.ndarray,
    p: float,
    n_episodes: int = 1000,
    batch_size: int = 128,
    model: str = "tabular",
    device: str = "cpu",
    agent_kwargs: dict = None,
) -> dict:
    """Train one agent for a single (env, weight, p) configuration.

    Runs ``n_episodes`` full episodes, calling ``agent.simulate()`` then
    ``agent.learn(batch_size)`` each episode.  Results are saved to:

        outputs/<env_name>/<model>/p_<p>/w_<w0>_<w1>/rewards.npy  — shape (n_episodes, reward_dim)
        outputs/<env_name>/<model>/p_<p>/w_<w0>_<w1>/lengths.npy  — shape (n_episodes,)

    Parameters
    ----------
    env_name : str
        MO-Gymnasium environment id, e.g. ``"deep-sea-treasure-v0"``.
    weight : np.ndarray, shape (reward_dim,)
        Scalarization weight vector (strictly positive orthant).
    p : float
        Lp-norm exponent.  Use ``float("inf")`` for Chebyshev.
    n_episodes : int
        Number of training episodes.  Default: ``1000``.
    batch_size : int
        Replay buffer sample size per ``learn()`` call (DQN only).  Default: ``128``.
    model : str
        Agent architecture: ``"tabular"``, ``"cem"``, or ``"dqn"``.
        Default: ``"tabular"``.
    device : str
        Torch device string (DQN only).  Default: ``"cpu"``.
    agent_kwargs : dict or None
        Extra keyword arguments forwarded to the agent ``__init__``.
        Default: ``None`` (uses agent defaults).

    Returns
    -------
    dict with keys:
        ``"rewards"``  — np.ndarray, shape (n_episodes, reward_dim)
        ``"lengths"``  — np.ndarray, shape (n_episodes,)
        ``"agent"``    — trained agent instance
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    env = mo_gym.make(env_name)

    if model == "tabular":
        agent = TabularAgent(env=env, weight=weight, p=p, **agent_kwargs)
    elif model == "cem":
        agent = CEMAgent(env=env, weight=weight, p=p, **agent_kwargs)
    else:
        agent = DQNAgent(env=env, weight=weight, p=p, device=device, **agent_kwargs)

    rewards_log = []
    lengths_log = []

    for ep in range(1, n_episodes + 1):
        total_reward, ep_len = agent.simulate()
        agent.learn(batch_size=batch_size)

        rewards_log.append(total_reward)
        lengths_log.append(ep_len)

        if ep % max(1, n_episodes // 10) == 0:
            eps_str = f"  ε={agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
            print(
                f"  [{env_name}] {model} p={p} w={np.round(weight, 2)}  "
                f"ep {ep:>4}/{n_episodes}  "
                f"reward={np.round(total_reward, 1)}  "
                f"len={ep_len}{eps_str}"
            )

    rewards_arr = np.array(rewards_log, dtype=np.float32)   # (n_episodes, reward_dim)
    lengths_arr = np.array(lengths_log, dtype=np.int32)     # (n_episodes,)

    p_tag = "inf" if p == float("inf") else str(p)
    w_tag = "_".join(f"{wi:.2f}" for wi in weight)
    out_dir = Path("outputs") / env_name / model / f"p_{p_tag}" / f"w_{w_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "rewards.npy", rewards_arr)
    np.save(out_dir / "lengths.npy", lengths_arr)

    env.close()
    return {"rewards": rewards_arr, "lengths": lengths_arr, "agent": agent}



def main() -> None:
    """Parse CLI arguments and run the full (env × p × weight) sweep."""
    parser = argparse.ArgumentParser(
        description="Lp-norm scalarization sweep on Deep Sea Treasure."
    )
    parser.add_argument(
        "--env",
        nargs="+",
        default=["deep-sea-treasure-v0", "deep-sea-treasure-concave-v0"],
        help="Environment id(s) to train on.",
    )
    parser.add_argument(
        "--p",
        nargs="+",
        type=lambda x: float("inf") if x in ("inf", "Inf", "INF") else float(x),
        default=[1.0, 2.0, 4.0, 8.0, float("inf")],
        help="Lp-norm exponent(s).  Use 'inf' for Chebyshev.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=1_000,
        help="Number of training episodes per configuration.  Default: 1000.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Replay buffer sample size per learn() call.  Default: 128.",
    )
    parser.add_argument(
        "--n_weights",
        type=int,
        default=38,
        help="Number of weight vectors (linearly spaced α in [0.05, 0.95]).  Default: 38.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tabular",
        choices=["tabular", "cem", "dqn"],
        help="Agent architecture: 'tabular' (MC Q-table), 'cem', or 'dqn'.  Default: 'tabular'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (DQN only), e.g. 'cpu' or 'cuda'.  Default: 'cpu'.",
    )
    args = parser.parse_args()

    # weight grid: α ∈ [0.05, 0.95] with step 0.1, w = [α, 1-α]
    alphas  = np.linspace(0.05, 0.95, args.n_weights)
    weights = [np.array([a, 1.0 - a], dtype=np.float32) for a in alphas]

    total = len(args.env) * len(args.p) * len(weights)
    run   = 0

    # Sweeping over environments, p values, and weight vectors.
    for env_name in args.env:
        for p in args.p:
            for weight in weights:
                run += 1
                p_tag = "inf" if p == float("inf") else p
                print(
                    f"\n[{run}/{total}] env={env_name}  p={p_tag}  "
                    f"w={np.round(weight, 2)}"
                )
                train_agent(
                    env_name=env_name,
                    weight=weight,
                    p=p,
                    n_episodes=args.n_episodes,
                    batch_size=args.batch_size,
                    model=args.model,
                    device=args.device,
                )

    print(f"\nSweep complete. Results saved under outputs/")


if __name__ == "__main__":
    main()
