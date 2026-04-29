"""
Pareto front plotting utilities for Deep Sea Treasure (MO-Gymnasium).

Functions
---------
plot_pareto_front(env_name, ax, gamma)
    Plot the true Pareto front as a step-like thick black line with 'x' markers.

Usage
-----
    python plot.py
    python plot.py --env deep-sea-treasure-v0 --save
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
import numpy as np

from utils import compute_utopian, scalarize_vec


def plot_pareto_front(
    env_name: str,
    ax: plt.Axes | None = None,
    gamma: float = 1.0,
) -> plt.Axes:
    """Plot the true Pareto front as a staircase line with 'x' markers.

    The step-like (staircase) shape reflects the dominance structure of the
    Pareto front: each horizontal segment shows the range of treasure values
    for which a given time-penalty is not yet dominated, and the vertical drop
    shows the cost of moving to the next Pareto point.

    Parameters
    ----------
    env_name : str
        MO-Gymnasium environment id, e.g. ``"deep-sea-treasure-v0"``.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure and axes are created if ``None``.
        Default: ``None``.
    gamma : float
        Discount factor passed to ``pareto_front()``.  Default: ``1.0``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot (useful for embedding in larger figures).
    """
    env = mo_gym.make(env_name)
    pf_list = env.unwrapped.pareto_front(gamma=gamma)
    env.close()

    pf = np.array(pf_list, dtype=np.float64)
    pf = pf[np.argsort(pf[:, 0])]          # sort by treasure r_0 ascending

    if ax is None:
        _, ax = plt.subplots(figsize=(20, 10))

    # Step-like Pareto front line
    ax.step(pf[:, 0], pf[:, 1],
            where="post",
            color="black", linewidth=2.5,
            label="Pareto front", zorder=2)

    # Mark each Pareto point with an 'x'
    ax.scatter(pf[:, 0], pf[:, 1],
               marker="x", color="black", s=70, linewidths=2.0,
               zorder=3)

    ax.set_xlabel("Treasure reward  $r_0$", fontsize=12)
    ax.set_ylabel("Time penalty  $r_1$", fontsize=12)
    ax.set_title(f"Pareto Front — {env_name}", fontsize=12)
    ax.grid(True, alpha=0.3)

    r0_pad = (pf[:, 0].max() - pf[:, 0].min()) * 0.05
    r1_pad = abs(pf[:, 1].max() - pf[:, 1].min()) * 0.05
    ax.set_xlim(pf[:, 0].min() - r0_pad, pf[:, 0].max() + r0_pad)
    ax.set_ylim(pf[:, 1].min() - r1_pad, pf[:, 1].max() + r1_pad)

    return ax, pf


def plot_reward(
    ax: plt.Axes,
    reward: np.ndarray,
    color: str = "red",
    label: str = "Achieved reward",
) -> plt.Axes:
    """Scatter a reward vector on an existing axes as a hollow circle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on (typically one already containing a Pareto front).
    reward : np.ndarray, shape (2,)
        Reward vector ``[r_0, r_1]`` to mark.
    color : str
        Edge colour of the hollow circle.  Default: ``"red"``.
    label : str
        Legend label.  Default: ``"Achieved reward"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    reward = np.asarray(reward, dtype=np.float64).ravel()
    ax.scatter(reward[0], reward[1],
               marker="o", facecolors="none", edgecolors=color,
               s=100, linewidths=4.0,
               zorder=4, label=label)
    return ax

def plot_scalarization(
    ax: plt.Axes,
    weight: np.ndarray,
    p: float,
    rewards: np.ndarray,
    utopian: np.ndarray,
) -> plt.Axes:
    """
        Plot the scalarization as a red line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    weight : np.ndarray, shape (2,)
        Weight vector.
    p : float
        Lp-norm exponent.
    rewards : np.ndarray, shape (n_episodes, 2)
        Reward vector.
    utopian : np.ndarray, shape (2,)
        Utopian point.
    """
    c = float(scalarize_vec(rewards, weight, p, utopian))
    label = "Scalarization curves"

    if p == float("inf"):
        # L-shape: corner at (utopian[0] - c/w0, utopian[1] - c/w1)
        # Horizontal arm: r0 ∈ [corner_r0, utopian[0]], r1 = corner_r1
        # Vertical arm:   r0 = corner_r0, r1 ∈ [corner_r1, utopian[1]]
        corner_r0 = utopian[0] - c / weight[0]
        corner_r1 = utopian[1] - c / weight[1]
        ax.plot([corner_r0, utopian[0]], [corner_r1, corner_r1],
                '--', color="red", linewidth=1, label=label)
        ax.plot([corner_r0, corner_r0], [corner_r1, utopian[1]],
                '--', color="red", linewidth=1, label="_nolegend_")
    else:
        # Smooth arc: r1 = utopian[1] - (c^p - (w0*(utopian[0]-r0))^p)^(1/p) / w1
        # Valid for r0 ∈ [utopian[0] - (c/w0), utopian[0]]
        r0_lo = utopian[0] - c / weight[0]
        r0 = np.linspace(r0_lo, utopian[0], 1000)
        inner = c ** p - (weight[0] * (utopian[0] - r0)) ** p
        mask = inner >= 0
        r1 = utopian[1] - inner[mask] ** (1.0 / p) / weight[1]
        ax.plot(r0[mask], r1, '--', color="red", linewidth=1, label=label)
    return ax

def plot_utopian(
    env_name: str,
    ax: plt.Axes,
) -> tuple[plt.Axes, np.ndarray]:
    """
        Plot the utopian point as a blue star.
    """
    env = mo_gym.make(env_name)
    utopian = compute_utopian(env)
    env.close()

    ax.scatter(
        *utopian, 
        marker="*", 
        s=300, 
        color="blue",
        label=f"Utopian r* = {np.round(utopian, 1)}"
    )
    return ax, utopian

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the true Pareto front and overlay best achieved rewards."
    )
    parser.add_argument(
        "--env",
        nargs="+",
        default=["deep-sea-treasure-v0", "deep-sea-treasure-concave-v0"],
        help="Environment id(s).  Default: both DST variants.",
    )
    parser.add_argument(
        "--p",
        type=lambda x: float("inf") if x in ("inf", "Inf", "INF") else float(x),
        default=None,
        help="Lp-norm exponent used during training.  Default: sweep all p dirs found.",
    )
    parser.add_argument(
        "--model",
        default="tabular",
        choices=["tabular", "cem", "dqn"],
        help="Agent architecture.  Default: tabular",
    )
    parser.add_argument(
        "--outputs_dir",
        default="outputs",
        help="Root outputs directory.  Default: outputs",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save each figure to outputs/<env>/pareto_front_p_<p>.png",
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=50,
        help="Number of episodes to average for the achieved reward.  Default: 50",
    )
    args = parser.parse_args()

    for env_name in args.env:
        # Discover available p directories
        base = Path(args.outputs_dir) / env_name / args.model
        if args.p is None:
            p_dirs = sorted(base.glob("p_*/"))
        else:
            p_tag_single = "inf" if args.p == float("inf") else str(args.p)
            p_dirs = sorted(base.glob(f"p_{p_tag_single}/"))

        for p_dir in p_dirs:
            best_rewards = []
            best_weights = []

            # Parse p value from directory name
            raw = p_dir.name[2:]          # strip leading "p_"
            p_val = float("inf") if raw == "inf" else float(raw)
            p_tag = raw
            p_label = "∞" if p_val == float("inf") else p_tag

            fig, ax = plt.subplots(figsize=(20, 10))
            ax, pf = plot_pareto_front(env_name, ax=ax)
            ax, utopian = plot_utopian(env_name, ax=ax)

            weight_dirs = sorted(p_dir.glob("w_*/"))
            for wdir in weight_dirs:
                match = re.search(r"w_([\d.]+)_([\d.]+)$", wdir.name)
                if match is None:
                    continue
                weight = np.array([float(match.group(1)), float(match.group(2))],
                                  dtype=np.float32)
                rewards = np.load(wdir / "rewards.npy")   # (n_episodes, 2)

                # Choosing the last n episodes
                rewards = rewards[-args.last_n:]

                scores = scalarize_vec(rewards, weight, p_val, utopian)
                best_reward = rewards[np.argmin(scores)]

                on_front = np.linalg.norm(pf - best_reward, axis=1).min() < 0.5
                if on_front:
                    plot_reward(ax, best_reward, color="red",
                                label="On Pareto front")
                    if not any(np.array_equal(best_reward, r) for r in best_rewards):
                        best_rewards.append(best_reward)
                        best_weights.append(weight)
                else:
                    plot_reward(ax, best_reward, color="gray",
                                label="Off Pareto front")
            
            for weight, reward in zip(best_weights, best_rewards):
                plot_scalarization(ax, weight, p_val, reward, utopian)

            # Deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, labels):
                seen.setdefault(l, h)
            ax.legend(seen.values(), seen.keys(), fontsize=10)

            ax.set_title(
                f"Pareto Front — {env_name}  |  p={p_label}  |  {args.model}",
                fontsize=11,
            )
            fig.tight_layout()

            if args.save:
                out_dir = Path(args.outputs_dir) / env_name / args.model / f"p_{p_tag}"
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / "pareto_front.png"
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved to {save_path}")

            plt.show()
