"""
Evaluation utilities for Lp-norm scalarization on Deep Sea Treasure.

Functions
---------
compute_vref(env)
    Compute the worst-case hypervolume reference point from the environment's
    true Pareto front (nadir point minus a small epsilon).

compute_hypervolume(env_name, model, p, outputs_dir, last_n)
    Compute the hypervolume indicator of the coverage set obtained by sweeping
    all weight vectors for a fixed (env, model, p) configuration.
"""

import re
from pathlib import Path

import mo_gymnasium as mo_gym
import numpy as np

from utils import compute_utopian, scalarize_vec

def compute_vref(
    env,
    max_steps: int = 100,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Compute the worst-case hypervolume reference point V_ref.

    V_ref is set to the **nadir point**: component-wise minimum across all
    Pareto-optimal solutions, using the known worst-case time penalty
    (−max_steps = −100) for the time dimension.  A small epsilon is subtracted
    to ensure V_ref is strictly dominated by every Pareto solution, keeping the
    hypervolume indicator positive and well-defined.

    Parameters
    ----------
    env : gymnasium.Env
        An MO-Gymnasium environment whose ``unwrapped`` instance exposes
        ``pareto_front(gamma)``.  The environment is not modified or closed.
    max_steps : int
        Maximum number of steps in an episode.  Default: 100.
    epsilon: float
        Small epsilon to shift the nadir point strictly below the Pareto front.  Default: 1e-3.
    
    Returns
    -------
    np.ndarray, shape (reward_dim,)
        V_ref = component-wise min over Pareto front − epsilon
    """
    pf = np.array(env.unwrapped.pareto_front(gamma=1.0), dtype=np.float64)
    # Nadir: worst treasure from Pareto front; worst time penalty is -max_steps=-100
    nadir = np.array([pf[:, 0].min(), -max_steps])
    v_ref = nadir - epsilon
    return v_ref


# ---------------------------------------------------------------------------
# True Pareto-front hypervolume (upper bound)
# ---------------------------------------------------------------------------

def compute_pareto_hv(env) -> float:
    """Compute the hypervolume of the environment's true Pareto front.

    This value is the theoretical upper bound for any coverage-set hypervolume
    on the same environment.  Dividing a coverage-set HV by this value gives
    the **normalised hypervolume** ∈ [0, 1].

    Parameters
    ----------
    env : gymnasium.Env
        MO-Gymnasium environment (not modified or closed).

    Returns
    -------
    float
        Hypervolume of the true Pareto front w.r.t. ``compute_vref(env)``.
    """
    v_ref = compute_vref(env)
    pf    = np.array(env.unwrapped.pareto_front(gamma=1.0), dtype=np.float64)
    pf    = pf[np.argsort(pf[:, 0])]   # sort by r_0 ascending

    hv      = 0.0
    r0_prev = v_ref[0]
    for r0, r1 in pf:
        hv      += (r0 - r0_prev) * (r1 - v_ref[1])
        r0_prev  = r0
    return float(hv)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _non_dominated(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of ``points`` for 2D maximization.

    Point A dominates point B when ``A >= B`` component-wise with at least one
    strict inequality.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)

    Returns
    -------
    np.ndarray, shape (M, 2)  where M <= N
    """
    dominated = np.zeros(len(points), dtype=bool)
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                continue
            if np.all(b >= a) and np.any(b > a):
                dominated[i] = True
                break
    return points[~dominated]


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------

def compute_hypervolume(
    env_name: str,
    model: str,
    p: float,
    outputs_dir: str = "outputs",
    last_n: int = 50,
) -> float:
    """Compute the hypervolume indicator of the coverage set.

    For a fixed ``(env_name, model, p)`` configuration the function:

    1. Iterates over every stored weight directory.
    2. For each weight, selects the episode with the lowest Lp-distance to the
       utopian point (best scalarized return) among the last ``last_n`` episodes
       — the same selection rule used in ``plots.py``.
    3. Collects these best rewards into a **coverage set** CS.
    4. Filters CS to its non-dominated subset.
    5. Computes the hypervolume via the 2-D sweepline formula:

       .. math::
           \\text{HV}(\\text{CS}, V_{\\text{ref}}) =
               \\bigcup_{\\pi \\in \\text{CS}}
               \\text{Volume}(V_{\\text{ref}},\\, V_\\pi)

       Concretely, after sorting the non-dominated points by :math:`r_0`
       ascending (so :math:`r_1` is descending):

       .. math::
           \\text{HV} = \\sum_i (r_0^{(i)} - r_0^{(i-1)})
                                \\cdot (r_1^{(i)} - V_{\\text{ref},1})

       where :math:`r_0^{(0)} = V_{\\text{ref},0}`.

    Parameters
    ----------
    env_name : str
        MO-Gymnasium environment id, e.g. ``"deep-sea-treasure-v0"``.
    model : str
        Agent architecture: ``"tabular"``, ``"cem"``, or ``"dqn"``.
    p : float
        Lp-norm exponent used during training.  Use ``float("inf")`` for
        Chebyshev.
    outputs_dir : str
        Root outputs directory created by ``train.py``.  Default: ``"outputs"``.
    last_n : int
        Number of final episodes to consider when selecting the best reward.
        Default: ``50``.

    Returns
    -------
    float
        Hypervolume indicator value.  Returns ``0.0`` if no weight directories
        are found.
    """
    env = mo_gym.make(env_name)
    v_ref   = compute_vref(env)
    utopian = compute_utopian(env)
    env.close()

    p_tag       = "inf" if p == float("inf") else str(p)
    weight_dirs = sorted(Path(outputs_dir).glob(
        f"{env_name}/{model}/p_{p_tag}/w_*/"
    ))

    if not weight_dirs:
        return 0.0

    coverage = []
    for wdir in weight_dirs:
        match = re.search(r"w_([\d.]+)_([\d.]+)$", wdir.name)
        if match is None:
            continue
        weight  = np.array([float(match.group(1)), float(match.group(2))],
                            dtype=np.float32)
        rewards = np.load(wdir / "rewards.npy")       # (n_episodes, 2)
        rewards = rewards[-last_n:]
        scores  = scalarize_vec(rewards, weight, p, utopian)
        coverage.append(rewards[np.argmin(scores)])

    if not coverage:
        return 0.0

    cs = np.array(coverage, dtype=np.float64)         # (N, 2)
    nd = _non_dominated(cs)                           # (M, 2)

    # Sort by r_0 ascending → r_1 is descending for a non-dominated set
    nd = nd[np.argsort(nd[:, 0])]

    hv       = 0.0
    r0_prev  = v_ref[0]
    for r0, r1 in nd:
        hv      += (r0 - r0_prev) * (r1 - v_ref[1])
        r0_prev  = r0

    return float(hv)


if __name__ == "__main__":
    envs   = ["deep-sea-treasure-v0", "deep-sea-treasure-concave-v0"]
    models = ["tabular", "cem"]
    ps     = [1.0, 2.0, 4.0, 8.0, float("inf")]

    # V_ref check
    for env_name in envs:
        env   = mo_gym.make(env_name)
        v_ref = compute_vref(env)
        env.close()
        print(f"{env_name}: V_ref = {np.round(v_ref, 4)}")

    print()

    # True Pareto-front hypervolumes (upper bounds)
    pf_hvs = {}
    for env_name in envs:
        env = mo_gym.make(env_name)
        pf_hvs[env_name] = compute_pareto_hv(env)
        env.close()
        print(f"Pareto-front HV  {env_name}: {pf_hvs[env_name]:.4f}")

    print()

    # Coverage-set hypervolume + normalised
    for env_name in envs:
        for model in models:
            for p in ps:
                p_label = "∞" if p == float("inf") else str(p)
                p_tag   = "inf" if p == float("inf") else str(p)
                hv      = compute_hypervolume(env_name, model, p)
                hv_norm = hv / pf_hvs[env_name]
                print(
                    f"HV  env={env_name:<35s}  model={model:<8s}  p={p_label:<4s}"
                    f"  → {hv:.4f}  (norm: {hv_norm:.4f})"
                )

                out_dir = Path("outputs") / env_name / model / f"p_{p_tag}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "hypervolume.npy",            np.array(hv))
                np.save(out_dir / "hypervolume_normalized.npy", np.array(hv_norm))
