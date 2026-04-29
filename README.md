# Evaluating Lp-Norm Scalarization Methods for Pareto Front Approximation in Multi-Objective Reinforcement Learning

**ECE 406 — Introduction to Multi-Objective Machine Learning**
**University of Rochester, Spring 2026**
**Authors:** Jiajun Wu, Venkatakrishnan V K
**Instructor:** Lisha Chen

---

## Problem Description

This project systematically evaluates how Lp-norm scalarization methods affect the quality and diversity of solutions learned in a Multi-Objective Reinforcement Learning (MORL) setting.

We train RL agents on the **Deep Sea Treasure** benchmark from [MO-Gymnasium](https://github.com/Farama-Foundation/momaland), using both:

- `deep-sea-treasure-v0` — convex Pareto front
- `deep-sea-treasure-concave-v0` — concave Pareto front

and sweep over the scalarization parameter:

$$p \in [1, \infty]$$

where $p = 1$ recovers **linear scalarization** and $p = \infty$ recovers **Chebyshev (Tchebyshev) scalarization**.

---

## Background and Motivation

Many real-world decision-making problems require balancing multiple competing objectives simultaneously. Standard RL methods assume a single scalar reward, which encodes all preferences in advance and limits post-hoc explainability. MORL addresses this by modeling rewards as vectors and computing policies that represent diverse trade-offs among objectives.

**Scalarization** is one of the most practical strategies in MORL: the vector reward is collapsed into a scalar objective and then standard RL is applied. However, the choice of scalarization function strongly influences which parts of the Pareto front can be recovered.

- **Linear scalarization** ($p=1$): easy to interpret and widely used, but can only recover *supported* Pareto-optimal solutions. It fails to identify Pareto-optimal points on non-convex regions of the front.
- **Chebyshev scalarization** ($p=\infty$): provides a different geometric bias and can recover solutions in non-convex regions.
- **Lp-norm scalarization** ($1 < p < \infty$): interpolates between the two extremes, offering a continuum of biases.

The Deep Sea Treasure environment is ideal for this study because:
1. The reward function is 2-dimensional, making the Pareto front easy to visualize.
2. Both convex and concave Pareto front variants are available, allowing direct comparison against theoretical predictions from multi-objective optimization.

---

## Environment

| Property | Value |
|---|---|
| Environment | Deep Sea Treasure (MO-Gymnasium) |
| Variants | `deep-sea-treasure-v0` (convex), `deep-sea-treasure-concave-v0` (concave) |
| Objectives | Time penalty, Treasure value |
| Scalarization parameter | $p \in \{1, 2, 4, 8, \infty\}$ |

---

## Methodology

For each value of $p$ and each weight vector $\mathbf{w} \succ 0$ (i.e., $\mathbf{w}$ lies in the strictly positive orthant cone, $w_i > 0$ for all $i$), we define the Lp-norm scalarization using the **utopian point** $\mathbf{r}^\star$ (component-wise maximum over the Pareto front) as:

$$f_{\mathbf{w}}(\mathbf{r}) = \left( \sum_i \left| w_i \left( r_i - r^\star_i \right) \right|^p \right)^{1/p}$$

with the limit $p \to \infty$ (Chebyshev scalarization) corresponding to:

$$f_{\mathbf{w}}(\mathbf{r}) = \max_i \left| w_i \left( r_i - r^\star_i \right) \right|$$

The utopian point $\mathbf{r}^\star$ is computed once from the environment's true Pareto front and cached to `outputs/<env_name>/utopian.npy`.

### SER vs ESR

We adopt the **Scalarized Expected Returns (SER)** objective rather than Expected Scalarized Returns (ESR).  Under SER, the agent first accumulates the full episode's cumulative vector return:

$$\mathbf{G}_0 = \sum_{t \geq 0} \gamma^t \, \mathbf{r}_t \qquad (\text{vector return})$$

and then applies the scalarization to the total:

$$G_0^{\text{scalar}} = f_{\mathbf{w}}(\mathbf{G}_0)$$

This is opposed to ESR, which would scalarize each per-step reward $f_{\mathbf{w}}(\mathbf{r}_t)$ and sum the scalars. In Deep Sea Treasure, all intermediate steps yield the same non-terminal reward vector $[0, -1]$; ESR therefore collapses every non-terminal step to the same scalar, destroying credit-assignment signal. SER preserves the full vector structure until the end of the episode, giving the scalarization function access to the cumulative trade-off actually realized by the policy.

### Agents

We train two families of agents for each $(p, \mathbf{w})$ configuration:

- **Tabular Monte Carlo Q-learning** — maintains a table $Q[s, \text{reward-dim}, \text{action}]$ of expected cumulative vector rewards. After each episode, all state–action pairs are updated via first-visit Monte Carlo returns (vector returns, never scalarized during learning). Scalarization enters only at action selection, where the action minimizing $f_{\mathbf{w}}(Q[s, :, \cdot])$ is chosen.

- **Cross-Entropy Method (CEM)** — a population-based policy-search algorithm. Each iteration generates $N$ full episode rollouts under a softmax tabular policy, computes the cumulative vector return $\mathbf{G}_0$ for each rollout, scalarizes with $f_{\mathbf{w}}$, and selects the top-$k$ elite rollouts (lowest Lp-distance to $\mathbf{r}^\star$). The policy logits are then updated by maximum-likelihood estimation on the elite trajectories.

**Evaluation metrics [1]:**
- Coverage of the true Pareto front
- Hypervolume indicator
- Diversity of recovered solutions across different weight vectors

---

## Project Structure

```
.
├── pyproject.toml          # Project dependencies (managed with uv)
├── README.md
├── agent.py                # BaseAgent, TabularAgent, CEMAgent, DQNAgent
├── dqn.py                  # MultiObjectiveDQN network architecture
├── utils.py                # Scalarization functions, ReplayBuffer, utopian computation
├── train.py                # Sweep training script (env × p × weight)
├── example_plots/          # Representative result figures (used in README)
├── documents/              # Reports, notes, and references
└── MOML_Final_Project_Proposal.pdf
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv sync
source .venv/bin/activate
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mo-gym` | Deep Sea Treasure MORL environment |
| `torch` | Neural network policies |
| `numpy` | Array operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Evaluation utilities |
| `scipy` | Numerical routines |

---

## Results

Each plot shows the true Pareto front (thick black staircase), the utopian point $\mathbf{r}^*$ (blue star), the best achieved reward per weight vector (red hollow circle = on Pareto front, grey hollow circle = suboptimal), and the Lp-norm scalarization level-set curve (red dashed) that passes through each achieved point.

---

### Convex Pareto Front (`deep-sea-treasure-v0`) — Tabular MC

| $p = 1$ (Linear) | $p = 2$ (Euclidean) | $p = \infty$ (Chebyshev) |
|:---:|:---:|:---:|
| ![p=1](example_plots/convex_tabular_p1.png) | ![p=2](example_plots/convex_tabular_p2.png) | ![p=inf](example_plots/convex_tabular_pinf.png) |

**Observation:** On the convex front, all three norms recover most Pareto-optimal solutions. Linear scalarization ($p=1$) produces diagonal level-set curves and struggles with weight vectors that select dense mid-range solutions. Chebyshev ($p=\infty$) produces L-shaped contours that visibly "pin" each solution to a corner of the front.

---

### Convex Pareto Front (`deep-sea-treasure-v0`) — CEM

| $p = 2$ (Euclidean) |
|:---:|
| ![p=2 CEM](example_plots/convex_cem_p2.png) |

---

### Concave Pareto Front (`deep-sea-treasure-concave-v0`) — Tabular MC

| $p = 1$ (Linear) | $p = 2$ (Euclidean) | $p = \infty$ (Chebyshev) |
|:---:|:---:|:---:|
| ![p=1](example_plots/concave_tabular_p1.png) | ![p=2](example_plots/concave_tabular_p2.png) | ![p=inf](example_plots/concave_tabular_pinf.png) |

**Observation:** On the concave front, linear scalarization ($p=1$) fails to recover solutions on the concave segments — the diagonal level-set curves cannot be tangent to the concave region, so many weight vectors collapse to the same supported extreme points (grey circles cluster away from the front). Higher-order norms ($p \geq 2$) and Chebyshev scalarization recover significantly more of the concave Pareto front by using curved or L-shaped contours that can be tangent to non-convex regions.

---

### Concave Pareto Front (`deep-sea-treasure-concave-v0`) — CEM

| $p = 2$ (Euclidean) | $p = \infty$ (Chebyshev) |
|:---:|:---:|
| ![p=2 CEM](example_plots/concave_cem_p2.png) | ![p=inf CEM](example_plots/concave_cem_pinf.png) |

**Observation:** CEM achieves reasonable coverage on the concave front but underperforms Tabular MC, particularly for Chebyshev scalarization ($p=\infty$) where Tabular MC's exact Q-table gives it a stronger advantage — CEM's population-based search introduces more variance and may converge to suboptimal policies before the L-shaped contour aligns with the correct Pareto point.

## LLM Usage Disclosure

The completion of this project was aided by the following LLM:
- **Model:** Claude 4.6 Sonnet
- **API:** Anthropic
The content of this repository has been verified by us manually for accuracy, originality, and completeness. Any errors or omissions are solely our responsibility.

---

## References

[1] Conor F. Hayes, Roxana Rădulescu, Eugenio Bargiacchi, Johan Källström, Matthew Macfarlane, Mathieu Reymond, Timothy Verstraeten, Luisa M. Zintgraf, Richard Dazeley, Fredrik Heintz, et al. *A practical guide to multi-objective reinforcement learning and planning.* Autonomous Agents and Multi-Agent Systems, 36(1):26, 2022.
