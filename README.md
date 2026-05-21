# MDP & Reinforcement Learning Agent

**Course:** CS 4033 — Artificial Intelligence · University of Oklahoma  
**Year:** 2025  
**Language:** Python

---

## Overview

This project implements two foundational reinforcement learning algorithms — **Value Iteration** and **Q-Learning** — from scratch in Python, applied across three environments of increasing complexity: a custom GridWorld, a simulated Crawler robot, and a Pacman AI agent.

The goal was to develop strong intuition for the core tradeoffs in RL: model-based vs. model-free learning, exploration vs. exploitation, and how policy convergence behaves across different state spaces.

---

## Algorithms Implemented

### Value Iteration (Model-Based)
- Implements the Bellman optimality equation iteratively until convergence
- Computes the optimal value function `V*(s)` and derives the optimal policy `π*(s)`
- Applied to GridWorld with configurable reward structures, discount factors (`γ`), and noise

### Q-Learning (Model-Free)
- Learns action-value estimates `Q(s, a)` directly from environment interaction
- No prior knowledge of transition probabilities required
- Implements ε-greedy exploration with tunable learning rate (`α`) and discount factor (`γ`)
- Applied to GridWorld, Crawler, and Pacman environments

---

## Environments

### GridWorld
A configurable grid environment used to analyze and visualize policy behavior.

- Tests agent performance under different reward structures (positive/negative living rewards)
- Demonstrates how noise in transitions affects optimal policy selection
- Visualizes value functions and policies at each iteration step

### Crawler
A simulated two-joint robot arm that learns locomotion through Q-Learning.

- Continuous state space discretized into joint angle buckets
- Agent learns to move forward without any hardcoded motion rules
- Demonstrates emergent behavior from reward-only feedback

### Pacman
A fully implemented Pacman game environment used as a Q-Learning testbed.

- Agent learns to navigate mazes, avoid ghosts, and eat food using only reward signals
- Compared performance of Q-Learning against approximate Q-Learning with feature extraction
- Evaluated generalization: trained on small grids, tested on unseen layouts

---

## Key Concepts Explored

| Concept | Description |
|---|---|
| **Bellman Equation** | Recursive relationship used to compute optimal value functions |
| **Discount Factor (γ)** | Controls how much the agent values future vs. immediate rewards |
| **Learning Rate (α)** | Controls how aggressively Q-values are updated per step |
| **ε-Greedy Exploration** | Balances exploration of unknown states vs. exploitation of learned policy |
| **Policy Convergence** | Analyzed how quickly value iteration stabilizes across grid sizes |
| **Model-Based vs. Model-Free** | Compared sample efficiency and convergence of VI vs. Q-Learning |

---

## Results & Observations

- Value Iteration converges to the optimal policy but requires full knowledge of the transition model — impractical for large or unknown environments
- Q-Learning converges more slowly but learns purely from experience, making it applicable to real-world settings
- In GridWorld, noise in transitions caused the agent to prefer "safer" paths even when a shorter high-risk path existed — demonstrating risk-averse behavior emerging from the reward structure
- In Pacman, the agent learned effective ghost-avoidance strategies after sufficient training episodes without any explicit rules

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/777natalie/AI_Project3.git
cd AI_Project3

# No external dependencies required — uses Python standard library
python gridworld.py        # Run GridWorld with value iteration
python crawler.py          # Run Crawler with Q-Learning
python pacman.py           # Run Pacman with Q-Learning
```

To adjust hyperparameters:
```bash
python gridworld.py --discount 0.9 --noise 0.2 --livingReward -0.1
python pacman.py --numTraining 2000 --epsilon 0.1 --alpha 0.5 --gamma 0.9
```

---

## File Structure

```
AI_Project3/
├── valueIterationAgents.py   # Value iteration implementation
├── qlearningAgents.py        # Q-Learning and approximate Q-Learning
├── mdp.py                    # MDP interface and base classes
├── gridworld.py              # GridWorld environment + visualization
├── crawler.py                # Crawler robot environment
├── pacman.py                 # Pacman game runner
├── game.py                   # Core game logic
├── util.py                   # Shared utilities (Counter, Queue, etc.)
└── README.md
```

---

## Concepts Reference

**Markov Decision Process (MDP):** A mathematical framework for modeling sequential decision-making where outcomes are partly random and partly controlled by an agent. Defined by states `S`, actions `A`, transition probabilities `T(s, a, s')`, and a reward function `R(s, a, s')`.

**Optimal Policy:** The policy `π*` that maximizes expected cumulative discounted reward from any starting state.

**Q-Value:** `Q(s, a)` represents the expected return of taking action `a` in state `s` and following the optimal policy thereafter.

---

## Author

**Natalie Roman** · [github.com/777natalie](https://github.com/777natalie)  
**Isabela Najera** · [github.com/isabelanajera](https://github.com/isabelanajera)


Computer Science, University of Oklahoma · Class of 2025
