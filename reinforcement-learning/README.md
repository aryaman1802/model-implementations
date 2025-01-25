# Reinforcement Learning

This directory contains implementations of Reinforcement Learning models.

Helpful resources:
- [Deep RL course by HuggingFace](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)
- [Nimish Sanghi's book on Deep RL](https://link.springer.com/book/10.1007/979-8-8688-0273-7)
- [Maxim Lapan's book on Deep RL](https://amzn.in/d/53wRETg)

## Contents (Algorithms List)

- **Action Selection Strategies:**
    - Greedy
    - Epsilon-Greedy
    - Probability
    - Epsilon-Soft *(coming soon)*
    - Upper-Confidence Bound *(coming soon)*

- **Model-Based Algorithms (uses dynamic programming):**
    - Policy Iteration (includes policy evaluation and policy improvement)
    - Value Iteration

- **Model-Free Algorithms:**
    - MC (Monte Carlo) Prediction/Control
    - GLIE (Greedy in the Limit with Infinite Exploration) MC Control
    - Off-policy MC Control
    - TD(0) (aka one-step TD) for Estimation
    - SARSA on-policy TD (Temporal Difference) control
    - Q-learning off-policy TD Control
    - Expected SARSA TD Control (on-policy version uploaded;  off-policy version coming soon)
    - Q-learning with Replay Buffer
    - Q-learning for continous state spaces
    - n-step SARSA *(coming soon)*
    - $\lambda$-SARSA *(coming soon)*

- **Function Approximation Algorithms:**
    - Semi-gradient n-step SARSA Control
    - Semi-gradient SARSA($\lambda$) Control

- **Deep Q-Learning Algorithm (and its variants)** *(coming soon)*
    - Simple DQN (Deep Q-Network)
    - Replay Buffer
    - TD Loss
    - Prioritized Replay Buffer
    - TD Loss with Prioritized Replay
    - Double Q-learning
    - TD Loss with Double Q-learning
    - Dueling DQN
    - NoisyNets DQN
    - Hindsight Experience Replay

- **Policy Gradient Algorithms**
    - Coming soon...

