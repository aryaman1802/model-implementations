# Reinforcement Learning

This directory contains implementations of Reinforcement Learning models.

Helpful resources:
- [Deep RL course by HuggingFace](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)
- [Nimish Sanghi's book on Deep RL](https://link.springer.com/book/10.1007/979-8-8688-0273-7)
- [Maxim Lapan's book on Deep RL](https://amzn.in/d/53wRETg)

## Contents (List of algorithms)

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

- **Deep Q-Networks:**
    - Simple Deep Q-Network (DQN)
    - DQN with Prioritized Replay
    - Double DQN
    - Dueling DQN
    - NoisyNets DQN
    - Categorical 51-Atom DQN
    - Quantile Regression DQN
    - DQN with Hindsight Experience Replay

- **Policy Gradient Algorithms:**
    - REINFORCE
    - Advantage Actor-Critic (A2C)

- **Combining Policy Gradient Algorithms & Deep Q-Networks:**
    - *coming soon*
