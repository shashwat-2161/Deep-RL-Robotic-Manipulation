# Deep Reinforcement Learning for Robotic Manipulation
### 7-DOF Robotic Arm Control with Soft Actor-Critic (SAC)

## 📄 Abstract
Robotic control involves high-dimensional, continuous action spaces that are difficult for standard RL algorithms (like DQN or PPO) to master. This project implements **Soft Actor-Critic (SAC)**, an off-policy algorithm that optimizes a stochastic policy for both maximum reward and maximum entropy.

To solve the sparse-reward "Pick and Place" task, **Hindsight Experience Replay (HER)** was utilized, allowing the agent to learn from failed attempts by re-labeling them as successful visualizations of unintended goals.

## 🎥 Results
**Task: Pick and Place**
* **Success Rate:** 100%
* **Convergence:** ~200k steps
* **Behavior:** The agent learned to smoothly coordinate 7 joints to grasp, lift, and place a dynamic object.

![Demo](videos\rl-video-episode-21.mp4)

## 🧠 Methodology

### 1. Soft Actor-Critic (SAC)
Unlike PPO (on-policy), SAC is **off-policy**, meaning it can reuse old data effectively. It maximizes the objective:
$$J(\pi) = \sum_{t} E [r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$$
Where $H$ is entropy. This encourages the robot to explore diverse movement strategies, preventing it from getting stuck in local optima (e.g., rigid, twitchy movements).

### 2. Hindsight Experience Replay (HER)
The "Pick and Place" task is a **Sparse Reward** environment (Reward = 0 mostly).
* **Problem:** The robot rarely succeeds by random chance, so it rarely gets a reward signal.
* **Solution:** HER stores failed episodes in the replay buffer but modifies the "Goal" to be the state the robot *actually* achieved. This teaches the robot: *"I didn't hit the target, but I now know how to reach this other point."*


## 📊 Performance
| Metric | Value |
| :--- | :--- |
| **Algorithm** | SAC + HER |
| **Training Steps** | 1000,000 |
| **Mean Reward** | -11.4 (Solved) |
| **Device** | NVIDIA RTX 3050 |
