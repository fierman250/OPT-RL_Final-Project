# SpaceShip Reinforcement Learning Project

A comprehensive reinforcement learning implementation using Double Deep Q-Network (DDQN) with Prioritized Experience Replay to train an AI agent to play a SpaceShip game. This project demonstrates advanced RL techniques including frame stacking, reward shaping, and neural network optimization.

## �� Project Overview

This project implements a sophisticated reinforcement learning system that trains an AI agent to master a SpaceShip arcade game. The agent learns to navigate through asteroid fields, shoot enemies, collect power-ups, and maximize its score through trial and error.

### Key Features

- **Double Deep Q-Network (DDQN)** with Prioritized Experience Replay
- **Frame Stacking** for temporal information processing
- **Advanced Reward Shaping** with multiple reward components
- **Convolutional Neural Network** architecture for visual processing
- **Checkpoint System** for training resumption
- **Comprehensive Logging** and visualization tools
- **Real-time Gameplay Recording** with video output

## �� Game Environment

The SpaceShip game environment features:

- **Player Spaceship**: Controllable ship with health and weapon systems
- **Asteroids**: Randomly generated obstacles with varying sizes and speeds
- **Power-ups**: Shield and weapon upgrades that spawn from destroyed asteroids
- **Scoring System**: Points awarded for destroying asteroids
- **Health System**: Player health decreases on collision with asteroids

### Game Mechanics

- **Actions**: Stay, Move Left, Move Right, Shoot
- **Objectives**: Survive as long as possible while maximizing score
- **Challenges**: Avoid asteroids, shoot enemies, collect power-ups
- **Termination**: Game ends when health reaches zero or score reaches 10,000

## 🧠 AI Architecture

### Neural Network Design

The DDQN architecture consists of:

1. **Feature Extraction Layers**:
   - Conv2D (4→32, 8×8, stride 4)
   - Conv2D (32→64, 4×4, stride 2)
   - Conv2D (64→64, 3×3, stride 1)
   - Flatten layer

2. **Q-Value Estimation Layers**:
   - Linear (3136→512)
   - Linear (512→256)
   - Linear (256→4 actions)

### Training Features

- **Frame Stacking**: 4 consecutive frames for temporal awareness
- **Image Preprocessing**: Grayscale conversion and resizing to 84×84
- **Prioritized Experience Replay**: Efficient sampling of important experiences
- **Target Network**: Stable learning with periodic updates
- **Epsilon-Greedy Exploration**: Balanced exploration vs exploitation

## �� Reward System

The reward function incorporates multiple components:

```python
reward = 0.1  # Base survival reward

# Score-based reward
if score_delta > 0:
    reward += score_delta * 0.5

# Collision penalty
if collision:
    reward -= 25.0

# Near miss reward
if rock_passed_close:
    reward += 2.0

# Power-up reward
if power_up_collected:
    reward += 20.0
```

## 🛠️ Installation & Setup

### Prerequisites

```bash
pip install torch torchvision
pip install pygame
pip install numpy matplotlib pandas
pip install imageio scipy
```

### Project Structure
