SpaceShip Reinforcement Learning Project
========================================

A comprehensive reinforcement learning implementation using Double Deep Q-Network (DDQN) with Prioritized Experience Replay to train an AI agent to play a SpaceShip game. This project demonstrates advanced RL techniques including frame stacking, reward shaping, and neural network optimization.

PROJECT OVERVIEW
================

This project implements a sophisticated reinforcement learning system that trains an AI agent to master a SpaceShip arcade game. The agent learns to navigate through asteroid fields, shoot enemies, collect power-ups, and maximize its score through trial and error.

Key Features:
- Double Deep Q-Network (DDQN) with Prioritized Experience Replay
- Frame Stacking for temporal information processing
- Advanced Reward Shaping with multiple reward components
- Convolutional Neural Network architecture for visual processing
- Checkpoint System for training resumption
- Comprehensive Logging and visualization tools
- Real-time Gameplay Recording with video output

GAME ENVIRONMENT
================

The SpaceShip game environment features:

Player Spaceship: Controllable ship with health and weapon systems
Asteroids: Randomly generated obstacles with varying sizes and speeds
Power-ups: Shield and weapon upgrades that spawn from destroyed asteroids
Scoring System: Points awarded for destroying asteroids
Health System: Player health decreases on collision with asteroids

Game Mechanics:
- Actions: Stay, Move Left, Move Right, Shoot
- Objectives: Survive as long as possible while maximizing score
- Challenges: Avoid asteroids, shoot enemies, collect power-ups
- Termination: Game ends when health reaches zero or score reaches 10,000

AI ARCHITECTURE
===============

Neural Network Design:

The DDQN architecture consists of:

1. Feature Extraction Layers:
   - Conv2D (4→32, 8×8, stride 4)
   - Conv2D (32→64, 4×4, stride 2)
   - Conv2D (64→64, 3×3, stride 1)
   - Flatten layer

2. Q-Value Estimation Layers:
   - Linear (3136→512)
   - Linear (512→256)
   - Linear (256→4 actions)

Training Features:
- Frame Stacking: 4 consecutive frames for temporal awareness
- Image Preprocessing: Grayscale conversion and resizing to 84×84
- Prioritized Experience Replay: Efficient sampling of important experiences
- Target Network: Stable learning with periodic updates
- Epsilon-Greedy Exploration: Balanced exploration vs exploitation

REWARD SYSTEM
=============

The reward function incorporates multiple components:

Base survival reward: 0.1 per step
Score-based reward: +0.5 * score_delta when score increases
Collision penalty: -25.0 when hitting asteroids
Near miss reward: +2.0 when rocks pass close by without collision
Power-up reward: +20.0 when collecting power-ups

INSTALLATION & SETUP


Prerequisites:
pip install torch torchvision
pip install pygame
pip install numpy matplotlib pandas
pip install imageio scipy

Project Structure:
OPT-RL/
├── Final_Project/
│   ├── RL_Final_Project_v18G.py    # Main training script
│   ├── Game_EnvB.py                # Game environment
│   ├── space_ship_game_RL_V2/      # Game assets
│   │   ├── img/                    # Game images
│   │   ├── sound/                  # Game sounds
│   │   └── font.ttf               # Game font
│   └── runs_v18G/                 # Training outputs
│       ├── best_ddqn_space_ship_v18G.pth
│       ├── checkpoint.pth
│       ├── training_data.xlsx
│       └── *.mp4                  # Gameplay videos

USAGE
=====

Training Mode:
# Run the main script
python RL_Final_Project_v18G.py
# Select mode: TRAIN

Evaluation Mode:
# Run the main script
python RL_Final_Project_v18G.py
# Select mode: EVAL

Training + Evaluation:
# Run the main script
python RL_Final_Project_v18G.py
# Select mode: TRAIN_EVAL

HYPERPARAMETERS
===============

Parameter          Value    Description
-----------        -----    -----------
Episodes           4000     Total training episodes
Batch Size         256      Training batch size
Learning Rate      1e-4     Adam optimizer learning rate
Gamma              0.99     Discount factor
Epsilon Start      1.0      Initial exploration rate
Epsilon End        0.05     Final exploration rate
Memory Size        50000    Replay buffer capacity
Target Update      1000     Target network update frequency

TRAINING RESULTS
================

The training process includes:

- Real-time Logging: Episode rewards, scores, and metrics
- Checkpoint System: Automatic model saving and resumption
- Visualization: Training curves and performance plots
- Video Recording: Gameplay demonstrations

Performance Metrics:
- Total Reward: Cumulative reward per episode
- Average Reward: Mean reward per step
- Game Score: In-game score achieved
- Training Loss: Neural network loss during training
- Q-Value Statistics: Max and mean Q-values

KEY INNOVATIONS
===============

1. Advanced Reward Shaping: Multi-component reward system for better learning
2. Frame Stacking: Temporal information processing for better decision making
3. Prioritized Experience Replay: Efficient learning from important experiences
4. Checkpoint System: Robust training with automatic resumption
5. Comprehensive Logging: Detailed tracking of training progress

OUTPUT FILES
============

After training, the system generates:

- Model Files: Best performing model weights
- Checkpoints: Training state for resumption
- Training Data: Excel file with all metrics
- Visualizations: Training curves and performance plots
- Gameplay Videos: MP4 recordings of agent performance

CUSTOMIZATION
=============

Modifying Reward Function:
Edit the _calculate_reward method in SpaceShipEnv class:

def _calculate_reward(self, action, score):
    reward = 0.1  # Base reward
    # Add your custom reward components
    return reward

Adjusting Hyperparameters:
Modify the hyperparameters section in the main script:

EPISODES = 4000
BATCH_SIZE = 256
GAMMA = 0.99
# ... other parameters

Changing Network Architecture:
Modify the DDQN class to experiment with different architectures:

class DDQN(nn.Module):
    def __init__(self, action_dim, input_channels=4):
        # Customize your network layers
        pass

CONTRIBUTING
============

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- New features
- Documentation updates

LICENSE
=======

This project is part of the Optimization and Reinforcement Learning course. Please refer to your course guidelines for usage and distribution.

ACKNOWLEDGMENTS
===============

- Course instructors for guidance on RL concepts
- PyTorch community for excellent documentation
- Pygame developers for the game framework
- OpenAI Gym for inspiration on environment design

---

Note: This project demonstrates advanced reinforcement learning techniques and serves as a comprehensive example of implementing DDQN with modern best practices for game AI development.
