# Notes:
# Starting this code is using the new Environment B (Updated Version 2)
# This code is using DDQN with Prioritized Experience Replay

# %% [markdown]

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

# Core libraries
import os, sys, random, logging
import numpy as np
import pygame
import imageio
from collections import deque

# Machine learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Visualization and analysis
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import math

# Game environment
from Game_EnvB import Game, WIDTH, HEIGHT, FPS

# ------------ ENVIRONMENT SETUP ------------
def setup_environment():
    """Configure game environment path and create necessary directories"""
    script_dir = os.path.join(os.getcwd(), 'space_ship_game_RL_V2')
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    return script_dir

# ------------ PROJECT INITIALIZATION ------------
def initialize_project(version):
    """Initialize project directories and logging configuration"""
    runs_dir = f"runs_{version}"
    os.makedirs(runs_dir, exist_ok=True)
    
    log_file = os.path.join(runs_dir, f"history_{version}.txt")
    with open(log_file, 'a') as file:
        pass  # Create file if it doesn't exist
    
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    return runs_dir, version

# ------------ EXECUTE SETUP FUNCTIONS ------------
setup_environment()
RUNS_DIR, version = initialize_project(version="v18G")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_MODE = False

# ------------ HYPERPARAMETERS ------------
EPISODES = 1 + 4000
BATCH_SIZE = 256
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05  # Increased minimum exploration
EPSILON_DECAY = math.exp(math.log(EPSILON_END/EPSILON_START) / EPISODES)  # Computed decay for smooth schedule
LR = 1e-4  # Adjusted learning rate
TARGET_UPDATE = 1000  # Steps between target-network updates
MEMORY_SIZE = 50000

#%%
# =============================================================================
# AGENT DEVELOPMENT
# =============================================================================

# ------------ ENVIRONMENT PREPROCESSING ------------
logging.info("Environment class is starting...")

transform = T.Compose([
            T.ToPILImage(),      # Convert ndarray to PIL
            T.Grayscale(),       # RGB → Gray
            T.Resize((84, 84)),  # Resize to standard size
            T.ToTensor(),        # Convert to tensor [0,1]
                      ])

# ------------ IMAGE PREPROCESSING ------------
def preprocess(image):
    """Preprocess image for neural network input"""
    # Convert pygame surface to array if needed
    if isinstance(image, pygame.Surface):
        image = pygame.surfarray.array3d(image)
    
    # Transpose and convert to tensor
    image = np.transpose(image, (1, 0, 2)).copy()
    tensor = transform(image)
    return tensor.to(device, non_blocking=True)

# ------------ ENVIRONMENT CLASS ------------
class SpaceShipEnv():
    """Reinforcement Learning environment for SpaceShip game with frame stacking and reward shaping"""
    
    def __init__(self, stack_size=4, render_mode=False):
        """Initialize environment with frame stacking and optional rendering"""
        pygame.init()
        pygame.font.init()
        
        # Display setup (delayed until render)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.fps = FPS
        self.render_mode = render_mode
        
        # Game and state tracking
        self.game = Game()
        self.action_space = [0, 1, 2, 3]  # [stay, left, right, shoot]
        self.prev_score = self.prev_health = self.game.player.sprite.health
        self.prev_gun = self.game.player.sprite.gun
        
        # Frame stacking for temporal information
        self.frame_stack = torch.zeros((stack_size, 1, 84, 84), device=device)
        self.current_pos = 0
        self.action_history = deque(maxlen=10)
        
    def step(self, action):
        """Execute action and return (state, reward, done, info, score)"""
        # Handle pygame events if rendering
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            self._render_frame()
        
        self.game.update(action)
        state, score = self.game.state, self.game.score
        
        # Calculate reward using multiple components
        reward = self._calculate_reward(action, score)
        
        # Update tracking variables
        self.prev_health = self.game.player.sprite.health
        self.prev_gun = self.game.player.sprite.gun
        self.prev_score = score
        self.action_history.append(action)
        
        # Update frame stack and return
        self._update_frame_stack(state)
        done = not self.game.running or score >= 10000
        return torch.roll(self.frame_stack, -self.current_pos, dims=0), reward, done, {}, score

    def _render_frame(self):
        """Render current game state to screen"""
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("SpaceShip RL Environment")
        self.game.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _calculate_reward(self, action, score):
        """Calculate reward based on multiple factors"""
        reward = 0.1  # Base survival reward
        
        # 1. Score-based reward
        if (score_delta := score - self.prev_score) > 0:
            reward += score_delta * 0.5

        # 2. Penalty for collision the rock
        if self.game.is_collided:
            reward -= 25.0
        
        # 3. Near miss reward for evading rocks
        if not self.game.is_collided and not self.game.is_hit_rock:
            # reward agent for rocks that just passed close by without collision
            player_center_x = self.game.player.sprite.rect.centerx
            player_top = self.game.player.sprite.rect.top
            for rock in self.game.rocks:
                bottom = rock.rect.bottom
                # rock just passed the player's top this step
                if bottom > player_top and bottom - rock.speedy <= player_top:
                    if abs(rock.rect.centerx - player_center_x) < 45:
                        reward += 2.0

        # 4. If collide Health OR power-up
        if self.game.is_power:
            reward += 20.0
        
        # print(f"Reward: {reward}")
        return reward

    def _update_frame_stack(self, state):
        """Update frame stack with new processed frame"""
        processed_frame = preprocess(state)
        self.frame_stack[self.current_pos] = processed_frame
        self.current_pos = (self.current_pos + 1) % self.frame_stack.size(0)

    def reset(self):
        """Reset environment to initial state and return first observation"""
        self.game = Game()
        self.prev_score, self.prev_health, self.prev_gun = 0, self.game.player.sprite.health, self.game.player.sprite.gun
        self.action_history.clear()
        
        # Initialize frame stack with preprocessed initial state
        initial_frame = preprocess(self.game.state)
        # Fill each frame in the stack with the initial frame
        for i in range(self.frame_stack.size(0)):
            self.frame_stack[i] = initial_frame
        self.current_pos = 0
        return self.frame_stack

    def render(self):
        """Render current game state to display"""
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("SpaceShip RL Environment")
        self.game.draw(self.screen)
        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        """Clean up pygame resources"""
        pygame.quit()

# %%
# =============================================================================
# MODEL DEVELOPMENT
# =============================================================================
logging.info("DDQN class is starting...")

class DDQN(nn.Module):
    """
    Double Deep Q-Network (DDQN) architecture for SpaceShip RL environment.
    Args:
        action_dim (int): Number of possible actions
        input_channels (int): Number of input channels (default: 4 for frame stacking)
    """
    
    def __init__(self, action_dim, input_channels=4):
        super().__init__()
        
        # Feature extraction: CNN layers for spatial feature learning
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # 9x9 -> 7x7
            nn.ReLU(),
            nn.Flatten()                                             # 7x7x64 = 3136 features
        )
        
        # Q-value estimation: Fully connected layers for action-value mapping
        self.q_values = nn.Sequential(
            nn.Linear(3136, 512),    # Feature compression
            nn.ReLU(),
            nn.Linear(512, 256),     # Hidden layer
            nn.ReLU(),
            nn.Linear(256, action_dim)  # Output Q-values for each action
        )

    def forward(self, x):
        """Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape [batch, frames, channels, height, width]
                              or [batch, channels, height, width]
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Handle frame-stacked input by reshaping to single channel dimension
        if len(x.shape) == 5:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1, x.shape[-2], x.shape[-1])
        
        features = self.features(x)
        return self.q_values(features)

# ------------ REPLAY MEMORY ------------
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for storing and sampling transitions"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.memory = []
        self.priorities = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.pos = 0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            old_state, old_action, old_reward, old_next_state, old_done = self.memory[self.pos]
            del old_state, old_next_state
            self.memory[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.memory)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = torch.multinomial(probs, batch_size, replacement=False)
        samples = [self.memory[idx] for idx in indices]
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        weights = weights.to(device)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        if not isinstance(priorities, torch.Tensor):
            priorities = torch.tensor(priorities, device=device)
        elif priorities.device != device:
            priorities = priorities.to(device)
        priorities = priorities.squeeze(-1)
        self.priorities[indices] = priorities

    def __len__(self):
        return len(self.memory)

# =============================================================================
# TRAINING
# =============================================================================

# ------------ INITIALIZATION ------------
def initialize_training_components(TRAIN_MODE):
    """Initialize DQN networks, optimizer, and training components"""
    logging.info("Initialization is starting...")
    
    # Environment and action space
    env = SpaceShipEnv(render_mode=TRAIN_MODE)
    action_dim = len(env.action_space)
    
    # Neural networks
    policy_net = DDQN(action_dim, input_channels=4).to(device)
    target_net = DDQN(action_dim, input_channels=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Optimizer and scheduler
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Replay memory
    memory = PrioritizedReplayBuffer(MEMORY_SIZE)
    
    print(f"Training on device: {device}")
    return device, env, policy_net, target_net, optimizer, scheduler, memory

# Initialize components
device, env, policy_net, target_net, optimizer, scheduler, memory = initialize_training_components(TRAIN_MODE)

# ------------ TRAINING LOOP ------------
def train_model():
    """Train the DQN model with a main training loop and checkpoint management."""
    # Resume training if a saved checkpoint exists
    epsilon, start_episode, total_steps, best_reward, best_score = EPSILON_START, 0, 0, -float("inf"), -float("inf")
    reward_history, score_history, steps_history, avg_rewards = [], [], [], []
    episode_losses, episode_max_qs, episode_mean_qs = [], [], []  # Per-episode aggregates

    def select_action(state, epsilon):
        """Select an action using an epsilon-greedy strategy."""
        if random.random() < epsilon:
            if len(env.action_history) >= 2:
                last_action = env.action_history[-1]
                weights = [1.0 if a != last_action else 0.1 for a in env.action_space]
                player_x = env.game.player.sprite.rect.centerx
                if player_x < 100: weights[1] *= 0.5
                elif player_x > WIDTH - 100: weights[2] *= 0.5
                return random.choices(env.action_space, weights=weights)[0]
            return random.choice(env.action_space)
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0))
            noise = torch.randn_like(q_values) * 0.3
            return torch.argmax(q_values + noise).item()

    def train_network():
        """Train the DQN network using sampled experiences from memory."""
        states, actions, rewards, next_states, dones, indices, weights = memory.sample(BATCH_SIZE)
        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards.unsqueeze(1) + GAMMA * next_q * (1 - dones.unsqueeze(1))
        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
        td_errors = target - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        # Update priorities in replay buffer
        new_priorities = td_errors.abs().detach().squeeze().cpu().numpy()
        memory.update_priorities(indices, new_priorities)
        return loss.item()  # Return loss for episode aggregation

    def save_model_if_best(total_reward, best_reward, episode, best_score, score):
        """Save the model if the current total reward is the best so far."""
        if total_reward > best_reward or score > best_score:
            best_reward = total_reward
            best_score = score
            torch.save(policy_net.state_dict(), f"{RUNS_DIR}/best_ddqn_space_ship_{version}.pth")
            save_checkpoint(episode, best_reward, best_score)  # Save full checkpoint
        elif episode % 10 == 0:
            save_checkpoint(episode, best_reward, best_score)  # Periodic full checkpoint
        
        return best_reward, best_score

    def save_checkpoint(episode, best_reward, best_score):
        """Save the current model and training state to a checkpoint."""
        torch.save({
            "policy_net": policy_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon,
            "episode": episode,
            "total_steps": total_steps,
            "best_reward": best_reward,
            "best_score": best_score,
            "reward_history": reward_history,
            "score_history": score_history,
            "steps_history": steps_history,
            "avg_rewards": avg_rewards,
            "episode_losses": episode_losses,
            "episode_max_qs": episode_max_qs,
            "episode_mean_qs": episode_mean_qs}, f"{RUNS_DIR}/checkpoint.pth")
        save_msg = f"✔ Episode {episode} save model Total Reward = {total_reward:.2f}, Best_reward = {best_reward:.2f}, Best_score = {best_score:.2f}"
        logging.info(save_msg); print(save_msg)

    def log_episode_results(episode, total_reward, steps, score, epsilon, episode_loss, episode_max_q, episode_mean_q):
        """Log the results of the current episode."""
        reward_history.append(total_reward)
        steps_history.append(steps)
        score_history.append(score)
        avg_reward = total_reward / steps
        avg_rewards.append(avg_reward)
        
        # Store per-episode aggregates
        episode_losses.append(episode_loss)
        episode_max_qs.append(episode_max_q)
        episode_mean_qs.append(episode_mean_q)
        
        episode_msg = f"Episode {episode}, Total Reward: {total_reward:.2f}, Score: {score}, Avg Reward: {avg_reward:.2f}"
        logging.info(episode_msg); print(episode_msg)  

    def save_intermediate_plots(episode):
        """Save intermediate plots every n episodes to monitor training progress."""
        if episode % 50 == 0 and episode > 0:
            plt.figure(figsize=(12, 8))
            smoothed_reward = gaussian_filter1d(reward_history, sigma=10)
            smoothed_score = gaussian_filter1d(score_history, sigma=10)
            plt.plot(reward_history, 'lightblue', alpha=0.7, label='Total Reward')
            plt.plot(smoothed_reward, 'blue', alpha=0.7, label='Total Reward (smoothed)')
            plt.plot(score_history, 'lightcoral', alpha=0.7, label='Game Score')
            plt.plot(smoothed_score, 'red', alpha=0.7, label='Game Score (smoothed)')
            plt.xlabel('Episode'); plt.ylabel('Reward/Score/Avg Reward')
            plt.title('Training Progress'); plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS_DIR, f'training_progress.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # Resume training if a saved checkpoint exists  
    checkpoint_path = os.path.join(RUNS_DIR, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint["policy_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epsilon = checkpoint.get("epsilon", EPSILON_START)
        start_episode = checkpoint.get("episode", 0) + 1
        total_steps = checkpoint.get("total_steps", 0)
        best_reward = checkpoint.get("best_reward", best_reward)
        best_score = checkpoint.get("best_score", best_score)
        reward_history = checkpoint.get("reward_history", [])
        score_history = checkpoint.get("score_history", [])
        steps_history = checkpoint.get("steps_history", [])
        avg_rewards = checkpoint.get("avg_rewards", [])
        episode_losses = checkpoint.get("episode_losses", [])
        episode_max_qs = checkpoint.get("episode_max_qs", [])
        episode_mean_qs = checkpoint.get("episode_mean_qs", [])
        print(f"Loaded checkpoint: start_episode={start_episode}, best_reward={best_reward:.2f}, epsilon={epsilon:.3f}")

    # Main training loop
    logging.info("Main training loop is starting...")
    for episode in range(start_episode, EPISODES):
        state, total_reward, steps, done = env.reset(), 0, 0, False
        
        # Episode-level tracking for aggregates
        episode_loss_list = []
        episode_q_values = []

        while not done:
            steps += 1
            action = select_action(state, epsilon)  # Action selection
            next_state, step_reward, done, _, score = env.step(action)
            next_state = next_state.to(device)
            total_reward += step_reward
            memory.push(state, action, step_reward, next_state, done)
            state = next_state

            if len(memory) >= BATCH_SIZE:
                loss = train_network()  # Train the network and get loss
                episode_loss_list.append(loss)
                
                # Get Q-values for this step (for episode aggregation)
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0))
                    episode_q_values.append(q_values.cpu().numpy())

            total_steps += 1
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Calculate episode aggregates
        episode_loss = np.mean(episode_loss_list) if episode_loss_list else 0.0
        episode_max_q = np.max([q.max() for q in episode_q_values]) if episode_q_values else 0.0
        episode_mean_q = np.mean([q.mean() for q in episode_q_values]) if episode_q_values else 0.0

        best_reward, best_score = save_model_if_best(total_reward, best_reward, episode, best_score, score)  # Save model if best reward
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)  # Update epsilon
        log_episode_results(episode, total_reward, steps, score, epsilon, episode_loss, episode_max_q, episode_mean_q)  # Log results
        save_intermediate_plots(episode)  # Save intermediate plots every 2 episodes
        scheduler.step(total_reward)  # Step the scheduler
        torch.cuda.empty_cache()  # Clear episode memory

    # Create the DataFrame with episode-level metrics (now directly available)
    saved_df = pd.DataFrame({
        'Episode': range(len(reward_history)),
        'Total Reward': reward_history,
        'Avg Reward': avg_rewards,
        'Score': score_history,
        'Loss': episode_losses,
        'Max Q': episode_max_qs,
        'Mean Q': episode_mean_qs
    })
    saved_df.to_excel(f"{RUNS_DIR}/training_data.xlsx", index=False)
    env.close()

    # Plotting functions
    def create_plot(data, title, ylabel, filename, colors=('lightgray', 'blue')):
            """Helper function to create and save plots"""
            plt.figure(figsize=(10, 6))
            smoothed = gaussian_filter1d(data, sigma=10)
            plt.plot(data, color=colors[0], alpha=0.5, label='Raw')
            plt.plot(smoothed, color=colors[1], label='Smoothed')
            plt.xlabel("Episode"); plt.ylabel(ylabel); plt.title(title)
            plt.grid(True); plt.legend()
            plt.savefig(f"{RUNS_DIR}/{filename}.png"); plt.show()
        
    # Plot reward curves
    create_plot(reward_history, "Total Reward per Episode", "Total Reward", "total_reward_curves")
    create_plot(score_history, "Score per Episode", "Score", "score_curves", ('lightgray', 'blue'))
    create_plot(avg_rewards, "Average Reward per Step", "Average Reward", "avg_reward_curves", ('lightgray', 'red'))
    create_plot(episode_losses, "Training Loss", "Loss", "loss_curve", ('lightgray', 'green'))
    epsilon_list = [max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** i)) for i in range(len(reward_history))]
    create_plot(epsilon_list, "Epsilon Decay", "Epsilon", "epsilon_curve", ('lightgray', 'purple'))

    # Plot Q-value statistics
    plt.figure(figsize=(10, 6))
    smoothed_max_q = gaussian_filter1d(episode_max_qs, sigma=10)
    smoothed_mean_q = gaussian_filter1d(episode_mean_qs, sigma=10)
    plt.plot(episode_max_qs, color='lightgray', alpha=0.5, label='Raw Max Q')
    plt.plot(smoothed_max_q, color='purple', label='Smoothed Max Q')
    plt.plot(episode_mean_qs, color='lightgray', alpha=0.5, label='Raw Mean Q')
    plt.plot(smoothed_mean_q, color='orange', label='Smoothed Mean Q')
    plt.xlabel("Episode"); plt.ylabel("Q-Value"); plt.title("Q-Value Statistics")
    plt.grid(True); plt.legend()
    plt.savefig(f"{RUNS_DIR}/q_value_curves.png"); plt.show()

# =============================================================================
# EVALUATION
# =============================================================================
def play_with_trained_model(i):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SpaceShipEnv(render_mode=True)
    policy_net = DDQN(len(env.action_space), input_channels=4)
    state_dict = torch.load(f"{RUNS_DIR}/best_ddqn_space_ship_{version}.pth", map_location="cpu")
    policy_net.load_state_dict(state_dict)
    policy_net.to(device)
    policy_net.eval()

    state = env.reset()
    state = state.to(device)
    done = False
    total_reward = 0
    frames = []
    
    # Add action history tracking
    action_history = []

    while not done:
        env.render()
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            # Print Q-values for debugging
            print("Q-values:", q_values.cpu().numpy())
            action = torch.argmax(q_values).item()
            action_history.append(action)

        frame = pygame.surfarray.array3d(env.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)

        next_state, reward, done, _, score = env.step(action)
        next_state = next_state.to(device)
        total_reward += reward
        state = next_state
        print(f"Action: {action}, Reward: {reward}, Total reward: {total_reward}, Score: {score}")

    env.close()

    # Print action distribution
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for action in action_history:
        action_counts[action] += 1
    print("\nAction distribution:")
    for action, count in action_counts.items():
        print(f"Action {action}: {count} times")

    imageio.mimsave(f"{RUNS_DIR}/{score}_space_ship_{version}.mp4", frames, fps=60, format='mp4')
    saved_msg = f"Saved gameplay video to: {RUNS_DIR}/{score}_space_ship_{version}.mp4"
    logging.info(saved_msg); print(saved_msg)
    print(f"Test{i}: Total reward: {total_reward}, Score: {score}")

# Call test function
def evaluate(num_test):
    # Run your play_with_trained_model(i) loop here
    for i in range(num_test):
        play_with_trained_model(i)

#%%
# =============================================================================
# MAIN FUNCTION
# =============================================================================
num_test = 100

if __name__ == "__main__":
    mode = input("Enter mode (TRAIN, EVAL, TRAIN_EVAL): ").strip().upper()
    # mode = "TRAIN"

    if mode == "TRAIN":
        train_model()
    elif mode == "EVAL":
        evaluate(num_test)
    elif mode == "TRAIN_EVAL":
        train_model()
        evaluate(num_test)
    else:
        raise ValueError(f"Invalid mode: {mode}")
