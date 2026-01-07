# DQN Agent - Training and Inference

"""
Deep Q-Learning Agent with Dueling Double DQN.
Handles training, inference, model saving/loading, and visualization.
"""

import os
import sys
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import itertools

import torch
import torch.nn as nn
import yaml
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RUNS_DIR, MODELS_DIR
from rl_agent.dqn import DQN
from rl_agent.experience_replay import ReplayMemory

# Use non-interactive backend for saving plots
matplotlib.use('Agg')

# Date format for logging
DATE_FORMAT = "%m-%d %H:%M:%S"

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DQNAgent:
    """
    Deep Q-Learning Agent with support for:
    - Dueling DQN architecture
    - Double DQN for reduced overestimation
    - Experience Replay for stable learning
    - Target network for training stability
    """
    
    def __init__(self, hyperparameter_set: str = 'flappybird', 
                 config_path: Optional[str] = None):
        """
        Initialize agent with hyperparameters.
        
        Args:
            hyperparameter_set: Name of config section to use
            config_path: Optional custom path to hyperparameters.yml
        """
        # Load hyperparameters
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'hyperparameters.yml'
            )
            
        with open(config_path, 'r') as f:
            all_hyperparameters = yaml.safe_load(f)
            hyperparameters = all_hyperparameters[hyperparameter_set]
            
        self.hyperparameter_set = hyperparameter_set
        
        # Store hyperparameters
        self.env_id = hyperparameters['env_id']
        self.learning_rate = hyperparameters['learning_rate_a']
        self.discount_factor = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.save_interval = hyperparameters.get('save_interval', 500)
        self.env_make_params = hyperparameters.get('env_make_params', {})
        
        # Neural network components
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.policy_dqn = None
        self.target_dqn = None
        
        # Training state
        self.epsilon = self.epsilon_init
        self.memory = None
        self.step_count = 0
        self.episode_rewards = []
        self.epsilon_history = []
        self.best_reward = -float('inf')
        
        # File paths
        os.makedirs(RUNS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.log_file = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.model_file = os.path.join(MODELS_DIR, f'{hyperparameter_set}.pt')
        self.graph_file = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')
        
    def initialize_networks(self, state_dim: int, action_dim: int):
        """
        Initialize policy and target networks.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.policy_dqn = DQN(
            state_dim, action_dim, 
            self.fc1_nodes, 
            self.enable_dueling_dqn
        ).to(device)
        
        self.target_dqn = DQN(
            state_dim, action_dim,
            self.fc1_nodes,
            self.enable_dueling_dqn
        ).to(device)
        
        # Initialize target network with policy network weights
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_dqn.parameters(), 
            lr=self.learning_rate
        )
        
        # Initialize replay memory
        self.memory = ReplayMemory(self.replay_memory_size)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, 1)  # 0 = no-op, 1 = flap
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                q_values = self.policy_dqn(state_tensor)
                return q_values.argmax(dim=1).item()
                
    def store_transition(self, state, action, next_state, reward, done):
        """Store a transition in replay memory."""
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        action_t = torch.tensor(action, dtype=torch.int64, device=device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
        
        self.memory.append((state_t, action_t, next_state_t, reward_t, done))
        self.step_count += 1
        
    def optimize(self):
        """
        Perform one optimization step.
        
        Uses Double DQN if enabled:
        - Policy network selects best action
        - Target network evaluates that action
        """
        if len(self.memory) < self.mini_batch_size:
            return
            
        # Sample mini-batch
        batch = self.memory.sample(self.mini_batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Stack tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        
        # Calculate target Q-values
        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN: use policy network to select action,
                # target network to evaluate
                best_actions = self.policy_dqn(next_states).argmax(dim=1)
                target_q = rewards + (1 - dones) * self.discount_factor * \
                    self.target_dqn(next_states).gather(
                        dim=1, 
                        index=best_actions.unsqueeze(dim=1)
                    ).squeeze()
            else:
                # Standard DQN: use target network for both
                target_q = rewards + (1 - dones) * self.discount_factor * \
                    self.target_dqn(next_states).max(dim=1)[0]
                    
        # Calculate current Q-values
        current_q = self.policy_dqn(states).gather(
            dim=1, 
            index=actions.unsqueeze(dim=1)
        ).squeeze()
        
        # Calculate loss and backpropagate
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.epsilon_history.append(self.epsilon)
        
        # Sync target network
        if self.step_count % self.network_sync_rate == 0:
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            
    def train(self, env, num_episodes: int = None, render: bool = False, 
              callback = None):
        """
        Train the agent.
        
        Args:
            env: Gymnasium-compatible environment
            num_episodes: Number of episodes (None for indefinite)
            render: Whether to render during training
            callback: Optional callback function(episode, reward, info)
        """
        start_time = datetime.now()
        last_graph_update = start_time
        
        # Log start
        log_msg = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_msg)
        with open(self.log_file, 'w') as f:
            f.write(log_msg + '\n')
            
        # Initialize networks if not already done
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        if self.policy_dqn is None:
            self.initialize_networks(num_states, num_actions)
            
        # Training loop
        episode_iter = range(num_episodes) if num_episodes else itertools.count()
        
        for episode in episode_iter:
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done and episode_reward < self.stop_on_reward:
                if render:
                    env.render()
                    
                # Select and perform action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.store_transition(state, action, next_state, reward, done)
                
                # Optimize
                self.optimize()
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
            # Episode complete
            self.episode_rewards.append(episode_reward)
            
            # Check for new best
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model()
                log_msg = f"{datetime.now().strftime(DATE_FORMAT)}: New best {episode_reward:.1f} at episode {episode}"
                print(log_msg)
                with open(self.log_file, 'a') as f:
                    f.write(log_msg + '\n')
                    
            # Update graph periodically
            if datetime.now() - last_graph_update > timedelta(seconds=10):
                self.save_graph()
                last_graph_update = datetime.now()
                
            # Callback
            if callback:
                callback(episode, episode_reward, {
                    'epsilon': self.epsilon,
                    'best_reward': self.best_reward,
                    'memory_size': len(self.memory)
                })
                
    def evaluate(self, env, num_episodes: int = 10, render: bool = True) -> Dict:
        """
        Evaluate the trained agent.
        
        Args:
            env: Gymnasium-compatible environment
            num_episodes: Number of evaluation episodes
            render: Whether to render
            
        Returns:
            Dictionary with evaluation statistics
        """
        if self.policy_dqn is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        self.policy_dqn.eval()
        rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                if render:
                    env.render()
                    
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")
            
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'rewards': rewards
        }
        
    def save_model(self, path: Optional[str] = None):
        """Save model weights."""
        path = path or self.model_file
        torch.save(self.policy_dqn.state_dict(), path)
        
    def load_model(self, path: Optional[str] = None, state_dim: int = 12, 
                   action_dim: int = 2):
        """
        Load model weights.
        
        Args:
            path: Path to model file
            state_dim: State dimension (will be auto-detected from model if possible)
            action_dim: Action dimension
        """
        path = path or self.model_file
        
        # Load state dict to check architecture
        state_dict = torch.load(path, map_location=device, weights_only=True)
        
        # Auto-detect state dimension from fc1 layer weight shape
        if 'fc1.weight' in state_dict:
            detected_state_dim = state_dict['fc1.weight'].shape[1]
            print(f"Auto-detected state dimension: {detected_state_dim}")
            state_dim = detected_state_dim
        
        # Detect architecture from state dict keys
        has_dueling = 'fc_value.weight' in state_dict or 'value.weight' in state_dict
        has_standard = 'output.weight' in state_dict
        
        if has_standard and not has_dueling:
            # Model was trained with standard DQN
            print("Detected standard DQN architecture in saved model")
            self.enable_dueling_dqn = False
        elif has_dueling:
            print("Detected dueling DQN architecture in saved model")
            self.enable_dueling_dqn = True
        
        # Store detected state dim for padding
        self._model_state_dim = state_dim
            
        if self.policy_dqn is None:
            self.initialize_networks(state_dim, action_dim)
            
        self.policy_dqn.load_state_dict(state_dict)
        self.policy_dqn.eval()
        
    def save_graph(self, path: Optional[str] = None):
        """Save training progress graphs."""
        path = path or self.graph_file
        
        fig = plt.figure(figsize=(12, 5))
        
        # Reward plot
        plt.subplot(121)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        if self.episode_rewards:
            plt.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
            # Moving average
            window = min(100, len(self.episode_rewards))
            if len(self.episode_rewards) >= window:
                moving_avg = np.convolve(
                    self.episode_rewards, 
                    np.ones(window) / window, 
                    mode='valid'
                )
                plt.plot(range(window - 1, len(self.episode_rewards)), 
                        moving_avg, label=f'Moving Avg ({window})')
            plt.legend()
            
        # Epsilon plot
        plt.subplot(122)
        plt.title('Epsilon Decay')
        plt.xlabel('Training Step')
        plt.ylabel('Epsilon')
        if self.epsilon_history:
            plt.plot(self.epsilon_history)
            
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)


# Entry point for standalone training
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test DQN agent')
    parser.add_argument('config', type=str, help='Hyperparameter config name')
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render environment')
    args = parser.parse_args()
    
    # Create agent
    agent = DQNAgent(args.config)
    
    # Create environment
    try:
        import gymnasium as gym
        import flappy_bird_gymnasium
        env = gym.make(agent.env_id, render_mode='human' if args.render else None,
                      **agent.env_make_params)
    except ImportError:
        from game.game_engine import GymnasiumWrapper
        env = GymnasiumWrapper(use_gymnasium=False, 
                               render_mode='human' if args.render else None)
        
    if args.train:
        agent.train(env, num_episodes=args.episodes, render=args.render)
    else:
        agent.load_model()
        agent.evaluate(env, render=True)
        
    env.close()
