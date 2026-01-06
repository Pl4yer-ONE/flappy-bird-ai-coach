# Experience Replay Buffer

"""
Experience Replay buffer for DQN training.
Stores transitions and provides random sampling for stable learning.
"""

from collections import deque
import random
from typing import List, Tuple, Any, Optional
import numpy as np


class ReplayMemory:
    """
    Experience Replay buffer using a deque with fixed maximum size.
    
    Stores transitions (state, action, next_state, reward, done) and
    provides random sampling for training mini-batches.
    """
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Optional random seed for reproducibility
        """
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def push(self, state, action, next_state, reward, done):
        """
        Store a transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
            done: Whether episode ended
        """
        self.memory.append((state, action, next_state, reward, done))
        
    def append(self, transition: Tuple):
        """Alias for push() for compatibility with reference code."""
        self.memory.append(transition)
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        return random.sample(self.memory, batch_size)
        
    def sample_tensors(self, batch_size: int, device: str = 'cpu'):
        """
        Sample a batch and return as PyTorch tensors.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones) tensors
        """
        import torch
        
        batch = self.sample(batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Stack tensors if they're already tensors, otherwise convert
        if isinstance(states[0], torch.Tensor):
            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            next_states = torch.stack(next_states).to(device)
            rewards = torch.stack(rewards).to(device)
        else:
            states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.int64, device=device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        
        return states, actions, next_states, rewards, dones
        
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.memory)
        
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.memory) >= batch_size
        
    def clear(self):
        """Clear all stored transitions."""
        self.memory.clear()


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions with probability proportional to their TD error,
    allowing more frequent learning from surprising transitions.
    
    Note: This is a simplified implementation. For production use,
    consider using a sum-tree data structure for O(log n) sampling.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed to 1)
            beta_increment: How much to increase beta per sample
            epsilon: Small constant to prevent zero priority
        """
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
    def push(self, state, action, next_state, reward, done):
        """Store a transition with max priority."""
        self.memory.append((state, action, next_state, reward, done))
        self.priorities.append(self.max_priority)
        
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample a prioritized batch.
        
        Returns:
            Tuple of (transitions, indices, importance_weights)
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get transitions
        transitions = [self.memory[i] for i in indices]
        
        return transitions, indices, weights
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self) -> int:
        return len(self.memory)
