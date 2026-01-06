# Dueling DQN Network Architecture

"""
Dueling Deep Q-Network implementation with PyTorch.
Supports both standard DQN and Dueling DQN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DQN(nn.Module):
    """
    Deep Q-Network with optional Dueling architecture.
    
    Standard DQN: state -> fc1 -> fc2 -> Q(s, a)
    Dueling DQN: state -> fc1 -> [value_stream, advantage_stream] -> Q(s, a)
    
    The Dueling architecture separates the estimation of state value V(s)
    and action advantages A(s, a), which can improve learning efficiency.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 512,
                 enable_dueling: bool = True):
        """
        Initialize DQN.
        
        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers
            enable_dueling: Whether to use dueling architecture
        """
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.enable_dueling = enable_dueling
        
        # Shared feature layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        if self.enable_dueling:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)
            
            # Advantage stream
            self.fc_advantage = nn.Linear(hidden_dim, 256)
            self.advantage = nn.Linear(256, action_dim)
        else:
            # Standard DQN output (matches reference model: fc1 -> output)
            self.output = nn.Linear(hidden_dim, action_dim)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        # Shared feature extraction
        x = F.relu(self.fc1(x))
        
        if self.enable_dueling:
            # Value stream
            v = F.relu(self.fc_value(x))
            V = self.value(v)  # Shape: (batch_size, 1)
            
            # Advantage stream
            a = F.relu(self.fc_advantage(x))
            A = self.advantage(a)  # Shape: (batch_size, action_dim)
            
            # Combine: Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
            # Subtracting mean ensures identifiability
            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            # Standard DQN: direct output
            Q = self.output(x)
            
        return Q
        
    def get_action(self, state: torch.Tensor) -> int:
        """
        Get best action for a single state.
        
        Args:
            state: State tensor of shape (state_dim,) or (1, state_dim)
            
        Returns:
            Best action index
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class DuelingDQN(DQN):
    """Alias for DQN with dueling enabled."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__(state_dim, action_dim, hidden_dim, enable_dueling=True)


class StandardDQN(DQN):
    """Alias for DQN without dueling."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__(state_dim, action_dim, hidden_dim, enable_dueling=False)


# Test the network architecture
if __name__ == '__main__':
    # Test both architectures
    state_dim = 12
    action_dim = 2
    batch_size = 32
    
    # Create test input
    test_state = torch.randn(batch_size, state_dim)
    single_state = torch.randn(state_dim)
    
    # Test Dueling DQN
    dueling_net = DuelingDQN(state_dim, action_dim)
    q_values = dueling_net(test_state)
    print(f"Dueling DQN output shape: {q_values.shape}")
    action = dueling_net.get_action(single_state)
    print(f"Best action for single state: {action}")
    
    # Test Standard DQN
    standard_net = StandardDQN(state_dim, action_dim)
    q_values = standard_net(test_state)
    print(f"Standard DQN output shape: {q_values.shape}")
    
    # Count parameters
    dueling_params = sum(p.numel() for p in dueling_net.parameters())
    standard_params = sum(p.numel() for p in standard_net.parameters())
    print(f"Dueling DQN parameters: {dueling_params:,}")
    print(f"Standard DQN parameters: {standard_params:,}")
