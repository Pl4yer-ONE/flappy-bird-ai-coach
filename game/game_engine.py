# Game Engine - Main Game Loop and State Management

"""
Flappy Bird game engine with state logging for RL and coaching.
Provides both human-playable and headless training modes.
"""

import pygame
import sys
import os
import time
import csv
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, GROUND_Y, GROUND_HEIGHT,
    SKY_COLOR, GROUND_COLOR, TEXT_COLOR,
    BIRD_START_X, BIRD_START_Y, LOGS_DIR,
    REWARD_ALIVE, REWARD_PASS_PIPE, REWARD_DEATH
)
from game.bird import Bird
from game.pipe import PipeManager
from game.assets import AssetManager, get_asset_manager


class FlappyBirdGame:
    """
    Main Flappy Bird game engine.
    
    Supports multiple modes:
    - Human play with rendering
    - AI agent play with optional rendering
    - Headless training (no rendering for speed)
    
    Logs game state for RL training and coach analysis.
    """
    
    def __init__(self, 
                 render_mode: Optional[str] = "human",
                 enable_logging: bool = True):
        """
        Initialize game engine.
        
        Args:
            render_mode: "human" for visual, None for headless
            enable_logging: Whether to log state history
        """
        self.render_mode = render_mode
        self.enable_logging = enable_logging
        
        # Initialize pygame
        if render_mode:
            if not pygame.get_init():
                pygame.init()
            
            if render_mode == "human":
                pygame.display.set_caption("Flappy Bird AI Coach")
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.clock = pygame.time.Clock()
            else:
                # For rgb_array or others, use offscreen surface
                # This prevents clobbering an existing window (e.g. Dashboard)
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.clock = None 
            
            self.assets = get_asset_manager()
            self.assets.load_all()
        else:
            self.screen = None
            self.clock = None
            self.assets = None
            
        # Game objects
        self.bird = Bird()
        self.pipes = PipeManager()
        
        # Game state
        self.score = 0
        self.frame = 0
        self.game_over = False
        self.started = False
        
        # State history for coaching
        self.state_history: List[Dict[str, Any]] = []
        self.episode_number = 0
        
        # Fonts (initialized lazily)
        self._fonts_initialized = False
        self._score_font = None
        self._small_font = None
        
    def _init_fonts(self):
        """Initialize fonts lazily."""
        if not self._fonts_initialized and self.render_mode:
            self._score_font = pygame.font.Font(None, 48)
            self._small_font = pygame.font.Font(None, 24)
            self._fonts_initialized = True
            
    def reset(self) -> Dict[str, Any]:
        """
        Reset game to initial state.
        
        Returns:
            Initial observation state
        """
        self.bird.reset()
        self.pipes.reset()
        self.score = 0
        self.frame = 0
        self.game_over = False
        self.started = True
        
        # Clear state history
        if self.enable_logging:
            self.state_history = []
            
        self.episode_number += 1
        
        return self._get_observation()
        
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """
        Take a game step.
        
        Args:
            action: 0 = do nothing, 1 = flap
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        
        # Handle action
        if action == 1:
            self.bird.flap(self.frame)
            if self.assets:
                self.assets.play_sound('flap', 0.3)
                
        # Update game objects
        self.bird.update()
        score_increment = self.pipes.update(self.bird.x)
        
        # Check for scoring
        if score_increment > 0:
            self.score += score_increment
            reward += REWARD_PASS_PIPE * score_increment
            if self.assets:
                self.assets.play_sound('score', 0.5)
                
        # Check collisions
        if self.pipes.check_collision(self.bird):
            self.bird.alive = False
            self.game_over = True
            reward = REWARD_DEATH
            if self.assets:
                self.assets.play_sound('hit', 0.5)
        elif not self.bird.alive:
            self.game_over = True
            reward = REWARD_DEATH
        else:
            # Alive reward
            reward += REWARD_ALIVE
            
        # Increment frame
        self.frame += 1
        
        # Log state
        if self.enable_logging:
            state = self._get_full_state()
            self.state_history.append(state)
            
        # Get observation
        observation = self._get_observation()
        
        info = {
            'score': self.score,
            'frame': self.frame,
            'alive': self.bird.alive
        }
        
        return observation, reward, self.game_over, False, info
        
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current observation for RL agent.
        
        Returns observation compatible with flappy-bird-gymnasium format.
        """
        pipe_state = self.pipes.get_state(self.bird.x)
        
        return {
            'bird_y': self.bird.y,
            'bird_velocity': self.bird.velocity,
            'distance_to_next_pipe': pipe_state['distance_to_next_pipe'],
            'next_pipe_gap_y': pipe_state['next_pipe_gap_y'],
            'next_pipe_upper_height': pipe_state['next_pipe_upper_height'],
            'next_pipe_lower_top': pipe_state['next_pipe_lower_top']
        }
        
    def _get_full_state(self) -> Dict[str, Any]:
        """Get complete game state for logging and coaching."""
        obs = self._get_observation()
        bird_state = self.bird.get_state()
        
        return {
            **obs,
            **bird_state,
            'score': self.score,
            'frame': self.frame,
            'game_over': self.game_over
        }
        
    def render(self):
        """Render the current game state."""
        if not self.render_mode or not self.screen:
            return
            
        self._init_fonts()
        
        # Draw background
        self.screen.fill(SKY_COLOR)
        
        # Draw pipes
        self.pipes.render(self.screen)
        
        # Draw ground
        ground_rect = pygame.Rect(0, GROUND_Y, SCREEN_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.screen, GROUND_COLOR, ground_rect)
        # Ground detail lines
        for x in range((self.frame * 3) % 20 - 20, SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, (200, 190, 130), 
                           (x, GROUND_Y), (x + 10, GROUND_Y + 5), 2)
        
        # Draw bird
        self.bird.render(self.screen)
        
        # Draw score
        self._draw_score()
        
        # Update display
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(FPS)
            
    def _draw_score(self):
        """Draw the current score."""
        if not self._score_font:
            return
            
        # Score shadow
        score_text = self._score_font.render(str(self.score), True, (0, 0, 0))
        score_rect = score_text.get_rect(centerx=SCREEN_WIDTH // 2 + 2, top=22)
        self.screen.blit(score_text, score_rect)
        
        # Score
        score_text = self._score_font.render(str(self.score), True, TEXT_COLOR)
        score_rect = score_text.get_rect(centerx=SCREEN_WIDTH // 2, top=20)
        self.screen.blit(score_text, score_rect)
        
    def handle_events(self) -> Tuple[bool, bool]:
        """
        Handle pygame events for human play.
        
        Returns:
            Tuple of (should_quit, did_flap)
        """
        should_quit = False
        did_flap = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    did_flap = True
                elif event.key == pygame.K_ESCAPE:
                    should_quit = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                did_flap = True
                
        return should_quit, did_flap
        
    def save_episode_log(self, filename: Optional[str] = None):
        """Save state history to CSV file."""
        if not self.state_history:
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(LOGS_DIR, f"episode_{self.episode_number}_{timestamp}.csv")
            
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            if self.state_history:
                writer = csv.DictWriter(f, fieldnames=self.state_history[0].keys())
                writer.writeheader()
                writer.writerows(self.state_history)
                
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get the state history for coaching analysis."""
        return self.state_history.copy()
        
    def close(self):
        """Clean up resources."""
        if self.render_mode:
            pygame.quit()
            
    @property
    def observation_space_size(self) -> int:
        """Get size of observation space (for DQN input)."""
        return 6  # bird_y, velocity, dist, gap_y, upper_h, lower_top
        
    @property
    def action_space_size(self) -> int:
        """Get size of action space (for DQN output)."""
        return 2  # no-op, flap


class GymnasiumWrapper:
    """
    Wrapper to make FlappyBirdGame compatible with gymnasium-style interface.
    Also supports using actual flappy-bird-gymnasium when available.
    """
    
    def __init__(self, use_gymnasium: bool = True, render_mode: Optional[str] = None):
        """
        Initialize wrapper.
        
        Args:
            use_gymnasium: If True, try to use flappy-bird-gymnasium package
            render_mode: "human" for visual, None for headless
        """
        self.use_gymnasium = use_gymnasium
        self.render_mode = render_mode
        self.env = None
        
        if use_gymnasium:
            try:
                import gymnasium as gym
                import flappy_bird_gymnasium
                self.env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
                self._gymnasium_available = True
            except ImportError:
                print("flappy-bird-gymnasium not available, using built-in engine")
                self._gymnasium_available = False
                
        if self.env is None:
            self.env = FlappyBirdGame(render_mode=render_mode)
            self._gymnasium_available = False
            
    @property
    def observation_space(self):
        """Get observation space."""
        if self._gymnasium_available:
            return self.env.observation_space
        return type('Space', (), {'shape': (6,)})()
        
    @property
    def action_space(self):
        """Get action space."""
        if self._gymnasium_available:
            return self.env.action_space
        return type('Space', (), {'n': 2, 'sample': lambda: 0})()
        
    def reset(self):
        """Reset environment."""
        if self._gymnasium_available:
            return self.env.reset()
        obs = self.env.reset()
        return self._dict_to_array(obs), {}
        
    def step(self, action):
        """Take a step."""
        if self._gymnasium_available:
            return self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._dict_to_array(obs), reward, terminated, truncated, info
        
    def render(self):
        """Render the environment."""
        if self._gymnasium_available:
            return self.env.render()
        return self.env.render()
        
    def close(self):
        """Close the environment."""
        if self.env:
            self.env.close()
            
    def _dict_to_array(self, obs_dict: Dict) -> List[float]:
        """Convert observation dict to array."""
        import numpy as np
        return np.array([
            obs_dict.get('bird_y', 0) / SCREEN_HEIGHT,
            obs_dict.get('bird_velocity', 0) / 10.0,
            obs_dict.get('distance_to_next_pipe', 0) / SCREEN_WIDTH,
            obs_dict.get('next_pipe_gap_y', 0) / SCREEN_HEIGHT,
            obs_dict.get('next_pipe_upper_height', 0) / SCREEN_HEIGHT,
            obs_dict.get('next_pipe_lower_top', 0) / SCREEN_HEIGHT
        ], dtype=np.float32)
        
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get state history (only available with built-in engine)."""
        if hasattr(self.env, 'get_state_history'):
            return self.env.get_state_history()
        return []
