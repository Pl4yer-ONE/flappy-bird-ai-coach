# Pipe Class - Procedural Generation and Collision

"""
Pipe class for Flappy Bird obstacles.
Handles procedural gap generation, scrolling, and collision detection.
"""

import pygame
import random
from typing import Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PIPE_WIDTH, PIPE_GAP, PIPE_VELOCITY, SCREEN_HEIGHT,
    PIPE_MIN_HEIGHT, PIPE_MAX_HEIGHT, GROUND_Y,
    PIPE_COLOR, PIPE_BORDER_COLOR
)


class Pipe:
    """
    A pair of pipes (top and bottom) with a gap for the bird to fly through.
    
    The gap position is randomly generated within allowed bounds.
    Pipes scroll left at constant velocity.
    """
    
    def __init__(self, x: float, gap_y: Optional[float] = None):
        """
        Initialize pipe pair.
        
        Args:
            x: Initial X position (right edge of screen for spawning)
            gap_y: Optional gap center Y position. Random if not specified.
        """
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP
        self.velocity = PIPE_VELOCITY
        
        # Calculate gap position
        if gap_y is None:
            # Random gap position within bounds
            min_gap_y = PIPE_MIN_HEIGHT + self.gap // 2
            max_gap_y = GROUND_Y - PIPE_MIN_HEIGHT - self.gap // 2
            self.gap_y = random.randint(min_gap_y, max_gap_y)
        else:
            self.gap_y = gap_y
            
        # Calculate pipe heights
        self.upper_height = self.gap_y - self.gap // 2
        self.lower_top = self.gap_y + self.gap // 2
        self.lower_height = GROUND_Y - self.lower_top
        
        # Scoring
        self.passed = False
        
    def update(self):
        """Move pipe left."""
        self.x -= self.velocity
        
    def is_off_screen(self) -> bool:
        """Check if pipe has scrolled completely off screen."""
        return self.x + self.width < 0
        
    def collides_with(self, bird) -> bool:
        """
        Check collision with bird using circle-rectangle intersection.
        
        Args:
            bird: Bird object with x, y, and radius attributes
            
        Returns:
            True if collision detected
        """
        # Get bird properties
        bx, by, br = bird.x, bird.y, bird.radius
        
        # Check horizontal overlap first (early exit)
        if bx + br < self.x or bx - br > self.x + self.width:
            return False
            
        # Check collision with upper pipe
        if by - br < self.upper_height:
            # Bird overlaps with upper pipe height
            if bx + br > self.x and bx - br < self.x + self.width:
                return True
                
        # Check collision with lower pipe
        if by + br > self.lower_top:
            # Bird overlaps with lower pipe height
            if bx + br > self.x and bx - br < self.x + self.width:
                return True
                
        return False
        
    def get_gap_center(self) -> Tuple[float, float]:
        """Get the center point of the gap."""
        return (self.x + self.width / 2, self.gap_y)
        
    def render(self, screen: pygame.Surface, sprites: Optional[dict] = None):
        """
        Render the pipe pair.
        
        Args:
            screen: Pygame surface to render on
            sprites: Optional dict of sprite images
        """
        if sprites and 'pipe' in sprites:
            self._render_sprites(screen, sprites['pipe'])
        else:
            self._render_simple(screen)
            
    def _render_simple(self, screen: pygame.Surface):
        """Simple rendering when sprites are not available."""
        # Upper pipe
        upper_rect = pygame.Rect(self.x, 0, self.width, self.upper_height)
        pygame.draw.rect(screen, PIPE_COLOR, upper_rect)
        pygame.draw.rect(screen, PIPE_BORDER_COLOR, upper_rect, 3)
        
        # Upper pipe cap
        cap_height = 20
        cap_width = self.width + 6
        upper_cap = pygame.Rect(
            self.x - 3,
            self.upper_height - cap_height,
            cap_width,
            cap_height
        )
        pygame.draw.rect(screen, PIPE_COLOR, upper_cap)
        pygame.draw.rect(screen, PIPE_BORDER_COLOR, upper_cap, 3)
        
        # Lower pipe
        lower_rect = pygame.Rect(self.x, self.lower_top, self.width, self.lower_height)
        pygame.draw.rect(screen, PIPE_COLOR, lower_rect)
        pygame.draw.rect(screen, PIPE_BORDER_COLOR, lower_rect, 3)
        
        # Lower pipe cap
        lower_cap = pygame.Rect(
            self.x - 3,
            self.lower_top,
            cap_width,
            cap_height
        )
        pygame.draw.rect(screen, PIPE_COLOR, lower_cap)
        pygame.draw.rect(screen, PIPE_BORDER_COLOR, lower_cap, 3)
        
    def _render_sprites(self, screen: pygame.Surface, pipe_sprite: pygame.Surface):
        """Render using sprite images."""
        # Scale sprite to pipe width
        scaled = pygame.transform.scale(pipe_sprite, (self.width, pipe_sprite.get_height()))
        
        # Upper pipe (flipped)
        upper_pipe = pygame.transform.flip(scaled, False, True)
        # Tile or stretch to fill upper area
        upper_y = self.upper_height - upper_pipe.get_height()
        screen.blit(upper_pipe, (self.x, upper_y))
        
        # Lower pipe
        screen.blit(scaled, (self.x, self.lower_top))
        
    def get_state(self) -> dict:
        """Get pipe state for logging."""
        return {
            'pipe_x': self.x,
            'gap_y': self.gap_y,
            'upper_height': self.upper_height,
            'lower_top': self.lower_top,
            'passed': self.passed
        }
        
    def __repr__(self) -> str:
        return f"Pipe(x={self.x:.1f}, gap_y={self.gap_y}, passed={self.passed})"


class PipeManager:
    """
    Manages the collection of pipes in the game.
    Handles spawning, removal, and scoring.
    """
    
    def __init__(self, spawn_distance: int = 200):
        """
        Initialize pipe manager.
        
        Args:
            spawn_distance: Horizontal distance between pipes
        """
        from config import SCREEN_WIDTH, PIPE_SPAWN_DISTANCE
        self.pipes = []
        self.spawn_distance = spawn_distance if spawn_distance else PIPE_SPAWN_DISTANCE
        self.screen_width = SCREEN_WIDTH
        self.next_spawn_x = self.screen_width + 100
        
    def reset(self):
        """Clear all pipes and reset spawn position."""
        self.pipes = []
        self.next_spawn_x = self.screen_width + 100
        
    def update(self, bird_x: float) -> int:
        """
        Update all pipes and handle spawning/removal.
        
        Args:
            bird_x: Bird's X position for scoring
            
        Returns:
            Number of pipes passed this frame (for scoring)
        """
        score_increment = 0
        
        # Update existing pipes
        for pipe in self.pipes:
            pipe.update()
            
            # Check if bird passed pipe
            if not pipe.passed and pipe.x + pipe.width < bird_x:
                pipe.passed = True
                score_increment += 1
                
        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if not p.is_off_screen()]
        
        # Spawn new pipes
        if len(self.pipes) == 0 or self.pipes[-1].x < self.screen_width - self.spawn_distance:
            self.pipes.append(Pipe(self.screen_width))
            
        return score_increment
        
    def get_next_pipe(self, bird_x: float) -> Optional[Pipe]:
        """Get the next pipe the bird needs to pass."""
        for pipe in self.pipes:
            if pipe.x + pipe.width > bird_x:
                return pipe
        return None
        
    def check_collision(self, bird) -> bool:
        """Check if bird collides with any pipe."""
        for pipe in self.pipes:
            if pipe.collides_with(bird):
                return True
        return False
        
    def render(self, screen: pygame.Surface, sprites: Optional[dict] = None):
        """Render all pipes."""
        for pipe in self.pipes:
            pipe.render(screen, sprites)
            
    def get_state(self, bird_x: float) -> dict:
        """Get pipe state relative to bird."""
        next_pipe = self.get_next_pipe(bird_x)
        if next_pipe:
            return {
                'distance_to_next_pipe': next_pipe.x - bird_x,
                'next_pipe_gap_y': next_pipe.gap_y,
                'next_pipe_upper_height': next_pipe.upper_height,
                'next_pipe_lower_top': next_pipe.lower_top
            }
        return {
            'distance_to_next_pipe': float('inf'),
            'next_pipe_gap_y': 0,
            'next_pipe_upper_height': 0,
            'next_pipe_lower_top': 0
        }
