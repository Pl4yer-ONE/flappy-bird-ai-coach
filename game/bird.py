# Bird Class - OG Flappy Bird Physics

"""
Bird class implementing original Flappy Bird physics and rendering.
Handles gravity, flapping, velocity limits, and collision detection.
"""

import pygame
import math
from typing import Tuple, Optional
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GRAVITY, FLAP_STRENGTH, BIRD_MAX_VELOCITY,
    BIRD_START_X, BIRD_START_Y, BIRD_RADIUS,
    GROUND_Y, BIRD_COLOR, SCREEN_HEIGHT
)


class Bird:
    """
    Flappy Bird with original physics.
    
    The bird is affected by gravity and can flap to gain upward velocity.
    Physics are tuned to match the original game feel.
    """
    
    def __init__(self, x: float = BIRD_START_X, y: float = BIRD_START_Y):
        """Initialize bird at starting position."""
        self.x = x
        self.y = y
        self.velocity = 0.0
        self.radius = BIRD_RADIUS
        self.alive = True
        
        # Animation state
        self.rotation = 0.0  # Rotation based on velocity
        self.flap_frame = 0
        self.animation_timer = 0
        
        # Tracking for coach analysis
        self.time_since_last_flap = 0
        self.total_flaps = 0
        self.flap_history = []  # Frame indices of flaps
        
    def reset(self):
        """Reset bird to starting position."""
        self.x = BIRD_START_X
        self.y = BIRD_START_Y
        self.velocity = 0.0
        self.alive = True
        self.rotation = 0.0
        self.flap_frame = 0
        self.animation_timer = 0
        self.time_since_last_flap = 0
        self.total_flaps = 0
        self.flap_history = []
        
    def flap(self, frame: int = 0):
        """
        Apply upward velocity (flap).
        
        Args:
            frame: Current game frame for history tracking
        """
        if self.alive:
            self.velocity = FLAP_STRENGTH
            self.time_since_last_flap = 0
            self.total_flaps += 1
            self.flap_history.append(frame)
            self.flap_frame = 0  # Reset animation
            
    def update(self):
        """Update bird physics and animation."""
        if not self.alive:
            return
            
        # Apply gravity
        self.velocity += GRAVITY
        
        # Clamp velocity
        self.velocity = min(self.velocity, BIRD_MAX_VELOCITY)
        
        # Update position
        self.y += self.velocity
        
        # Update rotation based on velocity (-90 to 90 degrees)
        target_rotation = self.velocity * 3  # Scale for visual effect
        target_rotation = max(-30, min(90, target_rotation))  # Clamp
        self.rotation = target_rotation
        
        # Increment time since last flap
        self.time_since_last_flap += 1
        
        # Update animation timer
        self.animation_timer += 1
        if self.animation_timer >= 5:  # Change frame every 5 ticks
            self.animation_timer = 0
            self.flap_frame = (self.flap_frame + 1) % 3
            
        # Check bounds
        if self.y - self.radius < 0:
            self.y = self.radius
            self.velocity = 0
        elif self.y + self.radius > GROUND_Y:
            self.y = GROUND_Y - self.radius
            self.velocity = 0
            self.alive = False
            
    def get_rect(self) -> pygame.Rect:
        """Get bounding rectangle for collision detection."""
        return pygame.Rect(
            self.x - self.radius,
            self.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )
        
    def collides_with_point(self, px: float, py: float) -> bool:
        """Check if a point is inside the bird's hitbox (circle)."""
        distance = math.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        return distance < self.radius
        
    def render(self, screen: pygame.Surface, sprites: Optional[dict] = None):
        """
        Render the bird.
        
        Args:
            screen: Pygame surface to render on
            sprites: Optional dict of sprite images
        """
        if sprites and 'bird' in sprites:
            # Use sprite
            bird_sprites = sprites['bird']
            current_sprite = bird_sprites[self.flap_frame % len(bird_sprites)]
            
            # Rotate sprite
            rotated = pygame.transform.rotate(current_sprite, -self.rotation)
            rect = rotated.get_rect(center=(int(self.x), int(self.y)))
            screen.blit(rotated, rect)
        else:
            # Fallback: Draw colored circle with basic animation
            self._render_simple(screen)
            
    def _render_simple(self, screen: pygame.Surface):
        """Simple rendering when sprites are not available."""
        # Draw bird body (yellow circle)
        pygame.draw.circle(screen, BIRD_COLOR, (int(self.x), int(self.y)), self.radius)
        
        # Draw outline
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius, 2)
        
        # Draw eye
        eye_x = int(self.x + self.radius * 0.4)
        eye_y = int(self.y - self.radius * 0.2)
        pygame.draw.circle(screen, (255, 255, 255), (eye_x, eye_y), 4)
        pygame.draw.circle(screen, (0, 0, 0), (eye_x + 1, eye_y), 2)
        
        # Draw beak
        beak_points = [
            (self.x + self.radius, self.y),
            (self.x + self.radius + 8, self.y + 2),
            (self.x + self.radius, self.y + 5)
        ]
        pygame.draw.polygon(screen, (255, 165, 0), beak_points)
        
        # Wing (animated)
        wing_y_offset = [-2, 0, 2][self.flap_frame % 3]
        wing_rect = pygame.Rect(
            self.x - self.radius * 0.5,
            self.y + wing_y_offset,
            self.radius * 0.8,
            self.radius * 0.5
        )
        pygame.draw.ellipse(screen, (230, 180, 0), wing_rect)
        
    def get_state(self) -> dict:
        """Get current state for RL and logging."""
        return {
            'bird_y': self.y,
            'bird_x': self.x,
            'bird_velocity': self.velocity,
            'time_since_last_flap': self.time_since_last_flap,
            'total_flaps': self.total_flaps,
            'alive': self.alive,
            'rotation': self.rotation
        }
        
    def __repr__(self) -> str:
        return f"Bird(x={self.x:.1f}, y={self.y:.1f}, vel={self.velocity:.2f}, alive={self.alive})"
