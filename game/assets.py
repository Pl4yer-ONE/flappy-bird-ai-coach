# Assets Module - Sprite and Sound Loading

"""
Asset loading and management for Flappy Bird.
Handles sprites, sounds, and fonts with fallback rendering.
"""

import pygame
import os
from typing import Dict, Optional, Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS_DIR


class AssetManager:
    """
    Manages game assets including sprites, sounds, and fonts.
    
    Provides fallback rendering when assets are not available,
    allowing the game to run without external asset files.
    """
    
    def __init__(self):
        """Initialize asset manager."""
        self.sprites: Dict[str, pygame.Surface] = {}
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.fonts: Dict[str, pygame.font.Font] = {}
        self.loaded = False
        
        # Asset directories
        self.sprite_dir = os.path.join(ASSETS_DIR, "sprites")
        self.sound_dir = os.path.join(ASSETS_DIR, "sounds")
        self.font_dir = os.path.join(ASSETS_DIR, "fonts")
        
    def load_all(self) -> bool:
        """
        Load all assets.
        
        Returns:
            True if all critical assets loaded successfully
        """
        self._load_sprites()
        self._load_sounds()
        self._load_fonts()
        self.loaded = True
        return True
        
    def _load_sprites(self):
        """Load sprite images (original Flappy Bird assets)."""
        sprite_files = {
            'bird': ['yellowbird-midflap.png', 'yellowbird-upflap.png', 'yellowbird-downflap.png'],
            'bird_blue': ['bluebird-midflap.png', 'bluebird-upflap.png', 'bluebird-downflap.png'],
            'bird_red': ['redbird-midflap.png', 'redbird-upflap.png', 'redbird-downflap.png'],
            'pipe': ['pipe-green.png'],
            'pipe_red': ['pipe-red.png'],
            'background': ['background-day.png'],
            'background_night': ['background-night.png'],
            'ground': ['base.png'],
            'gameover': ['gameover.png'],
            'message': ['message.png'],
            'numbers': [f'{i}.png' for i in range(10)]
        }
        
        for sprite_name, files in sprite_files.items():
            sprites = []
            for filename in files:
                path = os.path.join(self.sprite_dir, filename)
                if os.path.exists(path):
                    try:
                        img = pygame.image.load(path).convert_alpha()
                        sprites.append(img)
                    except pygame.error as e:
                        print(f"Warning: Could not load sprite {path}: {e}")
                        
            if sprites:
                if len(sprites) == 1:
                    self.sprites[sprite_name] = sprites[0]
                else:
                    self.sprites[sprite_name] = sprites
                    
    def _load_sounds(self):
        """Load sound effects."""
        sound_files = {
            'flap': 'wing.wav',
            'score': 'point.wav',
            'hit': 'hit.wav',
            'die': 'die.wav',
            'swoosh': 'swoosh.wav'
        }
        
        try:
            pygame.mixer.init()
        except pygame.error:
            print("Warning: Sound mixer not available")
            return
            
        for sound_name, filename in sound_files.items():
            path = os.path.join(self.sound_dir, filename)
            if os.path.exists(path):
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(path)
                except pygame.error as e:
                    print(f"Warning: Could not load sound {path}: {e}")
                    
    def _load_fonts(self):
        """Load fonts with fallback to system fonts."""
        font_sizes = [16, 24, 32, 48, 64]
        
        # Try to load custom font
        custom_font_path = os.path.join(self.font_dir, "flappy.ttf")
        
        for size in font_sizes:
            if os.path.exists(custom_font_path):
                try:
                    self.fonts[f'main_{size}'] = pygame.font.Font(custom_font_path, size)
                except pygame.error:
                    self.fonts[f'main_{size}'] = pygame.font.Font(None, size)
            else:
                self.fonts[f'main_{size}'] = pygame.font.Font(None, size)
                
    def get_sprite(self, name: str) -> Optional[pygame.Surface]:
        """Get a sprite by name."""
        return self.sprites.get(name)
        
    def get_sound(self, name: str) -> Optional[pygame.mixer.Sound]:
        """Get a sound by name."""
        return self.sounds.get(name)
        
    def get_font(self, size: int) -> pygame.font.Font:
        """Get a font of specified size."""
        key = f'main_{size}'
        if key in self.fonts:
            return self.fonts[key]
        # Fallback to closest size
        for s in [32, 24, 48, 16, 64]:
            if f'main_{s}' in self.fonts:
                return self.fonts[f'main_{s}']
        return pygame.font.Font(None, size)
        
    def play_sound(self, name: str, volume: float = 1.0):
        """Play a sound effect."""
        sound = self.sounds.get(name)
        if sound:
            sound.set_volume(volume)
            sound.play()
            
    def has_sprites(self) -> bool:
        """Check if any sprites are loaded."""
        return len(self.sprites) > 0
        
    def has_sounds(self) -> bool:
        """Check if any sounds are loaded."""
        return len(self.sounds) > 0


# Global asset manager instance
_asset_manager: Optional[AssetManager] = None


def get_asset_manager() -> AssetManager:
    """Get the global asset manager instance."""
    global _asset_manager
    if _asset_manager is None:
        _asset_manager = AssetManager()
    return _asset_manager


def create_fallback_sprites() -> Dict[str, pygame.Surface]:
    """
    Create simple fallback sprites for when assets are missing.
    These provide basic visual representation without external files.
    """
    sprites = {}
    
    # Bird sprites (3 frames for animation)
    bird_frames = []
    for i in range(3):
        surf = pygame.Surface((34, 24), pygame.SRCALPHA)
        # Body
        pygame.draw.ellipse(surf, (255, 204, 0), (0, 4, 28, 20))
        # Eye
        pygame.draw.circle(surf, (255, 255, 255), (22, 10), 5)
        pygame.draw.circle(surf, (0, 0, 0), (24, 10), 2)
        # Beak
        pygame.draw.polygon(surf, (255, 165, 0), [(28, 12), (34, 14), (28, 18)])
        # Wing (different positions for animation)
        wing_y = [8, 10, 12][i]
        pygame.draw.ellipse(surf, (230, 180, 0), (5, wing_y, 12, 8))
        bird_frames.append(surf)
    sprites['bird'] = bird_frames
    
    # Pipe sprite
    pipe_surf = pygame.Surface((52, 400), pygame.SRCALPHA)
    pygame.draw.rect(pipe_surf, (84, 174, 52), (0, 0, 52, 400))
    pygame.draw.rect(pipe_surf, (60, 128, 40), (0, 0, 52, 400), 3)
    # Cap
    pygame.draw.rect(pipe_surf, (84, 174, 52), (-3, 0, 58, 24))
    pygame.draw.rect(pipe_surf, (60, 128, 40), (-3, 0, 58, 24), 3)
    sprites['pipe'] = pipe_surf
    
    return sprites
