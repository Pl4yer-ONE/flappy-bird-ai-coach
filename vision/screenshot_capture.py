# Screenshot Capture - Gameplay Image Capture

"""
Captures gameplay screenshots for vision analysis.
Handles Pygame surface to image conversion and archival.
"""

import pygame
import os
import sys
from datetime import datetime
from typing import Optional, Tuple, List
from pathlib import Path
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGS_DIR


class ScreenshotCapture:
    """
    Captures and manages gameplay screenshots.
    
    Features:
    - Pygame surface to image conversion  
    - Automatic naming and archival
    - Death/mistake frame capture
    - Thumbnail generation
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize screenshot capture.
        
        Args:
            save_dir: Directory to save screenshots (default: logs/screenshots)
        """
        self.save_dir = Path(save_dir or os.path.join(LOGS_DIR, "screenshots"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.screenshot_count = 0
        
    def capture(self, surface: pygame.Surface, 
                label: str = "game",
                save: bool = True) -> Tuple[bytes, str]:
        """
        Capture a screenshot from Pygame surface.
        
        Args:
            surface: Pygame surface to capture
            label: Label for the screenshot (e.g., "death", "score")
            save: Whether to save to disk
            
        Returns:
            Tuple of (image bytes, filename)
        """
        self.screenshot_count += 1
        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"{self.session_id}_{label}_{self.screenshot_count:04d}_{timestamp}.png"
        
        # Convert surface to bytes
        image_bytes = self._surface_to_bytes(surface)
        
        # Save if requested
        if save:
            filepath = self.save_dir / filename
            pygame.image.save(surface, str(filepath))
            
        return image_bytes, filename
        
    def capture_death(self, surface: pygame.Surface, 
                      score: int,
                      mistake_type: str = "unknown") -> Tuple[bytes, str]:
        """
        Capture screenshot at death for analysis.
        
        Args:
            surface: Current game surface
            score: Score at death
            mistake_type: Type of mistake made
            
        Returns:
            Tuple of (image bytes, filename)
        """
        label = f"death_s{score}_{mistake_type}"
        return self.capture(surface, label=label, save=True)
        
    def _surface_to_bytes(self, surface: pygame.Surface, 
                          format: str = 'PNG') -> bytes:
        """Convert Pygame surface to image bytes."""
        buffer = io.BytesIO()
        pygame.image.save(surface, buffer, format)
        buffer.seek(0)
        return buffer.read()
        
    def _surface_to_pil(self, surface: pygame.Surface):
        """Convert Pygame surface to PIL Image."""
        try:
            from PIL import Image
            raw_str = pygame.image.tostring(surface, 'RGB')
            size = surface.get_size()
            return Image.frombytes('RGB', size, raw_str)
        except ImportError:
            return None
            
    def get_recent_screenshots(self, count: int = 5) -> List[Path]:
        """Get list of most recent screenshot paths."""
        screenshots = sorted(self.save_dir.glob("*.png"), reverse=True)
        return screenshots[:count]
        
    def get_death_screenshots(self) -> List[Path]:
        """Get all death screenshots from current session."""
        return sorted(self.save_dir.glob(f"{self.session_id}_death_*.png"))
        
    def cleanup_old(self, keep_count: int = 100):
        """Remove old screenshots keeping only the most recent."""
        screenshots = sorted(self.save_dir.glob("*.png"), reverse=True)
        for old_ss in screenshots[keep_count:]:
            old_ss.unlink()


# Global instance
_capture: Optional[ScreenshotCapture] = None


def get_capture() -> ScreenshotCapture:
    """Get global screenshot capture instance."""
    global _capture
    if _capture is None:
        _capture = ScreenshotCapture()
    return _capture
