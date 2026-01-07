# Vision Coach - LLaVA-Powered Screenshot Analysis

"""
Vision-based coaching using LLaVA model.
Analyzes gameplay screenshots to identify visual patterns and mistakes.
"""

import sys
import os
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLAVA_MODEL, LLM_MAX_TOKENS
from llm.ollama_client import get_client
from vision.screenshot_capture import get_capture


class VisionCoach:
    """
    Vision-based coaching using LLaVA.
    
    Analyzes gameplay screenshots to provide visual feedback,
    identifying patterns that might be missed by state-only analysis.
    
    Features:
    - Death frame analysis
    - Visual pattern recognition
    - Combined text + image feedback
    """
    
    def __init__(self, model: str = LLAVA_MODEL):
        """
        Initialize vision coach.
        
        Args:
            model: LLaVA model name
        """
        self.model = model
        self.client = get_client()
        self._capture = None  # Lazy init to avoid pygame conflicts
        self._available = None
        
    def is_available(self) -> bool:
        """Check if vision model is available."""
        if self._available is None:
            self._available = (self.client.is_available() and 
                             self.client.has_model(self.model))
        return self._available
        
    def analyze_death_screenshot(self, 
                                  image_path: str,
                                  score: int,
                                  mistake_type: str = "unknown") -> str:
        """
        Analyze a death screenshot to understand what went wrong.
        
        Args:
            image_path: Path to screenshot image
            score: Score at death
            mistake_type: Known mistake type from rule-based analysis
            
        Returns:
            Visual analysis feedback
        """
        if not self.is_available():
            return self._get_fallback_analysis(score, mistake_type)
            
        prompt = self._build_death_analysis_prompt(score, mistake_type)
        
        response = self.client.generate_with_image(
            prompt=prompt,
            image_path=image_path,
            model=self.model,
            temperature=0.7,
            max_tokens=LLM_MAX_TOKENS
        )
        
        return response if not response.startswith('Error') else \
               self._get_fallback_analysis(score, mistake_type)
               
    def analyze_death_bytes(self,
                            image_bytes: bytes,
                            score: int,
                            mistake_type: str = "unknown") -> str:
        """
        Analyze death screenshot from bytes.
        
        Args:
            image_bytes: Screenshot image bytes
            score: Score at death
            mistake_type: Known mistake type
            
        Returns:
            Visual analysis feedback
        """
        if not self.is_available():
            return self._get_fallback_analysis(score, mistake_type)
            
        prompt = self._build_death_analysis_prompt(score, mistake_type)
        
        response = self.client.generate_with_image_bytes(
            prompt=prompt,
            image_bytes=image_bytes,
            model=self.model,
            temperature=0.7,
            max_tokens=LLM_MAX_TOKENS
        )
        
        return response if not response.startswith('Error') else \
               self._get_fallback_analysis(score, mistake_type)
               
    def _build_death_analysis_prompt(self, score: int, mistake_type: str) -> str:
        """Build prompt for death screenshot analysis."""
        return f"""Analyze this Flappy Bird death screenshot.

Score: {score}
Known mistake: {mistake_type}

Looking at the image:
1. Where is the bird positioned relative to the pipe gap?
2. Is the bird too high, too low, or correctly positioned?
3. What specific adjustment would have helped avoid this death?

Provide a brief, actionable coaching tip (2 sentences max) based on what you see."""
        
    def analyze_gameplay_pattern(self, 
                                  screenshots: List[str]) -> str:
        """
        Analyze a series of screenshots to identify patterns.
        
        Args:
            screenshots: List of screenshot paths
            
        Returns:
            Pattern analysis feedback
        """
        if not self.is_available() or not screenshots:
            return "Pattern analysis unavailable"
            
        # Analyze the most recent screenshot with context
        latest = screenshots[-1]
        
        prompt = f"""Analyze this Flappy Bird gameplay screenshot.

This is part of a sequence of {len(screenshots)} frames.

Describe:
1. The current situation (bird position, nearest pipe)
2. Any obvious problems or good positioning
3. What the player should do next

Keep response brief and actionable (2-3 sentences)."""
        
        return self.client.generate_with_image(
            prompt=prompt,
            image_path=latest,
            model=self.model,
            temperature=0.6,
            max_tokens=150
        )
        
    def get_visual_tip(self, surface_bytes: bytes) -> str:
        """
        Get a quick visual tip from current game frame.
        
        Args:
            surface_bytes: Current game surface as bytes
            
        Returns:
            Quick visual tip
        """
        if not self.is_available():
            return ""
            
        prompt = """Look at this Flappy Bird game frame. 
In one short sentence, what should the player do right now?
Just the action, nothing else."""
        
        return self.client.generate_with_image_bytes(
            prompt=prompt,
            image_bytes=surface_bytes,
            model=self.model,
            temperature=0.5,
            max_tokens=30
        )
        
    def _get_fallback_analysis(self, score: int, mistake_type: str) -> str:
        """Fallback analysis when vision model unavailable."""
        analyses = {
            'panic_behavior': "Visual analysis suggests rapid, erratic movements near the pipe.",
            'overflapping': "The bird appears to be positioned too high, likely from too many flaps.",
            'late_reaction': "The collision occurred very close to the pipe - earlier action was needed.",
            'poor_anticipation': "Bird positioning was not aligned with the gap early enough.",
            'poor_centering': "The trajectory was off-center, leaving less margin for error."
        }
        return analyses.get(mistake_type, 
                           f"Score {score}: Focus on timing and positioning for improvement.")


# Singleton instance
_vision_coach: Optional[VisionCoach] = None


def get_vision_coach() -> VisionCoach:
    """Get global vision coach instance."""
    global _vision_coach
    if _vision_coach is None:
        _vision_coach = VisionCoach()
    return _vision_coach
