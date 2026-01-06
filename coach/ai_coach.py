# AI Coach - Main Orchestration Class

"""
Main AI Coach that coordinates mistake analysis and feedback generation.
Manages session tracking and provides the primary coaching interface.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coach.mistake_analyzer import MistakeAnalyzer, MistakeAnalysis, MistakeType
from coach.feedback_generator import FeedbackGenerator, CoachingFeedback


class AICoach:
    """
    Main AI coaching system that coordinates analysis and feedback.
    
    The coach observes gameplay and provides feedback - it does NOT
    control the bird. This separation ensures:
    - Player maintains full control
    - Feedback is advisory, not prescriptive
    - System is explainable and transparent
    """
    
    def __init__(self):
        """Initialize the AI coach."""
        self.analyzer = MistakeAnalyzer()
        self.feedback_generator = FeedbackGenerator()
        
        # Session tracking
        self.session_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        self.current_game_states: List[Dict[str, Any]] = []
        
    def reset_session(self):
        """Reset session history for a new coaching session."""
        self.session_history = []
        self.session_start = datetime.now()
        self.current_game_states = []
        
    def record_state(self, state: Dict[str, Any]):
        """
        Record a game state for analysis.
        
        Call this every frame during gameplay.
        
        Args:
            state: Current game state dictionary
        """
        self.current_game_states.append(state.copy())
        
    def analyze_and_coach(self, state_history: Optional[List[Dict]] = None,
                          score: int = 0) -> CoachingFeedback:
        """
        Analyze gameplay and generate coaching feedback.
        
        Call this when the player dies to get feedback.
        
        Args:
            state_history: Optional state history (uses recorded if not provided)
            score: Player's score this game
            
        Returns:
            CoachingFeedback with advice for the player
        """
        # Use provided history or recorded states
        history = state_history if state_history else self.current_game_states
        
        # Analyze the death
        analysis = self.analyzer.analyze_death(history)
        
        # Generate feedback
        feedback = self.feedback_generator.generate_feedback(analysis, score)
        
        # Record in session history
        self.session_history.append({
            'timestamp': datetime.now(),
            'score': score,
            'mistake_type': analysis.mistake_type.value,
            'confidence': analysis.confidence,
            'feedback': feedback
        })
        
        # Clear recorded states for next game
        self.current_game_states = []
        
        return feedback
        
    def get_quick_feedback(self, state_history: List[Dict], score: int) -> str:
        """
        Get a quick one-line feedback message.
        
        Useful for in-game overlay or quick display.
        
        Args:
            state_history: State history from the game
            score: Player's score
            
        Returns:
            Short feedback string
        """
        feedback = self.analyze_and_coach(state_history, score)
        return feedback.main_message
        
    def get_tts_feedback(self, state_history: List[Dict], score: int) -> str:
        """
        Get feedback optimized for text-to-speech.
        
        Args:
            state_history: State history from the game
            score: Player's score
            
        Returns:
            TTS-optimized feedback string
        """
        feedback = self.analyze_and_coach(state_history, score)
        return self.feedback_generator.get_tts_text(feedback)
        
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current coaching session.
        
        Returns:
            Dictionary with session statistics and recommendations
        """
        feedbacks = [h['feedback'] for h in self.session_history]
        summary = self.feedback_generator.generate_session_summary(feedbacks)
        
        # Add session metadata
        summary['session_start'] = self.session_start.isoformat()
        summary['session_duration'] = str(datetime.now() - self.session_start)
        
        return summary
        
    def get_improvement_trend(self) -> Dict[str, Any]:
        """
        Analyze improvement trend over the session.
        
        Returns:
            Trend analysis with score progression and mistake changes
        """
        if len(self.session_history) < 2:
            return {'error': 'Need at least 2 games for trend analysis'}
            
        scores = [h['score'] for h in self.session_history]
        mistakes = [h['mistake_type'] for h in self.session_history]
        
        # Calculate trends
        early_avg = sum(scores[:len(scores)//2]) / (len(scores)//2) if len(scores) >= 2 else 0
        late_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2) if len(scores) >= 2 else 0
        
        # Count mistake frequency change
        early_mistakes = mistakes[:len(mistakes)//2]
        late_mistakes = mistakes[len(mistakes)//2:]
        
        primary_early = max(set(early_mistakes), key=early_mistakes.count) if early_mistakes else None
        primary_late = max(set(late_mistakes), key=late_mistakes.count) if late_mistakes else None
        
        improving = late_avg > early_avg
        
        return {
            'early_average': early_avg,
            'late_average': late_avg,
            'score_improvement': late_avg - early_avg,
            'improving': improving,
            'primary_early_mistake': primary_early,
            'primary_late_mistake': primary_late,
            'mistake_changed': primary_early != primary_late
        }
        
    def get_mistake_breakdown(self) -> Dict[str, int]:
        """Get count of each mistake type in session."""
        breakdown = {}
        for h in self.session_history:
            mistake = h['mistake_type']
            breakdown[mistake] = breakdown.get(mistake, 0) + 1
        return breakdown
        
    def get_last_feedback(self) -> Optional[CoachingFeedback]:
        """Get the most recent feedback given."""
        if self.session_history:
            return self.session_history[-1]['feedback']
        return None


# Convenience function
def create_coach() -> AICoach:
    """Create and return a new AI coach instance."""
    return AICoach()
