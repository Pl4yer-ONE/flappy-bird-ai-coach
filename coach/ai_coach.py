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
    
    # =========================================================================
    # INTELLIGENT COACHING - Real-time predictive advice
    # =========================================================================
    
    def get_live_advice(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Get real-time coaching advice based on current game state.
        
        This is the SMART coach - predicts problems before they happen!
        
        Args:
            state: Current game state with bird_y, bird_velocity, pipe info
            
        Returns:
            Quick advice string or None if no advice needed
        """
        bird_y = state.get('bird_y', 256)
        velocity = state.get('bird_velocity', 0)
        dist_to_pipe = state.get('distance_to_next_pipe', 300)
        gap_y = state.get('next_pipe_gap_y', 256)
        
        # Calculate optimal position
        optimal_y = gap_y
        y_diff = bird_y - optimal_y
        
        # Predictive warnings
        if dist_to_pipe < 80:
            # Close to pipe - critical zone
            if abs(y_diff) > 40:
                return "⚠️ Adjust NOW!" if y_diff > 0 else "⚠️ Flap NOW!"
            elif velocity > 4:
                return "🛑 Stop flapping - too fast!"
            elif velocity < -4:
                return "✋ Let gravity help"
        
        elif dist_to_pipe < 150:
            # Approaching pipe - positioning zone
            if y_diff > 50:
                return "🔽 Too high - stop flapping"
            elif y_diff < -50:
                return "🔼 Too low - flap gently"
            elif abs(velocity) > 3:
                return "⚖️ Stabilize your height"
        
        elif dist_to_pipe < 250:
            # Far from pipe - planning zone  
            if abs(y_diff) > 80:
                direction = "down" if y_diff > 0 else "up"
                return f"📍 Start moving {direction} toward gap"
        
        # Top/bottom boundary warnings
        if bird_y < 50:
            return "🚫 Too close to ceiling!"
        elif bird_y > 350:
            return "🚫 Too close to ground!"
        
        return None
    
    def get_smart_tip(self, score: int, death_count: int) -> str:
        """
        Get context-aware coaching tip based on player performance.
        
        Args:
            score: Current/last score
            death_count: Number of deaths this session
            
        Returns:
            Smart contextual tip
        """
        # Beginner tips (struggling players)
        if death_count > 5 and score < 3:
            tips = [
                "💡 Focus on just passing ONE pipe. Small wins build confidence!",
                "💡 Try tapping rhythmically: tap... wait... tap... wait...",
                "💡 Watch the GAP, not the pipes. Aim for the center.",
                "💡 Relax your grip. Tense players tap too frantically.",
                "💡 The bird WANTS to fall. Use gravity - don't fight it!"
            ]
            return tips[death_count % len(tips)]
        
        # Intermediate tips (making progress)
        elif score >= 3 and score < 10:
            tips = [
                "🌟 Great progress! Now work on smoother transitions.",
                "🌟 You're getting it! Try to stay centered in gaps.",
                "🌟 Nice! Predict the NEXT pipe while passing current one.",
                "🌟 Good rhythm! Keep flaps gentle and consistent."
            ]
            return tips[score % len(tips)]
        
        # Advanced tips (skilled players)
        elif score >= 10:
            tips = [
                "🏆 Expert mode! Focus on pixel-perfect gap centering.",
                "🏆 Pro tip: The best players use FEWER flaps.",
                "🏆 Master move: Start positioning 3+ pipes ahead.",
                "🏆 Elite strategy: Maintain consistent altitude between pipes."
            ]
            return tips[score % len(tips)]
        
        # Default encouraging tip
        return "🎮 Keep practicing! Every attempt makes you better."
    
    def analyze_play_style(self) -> Dict[str, Any]:
        """
        Analyze overall play style from session history.
        
        Returns:
            Play style analysis with strengths and weaknesses
        """
        if len(self.session_history) < 3:
            return {'status': 'Need more games for analysis'}
        
        breakdown = self.get_mistake_breakdown()
        trend = self.get_improvement_trend()
        
        # Determine play style
        total = sum(breakdown.values()) or 1
        
        style = "Balanced"
        weakness = None
        strength = None
        
        if breakdown.get('panic_behavior', 0) / total > 0.3:
            style = "Reactive"
            weakness = "Tends to panic near obstacles"
            strength = "Quick reflexes, just need calming"
        elif breakdown.get('overflapping', 0) / total > 0.3:
            style = "Aggressive"
            weakness = "Flaps too frequently"
            strength = "High engagement, needs rhythm"
        elif breakdown.get('late_reaction', 0) / total > 0.3:
            style = "Cautious"
            weakness = "Reacts too late to obstacles"
            strength = "Patient approach, needs earlier action"
        elif breakdown.get('poor_centering', 0) / total > 0.3:
            style = "Imprecise"
            weakness = "Positioning not centered"
            strength = "Timing is good, needs accuracy"
        
        return {
            'style': style,
            'weakness': weakness,
            'strength': strength,
            'improving': trend.get('improving', False),
            'score_trend': trend.get('score_improvement', 0),
            'games_analyzed': len(self.session_history)
        }


# Convenience function
def create_coach() -> AICoach:
    """Create and return a new AI coach instance."""
    return AICoach()
