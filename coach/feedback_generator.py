# Feedback Generator - Deterministic Coaching Feedback

"""
Generates actionable coaching feedback based on mistake analysis.
Uses deterministic rules for explainable, reproducible feedback.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coach.mistake_analyzer import MistakeType, MistakeAnalysis


@dataclass
class CoachingFeedback:
    """Complete coaching feedback package."""
    mistake_type: str
    main_message: str
    specific_tips: List[str]
    encouragement: str
    score: int
    detailed_analysis: Optional[str] = None


class FeedbackGenerator:
    """
    Generates coaching feedback based on mistake analysis.
    
    Features:
    - Deterministic feedback templates (same mistake → same feedback)
    - Context-aware tips based on mistake details
    - Score-based encouragement system
    - Structured for both display and TTS
    """
    
    def __init__(self):
        """Initialize feedback templates."""
        self._init_templates()
        
    def _init_templates(self):
        """Initialize feedback templates for each mistake type."""
        self.templates = {
            MistakeType.PANIC_BEHAVIOR: {
                'main': "Stay calm! You're flapping too frantically near obstacles.",
                'tips': [
                    "Use single, controlled flaps instead of mashing the button.",
                    "Remember: One well-timed flap is better than multiple panic flaps.",
                    "Take a breath before each pipe - panicking makes things worse."
                ]
            },
            MistakeType.OVERFLAPPING: {
                'main': "You're flapping too frequently. Give the bird time to fall between flaps.",
                'tips': [
                    "Wait for the bird to start falling before flapping again.",
                    "Try to maintain a rhythm - flap, wait, flap, wait.",
                    "Watch your bird's trajectory, not just the pipes."
                ]
            },
            MistakeType.LATE_REACTION: {
                'main': "You're reacting too late to pipes. Start adjusting your height earlier.",
                'tips': [
                    "Look ahead to the next pipe and prepare your position in advance.",
                    "React when you're about 150-200 pixels away from the pipe.",
                    "Don't wait until the last moment to make adjustments."
                ]
            },
            MistakeType.POOR_ANTICIPATION: {
                'main': "Plan ahead! Position yourself for the next pipe while still far away.",
                'tips': [
                    "Always aim for the center of the gap when approaching.",
                    "Start adjusting your trajectory early, not at the last moment.",
                    "Think one pipe ahead at all times."
                ]
            },
            MistakeType.POOR_CENTERING: {
                'main': "Try to stay centered in the pipe gaps for maximum safety margin.",
                'tips': [
                    "Aim for the middle of the gap, not too high or too low.",
                    "Staying centered gives you more room for error.",
                    "Small adjustments are better than big corrections."
                ]
            },
            MistakeType.GENERAL: {
                'main': "Keep practicing! Every game is a learning opportunity.",
                'tips': [
                    "Focus on timing your flaps consistently.",
                    "Watch the gap center and guide the bird towards it.",
                    "Stay relaxed and maintain a steady rhythm."
                ]
            }
        }
        
    def generate_feedback(self, analysis: MistakeAnalysis, score: int) -> CoachingFeedback:
        """
        Generate coaching feedback from mistake analysis.
        
        Args:
            analysis: MistakeAnalysis from MistakeAnalyzer
            score: Player's score this game
            
        Returns:
            CoachingFeedback with message, tips, and encouragement
        """
        mistake_type = analysis.mistake_type
        template = self.templates.get(mistake_type, self.templates[MistakeType.GENERAL])
        
        # Get base message and tips
        main_message = template['main']
        tips = template['tips'][:2]  # Use first 2 tips
        
        # Add context-specific tip based on details
        specific_tip = self._generate_specific_tip(analysis)
        if specific_tip:
            tips.append(specific_tip)
            
        # Generate encouragement based on score
        encouragement = self._generate_encouragement(score)
        
        # Generate detailed analysis text
        detailed = self._generate_detailed_analysis(analysis)
        
        return CoachingFeedback(
            mistake_type=mistake_type.value,
            main_message=main_message,
            specific_tips=tips,
            encouragement=encouragement,
            score=score,
            detailed_analysis=detailed
        )
        
    def _generate_specific_tip(self, analysis: MistakeAnalysis) -> Optional[str]:
        """Generate context-specific tip based on mistake details."""
        details = analysis.details
        mistake_type = analysis.mistake_type
        
        if mistake_type == MistakeType.PANIC_BEHAVIOR:
            flap_count = details.get('flap_count', 0)
            if flap_count > 4:
                return f"You flapped {flap_count} times in just {details.get('window_size', 15)} frames - try to stay calmer!"
                
        elif mistake_type == MistakeType.OVERFLAPPING:
            min_interval = details.get('min_interval', 0)
            if min_interval < 5:
                return f"Your flaps were only {min_interval} frames apart - wait longer between flaps."
                
        elif mistake_type == MistakeType.LATE_REACTION:
            distance = details.get('flap_distance', 0)
            if distance < 30:
                return f"You flapped when only {distance:.0f}px from the pipe - way too close!"
            return f"React earlier! Flapping at {distance:.0f}px gives no time to adjust."
            
        elif mistake_type == MistakeType.POOR_ANTICIPATION:
            vertical_distance = details.get('vertical_distance', 0)
            return f"You were {vertical_distance:.0f}px away from the gap center when far from the pipe."
            
        elif mistake_type == MistakeType.POOR_CENTERING:
            avg_distance = details.get('average_distance', 0)
            return f"Your average distance from gap center was {avg_distance:.0f}px - aim for the middle!"
            
        return None
        
    def _generate_encouragement(self, score: int) -> str:
        """Generate score-based encouragement message."""
        if score == 0:
            return "Don't worry! Everyone starts here. Focus on passing just one pipe first."
        elif score < 3:
            return "You're getting the hang of it! Keep practicing the basics."
        elif score < 5:
            return "Good progress! You're building consistency."
        elif score < 10:
            return "Nice! You're getting better. Focus on maintaining your rhythm."
        elif score < 20:
            return "Excellent! You're playing well. Keep up the good work!"
        elif score < 50:
            return "Outstanding performance! You're mastering the game!"
        else:
            return "Incredible! You're playing at an expert level!"
            
    def _generate_detailed_analysis(self, analysis: MistakeAnalysis) -> str:
        """Generate detailed technical analysis for advanced users."""
        details = analysis.details
        lines = [f"Mistake Type: {analysis.mistake_type.value}"]
        lines.append(f"Confidence: {analysis.confidence:.1%}")
        lines.append(f"Frame: {analysis.frame_index}")
        
        for key, value in details.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
                
        return "\n".join(lines)
        
    def get_tts_text(self, feedback: CoachingFeedback) -> str:
        """
        Generate text optimized for text-to-speech.
        
        Shorter, more conversational version for voice output.
        """
        # Start with main message
        text = feedback.main_message
        
        # Add one tip
        if feedback.specific_tips:
            text += f" {feedback.specific_tips[0]}"
            
        # Add short encouragement if score is notable
        if feedback.score >= 5:
            if feedback.score >= 20:
                text += " Great job on the score!"
            elif feedback.score >= 10:
                text += " Nice score!"
                
        return text
        
    def generate_session_summary(self, feedbacks: List[CoachingFeedback]) -> Dict[str, Any]:
        """
        Generate a summary of the entire coaching session.
        
        Args:
            feedbacks: List of all feedback generated this session
            
        Returns:
            Summary with statistics and recommendations
        """
        if not feedbacks:
            return {'error': 'No feedback to summarize'}
            
        # Count mistake types
        mistake_counts = {}
        total_score = 0
        scores = []
        
        for fb in feedbacks:
            mistake_type = fb.mistake_type
            mistake_counts[mistake_type] = mistake_counts.get(mistake_type, 0) + 1
            total_score += fb.score
            scores.append(fb.score)
            
        # Find primary weakness
        primary_mistake = max(mistake_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate statistics
        avg_score = total_score / len(feedbacks)
        best_score = max(scores)
        
        # Generate recommendation
        recommendation = self._generate_session_recommendation(primary_mistake, mistake_counts)
        
        return {
            'games_played': len(feedbacks),
            'average_score': avg_score,
            'best_score': best_score,
            'mistake_distribution': mistake_counts,
            'primary_weakness': primary_mistake,
            'recommendation': recommendation
        }
        
    def _generate_session_recommendation(self, primary_mistake: str, 
                                          counts: Dict[str, int]) -> str:
        """Generate session-level recommendation."""
        recommendations = {
            'panic_behavior': "Focus on staying calm. Try taking slow breaths between games. "
                             "Remember: rushing makes everything harder.",
            'overflapping': "Work on your rhythm. Try counting 'one-two' between flaps. "
                           "Less is often more in Flappy Bird.",
            'late_reaction': "Practice looking ahead. Focus on the NEXT pipe, not the current one. "
                            "Early reactions give more control.",
            'poor_anticipation': "Think strategically. Start positioning early. "
                                "The best players plan 2-3 pipes ahead.",
            'poor_centering': "Precision practice needed. Aim for the exact center of each gap. "
                             "Small margins lead to big improvements.",
            'general': "Keep practicing! Focus on consistent timing and staying relaxed."
        }
        
        return recommendations.get(primary_mistake, recommendations['general'])
