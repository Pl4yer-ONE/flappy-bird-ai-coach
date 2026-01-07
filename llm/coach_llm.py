# Coach LLM - Natural Language Coaching with Multiple LLM Providers

"""
LLM-powered coaching that generates natural language feedback.
Supports multiple providers: Ollama (local), Groq, HuggingFace.
Falls back gracefully when no LLM is available.
"""

import sys
import os
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SYSTEM_PROMPT, LLAMA_MODEL, LLM_MAX_TOKENS

# Try universal client first, fall back to ollama
try:
    from llm.universal_client import get_universal_client, UniversalLLMClient
    USE_UNIVERSAL = True
except ImportError:
    from llm.ollama_client import get_client, OllamaClient
    USE_UNIVERSAL = False

from coach.ai_coach import AICoach
from coach.feedback_generator import CoachingFeedback


class CoachLLM:
    """
    LLM-powered coaching using Llama.
    
    Enhances rule-based feedback with natural language generation,
    making advice more conversational and personalized.
    
    Features:
    - Context-aware feedback based on game state
    - Interactive chat for Q&A with coach
    - Session-aware advice that remembers past mistakes
    """
    
    def __init__(self, model: str = LLAMA_MODEL):
        """
        Initialize LLM coach.
        
        Args:
            model: Model name (default: llama3.2)
        """
        self.model = model
        
        # Use universal client for multi-provider support
        if USE_UNIVERSAL:
            self.client = get_universal_client()
        else:
            self.client = get_client()
            
        self.base_coach = AICoach()
        self.chat_history: List[Dict[str, str]] = []
        self._available = None
        
        # Log provider info
        if USE_UNIVERSAL and hasattr(self.client, 'get_provider_info'):
            info = self.client.get_provider_info()
            print(f"🤖 LLM Coach: {info.get('name', 'unknown')} - {info.get('model', 'unknown')}")
        
    def is_available(self) -> bool:
        """Check if LLM is available."""
        if self._available is None:
            self._available = self.client.is_available()
        return self._available
        
    def enhance_feedback(self, feedback: CoachingFeedback, 
                         game_context: Optional[Dict] = None) -> str:
        """
        Enhance rule-based feedback with natural language.
        
        Args:
            feedback: Base CoachingFeedback from AI coach
            game_context: Optional additional context
            
        Returns:
            Natural language enhanced feedback
        """
        if not self.is_available():
            # Fallback to structured feedback
            return f"{feedback.main_message}\n\nTip: {feedback.specific_tips[0] if feedback.specific_tips else ''}"
            
        # Build prompt
        prompt = self._build_feedback_prompt(feedback, game_context)
        
        # Generate natural language
        response = self.client.generate(
            prompt=prompt,
            model=self.model,
            system=SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=LLM_MAX_TOKENS
        )
        
        return response if not response.startswith('Error') else feedback.main_message
        
    def _build_feedback_prompt(self, feedback: CoachingFeedback,
                                context: Optional[Dict] = None) -> str:
        """Build prompt for feedback enhancement."""
        prompt_parts = [
            f"The player just died in Flappy Bird with a score of {feedback.score}.",
            f"Their mistake was classified as: {feedback.mistake_type}",
            f"Base coaching message: {feedback.main_message}",
            f"Tips: {', '.join(feedback.specific_tips)}",
        ]
        
        if context:
            prompt_parts.append(f"Game context: {context}")
            
        if self.chat_history:
            prompt_parts.append("The player has been practicing for a while.")
            
        prompt_parts.append(
            "Generate a friendly, encouraging coaching message (2-3 sentences). "
            "Be specific, actionable, and supportive."
        )
        
        return "\n".join(prompt_parts)
        
    def generate_coaching(self, state_history: List[Dict], score: int) -> str:
        """
        Generate complete coaching feedback for a game.
        
        Args:
            state_history: Game state history
            score: Final score
            
        Returns:
            Natural language coaching feedback
        """
        # Get base analysis
        feedback = self.base_coach.analyze_and_coach(state_history, score)
        
        # Enhance with LLM
        return self.enhance_feedback(feedback)
        
    def chat(self, user_message: str) -> str:
        """
        Interactive chat with the coach.
        
        Args:
            user_message: Player's question or message
            
        Returns:
            Coach's response
        """
        if not self.is_available():
            return self._get_fallback_response(user_message)
            
        # Add user message to history
        self.chat_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Build messages with context
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT}
        ]
        
        # Add session context if available
        summary = self.base_coach.get_session_summary()
        if summary and 'games_played' in summary:
            context_msg = (
                f"Session context: Player has played {summary['games_played']} games "
                f"with average score {summary.get('average_score', 0):.1f}. "
                f"Primary weakness: {summary.get('primary_weakness', 'unknown')}."
            )
            messages.append({'role': 'system', 'content': context_msg})
            
        # Add chat history (keep last 10 exchanges)
        messages.extend(self.chat_history[-20:])
        
        # Generate response
        response = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.8,
            max_tokens=200
        )
        
        # Save response to history
        if not response.startswith('Error'):
            self.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
        return response
        
    def _get_fallback_response(self, message: str) -> str:
        """Get deterministic fallback response when LLM unavailable."""
        message_lower = message.lower()
        
        # Check for common questions
        if any(w in message_lower for w in ['how', 'improve', 'better', 'tip']):
            return ("Focus on timing your flaps consistently. Watch the gap center and "
                   "guide the bird towards it. Stay relaxed and maintain a steady rhythm!")
                   
        if any(w in message_lower for w in ['why', 'die', 'crash', 'hit']):
            summary = self.base_coach.get_session_summary()
            if summary and 'primary_weakness' in summary:
                return (f"Based on your session, your main challenge is "
                       f"{summary['primary_weakness']}. {summary.get('recommendation', '')}")
            return "Check your timing and positioning. The key is staying calm and centered."
            
        if any(w in message_lower for w in ['good', 'well', 'best']):
            return ("Great question! The best players stay calm, plan ahead, and maintain "
                   "a consistent rhythm. Focus on the gap center, not the pipes themselves.")
                   
        return ("I'm here to help you improve! Focus on staying calm, timing your flaps "
               "consistently, and aiming for the center of each gap.")
               
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        
    def get_motivational_message(self, score: int) -> str:
        """
        Generate a motivational message based on score.
        
        Args:
            score: Player's score
            
        Returns:
            Motivational message
        """
        if not self.is_available():
            if score == 0:
                return "Don't give up! Everyone starts somewhere. Try again!"
            elif score < 5:
                return "You're getting better! Keep practicing!"
            elif score < 10:
                return "Nice work! You're improving!"
            else:
                return "Excellent! You're becoming a pro!"
                
        prompt = (f"Generate a short, energetic motivational message for a Flappy Bird player "
                 f"who just scored {score}. One sentence, encouraging.")
                 
        return self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=0.9,
            max_tokens=50
        )
