# Voice Coach - Text-to-Speech Feedback

"""
Voice-based coaching using gTTS and pyttsx3 fallback.
Speaks coaching advice aloud for hands-free feedback.
"""

import os
import sys
import threading
import queue
from typing import Optional
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TTS_LANGUAGE, TTS_SLOW, TTS_CACHE_DIR, VOICE_RATE, VOICE_VOLUME


class VoiceCoach:
    """
    Text-to-Speech coaching using gTTS with pyttsx3 fallback.
    
    Features:
    - gTTS for high-quality cloud synthesis
    - pyttsx3 for offline fallback
    - Non-blocking audio playback
    - Audio queue management
    """
    
    def __init__(self):
        """Initialize voice coach."""
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self._worker_thread = None
        self._stop_flag = False
        
        # Check available TTS engines
        self._gtts_available = self._check_gtts()
        self._pyttsx3_available = self._check_pyttsx3()
        self._pygame_mixer_available = self._check_pygame_mixer()
        
        # Initialize pyttsx3 engine if available
        self._pyttsx3_engine = None
        if self._pyttsx3_available:
            self._init_pyttsx3()
            
        # Start worker thread
        self._start_worker()
        
    def _check_gtts(self) -> bool:
        """Check if gTTS is available."""
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False
            
    def _check_pyttsx3(self) -> bool:
        """Check if pyttsx3 is available."""
        try:
            import pyttsx3
            return True
        except ImportError:
            return False
            
    def _check_pygame_mixer(self) -> bool:
        """Check if pygame mixer is available."""
        try:
            import pygame.mixer
            return True
        except ImportError:
            return False
            
    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty('rate', VOICE_RATE)
            self._pyttsx3_engine.setProperty('volume', VOICE_VOLUME)
        except Exception as e:
            print(f"Failed to initialize pyttsx3: {e}")
            self._pyttsx3_available = False
            
    def _start_worker(self):
        """Start background worker thread for speech playback."""
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        
    def _worker(self):
        """Background worker that processes speech queue."""
        while not self._stop_flag:
            try:
                text = self.speech_queue.get(timeout=0.5)
                self.is_speaking = True
                self._notify_speaking(True)
                self._speak_now(text)
                self.is_speaking = False
                self._notify_speaking(False)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}")
                self.is_speaking = False
                self._notify_speaking(False)
                
    def _notify_speaking(self, speaking: bool):
        """Notify callback of speaking state change."""
        if hasattr(self, 'on_speaking') and self.on_speaking:
            try:
                self.on_speaking(speaking)
            except Exception as e:
                print(f"Speaking callback error: {e}")
                
    def speak(self, text: str):
        """
        Queue text for speech (non-blocking).
        
        Args:
            text: Text to speak
        """
        self.speech_queue.put(text)
        
    def speak_now(self, text: str):
        """
        Speak text immediately (blocking).
        
        Args:
            text: Text to speak
        """
        self._speak_now(text)
        
    def _speak_now(self, text: str):
        """Internal method to speak text."""
        # Try gTTS first (better quality)
        if self._gtts_available and self._pygame_mixer_available:
            if self._speak_gtts(text):
                return
                
        # Fallback to pyttsx3
        if self._pyttsx3_available:
            self._speak_pyttsx3(text)
            
    def _speak_gtts(self, text: str) -> bool:
        """Speak using gTTS."""
        try:
            from gtts import gTTS
            import pygame.mixer
            import time
            
            # Generate speech
            tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)
            
            # Save to temp file
            temp_file = os.path.join(TTS_CACHE_DIR, "temp_speech.mp3")
            tts.save(temp_file)
            
            # Play with pygame
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for completion
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # Cleanup
            pygame.mixer.music.unload()
            os.remove(temp_file)
            
            return True
            
        except Exception as e:
            print(f"gTTS error: {e}")
            return False
            
    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3."""
        try:
            if self._pyttsx3_engine:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
        except Exception as e:
            print(f"pyttsx3 error: {e}")
            
    def speak_feedback(self, main_message: str, tip: Optional[str] = None):
        """
        Speak coaching feedback.
        
        Args:
            main_message: Main coaching message
            tip: Optional additional tip
        """
        text = main_message
        if tip:
            text += f" {tip}"
        self.speak(text)
        
    def speak_encouragement(self, score: int):
        """
        Speak score-based encouragement.
        
        Args:
            score: Player's score
        """
        if score == 0:
            self.speak("Don't give up! Try again!")
        elif score < 5:
            self.speak(f"Score {score}! Keep practicing!")
        elif score < 10:
            self.speak(f"Nice! Score {score}!")
        elif score < 20:
            self.speak(f"Great job! Score {score}!")
        else:
            self.speak(f"Amazing! Score {score}!")
            
    def is_available(self) -> bool:
        """Check if any TTS engine is available."""
        return self._gtts_available or self._pyttsx3_available
        
    def stop(self):
        """Stop current speech and clear queue."""
        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
                
        # Stop pygame mixer if playing
        if self._pygame_mixer_available:
            try:
                import pygame.mixer
                pygame.mixer.music.stop()
            except:
                pass
                
    def shutdown(self):
        """Shutdown the voice coach."""
        self._stop_flag = True
        self.stop()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)


# Singleton instance
_voice_coach: Optional[VoiceCoach] = None


def get_voice_coach() -> VoiceCoach:
    """Get global voice coach instance."""
    global _voice_coach
    if _voice_coach is None:
        _voice_coach = VoiceCoach()
    return _voice_coach
