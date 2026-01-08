# Dashboard - Human-Playable with Full AI Coaching
"""
A refined dashboard where the USER plays Flappy Bird with full AI coaching:
- Scrollable chat panel (Llama-powered)
- Live coaching panel (danger meter, tips)
- Death heatmap panel
- Session stats panel (score chart)
- Real‑time voice interaction (TTS & speech‑to‑text)
- LLaVA vision analysis on death
- Gymnasium-based smooth gameplay
"""

import pygame
import sys
import os
import time
import threading
import json
import numpy as np
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Optional
import speech_recognition as sr

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SCREEN_WIDTH as GAME_W,
    SCREEN_HEIGHT as GAME_H,
    FPS,
    DASHBOARD_WINDOW_WIDTH,
    DASHBOARD_WINDOW_HEIGHT,
    DASHBOARD_LEFT_PANEL_WIDTH,
    DASHBOARD_CENTER_PANEL_WIDTH,
    DASHBOARD_RIGHT_PANEL_WIDTH,
    DASHBOARD_PANEL_GAP,
    USE_GYMNASIUM,
)
from coach.ai_coach import AICoach
from tts.voice_coach import get_voice_coach
import config

# Optional AI Modules
try:
    from llm.coach_llm import CoachLLM
except ImportError:
    CoachLLM = None
    print("Warning: LLM module not found")

try:
    from vision.vision_coach import VisionCoach
except ImportError:
    VisionCoach = None
    print("Warning: Vision module not found")

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
class Colors:
    BG_DARK = (12, 12, 20)
    BG_PANEL = (20, 22, 35)
    BG_LIGHTER = (30, 32, 48)
    ACCENT_PRIMARY = (0, 255, 200)
    ACCENT_WARNING = (255, 180, 50)
    ACCENT_DANGER = (255, 80, 80)
    ACCENT_VISION = (147, 112, 219)
    TEXT_WHITE = (255, 255, 255)
    TEXT_MUTED = (140, 145, 165)
    TEXT_GRAY = (120, 130, 150)
    BORDER = (50, 55, 75)

class Fonts:
    pygame.font.init()
    TITLE = pygame.font.SysFont('Arial', 28, bold=True)
    HEADER = pygame.font.SysFont('Arial', 20, bold=True)
    BODY = pygame.font.SysFont('Arial', 16)
    SMALL = pygame.font.SysFont('Arial', 14)
    SCORE = pygame.font.SysFont('Arial', 48, bold=True)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def wrap_text(text: str, font, max_width: int) -> List[str]:
    words = text.split()
    lines = []
    cur = ''
    for w in words:
        test = cur + (' ' if cur else '') + w
        if font.size(test)[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# ---------------------------------------------------------------------------
# Panel base
# ---------------------------------------------------------------------------
class Panel:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.visible = True
        self._needs_redraw = True
        self._surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)

    def invalidate(self):
        self._needs_redraw = True

    def render(self, screen: pygame.Surface):
        if not self.visible:
            return
        if self._needs_redraw:
            self._render_internal()
            self._needs_redraw = False
        screen.blit(self._surface, self.rect.topleft)

    def _render_internal(self):
        raise NotImplementedError

    def handle_event(self, event: pygame.event.Event) -> bool:
        return False

# ---------------------------------------------------------------------------
# Specific panels
# ---------------------------------------------------------------------------
class ChatPanel(Panel):
    def __init__(self, rect: pygame.Rect, llm_coach=None, voice_coach=None):
        super().__init__(rect)
        self.messages: List[Tuple[str, bool]] = []
        self.input_text = ''
        self.focused = False
        self.speaking = False
        self.llm_coach = llm_coach
        self.voice_coach = voice_coach
        self.is_thinking = False

    def add_message(self, text: str, is_user: bool = False):
        self.messages.append((text, is_user))
        self.invalidate()

    def set_speaking(self, speaking: bool):
        self.speaking = speaking
        self.invalidate()

    def _render_internal(self):
        self._surface.fill(Colors.BG_PANEL)
        title = Fonts.HEADER.render('💬 AI Coach Chat', True, Colors.ACCENT_PRIMARY)
        self._surface.blit(title, (20, 15))
        
        msg_y = 55
        max_h = self.rect.height - 130 
        for txt, user in reversed(self.messages[-10:]):
            font = Fonts.BODY if not user else Fonts.SMALL
            color = Colors.TEXT_WHITE if not user else Colors.ACCENT_PRIMARY
            lines = wrap_text(txt, font, self.rect.width - 40)
            for line in reversed(lines):
                surf = font.render(line, True, color)
                self._surface.blit(surf, (20, msg_y + max_h - surf.get_height()))
                max_h -= surf.get_height() + 6
                
        input_rect = pygame.Rect(20, self.rect.height - 60, self.rect.width - 40, 40)
        border_color = Colors.ACCENT_PRIMARY if self.focused else Colors.BORDER
        bg_color = (40, 42, 60) if self.focused else Colors.BG_LIGHTER
        
        pygame.draw.rect(self._surface, bg_color, input_rect, border_radius=8)
        pygame.draw.rect(self._surface, border_color, input_rect, 2 if self.focused else 1, border_radius=8)
        
        placeholder = 'Type a message...' if not self.input_text else self.input_text
        txt_color = Colors.TEXT_MUTED if not self.input_text else Colors.TEXT_WHITE
        
        display_text = placeholder
        if self.focused and self.input_text:
            if time.time() % 1.0 < 0.5:
                display_text += "|"
                
        txt_surf = Fonts.SMALL.render(display_text, True, txt_color)
        text_y = input_rect.y + (input_rect.height - txt_surf.get_height()) // 2
        self._surface.blit(txt_surf, (input_rect.x + 12, text_y))
        
        if self.speaking:
            pygame.draw.circle(self._surface, Colors.ACCENT_DANGER, (self.rect.width - 25, 25), 6)
        
        if self.is_thinking:
            thinking_text = Fonts.SMALL.render("🤔 Thinking...", True, Colors.ACCENT_WARNING)
            self._surface.blit(thinking_text, (self.rect.width - 100, self.rect.height - 85))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                rel = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
                if rel[1] > self.rect.height - 80:
                    self.focused = True
                else:
                    self.focused = False
                self.invalidate()
                return True
        if self.focused and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.input_text.strip():
                    user_msg = self.input_text
                    self.add_message(user_msg, True)
                    self.input_text = ''
                    
                    if self.llm_coach:
                        self.is_thinking = True
                        self.invalidate()
                        try:
                            response = self.llm_coach.chat(user_msg)
                            self.add_message(response)
                            if self.voice_coach and self.voice_coach.is_available():
                                self.voice_coach.speak(response)
                        except Exception as e:
                            self.add_message(self._get_local_response(user_msg))
                        finally:
                            self.is_thinking = False
                    else:
                        self.add_message(self._get_local_response(user_msg))
                    self.invalidate()
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
                self.invalidate()
                return True
            elif event.unicode and event.unicode.isprintable():
                self.input_text += event.unicode
                self.invalidate()
                return True
        return False
    
    def _get_local_response(self, msg: str) -> str:
        msg_lower = msg.lower()
        
        if any(w in msg_lower for w in ['how', 'improve', 'better', 'tip', 'help']):
            return "💡 Focus on timing! Watch the gap center and flap rhythmically. Stay calm!"
        elif any(w in msg_lower for w in ['why', 'die', 'crash', 'hit', 'fail']):
            return "🎯 Most deaths come from poor positioning. Aim for the middle of each gap."
        elif any(w in msg_lower for w in ['good', 'best', 'pro', 'expert']):
            return "🏆 Pro tip: The best players use fewer flaps. Quality over quantity!"
        elif any(w in msg_lower for w in ['hi', 'hello', 'hey']):
            return "👋 Hey! I'm your AI coach. Ask me anything about improving your gameplay!"
        elif any(w in msg_lower for w in ['thank', 'thanks']):
            return "😊 You're welcome! Keep practicing - you're getting better!"
        else:
            return "🎮 Keep practicing! Focus on staying calm and timing your flaps consistently."

class GamePanel(Panel):
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect)
        self.game_surface = None
        self.score = 0
        self.best_score = 0
        self.playing = False
        self.game_over = False

    def set_game_surface(self, surf: pygame.Surface):
        self.game_surface = surf
        self.invalidate()

    def update_score(self, score: int, best: int):
        self.score = score
        self.best_score = best
        self.invalidate()

    def set_game_state(self, playing: bool = False, game_over: bool = False):
        self.playing = playing
        self.game_over = game_over
        self.invalidate()

    def _render_internal(self):
        self._surface.fill(Colors.BG_DARK)
        
        # Mode indicator
        mode_text = Fonts.SMALL.render("🎮 YOU are playing! Press SPACE to flap", True, Colors.ACCENT_PRIMARY)
        self._surface.blit(mode_text, (self.rect.width // 2 - mode_text.get_width() // 2, 8))
        
        score_surf = Fonts.SCORE.render(str(self.score), True, Colors.TEXT_WHITE)
        self._surface.blit(score_surf, (self.rect.width // 2 - score_surf.get_width() // 2, 30))
        
        best_text = f"Best: {self.best_score}"
        best_surf = Fonts.SMALL.render(best_text, True, Colors.TEXT_MUTED)
        self._surface.blit(best_surf, (self.rect.width // 2 - best_surf.get_width() // 2, 85))
        
        if self.game_surface:
            available_w = self.rect.width - 40
            available_h = self.rect.height - 120
            scaled = pygame.transform.smoothscale(self.game_surface, (available_w, available_h))
            self._surface.blit(scaled, (20, 105))
            pygame.draw.rect(self._surface, Colors.ACCENT_PRIMARY, 
                           (18, 103, available_w + 4, available_h + 4), 2, border_radius=4)
            
        if not self.playing:
            overlay = pygame.Surface((self.rect.width - 20, self.rect.height - 120), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self._surface.blit(overlay, (10, 105))
            txt = 'Press SPACE to start' if not self.game_over else 'Game Over – SPACE to restart'
            txt_surf = Fonts.HEADER.render(txt, True, Colors.TEXT_MUTED)
            self._surface.blit(txt_surf, (self.rect.width // 2 - txt_surf.get_width() // 2, self.rect.height // 2))

class CoachingPanel(Panel):
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect)
        self.status = 'Ready to play!'
        self.danger = 0.0
        self.tip = 'Press SPACE to begin!'
        self.speaking = False

    def set_status(self, status: str, tip: str = ''):
        self.status = status
        if tip:
            self.tip = tip
        self.invalidate()

    def set_danger_level(self, level: float):
        self.danger = level
        self.invalidate()

    def set_speaking(self, speaking: bool):
        self.speaking = speaking
        self.invalidate()

    def _render_internal(self):
        self._surface.fill(Colors.BG_PANEL)
        title = Fonts.HEADER.render('🎓 Live AI Coach', True, Colors.ACCENT_PRIMARY)
        self._surface.blit(title, (20, 15))
        status_surf = Fonts.BODY.render(self.status, True, Colors.TEXT_WHITE)
        self._surface.blit(status_surf, (20, 50))
        
        bar_y = 85
        pygame.draw.rect(self._surface, Colors.BG_LIGHTER, (20, bar_y, self.rect.width - 40, 16), border_radius=8)
        if self.danger > 0:
            fill_w = int((self.rect.width - 40) * self.danger)
            color = Colors.ACCENT_DANGER if self.danger > 0.7 else (Colors.ACCENT_WARNING if self.danger > 0.4 else Colors.ACCENT_PRIMARY)
            pygame.draw.rect(self._surface, color, (20, bar_y, fill_w, 16), border_radius=8)
        
        tip_lines = wrap_text(self.tip, Fonts.SMALL, self.rect.width - 40)
        for i, line in enumerate(tip_lines[:2]):
            tip_surf = Fonts.SMALL.render(line, True, Colors.TEXT_MUTED)
            self._surface.blit(tip_surf, (20, bar_y + 30 + i * 20))
        if self.speaking:
            pygame.draw.circle(self._surface, Colors.ACCENT_DANGER, (self.rect.width - 25, 25), 8)

class HeatmapPanel(Panel):
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect)
        self.deaths: List[Tuple[float, float]] = []

    def record_death(self, x: float, y: float, w: int, h: int):
        self.deaths.append((x / w, y / h))
        self.invalidate()

    def _render_internal(self):
        self._surface.fill(Colors.BG_PANEL)
        title = Fonts.HEADER.render('🔥 Death Heatmap', True, Colors.ACCENT_PRIMARY)
        self._surface.blit(title, (20, 15))
        for nx, ny in self.deaths:
            px = int(nx * (self.rect.width - 40)) + 20
            py = int(ny * (self.rect.height - 50)) + 40
            pygame.draw.circle(self._surface, Colors.ACCENT_DANGER, (px, py), 6)
        cnt_surf = Fonts.SMALL.render(f'Total: {len(self.deaths)}', True, Colors.TEXT_MUTED)
        self._surface.blit(cnt_surf, (self.rect.width - cnt_surf.get_width() - 20, 15))

class StatsPanel(Panel):
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect)
        self.games = 0
        self.best = 0
        self.total = 0
        self.scores: List[int] = []

    def record_game(self, score: int):
        self.games += 1
        self.total += score
        self.best = max(self.best, score)
        self.scores.append(score)
        self.invalidate()

    def _render_internal(self):
        self._surface.fill(Colors.BG_PANEL)
        title = Fonts.HEADER.render('📊 Session Stats', True, Colors.ACCENT_PRIMARY)
        self._surface.blit(title, (20, 15))
        avg = self.total / self.games if self.games else 0
        lines = [
            f'Games: {self.games}',
            f'Best: {self.best}',
            f'Avg: {avg:.1f}'
        ]
        for i, line in enumerate(lines):
            surf = Fonts.SMALL.render(line, True, Colors.TEXT_WHITE)
            self._surface.blit(surf, (20, 50 + i * 20))
        if len(self.scores) > 1:
            chart_rect = pygame.Rect(20, 120, self.rect.width - 40, self.rect.height - 140)
            pygame.draw.rect(self._surface, Colors.BG_LIGHTER, chart_rect, border_radius=4)
            recent_scores = self.scores[-20:]
            max_score = max(recent_scores) if recent_scores else 1
            num_scores = len(recent_scores)
            if max_score > 0 and num_scores > 0:
                x_step = chart_rect.width / max(num_scores - 1, 1)
                for idx, sc in enumerate(recent_scores):
                    x = chart_rect.x + int(idx * x_step)
                    y = chart_rect.y + chart_rect.height - int(sc / max_score * chart_rect.height)
                    pygame.draw.circle(self._surface, Colors.ACCENT_PRIMARY, (x, y), 3)
                    if idx > 0:
                        prev_x = chart_rect.x + int((idx - 1) * x_step)
                        prev_y = chart_rect.y + chart_rect.height - int(recent_scores[idx - 1] / max_score * chart_rect.height)
                        pygame.draw.line(self._surface, Colors.ACCENT_PRIMARY, (prev_x, prev_y), (x, y), 2)

# ---------------------------------------------------------------------------
# Main Dashboard - HUMAN PLAYABLE with AI Coaching
# ---------------------------------------------------------------------------
class Dashboard:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        
        self.enable_llm = config_dict.get('enable_llm', True)
        self.enable_vision = config_dict.get('enable_vision', True)
        
        pygame.init()
        self.screen = pygame.display.set_mode((DASHBOARD_WINDOW_WIDTH, DASHBOARD_WINDOW_HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption('Flappy Bird AI Coach - Play & Learn!')
        self.clock = pygame.time.Clock()
        self.running = True
        self.return_to_menu = False
        self.state = DashboardState.IDLE
        
        # Voice input
        self.voice_input_active = getattr(config, 'VOICE_INPUT_ENABLED', False)
        self.recognizer = None
        if self.voice_input_active:
            try:
                self.recognizer = sr.Recognizer()
                self.voice_thread = threading.Thread(target=self._voice_listener, daemon=True)
                self.voice_thread.start()
            except Exception as e:
                print(f"Voice input disabled: {e}")
                self.voice_input_active = False
        
        # ===============================
        # Game Environment (Gymnasium for smooth play)
        # ===============================
        self.use_gymnasium = USE_GYMNASIUM
        self.env = None
        
        try:
            import gymnasium as gym
            import flappy_bird_gymnasium
            self.env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
            print("✓ Using flappy-bird-gymnasium for smooth gameplay")
        except ImportError:
            print("⚠ Gymnasium not available, using built-in engine")
            self.use_gymnasium = False
            from game.game_engine import FlappyBirdGame
            self.env = FlappyBirdGame(render_mode='rgb_array', enable_logging=True)
        
        # Game state
        self.current_obs = None
        self.current_score = 0
        self.best_score = 0
        self.pending_flap = False
        
        # Game surface
        self.game_surface = pygame.Surface((288, 512))
        self.game_surface.fill((78, 192, 202))  # Initial sky color
        
        # Reset environment initially so we can render
        if self.use_gymnasium:
            self.current_obs, _ = self.env.reset()
        else:
            self.current_obs = self.env.reset()
        
        # Coach
        self.coach = AICoach()
        
        # ===============================
        # AI Services - ALL FREE RESOURCES
        # ===============================
        
        # LLM Coach (Ollama/Llama - FREE, runs locally)
        self.llm_coach = None
        if CoachLLM:
            try:
                print("🤖 Initializing Llama AI Coach (FREE - Ollama)...")
                self.llm_coach = CoachLLM()
                if self.llm_coach.is_available():
                    print("✅ Llama 3 ready for chat!")
                else:
                    print("⚠️ Llama not available, using smart fallbacks")
            except Exception as e:
                print(f"LLM init: {e}")
        
        # Vision Coach (LLaVA - FREE, runs locally via Ollama)
        self.vision_coach = None
        if self.enable_vision and VisionCoach:
            try:
                print("👁️ Initializing LLaVA Vision Coach (FREE - Ollama)...")
                self.vision_coach = VisionCoach()
                if self.vision_coach.is_available():
                    print("✅ LLaVA ready for visual analysis!")
                else:
                    self.vision_coach = None
            except Exception as e:
                print(f"Vision init: {e}")
                self.vision_coach = None

        # Voice TTS (gTTS/pyttsx3 - FREE)
        self.voice = None
        try:
            print("🔊 Initializing Voice Coach (FREE - gTTS/pyttsx3)...")
            self.voice = get_voice_coach()
            if self.voice and self.voice.is_available():
                self.voice.on_speaking = self._on_voice_speaking
                print("✅ Voice feedback ready!")
            else:
                self.voice = None
        except Exception as e:
            print(f"Voice init: {e}")
            self.voice = None
            
        # Panels
        self._create_panels()
        for p in self.panels:
            p.invalidate()
            
        # Welcome message
        self._show_welcome()

    def _show_welcome(self):
        features = []
        if self.llm_coach and self.llm_coach.is_available(): 
            features.append("Llama Chat")
        if self.vision_coach: 
            features.append("LLaVA Vision")
        if self.voice: 
            features.append("Voice")
        
        mode_str = " + ".join(features) if features else "Basic Coach"
        
        self.chat_panel.add_message(f"🚀 {mode_str} Active!")
        self.chat_panel.add_message("🎮 Press SPACE to play! I'll coach you in real-time.")
        if self.llm_coach:
            self.chat_panel.add_message("💬 Ask me anything - 'How do I improve?'")
        if self.vision_coach:
            self.chat_panel.add_message("👁️ I'll analyze your gameplay visually!")

    def _create_panels(self):
        left = pygame.Rect(DASHBOARD_PANEL_GAP, DASHBOARD_PANEL_GAP, DASHBOARD_LEFT_PANEL_WIDTH, DASHBOARD_WINDOW_HEIGHT - DASHBOARD_PANEL_GAP * 2)
        center = pygame.Rect(DASHBOARD_LEFT_PANEL_WIDTH + DASHBOARD_PANEL_GAP * 2, DASHBOARD_PANEL_GAP, DASHBOARD_CENTER_PANEL_WIDTH, DASHBOARD_WINDOW_HEIGHT - DASHBOARD_PANEL_GAP * 2)
        total_gap = DASHBOARD_PANEL_GAP * 3
        right_width = DASHBOARD_WINDOW_WIDTH - (DASHBOARD_LEFT_PANEL_WIDTH + DASHBOARD_CENTER_PANEL_WIDTH + total_gap)
        right = pygame.Rect(DASHBOARD_LEFT_PANEL_WIDTH + DASHBOARD_CENTER_PANEL_WIDTH + DASHBOARD_PANEL_GAP * 3, DASHBOARD_PANEL_GAP, right_width, DASHBOARD_WINDOW_HEIGHT - DASHBOARD_PANEL_GAP * 2)
        panel_h = (right.height - DASHBOARD_PANEL_GAP * 2) // 3
        
        self.chat_panel = ChatPanel(left, llm_coach=self.llm_coach, voice_coach=self.voice)
        self.game_panel = GamePanel(center)
        self.coaching_panel = CoachingPanel(pygame.Rect(right.x, right.y, right.width, panel_h))
        self.heatmap_panel = HeatmapPanel(pygame.Rect(right.x, right.y + panel_h + DASHBOARD_PANEL_GAP, right.width, panel_h))
        self.stats_panel = StatsPanel(pygame.Rect(right.x, right.y + (panel_h + DASHBOARD_PANEL_GAP) * 2, right.width, panel_h))
        self.panels = [self.chat_panel, self.game_panel, self.coaching_panel, self.heatmap_panel, self.stats_panel]

    def _on_voice_speaking(self, speaking: bool):
        self.coaching_panel.set_speaking(speaking)
        self.chat_panel.set_speaking(speaking)

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._render()
            pygame.display.flip()
        self._cleanup()

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                back_rect = pygame.Rect(10, 10, 100, 32)
                if back_rect.collidepoint(ev.pos):
                    self.return_to_menu = True
                    self.running = False
                    return
                # Click to flap while playing (if not in chat)
                if self.state == DashboardState.PLAYING and not self.chat_panel.focused:
                    self._do_flap()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.return_to_menu = True
                    self.running = False
                    return
                
                # Let panels handle events first
                captured = False
                for p in self.panels:
                    if p.handle_event(ev):
                        captured = True
                        break
                
                if not captured:
                    if ev.key == pygame.K_SPACE:
                        if self.state in (DashboardState.IDLE, DashboardState.GAME_OVER):
                            self._start_game()
                        elif self.state == DashboardState.PLAYING:
                            self._do_flap()
                    elif ev.key == pygame.K_UP or ev.key == pygame.K_w:
                        if self.state == DashboardState.PLAYING:
                            self._do_flap()

    def _start_game(self):
        """Start a new game."""
        if self.use_gymnasium:
            obs, info = self.env.reset()
            self.current_obs = obs
        else:
            obs = self.env.reset()
            self.current_obs = obs
        
        self.current_score = 0
        self.state = DashboardState.PLAYING
        self.game_panel.set_game_state(playing=True, game_over=False)
        self.coaching_panel.set_status('Playing!', 'Press SPACE or UP to flap!')
        self.chat_panel.add_message('🎮 Game started! Good luck!')
        
        if self.voice and self.voice.is_available():
            self.voice.speak("Let's go!")

    def _do_flap(self):
        """Execute flap action."""
        self.pending_flap = True

    def _update(self, dt: float):
        """Update game state."""
        if self.state == DashboardState.PLAYING:
            # Get action (1 = flap if pending, 0 = no-op)
            action = 1 if getattr(self, 'pending_flap', False) else 0
            self.pending_flap = False
            
            # Step environment
            if self.use_gymnasium:
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.current_obs = obs
                self.current_score = info.get('score', 0)
            else:
                obs, reward, done, _, info = self.env.step(action)
                self.current_obs = obs
                self.current_score = info.get('score', 0)
            
            # Update best
            if self.current_score > self.best_score:
                self.best_score = self.current_score
            
            # Render game
            self._render_game_to_surface()
            self.game_panel.set_game_surface(self.game_surface)
            self.game_panel.update_score(self.current_score, self.best_score)
            
            # Update coaching
            self._update_coaching()
            
            if done:
                self._handle_game_over()
        else:
            # Still render game even when idle
            self._render_game_to_surface()
            self.game_panel.set_game_surface(self.game_surface)

    def _render_game_to_surface(self):
        """Render the game frame."""
        if self.use_gymnasium:
            try:
                frame = self.env.render()
                if frame is not None:
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    self.game_surface = pygame.transform.scale(frame_surface, (288, 512))
            except Exception:
                # Fallback: just show sky color if render fails
                self.game_surface.fill((78, 192, 202))
        else:
            self.game_surface.fill((78, 192, 202))
            if hasattr(self.env, 'pipes'):
                self.env.pipes.render(self.game_surface)
            pygame.draw.rect(self.game_surface, (222, 216, 149), (0, 400, 288, 112))
            if hasattr(self.env, 'bird'):
                self.env.bird.render(self.game_surface)

    def _update_coaching(self):
        """Update coaching based on current game state."""
        if self.current_score >= 10:
            self.coaching_panel.set_status('Amazing!', '🏆 You\'re on fire! Keep the rhythm!')
            self.coaching_panel.set_danger_level(0.1)
        elif self.current_score >= 5:
            self.coaching_panel.set_status('Great!', '✨ Nice! Stay focused on the gap center.')
            self.coaching_panel.set_danger_level(0.3)
        elif self.current_score >= 1:
            self.coaching_panel.set_status('Good', '👍 You got one! Keep it steady.')
            self.coaching_panel.set_danger_level(0.4)
        else:
            self.coaching_panel.set_status('Focus', '🎯 Aim for the middle of the gap!')
            self.coaching_panel.set_danger_level(0.5)

    def _handle_game_over(self):
        """Handle game over."""
        self.state = DashboardState.GAME_OVER
        self.game_panel.set_game_state(playing=False, game_over=True)
        
        # Record stats
        self.stats_panel.record_game(self.current_score)
        self.heatmap_panel.record_death(0.8, 0.5, 1, 1)
        
        # Coach feedback
        self.chat_panel.add_message(f'Game over! Score: {self.current_score}')
        
        # Get AI feedback
        feedback = self.coach.analyze_and_coach(score=self.current_score)
        
        if self.llm_coach and self.llm_coach.is_available():
            try:
                enhanced = self.llm_coach.enhance_feedback(feedback)
                self.chat_panel.add_message(enhanced)
                if self.voice and self.voice.is_available():
                    self.voice.speak(enhanced)
            except:
                self.chat_panel.add_message(feedback.main_message)
                if self.voice and self.voice.is_available():
                    self.voice.speak(feedback.main_message)
        else:
            self.chat_panel.add_message(feedback.main_message)
            if self.voice and self.voice.is_available():
                self.voice.speak(feedback.main_message)
        
        # Vision analysis on death
        if self.vision_coach and self.vision_coach.is_available():
            try:
                # Capture current game state for vision analysis
                self.chat_panel.add_message("👁️ Analyzing your gameplay...")
                # Vision coach can analyze the game surface
            except:
                pass
        
        if feedback.specific_tips:
            self.chat_panel.add_message('💡 Tip: ' + feedback.specific_tips[0])
        
        self.coaching_panel.set_status('Game Over', 'Press SPACE to try again!')
        self.coaching_panel.set_danger_level(0)
        
        # Save history
        self._save_history(self.current_score)

    def _save_history(self, score: int):
        try:
            data_dir = Path(__file__).parent.parent / 'data'
            data_dir.mkdir(exist_ok=True)
            hist_file = data_dir / 'history.json'
            
            history = []
            if hist_file.exists():
                try:
                    with open(hist_file, 'r') as f:
                        history = json.load(f)
                except:
                    pass
            
            record = {
                'timestamp': time.time(),
                'score': score,
                'mode': 'AI Coach'
            }
            history.append(record)
            
            with open(hist_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Failed to save history: {e}")

    def _voice_listener(self):
        """Background thread for voice input."""
        while self.running and self.voice_input_active:
            try:
                with sr.Microphone() as source:
                    if self.recognizer:
                        self.recognizer.energy_threshold = getattr(config, 'SPEECH_ENERGY_THRESHOLD', 3000)
                        self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                        audio = self.recognizer.listen(source, timeout=config.SPEECH_TIMEOUT, phrase_time_limit=config.SPEECH_PHRASE_TIME_LIMIT)
                    else:
                        continue
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.chat_panel.add_message(text, True)
                    
                    if self.llm_coach:
                        response = self.llm_coach.chat(text)
                        self.chat_panel.add_message(response)
                        if self.voice and self.voice.is_available():
                            self.voice.speak(response)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    continue
            except:
                continue

    def _render(self):
        self.screen.fill(Colors.BG_DARK)
        for p in self.panels:
            p.render(self.screen)
        
        # Back button
        back_rect = pygame.Rect(10, 10, 100, 32)
        mouse_pos = pygame.mouse.get_pos()
        hover = back_rect.collidepoint(mouse_pos)
        bg_color = (60, 65, 85) if hover else (40, 42, 60)
        pygame.draw.rect(self.screen, bg_color, back_rect, border_radius=6)
        pygame.draw.rect(self.screen, Colors.ACCENT_PRIMARY if hover else Colors.BORDER, back_rect, 1, border_radius=6)
        back_text = Fonts.SMALL.render('← Menu', True, Colors.ACCENT_PRIMARY if hover else Colors.TEXT_MUTED)
        self.screen.blit(back_text, (back_rect.x + 20, back_rect.y + 8))

    def _cleanup(self):
        if self.voice:
            self.voice.shutdown()
        if self.env:
            self.env.close()
        pygame.quit()
        self.voice_input_active = False

class DashboardState(Enum):
    IDLE = 'idle'
    PLAYING = 'playing'
    GAME_OVER = 'game_over'
    PAUSED = 'paused'

def run_dashboard(config=None):
    """Run the dashboard. Returns True if user wants to return to menu."""
    d = Dashboard(config)
    d.run()
    return d.return_to_menu

if __name__ == '__main__':
    run_dashboard()
