# Dashboard - Polished Multi‑Panel UI
"""
A refined dashboard that combines the original Flappy Bird game with:
- Scrollable chat panel
- Live coaching panel (danger meter, tips)
- Death heatmap panel
- Session stats panel (score chart)
- Real‑time voice interaction (TTS & speech‑to‑text)
"""

import pygame
import sys
import os
import time
import threading
import json
from pathlib import Path
from enum import Enum
from typing import List, Tuple
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
)
from game.game_engine import FlappyBirdGame
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
# Theme – lightweight copy of ui.theme for this file (keeps it self‑contained)
# ---------------------------------------------------------------------------
class Colors:
    BG_DARK = (12, 12, 20)
    BG_PANEL = (20, 22, 35)
    BG_LIGHTER = (30, 32, 48)
    ACCENT_PRIMARY = (0, 255, 200)
    ACCENT_WARNING = (255, 180, 50)
    ACCENT_DANGER = (255, 80, 80)
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
# Panel base (very small subset needed for this dashboard)
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
        self.messages: List[Tuple[str, bool]] = []  # (text, is_user)
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
        # Message area
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
        # Input box
        input_rect = pygame.Rect(20, self.rect.height - 60, self.rect.width - 40, 40)
        pygame.draw.rect(self._surface, Colors.BG_LIGHTER, input_rect, border_radius=8)
        pygame.draw.rect(self._surface, Colors.BORDER, input_rect, 1, border_radius=8)
        placeholder = 'Type a message...' if not self.input_text else self.input_text
        txt_surf = Fonts.SMALL.render(placeholder, True, Colors.TEXT_MUTED if not self.input_text else Colors.TEXT_WHITE)
        # Vertically center text in input box
        text_y = input_rect.y + (input_rect.height - txt_surf.get_height()) // 2
        self._surface.blit(txt_surf, (input_rect.x + 12, text_y))
        if self.speaking:
            pygame.draw.circle(self._surface, Colors.ACCENT_DANGER, (self.rect.width - 25, 25), 6)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                rel = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
                if pygame.Rect(15, self.rect.height - 55, self.rect.width - 30, 40).collidepoint(rel):
                    self.focused = True
                else:
                    self.focused = False
                return True
        if self.focused and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.input_text.strip():
                    user_msg = self.input_text
                    self.add_message(user_msg, True)
                    self.input_text = ''
                    
                    # Use LLM for response if available
                    if self.llm_coach:
                        self.is_thinking = True
                        self.invalidate()
                        try:
                            response = self.llm_coach.chat(user_msg)
                            self.add_message(response)
                            if self.voice_coach and self.voice_coach.is_available():
                                self.voice_coach.speak(response)
                        except Exception as e:
                            self.add_message(f"Sorry, I couldn't respond: {str(e)[:50]}")
                        finally:
                            self.is_thinking = False
                    else:
                        self.add_message('AI Coach is not available. Enable LLM mode!')
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
        score_surf = Fonts.SCORE.render(str(self.score), True, Colors.TEXT_WHITE)
        self._surface.blit(score_surf, (self.rect.width // 2 - score_surf.get_width() // 2, 10))
        if self.game_surface:
            # Maintain aspect ratio
            available_w = self.rect.width - 40
            available_h = self.rect.height - 80
            scale_w = available_w
            scale_h = available_h
            
            # Simple stretch for now to fill panel, or modify to keep aspect ratio if preferred
            # For professional look, let's keep it centered with padding
            scaled = pygame.transform.smoothscale(self.game_surface, (scale_w, scale_h))
            self._surface.blit(scaled, (20, 60))
        if not self.playing:
            overlay = pygame.Surface((self.rect.width - 20, self.rect.height - 80), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self._surface.blit(overlay, (10, 60))
            txt = 'Press SPACE to start' if not self.game_over else 'Game Over – SPACE to restart'
            txt_surf = Fonts.HEADER.render(txt, True, Colors.TEXT_MUTED)
            self._surface.blit(txt_surf, (self.rect.width // 2 - txt_surf.get_width() // 2, self.rect.height // 2))

class CoachingPanel(Panel):
    def __init__(self, rect: pygame.Rect):
        super().__init__(rect)
        self.status = 'Ready'
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
        title = Fonts.HEADER.render('🎓 Live Coach', True, Colors.ACCENT_PRIMARY)
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
# Main Dashboard class
# ---------------------------------------------------------------------------
class Dashboard:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        
        self.enable_llm = config_dict.get('enable_llm', False)
        self.enable_vision = config_dict.get('enable_vision', False)
        
        pygame.init()
        self.screen = pygame.display.set_mode((DASHBOARD_WINDOW_WIDTH, DASHBOARD_WINDOW_HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption('Flappy Bird AI Coach - Dashboard')
        self.clock = pygame.time.Clock()
        self.running = True
        self.return_to_menu = False  # Flag to indicate return to menu
        self.state = DashboardState.IDLE
        # Voice input flag
        self.voice_input_active = getattr(__import__('config'), 'VOICE_INPUT_ENABLED', False)
        self.recognizer = None
        # Initialize speech recognizer only if enabled
        if self.voice_input_active:
            try:
                self.recognizer = sr.Recognizer()
                self.voice_thread = threading.Thread(target=self._voice_listener, daemon=True)
                self.voice_thread.start()
            except Exception as e:
                print(f"Voice input disabled due to error: {e}")
                self.voice_input_active = False
        # Game
        self.game = FlappyBirdGame(render_mode='rgb_array', enable_logging=True)
        self.game_surface = pygame.Surface((GAME_W, GAME_H))
        # Coach
        self.coach = AICoach()
        
        # LLM Coach
        self.llm_coach = None
        if self.enable_llm and CoachLLM:
            try:
                print("Initializing Llama Coach...")
                self.llm_coach = CoachLLM()
                if not self.llm_coach.is_available():
                     print("Llama not available, disabling.")
                     self.llm_coach = None
            except Exception as e:
                print(f"Failed to init LLM: {e}")
        
        # Vision Coach - DISABLED to prevent segfault
        # The screenshot_capture or VisionCoach init conflicts with pygame
        self.vision_coach = None
        # Uncomment below to re-enable vision (may cause crashes on some systems)
        # if self.enable_vision and VisionCoach:
        #     try:
        #         print("Initializing Llava Vision Coach...")
        #         self.vision_coach = VisionCoach()
        #     except Exception as e:
        #         print(f"Failed to init Vision: {e}")

        # Voice - DISABLED to prevent segfault from audio conflicts
        # The voice coach uses pyttsx3/gTTS which conflicts with pygame audio
        self.voice = None
        # Uncomment below to re-enable voice (may cause crashes on some systems)
        # try:
        #     self.voice = get_voice_coach()
        #     if self.voice and self.voice.is_available():
        #         self.voice.on_speaking = self._on_voice_speaking
        # except Exception as e:
        #     print(f"Voice coach disabled due to error: {e}")
        # Panels
        self._create_panels()
        for p in self.panels:
            p.invalidate()
            
        # Initial greeting
        mode_str = "Standard"
        if self.enable_llm and self.enable_vision: mode_str = "Ultimate"
        elif self.enable_llm: mode_str = "Llama (NLP)"
        elif self.enable_vision: mode_str = "Llava (Vision)"
        
        self.chat_panel.add_message(f"Welcome to {mode_str} Mode!")
        if self.llm_coach:
            self.chat_panel.add_message("I am ready to chat!")

    def _create_panels(self):
        # Compute panel rectangles using dynamic right width to fit total window size
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
        # Optionally could sync other UI elements

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
                # Check back button click (top-left corner)
                back_rect = pygame.Rect(10, 10, 100, 32)
                if back_rect.collidepoint(ev.pos):
                    self.return_to_menu = True
                    self.running = False
                    return
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.return_to_menu = True  # Return to menu instead of just closing
                    self.running = False
                elif ev.key == pygame.K_SPACE:
                    # Ctrl+Space toggles voice input
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.voice_input_active = not self.voice_input_active
                        print(f'Voice input toggled: {self.voice_input_active}')
                    else:
                        if self.state in (DashboardState.IDLE, DashboardState.GAME_OVER):
                            self._start_game()
            for p in self.panels:
                if p.handle_event(ev):
                    break

    def _start_game(self):
        self.game.reset()
        self.state = DashboardState.PLAYING
        self.game_panel.set_game_state(playing=True, game_over=False)
        self.coaching_panel.set_status('Watching', 'Game started!')
        self.chat_panel.add_message('🎮 New game started! Good luck!')

    def _update(self, dt: float):
        if self.state == DashboardState.PLAYING:
            keys = pygame.key.get_pressed()
            action = 1 if keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w] else 0
            self.coach.record_state(self.game._get_full_state())
            obs, reward, done, _, info = self.game.step(action)
            self.game_panel.update_score(info['score'], self.game_panel.best_score)
            self._render_game_to_surface()
            self.game_panel.set_game_surface(self.game_surface)
            self._update_danger(info)
            if done:
                self._handle_game_over(info)

    def _render_game_to_surface(self):
        self.game_surface.fill((78, 192, 202))
        if hasattr(self.game, 'pipes'):
            self.game.pipes.render(self.game_surface)
        pygame.draw.rect(self.game_surface, (222, 216, 149), (0, GAME_H - 112, GAME_W, 112))
        if hasattr(self.game, 'bird'):
            self.game.bird.render(self.game_surface)
        # Additional: could overlay voice input status if needed


    def _update_danger(self, info):
        if not (self.game.bird and self.game.pipes.pipes):
            self.coaching_panel.set_danger_level(0)
            return
        bird_x = self.game.bird.x
        bird_y = self.game.bird.y
        for pipe in self.game.pipes.pipes:
            if pipe.x + pipe.width > bird_x:
                dist_x = pipe.x - bird_x
                gap_center = pipe.gap_y
                vert = abs(bird_y - gap_center)
                if dist_x < 100:
                    danger = min(1.0, (100 - dist_x) / 100 * 0.5 + vert / 100 * 0.5)
                else:
                    danger = 0
                self.coaching_panel.set_danger_level(danger)
                if danger > 0.7:
                    self.coaching_panel.set_status('Danger', 'Watch out!')
                elif danger > 0.4:
                    self.coaching_panel.set_status('Warning', 'Stay centered')
                else:
                    self.coaching_panel.set_status('Good', '')
                break
        # Voice input handling is separate thread

    def _voice_listener(self):
        """Background thread listening for speech and adding to chat panel."""
        while self.running and self.voice_input_active:
            try:
                with sr.Microphone() as source:
                    if self.recognizer:
                        # Reduced sensitivity: Higher threshold and dynamic adjustment
                        self.recognizer.energy_threshold = getattr(config, 'SPEECH_ENERGY_THRESHOLD', 3000)
                        self.recognizer.dynamic_energy_ratio = getattr(config, 'SPEECH_DYNAMIC_RATIO', 2.0)
                        self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                        audio = self.recognizer.listen(source, timeout=config.SPEECH_TIMEOUT, phrase_time_limit=config.SPEECH_PHRASE_TIME_LIMIT)
                    else:
                         continue
                try:
                    text = self.recognizer.recognize_google(audio)
                    # Add recognized text as user message
                    self.chat_panel.add_message(text, True)
                    
                    if self.llm_coach:
                        # Use LLM for response
                        response = self.llm_coach.chat(text)
                        self.chat_panel.add_message(response)
                        if self.voice and self.voice.is_available():
                            self.voice.speak(response)
                    else:
                        self.chat_panel.add_message('I heard you: ' + text)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    continue
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f'Voice listener error: {e}')
                continue

    def _handle_game_over(self, info):
        self.state = DashboardState.GAME_OVER
        self.game_panel.set_game_state(playing=False, game_over=True)
        self.coaching_panel.set_status('Game Over', '')
        self.chat_panel.add_message(f'Game over! Score: {info["score"]}')
        if hasattr(self.game, 'bird'):
            self.heatmap_panel.record_death(self.game.bird.x, self.game.bird.y, GAME_W, GAME_H)
        self.stats_panel.record_game(info['score'])
        self._save_history(info['score'])
        feedback = self.coach.analyze_and_coach(score=info['score'])
        if self.llm_coach:
             try:
                 # Enhance feedback with LLM
                 enhanced = self.llm_coach.enhance_feedback(feedback)
                 self.chat_panel.add_message(enhanced)
                 if self.voice and self.voice.is_available():
                    self.voice.speak(enhanced)
             except Exception as e:
                 print(f"LLM enhancement failed: {e}")
                 self.chat_panel.add_message(feedback.main_message)
                 if self.voice and self.voice.is_available():
                    self.voice.speak(feedback.main_message)
        else:
             self.chat_panel.add_message(feedback.main_message)
             if self.voice and self.voice.is_available():
                self.voice.speak(feedback.main_message)
        if feedback.specific_tips:
            self.chat_panel.add_message('💡 Tip: ' + feedback.specific_tips[0])

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
                'mode': 'AI Coach' if self.enable_llm else 'Standard'
            }
            history.append(record)
            
            with open(hist_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Failed to save history: {e}")

    def _render(self):
        self.screen.fill(Colors.BG_DARK)
        for p in self.panels:
            p.render(self.screen)
        
        # Draw back button (top-left corner)
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
        if self.game:
            self.game.close()
        pygame.quit()
        # Ensure voice thread stops
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

