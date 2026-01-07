# Flappy Bird AI Coach Configuration

"""
Centralized configuration for all game, RL, and coaching settings.
All constants and hyperparameters are defined here for easy experimentation.
"""

import os

# =============================================================================
# Path Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# Ensure directories exist
for dir_path in [ASSETS_DIR, MODELS_DIR, LOGS_DIR, RUNS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# Game Settings (OG Flappy Bird Physics)
# =============================================================================
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
FPS = 60

# Bird Physics
GRAVITY = 0.5
FLAP_STRENGTH = -9
BIRD_MAX_VELOCITY = 10
BIRD_START_X = 50
BIRD_START_Y = SCREEN_HEIGHT // 2
BIRD_RADIUS = 12

# Pipe Settings
PIPE_WIDTH = 52
PIPE_GAP = 100  # Gap between upper and lower pipes
PIPE_SPAWN_DISTANCE = 200  # Horizontal distance between pipes
PIPE_VELOCITY = 3  # Pixels per frame
PIPE_MIN_HEIGHT = 50
PIPE_MAX_HEIGHT = SCREEN_HEIGHT - PIPE_GAP - 50 - 112  # Account for ground

# Ground
GROUND_HEIGHT = 112
GROUND_Y = SCREEN_HEIGHT - GROUND_HEIGHT

# Colors (OG Flappy Bird palette)
SKY_COLOR = (78, 192, 202)
GROUND_COLOR = (222, 216, 149)
PIPE_COLOR = (84, 174, 52)
PIPE_BORDER_COLOR = (60, 128, 40)
BIRD_COLOR = (255, 204, 0)
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)

# =============================================================================
# RL Agent Hyperparameters (Dueling Double DQN)
# =============================================================================
# Network Architecture
STATE_DIM = 12  # FlappyBird-v0 observation space
ACTION_DIM = 2  # Flap or no-flap
HIDDEN_DIM = 512

# Training
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
REPLAY_MEMORY_SIZE = 100000
MINI_BATCH_SIZE = 32
NETWORK_SYNC_RATE = 10

# Exploration
EPSILON_START = 1.0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.05

# Training Control
STOP_ON_REWARD = 100000
SAVE_INTERVAL = 500  # Save model every N episodes

# DQN Variants
ENABLE_DOUBLE_DQN = True
ENABLE_DUELING_DQN = True

# =============================================================================
# AI Coach Settings
# =============================================================================
# Mistake Detection Thresholds
LATE_REACTION_THRESHOLD = 50      # Pixels from pipe (too close!)
OVERFLAP_WINDOW = 10              # Frames between flaps to consider overflapping
PANIC_FLAP_COUNT = 3              # Flaps in panic window to classify as panic
PANIC_WINDOW = 15                 # Frame window for panic detection
POOR_ANTICIPATION_DISTANCE = 150  # Pixels away from pipe to start positioning
POOR_CENTERING_THRESHOLD = PIPE_GAP / 3  # Max avg distance from center

# State History
STATE_HISTORY_LENGTH = 30  # Frames to analyze after death

# =============================================================================
# LLM Settings (Ollama / Llama)
# =============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
LLAMA_MODEL = "llama3:latest"
LLAVA_MODEL = "llava:latest"
LLM_TIMEOUT = 30  # seconds
LLM_MAX_TOKENS = 256

# Coaching Prompts
SYSTEM_PROMPT = """You are an expert Flappy Bird coach. Analyze the player's gameplay and provide:
1. Specific, actionable advice to improve
2. Encouragement based on their current score
3. Focus on one main area of improvement at a time

Keep responses concise (2-3 sentences max). Be supportive but direct."""

# =============================================================================
# TTS Settings (gTTS + pyttsx3)
# =============================================================================
TTS_LANGUAGE = "en"
TTS_SLOW = False
TTS_CACHE_DIR = os.path.join(BASE_DIR, "cache", "tts")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# Voice Settings
VOICE_RATE = 175  # Words per minute for pyttsx3
VOICE_VOLUME = 0.9

# =============================================================================
# UI Settings
VOICE_INPUT_ENABLED = False  # Disabled by default to prevent crashes
# =============================================================================
# Menu
MENU_BG_COLOR = (30, 30, 50)
MENU_ACCENT_COLOR = (0, 255, 128)
MENU_TEXT_COLOR = (255, 255, 255)
BUTTON_HOVER_COLOR = (60, 60, 90)

# HUD
HUD_FONT_SIZE = 32
HUD_SMALL_FONT_SIZE = 18
COACH_PANEL_WIDTH = 300
COACH_PANEL_HEIGHT = 150
COACH_PANEL_BG = (0, 0, 0, 180)

# Dashboard
DASHBOARD_BG = (20, 20, 35)

# =============================================================================
# Visualization Settings (Matplotlib)
# =============================================================================
PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_DPI = 100
REWARD_WINDOW = 100  # Window for moving average

# =============================================================================
# Game Modes
# =============================================================================
class GameMode:
    HUMAN_NO_COACH = "human_no_coach"
    HUMAN_WITH_COACH = "human_with_coach"
    HUMAN_WITH_NLP_COACH = "human_nlp_coach"
    WATCH_AI = "watch_ai"
    TRAINING = "training"
    ARCADE = "arcade"
    DASHBOARD = "dashboard"

# =============================================================================
# Dashboard Settings
# =============================================================================
DASHBOARD_WINDOW_WIDTH = 1280
DASHBOARD_WINDOW_HEIGHT = 720
DASHBOARD_LEFT_PANEL_WIDTH = 380
DASHBOARD_CENTER_PANEL_WIDTH = 500
DASHBOARD_RIGHT_PANEL_WIDTH = 380
DASHBOARD_PANEL_GAP = 12

# Speech Recognition Settings
SPEECH_TIMEOUT = 5  # seconds to wait for speech
SPEECH_PHRASE_TIME_LIMIT = 10  # max phrase length

# =============================================================================
# Reward Function
# =============================================================================
REWARD_ALIVE = 0.1         # Per frame survived
REWARD_PASS_PIPE = 1.0     # Successfully passing a pipe
REWARD_DEATH = -1.0        # Game over penalty
