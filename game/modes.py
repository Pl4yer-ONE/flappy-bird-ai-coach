"""
RL Game Modes - Watch AI and Training
Uses flappy-bird-gymnasium for authentic smooth gameplay.
"""

import os
import sys
import pygame
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    USE_GYMNASIUM, RL_FPS_NORMAL, RL_FPS_FAST, RL_FPS_SLOW,
    GYMNASIUM_MODEL_PATH, BUILTIN_MODEL_PATH, MODELS_DIR
)

# Colors - Modern dark theme
BG_DARK = (12, 12, 20)
ACCENT_PRIMARY = (0, 255, 200)
ACCENT_WARNING = (255, 180, 50)
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (80, 85, 100)
PANEL_BG = (25, 27, 40)


def watch_ai(speed: str = "normal"):
    """
    Watch trained AI agent play using flappy-bird-gymnasium.
    
    This uses the actual gymnasium environment for smooth, authentic gameplay.
    The pretrained model can score 100+ pipes consistently.
    
    Args:
        speed: "slow", "normal", or "fast"
    """
    pygame.init()
    
    # Window setup
    window_width, window_height = 650, 750
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("🎯 Flappy Bird AI Coach - RL Mode")
    clock = pygame.time.Clock()
    
    # Fonts
    font_title = pygame.font.SysFont('Arial', 26, bold=True)
    font_status = pygame.font.SysFont('Arial', 18)
    font_small = pygame.font.SysFont('Arial', 14)
    font_stats = pygame.font.SysFont('Arial', 16)
    
    # FPS settings based on speed
    fps_map = {"slow": RL_FPS_SLOW, "normal": RL_FPS_NORMAL, "fast": RL_FPS_FAST}
    current_fps = fps_map.get(speed, RL_FPS_NORMAL)
    speed_names = ["slow", "normal", "fast"]
    speed_idx = speed_names.index(speed) if speed in speed_names else 1
    
    # Try to use gymnasium
    env = None
    use_gym = USE_GYMNASIUM
    
    try:
        import gymnasium as gym
        import flappy_bird_gymnasium
        env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
        print("✓ Using flappy-bird-gymnasium for smooth gameplay")
        model_path = GYMNASIUM_MODEL_PATH
        state_dim = 12  # Gymnasium uses 12-dim observations
    except ImportError:
        print("⚠ flappy-bird-gymnasium not available, using built-in engine")
        use_gym = False
        from game.game_engine import FlappyBirdGame
        env = FlappyBirdGame(render_mode='rgb_array', enable_logging=False)
        model_path = BUILTIN_MODEL_PATH
        state_dim = 6
    
    # Load agent
    from rl_agent.agent import DQNAgent
    
    if use_gym:
        agent = DQNAgent('flappybird_gym')
    else:
        agent = DQNAgent('flappybird')
    
    # Try to load pretrained model
    model_loaded = False
    if os.path.exists(model_path):
        try:
            agent.load_model(model_path, state_dim=state_dim, action_dim=2)
            model_loaded = True
            model_status = f"✓ Trained model loaded"
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model_status = "⚠ Model load failed"
    else:
        # Try alternative path
        alt_path = os.path.join(MODELS_DIR, "flappybird.pt")
        if os.path.exists(alt_path):
            try:
                agent.load_model(alt_path, state_dim=state_dim, action_dim=2)
                model_loaded = True
                model_status = "✓ Alternative model loaded"
            except:
                model_status = "⚠ No trained model"
        else:
            model_status = "⚠ No trained model - random actions"
    
    if not model_loaded:
        agent.initialize_networks(state_dim=state_dim, action_dim=2)
    
    # UI elements
    back_rect = pygame.Rect(15, 15, 100, 32)
    speed_rect = pygame.Rect(window_width - 115, 15, 100, 32)
    
    # Game stats
    running = True
    return_to_menu = False
    episode = 0
    current_score = 0
    best_score = 0
    scores_history = []
    avg_score = 0.0
    
    # For smooth rendering
    game_surface = pygame.Surface((288, 512))
    
    while running:
        episode += 1
        
        # Reset environment
        if use_gym:
            obs, info = env.reset()
            state = obs
        else:
            obs = env.reset()
            state = _convert_obs_to_array(obs, state_dim)
        
        done = False
        current_score = 0
        frame_count = 0
        
        while not done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        return_to_menu = True
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        # Cycle speed
                        speed_idx = (speed_idx + 1) % 3
                        current_fps = fps_map[speed_names[speed_idx]]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if back_rect.collidepoint(event.pos):
                        running = False
                        return_to_menu = True
                    elif speed_rect.collidepoint(event.pos):
                        speed_idx = (speed_idx + 1) % 3
                        current_fps = fps_map[speed_names[speed_idx]]
            
            if not running:
                break
            
            # Get action from agent (no exploration in watch mode)
            action = agent.select_action(state, training=False)
            
            # Step environment
            if use_gym:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = obs
                current_score = info.get('score', 0)
            else:
                obs, reward, done, _, info = env.step(action)
                state = _convert_obs_to_array(obs, state_dim)
                current_score = info.get('score', 0)
            
            frame_count += 1
            
            if current_score > best_score:
                best_score = current_score
            
            # === Render ===
            screen.fill(BG_DARK)
            
            # Header
            title = font_title.render("🎯 RL Mode - AI Agent Playing", True, ACCENT_PRIMARY)
            screen.blit(title, (window_width // 2 - title.get_width() // 2, 55))
            
            # Model status
            status_color = (50, 205, 50) if "✓" in model_status else ACCENT_WARNING
            status = font_status.render(model_status, True, status_color)
            screen.blit(status, (window_width // 2 - status.get_width() // 2, 90))
            
            # Render game frame
            if use_gym:
                # Get RGB array from gymnasium
                frame = env.render()
                if frame is not None:
                    # Convert numpy array to pygame surface
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    # Scale to fit our display area
                    scaled = pygame.transform.scale(frame_surface, (288, 512))
                    game_x = (window_width - 288) // 2
                    game_y = 120
                    screen.blit(scaled, (game_x, game_y))
                    pygame.draw.rect(screen, ACCENT_PRIMARY, 
                                   (game_x - 2, game_y - 2, 292, 516), 2, border_radius=4)
            else:
                # Built-in engine rendering
                game_surface.fill((78, 192, 202))
                if hasattr(env, 'pipes'):
                    env.pipes.render(game_surface)
                pygame.draw.rect(game_surface, (222, 216, 149), (0, 400, 288, 112))
                if hasattr(env, 'bird'):
                    env.bird.render(game_surface)
                
                game_x = (window_width - 288) // 2
                game_y = 120
                screen.blit(game_surface, (game_x, game_y))
                pygame.draw.rect(screen, ACCENT_PRIMARY, 
                               (game_x - 2, game_y - 2, 292, 516), 2, border_radius=4)
            
            # Stats panel
            stats_y = 650
            stats_panel = pygame.Rect(game_x - 60, stats_y, 410, 80)
            pygame.draw.rect(screen, PANEL_BG, stats_panel, border_radius=8)
            pygame.draw.rect(screen, (40, 42, 55), stats_panel, 1, border_radius=8)
            
            # Score displays
            score_text = font_stats.render(f"Score: {current_score}", True, TEXT_WHITE)
            best_text = font_stats.render(f"Best: {best_score}", True, ACCENT_PRIMARY)
            ep_text = font_small.render(f"Episode: {episode}", True, TEXT_MUTED)
            avg_text = font_small.render(f"Avg: {avg_score:.1f}", True, TEXT_MUTED)
            
            screen.blit(score_text, (stats_panel.x + 20, stats_y + 15))
            screen.blit(best_text, (stats_panel.x + 140, stats_y + 15))
            screen.blit(ep_text, (stats_panel.x + 260, stats_y + 15))
            screen.blit(avg_text, (stats_panel.x + 20, stats_y + 45))
            
            # Speed indicator
            speed_label = f"Speed: {speed_names[speed_idx].upper()}"
            speed_text = font_small.render(speed_label, True, ACCENT_WARNING)
            screen.blit(speed_text, (stats_panel.x + 260, stats_y + 45))
            
            # Back button
            mouse_pos = pygame.mouse.get_pos()
            hover_back = back_rect.collidepoint(mouse_pos)
            bg_color = (60, 65, 85) if hover_back else (40, 42, 60)
            pygame.draw.rect(screen, bg_color, back_rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_PRIMARY if hover_back else (50, 55, 75), 
                           back_rect, 1, border_radius=6)
            back_text = font_small.render('← Menu', True, ACCENT_PRIMARY if hover_back else TEXT_MUTED)
            screen.blit(back_text, (back_rect.x + 20, back_rect.y + 10))
            
            # Speed toggle button
            hover_speed = speed_rect.collidepoint(mouse_pos)
            bg_color = (60, 65, 85) if hover_speed else (40, 42, 60)
            pygame.draw.rect(screen, bg_color, speed_rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_WARNING if hover_speed else (50, 55, 75), 
                           speed_rect, 1, border_radius=6)
            speed_btn_text = font_small.render('Speed ⟳', True, ACCENT_WARNING if hover_speed else TEXT_MUTED)
            screen.blit(speed_btn_text, (speed_rect.x + 15, speed_rect.y + 10))
            
            pygame.display.flip()
            
            # Frame rate control
            if current_fps > 0:
                clock.tick(current_fps)
        
        # Episode ended
        if current_score > 0:
            scores_history.append(current_score)
            if len(scores_history) > 20:
                scores_history = scores_history[-20:]
            avg_score = sum(scores_history) / len(scores_history)
        
        if running:
            print(f"Episode {episode} - Score: {current_score} | Best: {best_score} | Avg: {avg_score:.1f}")
    
    # Cleanup
    env.close()
    return return_to_menu


def train_agent_mode(episodes: int = 1000, render: bool = True):
    """
    Train DQN agent with visual feedback.
    
    Uses flappy-bird-gymnasium for proper RL training.
    """
    pygame.init()
    
    # Window setup
    window_width, window_height = 650, 750
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("📚 Flappy Bird AI Coach - Learning Mode")
    clock = pygame.time.Clock()
    
    # Fonts
    font_title = pygame.font.SysFont('Arial', 26, bold=True)
    font_status = pygame.font.SysFont('Arial', 18)
    font_small = pygame.font.SysFont('Arial', 14)
    font_stats = pygame.font.SysFont('Arial', 16)
    
    print(f"Starting training for {episodes} episodes...")
    
    # Try to use gymnasium
    use_gym = USE_GYMNASIUM
    try:
        import gymnasium as gym
        import flappy_bird_gymnasium
        env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
        state_dim = 12
        print("✓ Using flappy-bird-gymnasium for training")
    except ImportError:
        print("⚠ flappy-bird-gymnasium not available, using built-in engine")
        use_gym = False
        from game.game_engine import FlappyBirdGame
        env = FlappyBirdGame(render_mode='rgb_array', enable_logging=False)
        state_dim = 6
    
    # Create agent
    from rl_agent.agent import DQNAgent
    
    if use_gym:
        agent = DQNAgent('flappybird_gym')
    else:
        agent = DQNAgent('flappybird')
    
    agent.initialize_networks(state_dim=state_dim, action_dim=2)
    
    # UI elements
    back_rect = pygame.Rect(15, 15, 100, 32)
    
    # Training stats
    running = True
    return_to_menu = False
    episode = 0
    best_score = 0
    scores_history = []
    avg_score = 0.0
    
    # For rendering
    game_surface = pygame.Surface((288, 512))
    
    while running and episode < episodes:
        episode += 1
        
        # Reset
        if use_gym:
            obs, info = env.reset()
            state = obs
        else:
            obs = env.reset()
            state = _convert_obs_to_array(obs, state_dim)
        
        done = False
        episode_reward = 0
        current_score = 0
        
        while not done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        return_to_menu = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if back_rect.collidepoint(event.pos):
                        running = False
                        return_to_menu = True
            
            if not running:
                break
            
            # Get action with exploration
            action = agent.select_action(state, training=True)
            
            # Step
            if use_gym:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = obs
                current_score = info.get('score', 0)
            else:
                obs, reward, done, _, info = env.step(action)
                next_state = _convert_obs_to_array(obs, state_dim)
                current_score = info.get('score', 0)
            
            # Store and train
            agent.store_transition(state, action, next_state, reward, done)
            agent.optimize()
            
            state = next_state
            episode_reward += reward
            
            if current_score > best_score:
                best_score = current_score
            
            # === Render ===
            screen.fill(BG_DARK)
            
            # Header
            title = font_title.render("📚 Learning Mode - Training AI", True, ACCENT_WARNING)
            screen.blit(title, (window_width // 2 - title.get_width() // 2, 55))
            
            # Progress
            progress = f"Episode {episode}/{episodes}"
            progress_text = font_status.render(progress, True, TEXT_WHITE)
            screen.blit(progress_text, (window_width // 2 - progress_text.get_width() // 2, 90))
            
            # Render game
            if use_gym:
                frame = env.render()
                if frame is not None:
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    scaled = pygame.transform.scale(frame_surface, (288, 512))
                    game_x = (window_width - 288) // 2
                    game_y = 120
                    screen.blit(scaled, (game_x, game_y))
                    pygame.draw.rect(screen, ACCENT_WARNING, 
                                   (game_x - 2, game_y - 2, 292, 516), 2, border_radius=4)
            else:
                game_surface.fill((78, 192, 202))
                if hasattr(env, 'pipes'):
                    env.pipes.render(game_surface)
                pygame.draw.rect(game_surface, (222, 216, 149), (0, 400, 288, 112))
                if hasattr(env, 'bird'):
                    env.bird.render(game_surface)
                
                game_x = (window_width - 288) // 2
                game_y = 120
                screen.blit(game_surface, (game_x, game_y))
                pygame.draw.rect(screen, ACCENT_WARNING, 
                               (game_x - 2, game_y - 2, 292, 516), 2, border_radius=4)
            
            # Stats panel
            stats_y = 650
            stats_panel = pygame.Rect(game_x - 60, stats_y, 410, 80)
            pygame.draw.rect(screen, PANEL_BG, stats_panel, border_radius=8)
            
            # Stats
            score_text = font_stats.render(f"Score: {current_score}", True, TEXT_WHITE)
            best_text = font_stats.render(f"Best: {best_score}", True, ACCENT_PRIMARY)
            eps_text = font_small.render(f"ε: {agent.epsilon:.3f}", True, TEXT_MUTED)
            avg_text = font_small.render(f"Avg: {avg_score:.1f}", True, TEXT_MUTED)
            
            screen.blit(score_text, (stats_panel.x + 20, stats_y + 15))
            screen.blit(best_text, (stats_panel.x + 140, stats_y + 15))
            screen.blit(eps_text, (stats_panel.x + 260, stats_y + 15))
            screen.blit(avg_text, (stats_panel.x + 20, stats_y + 45))
            
            # Memory size
            mem_text = font_small.render(f"Memory: {len(agent.memory) if agent.memory else 0}", True, TEXT_MUTED)
            screen.blit(mem_text, (stats_panel.x + 140, stats_y + 45))
            
            # Back button
            mouse_pos = pygame.mouse.get_pos()
            hover = back_rect.collidepoint(mouse_pos)
            bg_color = (60, 65, 85) if hover else (40, 42, 60)
            pygame.draw.rect(screen, bg_color, back_rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_WARNING if hover else (50, 55, 75), 
                           back_rect, 1, border_radius=6)
            back_text = font_small.render('← Menu', True, ACCENT_WARNING if hover else TEXT_MUTED)
            screen.blit(back_text, (back_rect.x + 20, back_rect.y + 10))
            
            pygame.display.flip()
            clock.tick(60)  # Training runs at full speed but renders at 60 FPS
        
        # Episode ended
        if current_score > 0:
            scores_history.append(current_score)
            if len(scores_history) > 50:
                scores_history = scores_history[-50:]
            avg_score = sum(scores_history) / len(scores_history)
        
        if running and episode % 10 == 0:
            print(f"Episode {episode}/{episodes} - Score: {current_score}, ε: {agent.epsilon:.4f}, Avg: {avg_score:.1f}")
        
        # Save periodically
        if running and episode % 100 == 0:
            save_path = os.path.join(MODELS_DIR, 'flappybird_training.pt')
            agent.save_model(save_path)
            print(f"Model checkpoint saved to {save_path}")
    
    # Final save
    if running:
        final_path = os.path.join(MODELS_DIR, 'flappybird.pt')
        agent.save_model(final_path)
        print(f"Training complete! Model saved to {final_path}")
    
    env.close()
    return return_to_menu


def _convert_obs_to_array(obs, target_dim: int):
    """Convert observation dict to numpy array for agent."""
    if isinstance(obs, dict):
        from config import SCREEN_WIDTH, SCREEN_HEIGHT
        state = np.array([
            obs.get('bird_y', 0) / SCREEN_HEIGHT,
            obs.get('bird_velocity', 0) / 10.0,
            obs.get('distance_to_next_pipe', 0) / SCREEN_WIDTH,
            obs.get('next_pipe_gap_y', 0) / SCREEN_HEIGHT,
            obs.get('next_pipe_upper_height', 0) / SCREEN_HEIGHT,
            obs.get('next_pipe_lower_top', 0) / SCREEN_HEIGHT
        ], dtype=np.float32)
    else:
        state = np.array(obs, dtype=np.float32)
    
    # Pad if needed
    if len(state) < target_dim:
        state = np.pad(state, (0, target_dim - len(state)))
    
    return state
