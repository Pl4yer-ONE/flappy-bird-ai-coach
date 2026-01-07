import os
import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT

# Colors
BG_DARK = (12, 12, 20)
ACCENT_PRIMARY = (0, 255, 200)
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (80, 85, 100)


def watch_ai():
    """Watch trained AI agent play using built-in game engine."""
    pygame.init()
    
    # Create window matching menu style
    width, height = 600, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("🎯 Flappy Bird AI Coach - RL Mode")
    clock = pygame.time.Clock()
    
    # Fonts
    font_title = pygame.font.SysFont('Arial', 28, bold=True)
    font_status = pygame.font.SysFont('Arial', 18)
    font_small = pygame.font.SysFont('Arial', 14)
    
    print("Loading AI agent...")
    
    # Use our built-in game engine (not gymnasium separate window)
    from game.game_engine import FlappyBirdGame
    game = FlappyBirdGame(render_mode='rgb_array', enable_logging=False)
    game_surface = pygame.Surface((288, 512))
    
    # Load agent
    from rl_agent.agent import DQNAgent
    agent = DQNAgent('flappybird')
    
    # Try to load pre-trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'flappybird.pt')
    state_dim = 6
    action_dim = 2
    
    if os.path.exists(model_path):
        agent.load_model(model_path, state_dim=state_dim, action_dim=action_dim)
        print(f"Loaded model from {model_path}")
        model_status = "✓ Model loaded"
    else:
        print("No pre-trained model found. Agent will use random actions.")
        agent.initialize_networks(state_dim=state_dim, action_dim=action_dim)
        model_status = "⚠ Random actions (no model)"
    
    # Play loop
    running = True
    return_to_menu = False
    episode = 0
    total_episodes = 0
    current_score = 0
    best_score = 0
    
    # Back button
    back_rect = pygame.Rect(15, 15, 100, 32)
    
    while running:
        episode += 1
        total_episodes += 1
        
        # Reset game
        obs = game.reset()
        state = _obs_to_state(obs)
        done = False
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
            
            # Get action from agent
            action = agent.select_action(state, training=False)
            
            # Step game
            obs, reward, done, _, info = game.step(action)
            state = _obs_to_state(obs)
            current_score = info['score']
            
            if current_score > best_score:
                best_score = current_score
            
            # Render
            screen.fill(BG_DARK)
            
            # Title
            title = font_title.render("🎯 RL Mode - AI Agent Playing", True, ACCENT_PRIMARY)
            screen.blit(title, (width // 2 - title.get_width() // 2, 60))
            
            # Status
            status = font_status.render(model_status, True, TEXT_WHITE)
            screen.blit(status, (width // 2 - status.get_width() // 2, 100))
            
            # Render game to surface
            game_surface.fill((78, 192, 202))
            if hasattr(game, 'pipes'):
                game.pipes.render(game_surface)
            pygame.draw.rect(game_surface, (222, 216, 149), (0, 400, 288, 112))
            if hasattr(game, 'bird'):
                game.bird.render(game_surface)
            
            # Scale and center game
            game_x = (width - 288) // 2
            game_y = 130
            screen.blit(game_surface, (game_x, game_y))
            pygame.draw.rect(screen, ACCENT_PRIMARY, (game_x - 2, game_y - 2, 292, 516), 2, border_radius=4)
            
            # Stats
            stats_y = 660
            score_text = font_status.render(f"Score: {current_score}", True, TEXT_WHITE)
            best_text = font_status.render(f"Best: {best_score}", True, ACCENT_PRIMARY)
            ep_text = font_small.render(f"Episode: {total_episodes}", True, TEXT_MUTED)
            
            screen.blit(score_text, (game_x, stats_y))
            screen.blit(best_text, (game_x + 150, stats_y))
            screen.blit(ep_text, (game_x + 220, stats_y))
            
            # Back button
            mouse_pos = pygame.mouse.get_pos()
            hover = back_rect.collidepoint(mouse_pos)
            bg_color = (60, 65, 85) if hover else (40, 42, 60)
            pygame.draw.rect(screen, bg_color, back_rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_PRIMARY if hover else (50, 55, 75), back_rect, 1, border_radius=6)
            back_text = font_small.render('← Menu', True, ACCENT_PRIMARY if hover else TEXT_MUTED)
            screen.blit(back_text, (back_rect.x + 20, back_rect.y + 10))
            
            pygame.display.flip()
            clock.tick(60)
        
        if running:
            print(f"Episode {total_episodes} - Score: {current_score}")
    
    # Cleanup
    game.close()
    # Don't call pygame.quit() - let main.py handle it
    
    return return_to_menu


def _obs_to_state(obs):
    """Convert observation dict to state list for agent."""
    if isinstance(obs, dict):
        return [
            obs.get('bird_y', 0) / SCREEN_HEIGHT,
            obs.get('bird_velocity', 0) / 10.0,
            obs.get('distance_to_next_pipe', 0) / SCREEN_WIDTH,
            obs.get('next_pipe_gap_y', 0) / SCREEN_HEIGHT,
            obs.get('next_pipe_upper_height', 0) / SCREEN_HEIGHT,
            obs.get('next_pipe_lower_top', 0) / SCREEN_HEIGHT
        ]
    return list(obs)


def train_agent_mode(episodes: int = 100, render: bool = True):
    """Train DQN agent with visual feedback."""
    pygame.init()
    
    # Create window matching menu style
    width, height = 600, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("📚 Flappy Bird AI Coach - Learning Mode")
    clock = pygame.time.Clock()
    
    # Fonts
    font_title = pygame.font.SysFont('Arial', 28, bold=True)
    font_status = pygame.font.SysFont('Arial', 18)
    font_small = pygame.font.SysFont('Arial', 14)
    
    print(f"Starting training for {episodes} episodes...")
    
    # Use built-in game engine
    from game.game_engine import FlappyBirdGame
    game = FlappyBirdGame(render_mode='rgb_array', enable_logging=False)
    game_surface = pygame.Surface((288, 512))
    
    # Create agent
    from rl_agent.agent import DQNAgent
    agent = DQNAgent('flappybird')
    agent.initialize_networks(state_dim=6, action_dim=2)
    
    # Training variables
    running = True
    return_to_menu = False
    episode = 0
    best_score = 0
    total_reward = 0
    epsilon = 1.0
    
    # Back button
    back_rect = pygame.Rect(15, 15, 100, 32)
    
    while running and episode < episodes:
        episode += 1
        
        # Reset
        obs = game.reset()
        state = _obs_to_state(obs)
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
            
            # Get action (with exploration)
            action = agent.select_action(state, training=True)
            
            # Step
            obs, reward, done, _, info = game.step(action)
            next_state = _obs_to_state(obs)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            current_score = info['score']
            
            if current_score > best_score:
                best_score = current_score
            
            # Render
            screen.fill(BG_DARK)
            
            # Title
            title = font_title.render("📚 Learning Mode - Training AI", True, ACCENT_PRIMARY)
            screen.blit(title, (width // 2 - title.get_width() // 2, 60))
            
            # Status
            progress = f"Episode {episode}/{episodes}"
            status = font_status.render(progress, True, TEXT_WHITE)
            screen.blit(status, (width // 2 - status.get_width() // 2, 100))
            
            # Render game
            game_surface.fill((78, 192, 202))
            if hasattr(game, 'pipes'):
                game.pipes.render(game_surface)
            pygame.draw.rect(game_surface, (222, 216, 149), (0, 400, 288, 112))
            if hasattr(game, 'bird'):
                game.bird.render(game_surface)
            
            # Scale and center game
            game_x = (width - 288) // 2
            game_y = 130
            screen.blit(game_surface, (game_x, game_y))
            pygame.draw.rect(screen, (255, 180, 50), (game_x - 2, game_y - 2, 292, 516), 2, border_radius=4)
            
            # Stats
            stats_y = 660
            score_text = font_status.render(f"Score: {current_score}", True, TEXT_WHITE)
            best_text = font_status.render(f"Best: {best_score}", True, ACCENT_PRIMARY)
            eps_text = font_small.render(f"ε: {agent.epsilon:.2f}", True, TEXT_MUTED)
            
            screen.blit(score_text, (game_x, stats_y))
            screen.blit(best_text, (game_x + 120, stats_y))
            screen.blit(eps_text, (game_x + 200, stats_y))
            
            # Back button
            mouse_pos = pygame.mouse.get_pos()
            hover = back_rect.collidepoint(mouse_pos)
            bg_color = (60, 65, 85) if hover else (40, 42, 60)
            pygame.draw.rect(screen, bg_color, back_rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_PRIMARY if hover else (50, 55, 75), back_rect, 1, border_radius=6)
            back_text = font_small.render('← Menu', True, ACCENT_PRIMARY if hover else TEXT_MUTED)
            screen.blit(back_text, (back_rect.x + 20, back_rect.y + 10))
            
            pygame.display.flip()
            clock.tick(60)
        
        if running and episode % 10 == 0:
            print(f"Episode {episode}/{episodes} - Score: {current_score}, ε: {agent.epsilon:.3f}")
    
    # Save model
    if running:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'flappybird.pt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    # Cleanup
    game.close()
    # Don't call pygame.quit() - let main.py handle it
    
    return return_to_menu
