import os
import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT
from game.game_engine import FlappyBirdGame, GymnasiumWrapper

def watch_ai():
    """Watch trained AI agent play."""
    pygame.init()
    
    print("Loading AI agent...")
    
    # Try gymnasium first
    try:
        import gymnasium as gym
        import flappy_bird_gymnasium
        env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        use_gym = True
    except ImportError:
        env = FlappyBirdGame(render_mode="human")
        use_gym = False
        
    # Load agent
    from rl_agent.agent import DQNAgent
    agent = DQNAgent('flappybird')
    
    # Try to load pre-trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'flappybird.pt')
    if os.path.exists(model_path):
        state_dim = 12 if use_gym else 6
        agent.load_model(model_path, state_dim=state_dim, action_dim=2)
        print(f"Loaded model from {model_path}")
    else:
        print("No pre-trained model found. Agent will use random actions.")
        agent.initialize_networks(state_dim=12, action_dim=2)
        
    # Fonts
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24, bold=True)
    
    # Play loop
    running = True
    episode = 0
    max_episodes = 100
    
    while running and episode < max_episodes:
        episode += 1
        print(f"\n--- Episode {episode} ---")
        
        if use_gym:
            state, _ = env.reset()
        else:
            state = env.reset()
            # Convert dict state to list for agent
            state = [
                state['bird_y'] / SCREEN_HEIGHT,
                state['bird_velocity'] / 10.0,
                state['distance_to_next_pipe'] / SCREEN_WIDTH,
                state['next_pipe_gap_y'] / SCREEN_HEIGHT,
                state['next_pipe_upper_height'] / SCREEN_HEIGHT,
                state['next_pipe_lower_top'] / SCREEN_HEIGHT
            ]
            
        done = False
        total_reward = 0
        score = 0
        
        while not done and running:
            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    
            if not running:
                break
                
            # Get action from agent
            action = agent.select_action(state, training=False)
            
            # Step
            if use_gym:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # gymnasium flappy bird doesn't return score in info directly usually, need to check env wrapper
                score = info.get('score', 0)
            else:
                obs, reward, done, _, info = env.step(action)
                next_state = [
                    obs['bird_y'] / SCREEN_HEIGHT,
                    obs['bird_velocity'] / 10.0,
                    obs['distance_to_next_pipe'] / SCREEN_WIDTH,
                    obs['next_pipe_gap_y'] / SCREEN_HEIGHT,
                    obs['next_pipe_upper_height'] / SCREEN_HEIGHT,
                    obs['next_pipe_lower_top'] / SCREEN_HEIGHT
                ]
                
                # Render Game
                env.render()
                
                # Render Stats
                status_text = font.render(f"AI: DQN | Score: {info['score']}", True, (255, 255, 255))
                env.screen.blit(status_text, (10, 10))
                
                pygame.display.flip()
                score = info['score']
                
            state = next_state
            total_reward += reward
            
            if use_gym:
                 # Minimal render delay for visibility if using gym default render
                 env.render()
            
        print(f"Episode {episode} - Score: {score}, Reward: {total_reward:.1f}")
        
    env.close()
    pygame.quit()


def train_agent_mode(episodes: int = 1000, render: bool = True):
    """Train DQN agent."""
    print(f"Starting training for {episodes} episodes...")
    
    # Create environment
    try:
        import gymnasium as gym
        import flappy_bird_gymnasium
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
    except ImportError:
        print("Using built-in game engine")
        env = GymnasiumWrapper(use_gymnasium=False, render_mode="human" if render else None)
        
    # Create and train agent
    from rl_agent.agent import DQNAgent
    agent = DQNAgent('flappybird')
    
    def training_callback(episode, reward, info):
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={reward:.1f}, ε={info['epsilon']:.3f}, Best={info['best_reward']:.1f}")
            
    try:
        agent.train(env, num_episodes=episodes, render=render, callback=training_callback)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    env.close()
    print("Training complete!")
