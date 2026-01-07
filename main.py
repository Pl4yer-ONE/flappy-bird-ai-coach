# Flappy Bird AI Coach - Main Entry Point

"""
Main entry point for Flappy Bird AI Coach.
Launches the professional multi-panel dashboard with menu loop.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.menu import run_menu

def main():
    """Main entry point with menu loop support."""
    print("🚀 Launching Flappy Bird AI Coach...")
    
    while True:
        # 1. Show Launcher Menu
        config = run_menu()
        
        # 2. Exit if menu was closed
        if config is None:
            print("Exiting Flappy Bird AI Coach. Goodbye!")
            break
        
        # 3. Launch selected mode
        try:
            return_to_menu = False
            
            if config.get("mode") == "watch_ai":
                from game.modes import watch_ai
                print("Starting RL Mode...")
                return_to_menu = watch_ai()
                
            elif config.get("mode") == "train_ai":
                from game.modes import train_agent_mode
                print("Starting Learning Mode...")
                return_to_menu = train_agent_mode()
                
            elif config.get("mode") == "scoreboard":
                from ui.scoreboard import run_scoreboard
                print("Showing Scoreboard...")
                return_to_menu = run_scoreboard()
                
            elif config.get("mode") == "analysis":
                from ui.analysis import run_analysis
                print("Showing Analysis Mode...")
                return_to_menu = run_analysis()
                
            else:
                # Dashboard Mode (AI Coach Combined)
                from ui.dashboard import run_dashboard
                print(f"Starting AI Coach with config: {config}")
                return_to_menu = run_dashboard(config)
            
            # 4. Continue loop if returning to menu, otherwise exit
            if not return_to_menu:
                print("Exiting Flappy Bird AI Coach. Goodbye!")
                break
                
        except Exception as e:
            print(f"Error launching mode: {e}")
            import traceback
            traceback.print_exc()
            # Continue to menu on error
            continue

if __name__ == '__main__':
    main()
