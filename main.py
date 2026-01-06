# Flappy Bird AI Coach - Main Entry Point

"""
Main entry point for Flappy Bird AI Coach.
Launches the professional multi-panel dashboard.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.dashboard import run_dashboard
from ui.menu import run_menu

def main():
    """Main entry point."""
    print("🚀 Launching Flappy Bird AI Coach Launcher...")
    
    # 1. Show Launcher Menu
    config = run_menu()
    
    # 2. Launch Dashboard if config selected
    if config is not None:
        try:
            # Check for special modes
            if config.get("mode") == "watch_ai":
                from game.modes import watch_ai
                print("Starting Watch AI Mode...")
                watch_ai()
            elif config.get("mode") == "train_ai":
                from game.modes import train_agent_mode
                print("Starting Training Mode...")
                train_agent_mode()
            else:
                # Dashboard Mode
                print(f"Starting Dashboard with config: {config}")
                run_dashboard(config)
        except Exception as e:
            print(f"Error launching mode: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Launcher exited.")

if __name__ == '__main__':
    main()

