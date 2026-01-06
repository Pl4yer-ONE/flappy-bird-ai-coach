import sys
import os
import unittest
import pygame

# Set dummy video driver for headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.dashboard import Dashboard

class TestDashboard(unittest.TestCase):
    def test_dashboard_init(self):
        """Test that dashboard initializes without error."""
        try:
            d = Dashboard()
            print("Dashboard initialized successfully")
            d.running = False
            d.run() # Should exit immediately
        except Exception as e:
            self.fail(f"Dashboard initialization failed: {e}")

if __name__ == "__main__":
    unittest.main()
