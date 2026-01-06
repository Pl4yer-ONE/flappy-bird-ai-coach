import pygame
import json
from pathlib import Path
import sys

class Scoreboard:
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 500
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird AI Coach - Scoreboard")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 32, bold=True)
        self.font_header = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_row = pygame.font.SysFont('Arial', 20)
        
        # Data
        self.scores = self._load_scores()
        
        # Colors
        self.BG = (12, 12, 20)
        self.ACCENT = (0, 255, 200)
        self.TEXT = (255, 255, 255)
        self.ROW_BG = (30, 32, 48)

    def _load_scores(self):
        try:
            path = Path(__file__).parent.parent / 'data' / 'history.json'
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Sort desc
                    data.sort(key=lambda x: x['score'], reverse=True)
                    return data[:10]  # Top 10
        except Exception as e:
            print(f"Error loading scores: {e}")
        return []

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            self.screen.fill(self.BG)
            
            # Title
            title = self.font_title.render("🏆 Top 10 Scores", True, self.ACCENT)
            self.screen.blit(title, (self.width//2 - title.get_width()//2, 20))
            
            # Header
            pygame.draw.line(self.screen, self.ACCENT, (50, 70), (550, 70), 2)
            
            # List
            start_y = 90
            for i, entry in enumerate(self.scores):
                y = start_y + i * 35
                
                # Rank
                rank = self.font_row.render(f"#{i+1}", True, self.ACCENT)
                self.screen.blit(rank, (60, y))
                
                # Score
                sc = self.font_row.render(str(entry['score']), True, self.TEXT)
                self.screen.blit(sc, (150, y))
                
                # Mode
                mode = self.font_row.render(entry.get('mode', 'Standard'), True, (150, 150, 170))
                self.screen.blit(mode, (300, y))
            
            if not self.scores:
                msg = self.font_row.render("No games played yet!", True, (100, 100, 100))
                self.screen.blit(msg, (self.width//2 - msg.get_width()//2, 200))
                
            # Footer
            foot = self.font_row.render("Press ESC to return", True, (80, 80, 80))
            self.screen.blit(foot, (self.width//2 - foot.get_width()//2, self.height - 40))

            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()

def run_scoreboard():
    s = Scoreboard()
    s.run()

if __name__ == '__main__':
    run_scoreboard()
