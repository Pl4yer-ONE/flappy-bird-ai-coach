import pygame
import json
from pathlib import Path
from datetime import datetime
import sys

# Colors - Modern dark theme (consistent with menu)
BG_DARK = (12, 12, 20)
ACCENT_PRIMARY = (0, 255, 200)
ACCENT_GOLD = (255, 215, 0)
ACCENT_SILVER = (192, 192, 192)
ACCENT_BRONZE = (205, 127, 50)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (120, 130, 150)
TEXT_MUTED = (80, 85, 100)
ROW_BG = (30, 32, 48)
ROW_BG_ALT = (25, 27, 40)
ROW_HOVER = (40, 45, 60)
CARD_BG = (35, 38, 55)


class Scoreboard:
    def __init__(self):
        pygame.init()
        self.width = 700
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🏆 Flappy Bird AI Coach - Scoreboard")
        self.clock = pygame.time.Clock()
        self.return_to_menu = False
        
        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 36, bold=True)
        self.font_header = pygame.font.SysFont('Arial', 18, bold=True)
        self.font_row = pygame.font.SysFont('Arial', 18)
        self.font_score = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 12)
        
        # Data
        self.scores = self._load_scores()
        self.hovered_row = -1

    def _load_scores(self):
        try:
            path = Path(__file__).parent.parent / 'data' / 'history.json'
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Sort by score descending
                    data.sort(key=lambda x: x.get('score', 0), reverse=True)
                    return data[:10]  # Top 10
        except Exception as e:
            print(f"Error loading scores: {e}")
        return []
    
    def _get_rank_color(self, rank):
        """Get color based on rank."""
        if rank == 1:
            return ACCENT_GOLD
        elif rank == 2:
            return ACCENT_SILVER
        elif rank == 3:
            return ACCENT_BRONZE
        return TEXT_WHITE
    
    def _get_rank_emoji(self, rank):
        """Get emoji for top 3."""
        if rank == 1:
            return "🥇"
        elif rank == 2:
            return "🥈"
        elif rank == 3:
            return "🥉"
        return f"#{rank}"

    def _format_timestamp(self, timestamp):
        """Format timestamp to readable date."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%b %d, %H:%M")
        except:
            return "Unknown"

    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            self.hovered_row = -1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        self.return_to_menu = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    back_rect = pygame.Rect(15, 15, 100, 32)
                    if back_rect.collidepoint(event.pos):
                        running = False
                        self.return_to_menu = True

            self.screen.fill(BG_DARK)
            
            # Title with icon
            title = self.font_title.render("🏆 Top 10 Scores", True, ACCENT_PRIMARY)
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 25))
            
            # Subtitle
            if self.scores:
                total_games = len(self.scores)
                best_score = self.scores[0]['score'] if self.scores else 0
                subtitle = self.font_small.render(f"Best: {best_score} | Total tracked: {total_games}", True, TEXT_GRAY)
                self.screen.blit(subtitle, (self.width // 2 - subtitle.get_width() // 2, 70))
            
            # Column headers
            header_y = 100
            pygame.draw.line(self.screen, ACCENT_PRIMARY, (50, header_y + 25), (self.width - 50, header_y + 25), 2)
            
            headers = [("Rank", 70), ("Score", 180), ("Mode", 350), ("Date", 530)]
            for text, x in headers:
                surf = self.font_header.render(text, True, TEXT_GRAY)
                self.screen.blit(surf, (x, header_y))
            
            # Scores list
            if not self.scores:
                msg = self.font_row.render("No games played yet! Start playing to see your scores.", True, TEXT_MUTED)
                self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2, 250))
            else:
                start_y = 140
                row_height = 42
                
                for i, entry in enumerate(self.scores):
                    y = start_y + i * row_height
                    row_rect = pygame.Rect(50, y, self.width - 100, row_height - 4)
                    
                    # Check hover
                    if row_rect.collidepoint(mouse_pos):
                        self.hovered_row = i
                    
                    # Row background (alternating with hover effect)
                    if self.hovered_row == i:
                        bg_color = ROW_HOVER
                    else:
                        bg_color = ROW_BG if i % 2 == 0 else ROW_BG_ALT
                    
                    pygame.draw.rect(self.screen, bg_color, row_rect, border_radius=6)
                    
                    # Highlight border for top 3
                    if i < 3:
                        pygame.draw.rect(self.screen, self._get_rank_color(i + 1), row_rect, width=1, border_radius=6)
                    
                    # Rank
                    rank_text = self._get_rank_emoji(i + 1)
                    rank_color = self._get_rank_color(i + 1)
                    rank_surf = self.font_row.render(rank_text, True, rank_color)
                    self.screen.blit(rank_surf, (70, y + 10))
                    
                    # Score
                    score_color = ACCENT_GOLD if i == 0 else TEXT_WHITE
                    score_surf = self.font_score.render(str(entry.get('score', 0)), True, score_color)
                    self.screen.blit(score_surf, (180, y + 8))
                    
                    # Mode
                    mode = entry.get('mode', 'Standard')
                    mode_color = ACCENT_PRIMARY if 'AI' in mode else TEXT_GRAY
                    mode_surf = self.font_row.render(mode, True, mode_color)
                    self.screen.blit(mode_surf, (350, y + 10))
                    
                    # Timestamp
                    timestamp = entry.get('timestamp', 0)
                    date_str = self._format_timestamp(timestamp)
                    date_surf = self.font_small.render(date_str, True, TEXT_MUTED)
                    self.screen.blit(date_surf, (530, y + 12))
            
            # Back button
            back_rect = pygame.Rect(15, 15, 100, 32)
            hover = back_rect.collidepoint(mouse_pos)
            bg_color = (60, 65, 85) if hover else (40, 42, 60)
            pygame.draw.rect(self.screen, bg_color, back_rect, border_radius=6)
            pygame.draw.rect(self.screen, ACCENT_PRIMARY if hover else (50, 55, 75), back_rect, 1, border_radius=6)
            back_text = self.font_small.render('← Menu', True, ACCENT_PRIMARY if hover else TEXT_MUTED)
            self.screen.blit(back_text, (back_rect.x + 20, back_rect.y + 10))
            
            # Footer
            footer = self.font_small.render("Press ESC to return to menu", True, TEXT_MUTED)
            self.screen.blit(footer, (self.width // 2 - footer.get_width() // 2, self.height - 35))

            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()


def run_scoreboard():
    """Run scoreboard. Returns True if user wants to return to menu."""
    s = Scoreboard()
    s.run()
    return s.return_to_menu


if __name__ == '__main__':
    run_scoreboard()
