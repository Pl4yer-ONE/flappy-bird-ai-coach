import pygame
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np

# Colors - Modern dark theme (consistent with other UI)
BG_DARK = (12, 12, 20)
BG_PANEL = (30, 32, 48)
ACCENT_PRIMARY = (0, 255, 200)
ACCENT_SECONDARY = (255, 180, 50)
ACCENT_POSITIVE = (100, 220, 150)
ACCENT_NEGATIVE = (255, 100, 100)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (120, 130, 150)
TEXT_MUTED = (80, 85, 100)


class AnalysisDashboard:
    def __init__(self):
        pygame.init()
        self.width = 1100
        self.height = 750
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("📊 Flappy Bird AI Coach - Analysis Mode")
        self.clock = pygame.time.Clock()
        self.return_to_menu = False
        
        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 32, bold=True)
        self.font_header = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_body = pygame.font.SysFont('Arial', 16)
        self.font_metric = pygame.font.SysFont('Arial', 42, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 12)
        
        self.data = self._load_data()
        self.graph_surface = self._generate_graph()
        self.prediction = 0

    def _load_data(self):
        try:
            path = Path(__file__).parent.parent / 'data' / 'history.json'
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _generate_graph(self):
        if not self.data:
            return None
            
        scores = [x.get('score', 0) for x in self.data]
        indices = list(range(1, len(scores) + 1))
        
        fig = plt.figure(figsize=(9, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Style matching UI theme
        fig.patch.set_facecolor('#1e2030')
        ax.set_facecolor('#1e2030')
        ax.tick_params(colors='white', labelsize=9)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('#00ffc8')
        for spine in ax.spines.values():
            spine.set_color('#505565')
        
        # Main line with gradient fill
        ax.plot(indices, scores, color='#00ffc8', linewidth=2.5, marker='o', markersize=5, label='Score')
        ax.fill_between(indices, scores, alpha=0.2, color='#00ffc8')
        
        ax.set_title("Performance History", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Games Played", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(True, color='#303545', alpha=0.5, linestyle='--')
        
        # Trend / Predictive line
        if len(scores) > 1:
            z = np.polyfit(indices, scores, 1)
            p = np.poly1d(z)
            trend_color = '#64dc96' if z[0] > 0 else '#ff6464'
            ax.plot(indices, p(indices), color=trend_color, linestyle='--', linewidth=2, alpha=0.7, label='Trend')
            
            # Projection for next 3 games
            future_x = list(range(len(scores) + 1, len(scores) + 4))
            future_y = [p(x) for x in future_x]
            ax.plot(future_x, future_y, color=trend_color, linestyle=':', linewidth=1.5, alpha=0.5, marker='^', markersize=4)
            
            self.prediction = max(0, p(len(scores) + 1))
        else:
            self.prediction = scores[0] if scores else 0
        
        ax.legend(loc='upper left', facecolor='#1e2030', edgecolor='#505565', labelcolor='white')
        fig.tight_layout()
        
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        
        # Convert canvas to pygame surface (compatible with newer matplotlib)
        size = canvas.get_width_height()
        buf = canvas.buffer_rgba()
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        plt.close(fig)
        return surf

    def _draw_card(self, x, y, width, height, label, value, color=ACCENT_PRIMARY, subtitle=None):
        """Draw a metric card."""
        card_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, BG_PANEL, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, color, card_rect, width=1, border_radius=12)
        
        # Label
        label_surf = self.font_body.render(label, True, TEXT_GRAY)
        self.screen.blit(label_surf, (x + 20, y + 15))
        
        # Value
        value_surf = self.font_metric.render(str(value), True, color)
        self.screen.blit(value_surf, (x + 20, y + 45))
        
        # Subtitle if provided
        if subtitle:
            sub_surf = self.font_small.render(subtitle, True, TEXT_MUTED)
            self.screen.blit(sub_surf, (x + 20, y + height - 25))

    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
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
            
            # Title
            title = self.font_title.render("📊 Historical & Predictive Analysis", True, ACCENT_PRIMARY)
            self.screen.blit(title, (30, 25))
            
            if not self.data:
                msg = self.font_body.render("No data available. Play some games to see your analysis!", True, TEXT_GRAY)
                self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2, self.height // 2))
            else:
                scores = [x.get('score', 0) for x in self.data]
                avg = sum(scores) / len(scores) if scores else 0
                best = max(scores) if scores else 0
                recent = scores[-1] if scores else 0
                total_games = len(scores)
                
                # Trend analysis
                if len(scores) > 1:
                    recent_5 = scores[-5:] if len(scores) >= 5 else scores
                    first_5 = scores[:5] if len(scores) >= 5 else scores
                    trend_improving = sum(recent_5) / len(recent_5) > sum(first_5) / len(first_5)
                else:
                    trend_improving = True
                
                trend_text = "↗ Improving" if trend_improving else "↘ Declining"
                trend_color = ACCENT_POSITIVE if trend_improving else ACCENT_NEGATIVE
                
                # Metric cards - top row
                card_width = 230
                card_height = 115
                card_y = 90
                gap = 25
                start_x = 30
                
                self._draw_card(start_x, card_y, card_width, card_height, 
                               "Total Games", str(total_games), 
                               ACCENT_PRIMARY, f"Sessions tracked")
                
                self._draw_card(start_x + card_width + gap, card_y, card_width, card_height, 
                               "Best Score", str(best), 
                               (255, 215, 0), "Personal best")
                
                self._draw_card(start_x + (card_width + gap) * 2, card_y, card_width, card_height, 
                               "Average", f"{avg:.1f}", 
                               ACCENT_SECONDARY, f"Across all games")
                
                self._draw_card(start_x + (card_width + gap) * 3, card_y, card_width, card_height, 
                               "Trend", trend_text, 
                               trend_color, "Based on recent play")
                
                # Graph
                if self.graph_surface:
                    graph_x = (self.width - self.graph_surface.get_width()) // 2
                    self.screen.blit(self.graph_surface, (graph_x, 230))
                
                # Predictive insights section
                insights_y = 650
                pygame.draw.line(self.screen, BG_PANEL, (30, insights_y - 15), (self.width - 30, insights_y - 15), 2)
                
                pred_text = f"🔮 Projected next score: {self.prediction:.1f}"
                pred_color = ACCENT_POSITIVE if self.prediction > avg else ACCENT_SECONDARY
                pred_surf = self.font_header.render(pred_text, True, pred_color)
                self.screen.blit(pred_surf, (30, insights_y))
                
                # Additional insight
                if trend_improving:
                    insight = "Keep up the great work! Your skills are improving steadily."
                else:
                    insight = "Consider focusing on timing and positioning to improve your scores."
                insight_surf = self.font_body.render(insight, True, TEXT_GRAY)
                self.screen.blit(insight_surf, (30, insights_y + 30))
            
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
            self.screen.blit(footer, (self.width // 2 - footer.get_width() // 2, self.height - 25))

            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()


def run_analysis():
    """Run analysis. Returns True if user wants to return to menu."""
    a = AnalysisDashboard()
    a.run()
    return a.return_to_menu


if __name__ == '__main__':
    run_analysis()
