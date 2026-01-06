import pygame
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np

class AnalysisDashboard:
    def __init__(self):
        pygame.init()
        self.width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird AI Coach - Analysis Mode")
        self.clock = pygame.time.Clock()
        
        self.data = self._load_data()
        self.graph_surface = self._generate_graph()
        
        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 32, bold=True)
        self.font_body = pygame.font.SysFont('Arial', 18)
        self.font_big = pygame.font.SysFont('Arial', 48, bold=True)
        
        # Colors
        self.BG = (12, 12, 20)
        self.ACCENT = (0, 255, 200)
        self.TEXT = (255, 255, 255)
        self.CARD = (30, 32, 48)

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
            
        scores = [x['score'] for x in self.data]
        indices = list(range(1, len(scores) + 1))
        
        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Style
        fig.patch.set_facecolor('#1e2030')
        ax.set_facecolor('#1e2030')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#505565')
            
        ax.plot(indices, scores, color='#00ffc8', linewidth=2, marker='o', markersize=4)
        ax.set_title("Performance History")
        ax.set_xlabel("Games Played")
        ax.set_ylabel("Score")
        ax.grid(True, color='#303545')
        
        # Trend / Predictive
        if len(scores) > 1:
            z = np.polyfit(indices, scores, 1)
            p = np.poly1d(z)
            ax.plot(indices, p(indices), "r--", alpha=0.5, label='Trend')
            
            # Projection
            self.prediction = p(len(scores) + 1)
        else:
            self.prediction = 0
            
        fig.tight_layout()
        
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)
        return surf

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
            title = self.font_title.render("📊 Historical & Predictive Analysis", True, self.TEXT)
            self.screen.blit(title, (20, 20))
            
            if not self.data:
                msg = self.font_body.render("No data available. Play some games!", True, (150, 150, 150))
                self.screen.blit(msg, (self.width//2 - msg.get_width()//2, self.height//2))
            else:
                # Key Metrics Cards
                scores = [x['score'] for x in self.data]
                avg = sum(scores) / len(scores)
                best = max(scores)
                recent = scores[-1]
                trend = "Improving ↗" if getattr(self, 'prediction', 0) > avg else "Steady/Declining ↘"
                
                metrics = [
                    ("Total Games", str(len(scores))),
                    ("Best Score", str(best)),
                    ("Average", f"{avg:.1f}"),
                    ("Trend", trend)
                ]
                
                for i, (label, val) in enumerate(metrics):
                    card_rect = pygame.Rect(20 + i * 240, 80, 220, 100)
                    pygame.draw.rect(self.screen, self.CARD, card_rect, border_radius=8)
                    
                    lbl = self.font_body.render(label, True, (150,150,170))
                    self.screen.blit(lbl, (card_rect.x + 15, card_rect.y + 15))
                    
                    v = self.font_big.render(val, True, self.ACCENT)
                    if label == "Trend": # Smaller font for trend text
                        v = self.font_title.render(val, True, self.ACCENT if "Improving" in val else (255, 100, 100))
                    
                    self.screen.blit(v, (card_rect.x + 15, card_rect.y + 45))
                
                # Graph
                if self.graph_surface:
                    self.screen.blit(self.graph_surface, (100, 220))
                    
                # Predictive text
                if hasattr(self, 'prediction'):
                    pred_txt = f"🔮 Projected next score: {self.prediction:.1f}"
                    pred_surf = self.font_title.render(pred_txt, True, (100, 200, 255))
                    self.screen.blit(pred_surf, (self.width//2 - pred_surf.get_width()//2, 640))

            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()

def run_analysis():
    a = AnalysisDashboard()
    a.run()

if __name__ == '__main__':
    run_analysis()
