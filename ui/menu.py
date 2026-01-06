import pygame
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCREEN_WIDTH, SCREEN_HEIGHT

# Colors
BG_DARK = (12, 12, 20)
ACCENT_PRIMARY = (0, 255, 200)
ACCENT_SECONDARY = (255, 180, 50)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (120, 130, 150)
BTN_BG = (30, 32, 48)
BTN_HOVER = (40, 45, 65)

class Menu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 550))
        pygame.display.set_caption("Flappy Bird AI Coach - Launcher")
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont('Arial', 32, bold=True)
        self.font_btn = pygame.font.SysFont('Arial', 20)
        self.font_desc = pygame.font.SysFont('Arial', 14)
        
        self.options = [
            {
                "title": "Standard Dashboard",
                "desc": "Original Game + Rule-Based Coach (Fastest)",
                "config": {"enable_llm": False, "enable_vision": False}
            },
            {
                "title": "Llama Mode (NLP)",
                "desc": "Adds AI Chat & Conversational Coaching",
                "config": {"enable_llm": True, "enable_vision": False}
            },
            {
                "title": "Llava Mode (Vision)",
                "desc": "Adds Visual Analysis of Gameplay",
                "config": {"enable_llm": False, "enable_vision": True}
            },
            {
                "title": "Ultimate Mode",
                "desc": "All Features: Voice, Chat, Vision, & NLP",
                "config": {"enable_llm": True, "enable_vision": True}
            },
            {
                "title": "Watch AI Play (RL)",
                "desc": "Watch trained DQN agent play the game",
                "config": {"mode": "watch_ai"}
            },
            {
                "title": "Train AI Agent",
                "desc": "Train the RL agent (visual mode)",
                "config": {"mode": "train_ai"}
            }
        ]
        
        # Calculate button rects
        start_y = 60
        self.buttons = []
        for i, opt in enumerate(self.options):
            rect = pygame.Rect(50, start_y + i * 65, 500, 55)
            self.buttons.append((rect, opt))

    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for rect, opt in self.buttons:
                            if rect.collidepoint(mouse_pos):
                                return opt["config"]

            # Draw
            self.screen.fill(BG_DARK)
            
            # Title
            title_surf = self.font_title.render("Select Experience", True, TEXT_WHITE)
            self.screen.blit(title_surf, (50, 40))
            
            # Buttons
            for rect, opt in self.buttons:
                hover = rect.collidepoint(mouse_pos)
                color = BTN_HOVER if hover else BTN_BG
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, ACCENT_PRIMARY if hover else (60, 60, 80), rect, width=2, border_radius=8)
                
                # Text
                name_surf = self.font_btn.render(opt["title"], True, ACCENT_PRIMARY if hover else TEXT_WHITE)
                desc_surf = self.font_desc.render(opt["desc"], True, TEXT_GRAY)
                
                self.screen.blit(name_surf, (rect.x + 20, rect.y + 15))
                self.screen.blit(desc_surf, (rect.x + 20, rect.y + 40))
                
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        return None

def run_menu():
    m = Menu()
    return m.run()

if __name__ == "__main__":
    print(run_menu())
