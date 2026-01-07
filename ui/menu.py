import pygame
import sys
import os
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCREEN_WIDTH, SCREEN_HEIGHT

# Colors - Modern dark theme
BG_DARK = (12, 12, 20)
BG_GRADIENT_TOP = (20, 22, 35)
BG_GRADIENT_BOTTOM = (12, 12, 20)
ACCENT_PRIMARY = (0, 255, 200)
ACCENT_SECONDARY = (255, 180, 50)
ACCENT_VISION = (147, 112, 219)  # Purple for vision
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (120, 130, 150)
TEXT_MUTED = (80, 85, 100)
BTN_BG = (30, 32, 48)
BTN_HOVER = (45, 48, 70)
BTN_SELECTED = (35, 80, 70)
BORDER_DEFAULT = (50, 55, 75)
SUCCESS_GREEN = (50, 205, 50)
ERROR_RED = (255, 80, 80)


class Menu:
    def __init__(self):
        pygame.init()
        self.width = 720
        self.height = 650
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🎮 Flappy Bird AI Coach - Launcher")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 36, bold=True)
        self.font_subtitle = pygame.font.SysFont('Arial', 18)
        self.font_btn = pygame.font.SysFont('Arial', 22, bold=True)
        self.font_desc = pygame.font.SysFont('Arial', 14)
        self.font_status = pygame.font.SysFont('Arial', 12)
        
        # Service status (check async)
        self.llm_status = "checking..."
        self.vision_status = "checking..."
        self._check_services()
        
        self.selected_index = 0
        
        self.options = [
            {
                "title": "🤖 AI Coach Combined Mode",
                "desc": "Llama (NLP) + LLaVA (Vision) + gTTS (Voice)",
                "config": {"enable_llm": True, "enable_vision": True},
                "color": ACCENT_PRIMARY
            },
            {
                "title": "🎯 RL Mode",
                "desc": "Watch trained AI Agent play with DQN",
                "config": {"mode": "watch_ai"},
                "color": ACCENT_SECONDARY
            },
            {
                "title": "📚 Learning Mode",
                "desc": "Train the AI Agent from scratch",
                "config": {"mode": "train_ai"},
                "color": ACCENT_VISION
            },
            {
                "title": "🏆 Scoreboard",
                "desc": "View your top scores and achievements",
                "config": {"mode": "scoreboard"},
                "color": (100, 200, 255)
            },
            {
                "title": "📊 Analysis Mode",
                "desc": "Historical & predictive performance analysis",
                "config": {"mode": "analysis"},
                "color": (255, 150, 200)
            }
        ]
        
        # Calculate button rects with better spacing
        self.btn_width = 620
        self.btn_height = 70
        start_y = 160
        self.buttons = []
        for i, opt in enumerate(self.options):
            x = (self.width - self.btn_width) // 2
            rect = pygame.Rect(x, start_y + i * 85, self.btn_width, self.btn_height)
            self.buttons.append((rect, opt))
    
    def _check_services(self):
        """Check LLM and Vision service status asynchronously."""
        def check():
            try:
                from llm.ollama_client import get_client
                from config import LLAMA_MODEL, LLAVA_MODEL
                
                client = get_client()
                if client.is_available():
                    # Flexible check: exact match OR substring
                    if client.has_model(LLAMA_MODEL) or client.has_model("llama"):
                        self.llm_status = "✓ Ready"
                    else:
                        self.llm_status = "⚠ Model missing"
                        
                    if client.has_model(LLAVA_MODEL) or client.has_model("llava"):
                        self.vision_status = "✓ Ready"
                    else:
                        self.vision_status = "⚠ Model missing"
                else:
                    self.llm_status = "✗ Offline"
                    self.vision_status = "✗ Offline"
            except Exception as e:
                self.llm_status = "✗ Error"
                self.vision_status = "✗ Error"
        
        threading.Thread(target=check, daemon=True).start()

    def run(self):
        running = True
        hover_animation = {}  # Track hover states for animation
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return None
                    elif event.key == pygame.K_UP:
                        self.selected_index = (self.selected_index - 1) % len(self.options)
                    elif event.key == pygame.K_DOWN:
                        self.selected_index = (self.selected_index + 1) % len(self.options)
                    elif event.key == pygame.K_RETURN:
                        # Don't quit pygame - let dashboard reinitialize
                        return self.options[self.selected_index]["config"]
                        
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for i, (rect, opt) in enumerate(self.buttons):
                            if rect.collidepoint(mouse_pos):
                                # Don't quit pygame - let dashboard reinitialize
                                return opt["config"]

            # Update selection on hover
            for i, (rect, opt) in enumerate(self.buttons):
                if rect.collidepoint(mouse_pos):
                    self.selected_index = i
                    break

            # Draw
            self.screen.fill(BG_DARK)
            
            # Draw gradient background
            for y in range(100):
                alpha = y / 100
                color = tuple(int(BG_GRADIENT_TOP[i] * (1 - alpha) + BG_GRADIENT_BOTTOM[i] * alpha) for i in range(3))
                pygame.draw.line(self.screen, color, (0, y * 6), (self.width, y * 6))
            
            # Title with glow effect
            title_text = "Flappy Bird AI Coach"
            title_surf = self.font_title.render(title_text, True, ACCENT_PRIMARY)
            title_x = self.width // 2 - title_surf.get_width() // 2
            self.screen.blit(title_surf, (title_x, 35))
            
            # Subtitle
            subtitle_text = "Select your experience"
            subtitle_surf = self.font_subtitle.render(subtitle_text, True, TEXT_GRAY)
            self.screen.blit(subtitle_surf, (self.width // 2 - subtitle_surf.get_width() // 2, 80))
            
            # Service status indicators
            status_y = 115
            llm_color = SUCCESS_GREEN if "Ready" in self.llm_status else (ACCENT_SECONDARY if "checking" in self.llm_status else ERROR_RED)
            vision_color = SUCCESS_GREEN if "Ready" in self.vision_status else (ACCENT_SECONDARY if "checking" in self.vision_status else ERROR_RED)
            
            llm_text = self.font_status.render(f"Llama: {self.llm_status}", True, llm_color)
            vision_text = self.font_status.render(f"LLaVA: {self.vision_status}", True, vision_color)
            self.screen.blit(llm_text, (self.width // 2 - 90, status_y))
            self.screen.blit(vision_text, (self.width // 2 + 20, status_y))
            
            # Buttons
            for i, (rect, opt) in enumerate(self.buttons):
                is_selected = (i == self.selected_index)
                hover = rect.collidepoint(mouse_pos) or is_selected
                
                # Button background with selection highlight
                bg_color = BTN_SELECTED if is_selected else (BTN_HOVER if hover else BTN_BG)
                pygame.draw.rect(self.screen, bg_color, rect, border_radius=12)
                
                # Border with accent color
                border_color = opt["color"] if is_selected else (BORDER_DEFAULT if not hover else opt["color"])
                border_width = 2 if is_selected else 1
                pygame.draw.rect(self.screen, border_color, rect, width=border_width, border_radius=12)
                
                # Selection indicator
                if is_selected:
                    indicator_rect = pygame.Rect(rect.x + 8, rect.y + 20, 4, 30)
                    pygame.draw.rect(self.screen, opt["color"], indicator_rect, border_radius=2)
                
                # Title text
                title_color = opt["color"] if is_selected else TEXT_WHITE
                name_surf = self.font_btn.render(opt["title"], True, title_color)
                self.screen.blit(name_surf, (rect.x + 25, rect.y + 15))
                
                # Description text
                desc_surf = self.font_desc.render(opt["desc"], True, TEXT_GRAY)
                self.screen.blit(desc_surf, (rect.x + 25, rect.y + 45))
            
            # Footer
            footer_text = "Use ↑↓ arrows to navigate, ENTER to select, ESC to quit"
            footer_surf = self.font_status.render(footer_text, True, TEXT_MUTED)
            self.screen.blit(footer_surf, (self.width // 2 - footer_surf.get_width() // 2, self.height - 30))
            
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        return None


def run_menu():
    m = Menu()
    return m.run()


if __name__ == "__main__":
    print(run_menu())
