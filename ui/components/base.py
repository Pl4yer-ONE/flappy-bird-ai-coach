# UI Components - Reusable UI Building Blocks

"""
Reusable UI components for the dashboard.
Includes ScrollView, TextInput, Button, and styling utilities.
"""

import pygame
from typing import Optional, Callable, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MENU_ACCENT_COLOR, MENU_TEXT_COLOR, MENU_BG_COLOR


class Colors:
    """Color palette for consistent theming."""
    BG_DARK = (15, 20, 30)
    BG_PANEL = (25, 32, 45)
    BG_INPUT = (35, 45, 60)
    BORDER = (60, 80, 100)
    ACCENT = (100, 200, 150)
    TEXT = (220, 230, 240)
    TEXT_DIM = (120, 140, 160)
    SUCCESS = (100, 220, 120)
    WARNING = (255, 200, 80)
    ERROR = (255, 100, 100)
    CHAT_USER = (80, 140, 200)
    CHAT_AI = (100, 180, 130)


class ScrollView:
    """Scrollable container for content."""
    
    def __init__(self, rect: pygame.Rect, item_height: int = 30):
        self.rect = rect
        self.item_height = item_height
        self.scroll_offset = 0
        self.items: List[dict] = []
        self.max_visible = rect.height // item_height
        self.font = None
        
    def _init_font(self):
        if self.font is None:
            self.font = pygame.font.Font(None, 20)
            
    def add_item(self, text: str, color: Tuple = Colors.TEXT, icon: str = ""):
        """Add an item to the scroll view."""
        self.items.append({'text': text, 'color': color, 'icon': icon})
        # Auto-scroll to bottom
        if len(self.items) > self.max_visible:
            self.scroll_offset = len(self.items) - self.max_visible
            
    def clear(self):
        """Clear all items."""
        self.items.clear()
        self.scroll_offset = 0
        
    def scroll(self, direction: int):
        """Scroll by direction (-1 up, +1 down)."""
        self.scroll_offset = max(0, min(
            len(self.items) - self.max_visible,
            self.scroll_offset + direction
        ))
        
    def handle_event(self, event: pygame.event.Event):
        """Handle scroll wheel events."""
        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll(-event.y)
                
    def render(self, screen: pygame.Surface):
        """Render the scroll view."""
        self._init_font()
        
        # Background
        pygame.draw.rect(screen, Colors.BG_PANEL, self.rect, border_radius=5)
        pygame.draw.rect(screen, Colors.BORDER, self.rect, 1, border_radius=5)
        
        # Clip to rect
        clip = screen.get_clip()
        screen.set_clip(self.rect.inflate(-4, -4))
        
        # Render visible items
        visible_start = self.scroll_offset
        visible_end = min(len(self.items), visible_start + self.max_visible + 1)
        
        for i, item in enumerate(self.items[visible_start:visible_end]):
            y = self.rect.y + 5 + i * self.item_height
            text = f"{item['icon']} {item['text']}" if item['icon'] else item['text']
            
            # Word wrap if needed
            words = text.split()
            lines = []
            current = ""
            for word in words:
                test = current + " " + word if current else word
                if self.font.size(test)[0] < self.rect.width - 20:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)
                
            for line in lines[:2]:  # Max 2 lines per item
                surf = self.font.render(line, True, item['color'])
                screen.blit(surf, (self.rect.x + 10, y))
                y += 18
                
        # Restore clip
        screen.set_clip(clip)
        
        # Scrollbar
        if len(self.items) > self.max_visible:
            bar_height = max(20, self.rect.height * self.max_visible // len(self.items))
            bar_y = self.rect.y + (self.rect.height - bar_height) * self.scroll_offset // (len(self.items) - self.max_visible)
            bar_rect = pygame.Rect(self.rect.right - 8, bar_y, 5, bar_height)
            pygame.draw.rect(screen, Colors.BORDER, bar_rect, border_radius=2)


class TextInput:
    """Text input field with focus handling."""
    
    def __init__(self, rect: pygame.Rect, placeholder: str = "Type here..."):
        self.rect = rect
        self.placeholder = placeholder
        self.text = ""
        self.focused = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.font = None
        self.on_submit: Optional[Callable[[str], None]] = None
        
    def _init_font(self):
        if self.font is None:
            self.font = pygame.font.Font(None, 22)
            
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handle input events. Returns submitted text if Enter pressed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.focused = self.rect.collidepoint(event.pos)
            
        if self.focused and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.text.strip():
                    submitted = self.text
                    self.text = ""
                    if self.on_submit:
                        self.on_submit(submitted)
                    return submitted
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode and event.unicode.isprintable():
                if self.font and self.font.size(self.text + event.unicode)[0] < self.rect.width - 30:
                    self.text += event.unicode
                    
        return None
        
    def update(self):
        """Update cursor blink."""
        self.cursor_timer += 1
        if self.cursor_timer > 30:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
            
    def render(self, screen: pygame.Surface):
        """Render the input field."""
        self._init_font()
        
        # Background
        bg_color = Colors.BG_INPUT if self.focused else Colors.BG_PANEL
        pygame.draw.rect(screen, bg_color, self.rect, border_radius=5)
        
        # Border
        border_color = Colors.ACCENT if self.focused else Colors.BORDER
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=5)
        
        # Text or placeholder
        if self.text:
            text_surf = self.font.render(self.text, True, Colors.TEXT)
        else:
            text_surf = self.font.render(self.placeholder, True, Colors.TEXT_DIM)
            
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        screen.blit(text_surf, text_rect)
        
        # Cursor
        if self.focused and self.cursor_visible:
            cursor_x = text_rect.right + 2 if self.text else text_rect.left
            pygame.draw.line(screen, Colors.ACCENT, 
                           (cursor_x, self.rect.y + 8),
                           (cursor_x, self.rect.bottom - 8), 2)


class Button:
    """Interactive button with hover effects."""
    
    def __init__(self, rect: pygame.Rect, text: str, icon: str = "",
                 on_click: Optional[Callable] = None):
        self.rect = rect
        self.text = text
        self.icon = icon
        self.on_click = on_click
        self.hovered = False
        self.pressed = False
        self.font = None
        
    def _init_font(self):
        if self.font is None:
            self.font = pygame.font.Font(None, 20)
            
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle click events. Returns True if clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
            
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.pressed = True
                
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.hovered:
                self.pressed = False
                if self.on_click:
                    self.on_click()
                return True
            self.pressed = False
            
        return False
        
    def render(self, screen: pygame.Surface):
        """Render the button."""
        self._init_font()
        
        # Background
        if self.pressed:
            bg_color = Colors.ACCENT
        elif self.hovered:
            bg_color = (50, 70, 90)
        else:
            bg_color = Colors.BG_PANEL
            
        pygame.draw.rect(screen, bg_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, Colors.BORDER, self.rect, 1, border_radius=5)
        
        # Text
        label = f"{self.icon} {self.text}" if self.icon else self.text
        text_color = Colors.BG_DARK if self.pressed else Colors.TEXT
        text_surf = self.font.render(label, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)


class Panel:
    """Base panel class."""
    
    def __init__(self, rect: pygame.Rect, title: str = ""):
        self.rect = rect
        self.title = title
        self.title_height = 30 if title else 0
        self.content_rect = pygame.Rect(
            rect.x, rect.y + self.title_height,
            rect.width, rect.height - self.title_height
        )
        self.title_font = None
        
    def _init_fonts(self):
        if self.title_font is None:
            self.title_font = pygame.font.Font(None, 22)
            
    def handle_event(self, event: pygame.event.Event):
        """Override in subclasses."""
        pass
        
    def update(self):
        """Override in subclasses."""
        pass
        
    def render_background(self, screen: pygame.Surface):
        """Render panel background and title."""
        self._init_fonts()
        
        # Background
        pygame.draw.rect(screen, Colors.BG_PANEL, self.rect, border_radius=8)
        pygame.draw.rect(screen, Colors.BORDER, self.rect, 1, border_radius=8)
        
        # Title bar
        if self.title:
            title_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.title_height)
            pygame.draw.rect(screen, (35, 45, 60), title_rect, 
                           border_top_left_radius=8, border_top_right_radius=8)
            pygame.draw.line(screen, Colors.BORDER,
                           (self.rect.x, self.rect.y + self.title_height),
                           (self.rect.right, self.rect.y + self.title_height))
            
            title_surf = self.title_font.render(self.title, True, Colors.ACCENT)
            screen.blit(title_surf, (self.rect.x + 10, self.rect.y + 7))
            
    def render(self, screen: pygame.Surface):
        """Override in subclasses."""
        self.render_background(screen)
