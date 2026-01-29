"""Theme system for Ripperdoc CLI.

Provides color theming support with predefined themes and runtime switching.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ThemeColors:
    """Theme color definitions - all semantic color slots."""

    # === Brand/Primary colors ===
    primary: str = "cyan"  # Main accent (brand, borders, spinner)
    secondary: str = "green"  # Secondary accent (success, ready status)

    # === Semantic colors ===
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "blue"

    # === Text colors ===
    text_primary: str = "white"
    text_secondary: str = "dim"  # dim/grey for secondary text
    text_muted: str = "grey50"

    # === UI elements ===
    border: str = "cyan"
    spinner: str = "cyan"
    prompt: str = "bold green"

    # === Tool output ===
    tool_call: str = "dim cyan"
    tool_result: str = "dim"

    # === Message senders ===
    sender_user: str = "bold green"
    sender_assistant: str = "white"
    sender_system: str = "dim cyan"

    # === Context visualization ===
    ctx_system_prompt: str = "grey58"
    ctx_mcp: str = "cyan"
    ctx_tools: str = "green3"
    ctx_memory: str = "dark_orange3"
    ctx_messages: str = "medium_purple"
    ctx_free: str = "grey46"
    ctx_reserved: str = "yellow3"

    # === Thinking mode ===
    thinking_on: str = "ansicyan bold"
    thinking_off: str = "ansibrightblack"
    thinking_content: str = "dim italic"

    # === Headings and emphasis ===
    heading: str = "bold"
    emphasis: str = "bold cyan"


@dataclass
class Theme:
    """Complete theme definition."""

    name: str
    display_name: str
    description: str
    colors: ThemeColors = field(default_factory=ThemeColors)

    def get_color(self, slot: str) -> str:
        """Get color for a specific slot."""
        return getattr(self.colors, slot, "white")


# === Predefined themes ===

THEME_DARK = Theme(
    name="dark",
    display_name="Dark",
    description="Default dark theme with cyan accents",
    colors=ThemeColors(),
)

THEME_LIGHT = Theme(
    name="light",
    display_name="Light",
    description="Light theme for bright terminals",
    colors=ThemeColors(
        primary="blue",
        secondary="green",
        text_primary="black",
        text_secondary="grey37",
        text_muted="grey50",
        border="blue",
        spinner="blue",
        tool_call="dim blue",
        sender_user="bold blue",
        ctx_system_prompt="grey37",
        ctx_free="grey62",
        thinking_on="ansiblue bold",
        thinking_off="grey50",
        emphasis="bold blue",
    ),
)

THEME_MONOKAI = Theme(
    name="monokai",
    display_name="Monokai",
    description="Monokai-inspired color scheme",
    colors=ThemeColors(
        primary="#f92672",  # Monokai pink
        secondary="#a6e22e",  # Monokai green
        warning="#e6db74",  # Monokai yellow
        error="#f92672",  # Monokai pink
        info="#66d9ef",  # Monokai cyan
        border="#f92672",
        spinner="#66d9ef",
        tool_call="#75715e",  # Monokai comment
        sender_user="#a6e22e",
        emphasis="#f92672",
        ctx_mcp="#66d9ef",
        ctx_tools="#a6e22e",
        ctx_memory="#fd971f",  # Monokai orange
        ctx_messages="#ae81ff",  # Monokai purple
    ),
)

THEME_DRACULA = Theme(
    name="dracula",
    display_name="Dracula",
    description="Dracula color scheme",
    colors=ThemeColors(
        primary="#bd93f9",  # Dracula purple
        secondary="#50fa7b",  # Dracula green
        warning="#f1fa8c",  # Dracula yellow
        error="#ff5555",  # Dracula red
        info="#8be9fd",  # Dracula cyan
        border="#bd93f9",
        spinner="#bd93f9",
        tool_call="#6272a4",  # Dracula comment
        sender_user="#50fa7b",
        emphasis="#ff79c6",  # Dracula pink
        ctx_mcp="#8be9fd",
        ctx_tools="#50fa7b",
        ctx_memory="#ffb86c",  # Dracula orange
        ctx_messages="#bd93f9",
    ),
)

THEME_SOLARIZED_DARK = Theme(
    name="solarized_dark",
    display_name="Solarized Dark",
    description="Solarized dark color scheme",
    colors=ThemeColors(
        primary="#268bd2",  # Solarized blue
        secondary="#859900",  # Solarized green
        warning="#b58900",  # Solarized yellow
        error="#dc322f",  # Solarized red
        info="#2aa198",  # Solarized cyan
        text_secondary="#586e75",  # Solarized base01
        border="#268bd2",
        spinner="#2aa198",
        emphasis="#268bd2",
        ctx_mcp="#2aa198",
        ctx_tools="#859900",
        ctx_memory="#cb4b16",  # Solarized orange
        ctx_messages="#6c71c4",  # Solarized violet
    ),
)

THEME_NORD = Theme(
    name="nord",
    display_name="Nord",
    description="Arctic, bluish color scheme",
    colors=ThemeColors(
        primary="#88c0d0",  # Nord frost
        secondary="#a3be8c",  # Nord green
        warning="#ebcb8b",  # Nord yellow
        error="#bf616a",  # Nord red
        info="#81a1c1",  # Nord blue
        text_secondary="#4c566a",  # Nord polar night
        border="#88c0d0",
        spinner="#88c0d0",
        emphasis="#88c0d0",
        ctx_mcp="#81a1c1",
        ctx_tools="#a3be8c",
        ctx_memory="#d08770",  # Nord orange
        ctx_messages="#b48ead",  # Nord purple
    ),
)

# Theme registry
BUILTIN_THEMES: Dict[str, Theme] = {
    "dark": THEME_DARK,
    "light": THEME_LIGHT,
    "monokai": THEME_MONOKAI,
    "dracula": THEME_DRACULA,
    "solarized_dark": THEME_SOLARIZED_DARK,
    "nord": THEME_NORD,
}


class ThemeManager:
    """Theme manager - singleton pattern."""

    _instance: Optional["ThemeManager"] = None
    _current_theme: Theme
    _listeners: List[Callable[["Theme"], None]]

    def __new__(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._current_theme = THEME_DARK
            cls._instance._listeners = []
        return cls._instance

    @property
    def current(self) -> Theme:
        """Get current theme."""
        return self._current_theme

    def set_theme(self, theme_name: str) -> bool:
        """Set theme by name (does not persist)."""
        theme = BUILTIN_THEMES.get(theme_name)
        if theme:
            self._current_theme = theme
            self._notify_listeners()
            return True
        return False

    def get_color(self, slot: str) -> str:
        """Get color for a slot from current theme."""
        return self._current_theme.get_color(slot)

    def add_listener(self, callback: Callable[["Theme"], None]) -> None:
        """Add a theme change listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[["Theme"], None]) -> None:
        """Remove a theme change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self) -> None:
        """Notify all listeners of theme change."""
        for listener in self._listeners:
            try:
                listener(self._current_theme)
            except Exception:
                pass  # Don't let listener errors break theme switching

    def list_themes(self) -> List[str]:
        """List all available theme names."""
        return list(BUILTIN_THEMES.keys())


# Global accessor functions
_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


def theme_color(slot: str) -> str:
    """Convenience function: get color for a slot from current theme."""
    return get_theme_manager().get_color(slot)


def styled(text: str, slot: str) -> str:
    """Convenience function: wrap text with theme color markup."""
    color = theme_color(slot)
    # Handle colors that already include style modifiers (like "bold cyan")
    if " " in color:
        return f"[{color}]{text}[/]"
    return f"[{color}]{text}[/{color}]"


def get_current_theme() -> Theme:
    """Get the current active theme."""
    return get_theme_manager().current


__all__ = [
    "Theme",
    "ThemeColors",
    "ThemeManager",
    "BUILTIN_THEMES",
    "get_theme_manager",
    "theme_color",
    "styled",
    "get_current_theme",
]
