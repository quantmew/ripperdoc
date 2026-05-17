"""Rich-based bridge status display with QR code toggle.

Provides a live status display for the remote-control bridge,
including idle waiting, active session, reconnecting, and failed states.
Supports QR code rendering of the connect URL.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Literal, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ripperdoc.utils.log import get_logger

from .utils import build_connect_url

if TYPE_CHECKING:
    from .process import BridgeActivity

logger = get_logger()
console = Console()

StatusState = Literal["idle", "attached", "reconnecting", "failed"]
TOOL_DISPLAY_EXPIRY_SEC = 30


class BridgeStatusDisplay:
    """Rich-based bridge status display with QR code toggle."""

    def __init__(self) -> None:
        self._show_qr = False
        self._status_state: StatusState = "idle"
        self._connect_url: str = ""
        self._environment_id: str = ""
        self._repo_name: str = ""
        self._branch: str = ""
        self._session_title: str = ""
        self._session_elapsed: str = ""
        self._session_activity: str = ""
        self._session_trail: List[str] = []
        self._active_sessions: int = 0
        self._max_sessions: int = 1
        self._spawn_mode: str = "single-session"
        self._last_tool_time: float = 0.0

    def print_banner(
        self,
        config: Any,
        environment_id: str,
        connect_url: str,
    ) -> None:
        """Show the initial bridge banner."""
        self._connect_url = connect_url
        self._environment_id = environment_id

        lines = [
            f"[bold]Remote Control bridge started[/bold]",
            f"  Workspace: {config.directory}",
            f"  Environment: {environment_id}",
            f"  Connect URL: {connect_url}",
            "  Press Ctrl+C to stop.",
        ]
        console.print("\n".join(lines))

        if self._show_qr:
            self._print_qr(connect_url)

    def toggle_qr(self) -> None:
        """Toggle QR code visibility and re-render."""
        self._show_qr = not self._show_qr
        if self._show_qr and self._connect_url:
            self._print_qr(self._connect_url)
        else:
            console.print("[dim]QR code hidden[/dim]")

    def _print_qr(self, url: str) -> None:
        """Render QR code for the given URL."""
        try:
            import qrcode
            from io import StringIO

            qr = qrcode.QRCode(box_size=1, border=1)
            qr.add_data(url)
            qr.make(fit=True)
            buf = StringIO()
            qr.print_ascii(out=buf)
            console.print(Panel(buf.getvalue(), title="Scan to connect", border_style="blue"))
        except ImportError:
            console.print("[dim]Install qrcode package for QR display: pip install qrcode[pil][/dim]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[dim]QR render failed: {exc}[/dim]")

    def set_repo_info(self, repo_name: str, branch: str) -> None:
        self._repo_name = repo_name
        self._branch = branch

    def set_attached(self, session_id: str) -> None:
        """Transition to 'Attached' state when a session starts."""
        self._status_state = "attached"
        self._session_title = session_id[:8] if len(session_id) > 8 else session_id

    def update_idle_status(self) -> None:
        """Show idle status with repo/branch info."""
        self._status_state = "idle"
        parts = ["[dim]Waiting for connection...[/dim]"]
        if self._repo_name:
            branch_info = f" ({self._branch})" if self._branch else ""
            parts.append(f"[dim]  Repo: {self._repo_name}{branch_info}[/dim]")
        console.print("\n".join(parts))

    def update_session_status(
        self,
        session_id: str,
        elapsed: str,
        activity: Optional[BridgeActivity],
        trail: Optional[List[str]] = None,
    ) -> None:
        """Show active session status."""
        self._status_state = "attached"
        self._session_elapsed = elapsed
        if activity:
            self._session_activity = activity.summary
            self._last_tool_time = time.time()

        parts = [f"[green]Session {session_id[:8]}[/green] [dim]{elapsed}[/dim]"]
        if self._session_activity and (time.time() - self._last_tool_time) < TOOL_DISPLAY_EXPIRY_SEC:
            parts.append(f"  [cyan]{self._session_activity}[/cyan]")
        if trail:
            for t in trail[-3:]:
                parts.append(f"  [dim]{t}[/dim]")
        console.print("\n".join(parts))

    def update_reconnecting_status(self, delay_str: str, elapsed_str: str) -> None:
        """Show reconnecting status."""
        self._status_state = "reconnecting"
        console.print(f"[yellow]Reconnecting... ({elapsed_str}, next retry in {delay_str})[/yellow]")

    def update_failed_status(self, error: str) -> None:
        """Show failed status."""
        self._status_state = "failed"
        console.print(f"[red]Remote Control failed: {error}[/red]")

    def update_session_count(self, active: int, max_sessions: int, mode: str) -> None:
        """Update the session count indicator."""
        self._active_sessions = active
        self._max_sessions = max_sessions
        self._spawn_mode = mode
        if max_sessions > 1:
            console.print(f"[dim]{active} of {max_sessions} sessions ({mode})[/dim]")

    def set_spawn_mode_display(self, mode: Optional[str]) -> None:
        self._spawn_mode = mode or "single-session"

    def add_session(self, session_id: str, url: str) -> None:
        """Register a new session for multi-session display."""
        self._active_sessions += 1

    def update_session_activity(self, session_id: str, activity: BridgeActivity) -> None:
        """Update the per-session activity summary."""
        self._session_activity = activity.summary
        self._last_tool_time = time.time()

    def set_session_title(self, session_id: str, title: str) -> None:
        """Set a session's display title."""
        self._session_title = title

    def remove_session(self, session_id: str) -> None:
        """Remove a session from the display."""
        self._active_sessions = max(0, self._active_sessions - 1)

    def refresh_display(self) -> None:
        """Force a re-render of the status display."""
        if self._status_state == "idle":
            self.update_idle_status()

    def clear_status(self) -> None:
        """Clear the status display."""
        pass

    def log_session_start(self, session_id: str, prompt: str) -> None:
        console.print(f"[green]Session started: {session_id[:8]}[/green]")

    def log_session_complete(self, session_id: str, duration_ms: int) -> None:
        sec = duration_ms / 1000.0
        console.print(f"[green]Session completed: {session_id[:8]} ({sec:.1f}s)[/green]")

    def log_session_failed(self, session_id: str, error: str) -> None:
        console.print(f"[red]Session failed: {session_id[:8]}: {error}[/red]")

    def log_reconnected(self, disconnected_ms: float) -> None:
        sec = disconnected_ms / 1000.0
        console.print(f"[green]Reconnected after {sec:.1f}s[/green]")

    def log_status(self, message: str) -> None:
        console.print(f"[dim]{message}[/dim]")

    def log_verbose(self, message: str) -> None:
        logger.debug("[bridge:status] %s", message)

    def log_error(self, message: str) -> None:
        console.print(f"[red]{message}[/red]")

    def set_debug_log_path(self, path: str) -> None:
        pass
