"""
Blackboard Interactive TUI (Textual-based)

A full-featured terminal application for real-time agent debugging and intervention.

Features:
- Live 3-pane dashboard: Log, Artifacts, State
- Pause/Resume execution with Space
- Intervention mode: inject commands with I
- Human inbox for HumanProxyWorker

Usage:
    from blackboard.ui import BlackboardTUI
    
    app = BlackboardTUI(orchestrator)
    await app.run_async()

Requirements:
    pip install textual>=0.90.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Any, Dict, List

logger = logging.getLogger("blackboard.ui.textual")

# Guard against missing textual
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, Container
    from textual.widgets import Header, Footer, RichLog, Tree, Static, Input, Button, Label
    from textual.screen import ModalScreen
    from textual.binding import Binding
    from textual.reactive import reactive
    from textual import events
    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False
    App = object  # Stub for type hints

if TYPE_CHECKING:
    from blackboard.core import Orchestrator
    from blackboard.state import Blackboard


# =============================================================================
# Widgets
# =============================================================================

if _HAS_TEXTUAL:
    
    class LogPanel(RichLog):
        """Scrolling execution log panel."""
        
        BORDER_TITLE = "Execution Log"
        
        def __init__(self, **kwargs):
            super().__init__(highlight=True, markup=True, **kwargs)
        
        def add_event(self, event_type: str, message: str, timestamp: Optional[str] = None):
            """Add an event to the log."""
            ts = timestamp or datetime.now().strftime("%H:%M:%S")
            
            # Color code by event type
            colors = {
                "step": "cyan",
                "worker": "green", 
                "artifact": "blue",
                "feedback": "yellow",
                "error": "red",
                "pause": "orange3",
                "resume": "green",
                "done": "green bold",
            }
            color = colors.get(event_type, "white")
            
            self.write(f"[dim]{ts}[/dim] [{color}]{event_type.upper():>10}[/{color}] {message}")
    
    
    class ArtifactTree(Tree):
        """Tree view of artifacts."""
        
        BORDER_TITLE = "Artifacts"
        
        def __init__(self, **kwargs):
            super().__init__("📦 Artifacts", **kwargs)
            self.root.expand()
            self._artifacts: List[Dict] = []
        
        def update_artifacts(self, artifacts: List[Dict]):
            """Update the artifact tree."""
            self._artifacts = artifacts
            self.clear()
            self.root.expand()
            
            # Group by type
            by_type: Dict[str, List[Dict]] = {}
            for art in artifacts:
                art_type = art.get("type", "unknown")
                if art_type not in by_type:
                    by_type[art_type] = []
                by_type[art_type].append(art)
            
            for art_type, items in by_type.items():
                type_node = self.root.add(f"📁 {art_type} ({len(items)})")
                for item in items[-5:]:  # Show last 5 of each type
                    creator = item.get("creator", "?")
                    preview = str(item.get("content", ""))[:50]
                    type_node.add_leaf(f"[dim]{creator}:[/dim] {preview}...")
    
    
    class StatePanel(Static):
        """JSON state viewer."""
        
        BORDER_TITLE = "State"
        
        def __init__(self, **kwargs):
            super().__init__("", **kwargs)
        
        def update_state(self, state: "Blackboard"):
            """Update state display."""
            content = f"""[bold]Goal:[/bold] {state.goal[:80]}...
[bold]Status:[/bold] {state.status.value if state.status else 'N/A'}
[bold]Step:[/bold] {state.step_count}
[bold]Artifacts:[/bold] {len(state.artifacts)}
[bold]Feedback:[/bold] {len(state.feedback)}
"""
            self.update(content)
    
    
    class StatusBar(Static):
        """Bottom status bar."""
        
        paused = reactive(False)
        
        def __init__(self, **kwargs):
            super().__init__("", **kwargs)
        
        def watch_paused(self, paused: bool):
            """React to pause state changes."""
            if paused:
                self.update("[bold yellow]⏸ PAUSED[/bold yellow] - Press SPACE to resume, I to intervene")
            else:
                self.update("[bold green]▶ RUNNING[/bold green] - Press SPACE to pause")
    
    
    class InterventionModal(ModalScreen):
        """Modal for injecting commands."""
        
        BINDINGS = [
            ("escape", "cancel", "Cancel"),
        ]
        
        def compose(self) -> ComposeResult:
            yield Vertical(
                Label("Inject Command", id="modal-title"),
                Input(placeholder="Enter instructions for the next step...", id="intervention-input"),
                Horizontal(
                    Button("Inject", variant="primary", id="inject-btn"),
                    Button("Cancel", variant="default", id="cancel-btn"),
                    id="modal-buttons"
                ),
                id="intervention-modal"
            )
        
        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "inject-btn":
                input_widget = self.query_one("#intervention-input", Input)
                self.dismiss(input_widget.value)
            else:
                self.dismiss(None)
        
        def action_cancel(self):
            self.dismiss(None)


# =============================================================================
# Main Application
# =============================================================================

if _HAS_TEXTUAL:
    
    class BlackboardApp(App):
        """Interactive TUI for Blackboard orchestration."""
        
        CSS = """
        Screen {
            layout: grid;
            grid-size: 2 2;
            grid-columns: 2fr 1fr;
            grid-rows: 3fr 1fr;
        }
        
        LogPanel {
            column-span: 1;
            row-span: 2;
            border: solid green;
            padding: 0 1;
        }
        
        ArtifactTree {
            border: solid blue;
            padding: 0 1;
        }
        
        StatePanel {
            border: solid cyan;
            padding: 1;
        }
        
        StatusBar {
            dock: bottom;
            height: 1;
            background: $surface;
            padding: 0 1;
        }
        
        #intervention-modal {
            width: 60;
            height: 12;
            border: thick $primary;
            background: $surface;
            padding: 1 2;
        }
        
        #modal-title {
            text-align: center;
            text-style: bold;
            margin-bottom: 1;
        }
        
        #modal-buttons {
            margin-top: 1;
            align: center middle;
        }
        """
        
        BINDINGS = [
            Binding("space", "toggle_pause", "Pause/Resume", show=True),
            Binding("i", "intervene", "Intervene", show=True),
            Binding("q", "quit", "Quit", show=True),
            Binding("r", "refresh", "Refresh", show=False),
        ]
        
        paused = reactive(False)
        
        def __init__(
            self,
            orchestrator: Optional["Orchestrator"] = None,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.orchestrator = orchestrator
            self._pause_event = asyncio.Event()
            self._pause_event.set()  # Start unpaused
            self._intervention_command: Optional[str] = None
            self._current_state: Optional["Blackboard"] = None
        
        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield LogPanel(id="log")
            yield ArtifactTree(id="artifacts")
            yield StatePanel(id="state")
            yield StatusBar(id="status")
            yield Footer()
        
        def on_mount(self) -> None:
            """Called when app is mounted."""
            self.log_panel.add_event("start", "Blackboard TUI initialized")
            if self.orchestrator:
                self.log_panel.add_event("info", f"Orchestrator attached with {len(self.orchestrator.workers)} workers")
        
        @property
        def log_panel(self) -> LogPanel:
            return self.query_one("#log", LogPanel)
        
        @property
        def artifact_tree(self) -> ArtifactTree:
            return self.query_one("#artifacts", ArtifactTree)
        
        @property
        def state_panel(self) -> StatePanel:
            return self.query_one("#state", StatePanel)
        
        @property
        def status_bar(self) -> StatusBar:
            return self.query_one("#status", StatusBar)
        
        def action_toggle_pause(self) -> None:
            """Toggle pause state."""
            self.paused = not self.paused
            self.status_bar.paused = self.paused
            
            if self.paused:
                self._pause_event.clear()
                self.log_panel.add_event("pause", "Execution paused by user")
            else:
                self._pause_event.set()
                self.log_panel.add_event("resume", "Execution resumed")
        
        def action_intervene(self) -> None:
            """Open intervention modal."""
            if not self.paused:
                self.action_toggle_pause()
            
            def handle_intervention(command: Optional[str]) -> None:
                if command:
                    self._intervention_command = command
                    self.log_panel.add_event("intervention", f"Injected: {command[:50]}...")
                    self.action_toggle_pause()  # Resume after injection
            
            self.push_screen(InterventionModal(), handle_intervention)
        
        def action_refresh(self) -> None:
            """Refresh the display."""
            if self._current_state:
                self.update_state(self._current_state)
        
        # Public API for Orchestrator integration
        
        def update_state(self, state: "Blackboard") -> None:
            """Update the display with new state."""
            self._current_state = state
            
            self.state_panel.update_state(state)
            self.artifact_tree.update_artifacts([a.model_dump() for a in state.artifacts])
        
        def log_step(self, step: int, worker_name: str) -> None:
            """Log a step execution."""
            self.log_panel.add_event("step", f"Step {step}: {worker_name}")
        
        def log_artifact(self, artifact_type: str, creator: str) -> None:
            """Log artifact creation."""
            self.log_panel.add_event("artifact", f"{creator} created {artifact_type}")
        
        def log_feedback(self, worker: str, passed: bool) -> None:
            """Log feedback."""
            result = "✓ passed" if passed else "✗ failed"
            self.log_panel.add_event("feedback", f"{worker}: {result}")
        
        def log_error(self, message: str) -> None:
            """Log an error."""
            self.log_panel.add_event("error", message)
        
        def log_done(self, message: str = "Execution complete") -> None:
            """Log completion."""
            self.log_panel.add_event("done", message)
        
        async def wait_if_paused(self) -> Optional[str]:
            """Wait if paused, return any intervention command."""
            await self._pause_event.wait()
            cmd = self._intervention_command
            self._intervention_command = None
            return cmd
        
        def get_intervention(self) -> Optional[str]:
            """Get pending intervention command (non-blocking)."""
            cmd = self._intervention_command
            self._intervention_command = None
            return cmd


# =============================================================================
# Public API
# =============================================================================

def create_tui(orchestrator: Optional["Orchestrator"] = None) -> "BlackboardApp":
    """Create a Blackboard TUI application.
    
    Args:
        orchestrator: Optional Orchestrator instance
        
    Returns:
        BlackboardApp instance
        
    Raises:
        ImportError: If textual is not installed
    """
    if not _HAS_TEXTUAL:
        raise ImportError(
            "Textual is required for the interactive TUI. "
            "Install with: pip install blackboard-core[textual-tui]"
        )
    return BlackboardApp(orchestrator=orchestrator)


def is_headless() -> bool:
    """Check if running in headless/CI mode."""
    import os
    
    # Check for common CI environment variables
    ci_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "TRAVIS"]
    for var in ci_vars:
        if os.environ.get(var):
            return True
    
    # Check for explicit headless flag
    if os.environ.get("BLACKBOARD_HEADLESS", "").lower() in ("1", "true", "yes"):
        return True
    
    # Check if we have a TTY
    import sys
    if not sys.stdout.isatty():
        return True
    
    return False
