"""
Terminal UI for Blackboard Visualization

Real-time terminal visualization of blackboard state using rich.

Example:
    from blackboard import Orchestrator
    from blackboard.tui import BlackboardTUI
    from blackboard.events import EventBus
    
    event_bus = EventBus()
    tui = BlackboardTUI(event_bus)
    
    orchestrator = Orchestrator(llm=llm, workers=workers, event_bus=event_bus)
    
    # Run with live visualization
    with tui.live():
        await orchestrator.run(goal="Write a haiku")
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Blackboard
    from .core import SupervisorDecision

logger = logging.getLogger("blackboard.tui")


class BlackboardTUI:
    """
    Real-time terminal visualization of blackboard state.
    
    Uses the rich library to render colorful, updating displays
    of the orchestration progress.
    
    Args:
        event_bus: Event bus to subscribe to for updates
        show_artifacts: Whether to show artifact content
        show_reasoning: Whether to show supervisor reasoning
        max_content_length: Maximum length of content to display
        
    Example:
        from blackboard.events import EventBus
        from blackboard.tui import BlackboardTUI
        
        event_bus = EventBus()
        tui = BlackboardTUI(event_bus)
        
        # As context manager
        with tui.live():
            await orchestrator.run(goal="...")
    """
    
    def __init__(
        self,
        event_bus: Optional["EventBus"] = None,
        show_artifacts: bool = True,
        show_reasoning: bool = True,
        max_content_length: int = 200
    ):
        self.event_bus = event_bus
        self.show_artifacts = show_artifacts
        self.show_reasoning = show_reasoning
        self.max_content_length = max_content_length
        
        self._current_state: Optional["Blackboard"] = None
        self._current_decision: Optional["SupervisorDecision"] = None
        self._live: Optional[Live] = None
        self._step_count = 0
        
        # Enhanced tracking for dynamic updates
        self._current_worker: Optional[str] = None
        self._activity_log: list = []
        self._is_thinking: bool = False
        
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.layout import Layout
            from rich.spinner import Spinner
            self._rich_available = True
            self.console = Console()
        except ImportError:
            self._rich_available = False
            logger.warning("rich not installed. Install with: pip install 'blackboard-core[tui]'")
            return
        
        if event_bus:
            self._subscribe_events()
    
    def _subscribe_events(self) -> None:
        """Subscribe to relevant events."""
        from .events import EventType
        
        self.event_bus.subscribe(EventType.STEP_STARTED, self._on_step_started)
        self.event_bus.subscribe(EventType.STEP_COMPLETED, self._on_step_completed)
        self.event_bus.subscribe(EventType.WORKER_CALLED, self._on_worker_called)
        self.event_bus.subscribe(EventType.WORKER_COMPLETED, self._on_worker_completed)
        self.event_bus.subscribe(EventType.ARTIFACT_CREATED, self._on_artifact_created)
        self.event_bus.subscribe(EventType.ORCHESTRATOR_COMPLETED, self._on_completed)
    
    def _on_step_started(self, event) -> None:
        """Handle step started event."""
        self._step_count = event.data.get("step", 0)
        self._is_thinking = True
        self._add_activity(f"🧠 Step {self._step_count}: Supervisor thinking...")
        if "state" in event.data:
            self._current_state = event.data["state"]
        self._refresh()
    
    def _on_step_completed(self, event) -> None:
        """Handle step completed event."""
        self._is_thinking = False
        if "state" in event.data:
            self._current_state = event.data["state"]
        self._refresh()
    
    def _on_worker_called(self, event) -> None:
        """Handle worker called event."""
        worker_name = event.data.get("worker", "Unknown")
        self._current_worker = worker_name
        self._add_activity(f"⚡ Calling {worker_name}...")
        if "state" in event.data:
            self._current_state = event.data["state"]
        self._refresh()
    
    def _on_worker_completed(self, event) -> None:
        """Handle worker completed event."""
        worker_name = event.data.get("worker", self._current_worker or "Worker")
        
        # Check if this was a sub-agent with trace link
        if event.data.get("is_sub_agent"):
            trace_id = event.data.get("trace_id", "")
            self._add_activity(f"✓ {worker_name} completed [trace: {trace_id}]")
        else:
            self._add_activity(f"✓ {worker_name} completed")
        
        self._current_worker = None
        if "state" in event.data:
            self._current_state = event.data["state"]
        self._refresh()
    
    def _on_artifact_created(self, event) -> None:
        """Handle artifact created event."""
        artifact_type = event.data.get("artifact_type", "artifact")
        creator = event.data.get("creator", "Worker")
        self._add_activity(f"📄 New {artifact_type} from {creator}")
        if "state" in event.data:
            self._current_state = event.data["state"]
        self._refresh()
    
    def _on_completed(self, event) -> None:
        """Handle orchestrator completed event."""
        status = event.data.get("status", "done")
        self._add_activity(f"🏁 Completed: {status}")
        self._current_worker = None
        self._is_thinking = False
        if "state" in event.data:
            self._current_state = event.data["state"]
        self._refresh()
    
    def _add_activity(self, message: str) -> None:
        """Add an activity to the log (keep last 5)."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._activity_log.append(f"[dim]{timestamp}[/dim] {message}")
        self._activity_log = self._activity_log[-5:]  # Keep last 5
    
    def _refresh(self) -> None:
        """Refresh the display."""
        if self._live and self._current_state:
            self._live.update(self.render_state(self._current_state))
    
    def render_state(self, state: "Blackboard") -> "Panel":
        """
        Render the current state as a rich Panel.
        
        Args:
            state: Blackboard state to render
            
        Returns:
            Rich Panel with formatted state
        """
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.console import Group
        from rich.spinner import Spinner
        from rich.markdown import Markdown
        from rich import box
        
        components = []
        
        # Header with goal and status
        header = Text()
        header.append("Goal: ", style="bold cyan")
        goal_text = state.goal[:80] + "..." if len(state.goal) > 80 else state.goal
        header.append(goal_text)
        header.append("\n")
        header.append("Status: ", style="bold")
        header.append(state.status.value.upper(), style=f"bold {self._status_style(state.status.value)}")
        header.append(f" │ Step: {state.step_count}", style="dim")
        components.append(header)
        components.append(Text())  # Spacer
        
        # Activity log (shows real-time updates)
        if self._activity_log:
            activity = Text()
            activity.append("─── Activity ───\n", style="bold yellow")
            for log_entry in self._activity_log:
                activity.append_text(Text.from_markup(log_entry + "\n"))
            components.append(activity)
        
        # Current worker indicator with ANIMATED spinner
        if self._current_worker:
            spinner = Spinner("dots", text=Text.assemble(
                ("Running: ", "bold"),
                (self._current_worker, "bold green")
            ), style="cyan")
            components.append(spinner)
            components.append(Text())
        elif self._is_thinking:
            spinner = Spinner("dots", text=Text.assemble(
                ("Supervisor deciding next action...", "italic yellow")
            ), style="yellow")
            components.append(spinner)
            components.append(Text())
        
        # Artifacts - show as markdown panels for proper rendering
        if self.show_artifacts and state.artifacts:
            components.append(Text("📄 Artifacts", style="bold magenta"))
            components.append(Text())
            
            for artifact in state.artifacts[-2:]:  # Last 2 for space
                content = str(artifact.content)
                # Truncate very long content
                if len(content) > 400:
                    content = content[:400] + "\n\n*...content truncated...*"
                
                # Wrap long lines to prevent overflow
                lines = []
                for line in content.split('\n'):
                    # Wrap lines longer than 80 chars
                    while len(line) > 80:
                        lines.append(line[:80])
                        line = line[80:]
                    lines.append(line)
                content = '\n'.join(lines)
                
                # Render as markdown for proper formatting
                try:
                    md_content = Markdown(content)
                except:
                    md_content = Text(content[:300] + "...")
                
                artifact_panel = Panel(
                    md_content,
                    title=f"[cyan]{artifact.type}[/cyan] by [green]{artifact.creator}[/green]",
                    border_style="dim",
                    padding=(0, 1),
                    expand=False  # Don't expand, let content determine width
                )
                components.append(artifact_panel)
        
        # Feedback section
        if state.feedback:
            components.append(Text())
            components.append(Text("💬 Feedback", style="bold yellow"))
            
            for fb in state.feedback[-2:]:  # Last 2
                passed_icon = "✅" if fb.passed else "❌"
                passed_style = "green" if fb.passed else "red"
                
                fb_text = Text()
                fb_text.append(f"{passed_icon} ", style=passed_style)
                fb_text.append(f"{fb.source}: ", style="bold")
                
                # Show critique, truncated
                critique = fb.critique[:200] + "..." if len(fb.critique) > 200 else fb.critique
                critique = critique.replace("\n", " ")
                fb_text.append(critique, style="dim")
                
                components.append(fb_text)
        
        # Combine into panel
        content = Group(*components)
        
        # Dynamic border color based on state
        if self._current_worker:
            border_style = "green"
            title_icon = "⚡"
        elif self._is_thinking:
            border_style = "yellow"
            title_icon = "🧠"
        elif state.status.value == "done":
            border_style = "green"
            title_icon = "✅"
        elif state.status.value == "failed":
            border_style = "red"
            title_icon = "❌"
        else:
            border_style = "blue"
            title_icon = "🔲"
        
        return Panel(
            content,
            title=f"[bold {border_style}]{title_icon} Blackboard[/bold {border_style}]",
            border_style=border_style,
            padding=(1, 2),
            expand=True  # Dynamic width based on terminal size
        )

    
    def _status_style(self, status: str) -> str:
        """Get style for status."""
        styles = {
            "planning": "yellow",
            "generating": "cyan",
            "critiquing": "magenta",
            "refining": "blue",
            "paused": "yellow",
            "done": "green",
            "failed": "red"
        }
        return styles.get(status, "white")
    
    def live(self) -> "Live":
        """
        Create a live context for real-time updates.
        
        Returns:
            Rich Live context manager
            
        Example:
            with tui.live():
                await orchestrator.run(goal="...")
        """
        if not self._rich_available:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
        
        from rich.live import Live
        self._live = Live(
            self.render_state(self._create_empty_state()),
            console=self.console,
            refresh_per_second=10  # Smooth updates
        )
        return self._live
    
    def _create_empty_state(self) -> "Blackboard":
        """Create an empty state for initial render."""
        from .state import Blackboard
        return Blackboard(goal="Waiting...")
    
    def update_state(self, state: "Blackboard") -> None:
        """
        Update the displayed state.
        
        Args:
            state: New state to display
        """
        self._current_state = state
        self._refresh()
    
    def print_summary(self, state: "Blackboard") -> None:
        """
        Print a summary of the final state.
        
        Args:
            state: Final state to summarize
        """
        if not self._rich_available:
            print(f"Status: {state.status.value}")
            print(f"Steps: {state.step_count}")
            print(f"Artifacts: {len(state.artifacts)}")
            return
        
        from rich.panel import Panel
        from rich.text import Text
        
        text = Text()
        text.append(f"Status: ", style="bold")
        text.append(state.status.value, style=self._status_style(state.status.value))
        text.append(f"\nSteps: {state.step_count}")
        text.append(f"\nArtifacts: {len(state.artifacts)}")
        text.append(f"\nFeedback: {len(state.feedback)}")
        
        if state.artifacts:
            text.append("\n\nFinal Artifact:\n", style="bold")
            content = str(state.artifacts[-1].content)
            if len(content) > 500:
                content = content[:500] + "..."
            text.append(content)
        
        self.console.print(Panel(text, title="[bold green]✅ Complete[/bold green]"))


def watch(orchestrator, goal: str, **kwargs) -> "Blackboard":
    """
    Convenience function to run orchestrator with TUI visualization.
    
    Creates a shared Blackboard state that both the TUI and Orchestrator
    reference. Since Blackboard is mutable, the TUI sees live updates.
    
    Args:
        orchestrator: Orchestrator instance
        goal: Goal to accomplish
        **kwargs: Additional arguments for orchestrator.run()
        
    Returns:
        Final Blackboard state
        
    Example:
        from blackboard.tui import watch
        
        result = watch(orchestrator, goal="Write a poem")
    """
    from .state import Blackboard
    from .core import _run_sync
    
    # Create state BEFORE running - TUI holds reference to this mutable object
    state = Blackboard(goal=goal)
    
    # Create TUI and give it the shared state reference
    tui = BlackboardTUI(orchestrator.event_bus)
    tui.update_state(state)  # TUI now references the same object
    
    async def run_with_tui():
        with tui.live():
            # Pass the pre-created state to orchestrator
            # Orchestrator mutates this same object, TUI sees changes
            result = await orchestrator.run(state=state, **kwargs)
            tui.update_state(result)
        tui.print_summary(result)
        return result
    
    return _run_sync(run_with_tui())

