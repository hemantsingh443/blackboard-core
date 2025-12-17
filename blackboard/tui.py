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
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.layout import Layout
            self._rich_available = True
        except ImportError:
            self._rich_available = False
            logger.warning("rich not installed. Install with: pip install 'blackboard-core[tui]'")
            return
        
        self.console = Console()
        self.event_bus = event_bus
        self.show_artifacts = show_artifacts
        self.show_reasoning = show_reasoning
        self.max_content_length = max_content_length
        
        self._current_state: Optional["Blackboard"] = None
        self._current_decision: Optional["SupervisorDecision"] = None
        self._live: Optional[Live] = None
        self._step_count = 0
        
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
        self._step_count = event.data.get("step", 0)
        self._refresh()
    
    def _on_step_completed(self, event) -> None:
        self._refresh()
    
    def _on_worker_called(self, event) -> None:
        self._refresh()
    
    def _on_worker_completed(self, event) -> None:
        self._refresh()
    
    def _on_artifact_created(self, event) -> None:
        self._refresh()
    
    def _on_completed(self, event) -> None:
        self._refresh()
    
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
        
        # Header with goal and status
        header = Text()
        header.append(f"Goal: ", style="bold")
        header.append(state.goal[:80] + "..." if len(state.goal) > 80 else state.goal)
        header.append(f"\nStatus: ", style="bold")
        header.append(state.status.value, style=self._status_style(state.status.value))
        header.append(f" | Step: {state.step_count}")
        
        # Artifacts table
        artifacts_table = Table(title="📄 Artifacts", show_header=True, header_style="bold magenta")
        artifacts_table.add_column("Type", style="cyan", width=12)
        artifacts_table.add_column("Creator", style="green", width=15)
        artifacts_table.add_column("Content", style="white")
        
        if self.show_artifacts:
            for artifact in state.artifacts[-3:]:  # Last 3
                content = str(artifact.content)
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "..."
                artifacts_table.add_row(artifact.type, artifact.creator, content)
        
        # Feedback table
        feedback_table = Table(title="💬 Feedback", show_header=True, header_style="bold yellow")
        feedback_table.add_column("Source", style="green", width=15)
        feedback_table.add_column("Passed", style="white", width=8)
        feedback_table.add_column("Critique", style="white")
        
        for fb in state.feedback[-3:]:  # Last 3
            passed_str = "✅" if fb.passed else "❌"
            critique = fb.critique[:50] + "..." if len(fb.critique) > 50 else fb.critique
            feedback_table.add_row(fb.source, passed_str, critique)
        
        # Combine into panel
        content = Group(header, "", artifacts_table, "", feedback_table)
        
        return Panel(
            content,
            title=f"[bold blue]🔲 Blackboard[/bold blue]",
            border_style="blue"
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
            refresh_per_second=4
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
    import asyncio
    
    tui = BlackboardTUI(orchestrator.event_bus)
    
    async def run_with_tui():
        with tui.live():
            result = await orchestrator.run(goal=goal, **kwargs)
            tui.update_state(result)
        tui.print_summary(result)
        return result
    
    return asyncio.run(run_with_tui())
