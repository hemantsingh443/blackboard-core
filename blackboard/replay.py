"""
Session Recording and Replay System

Enables deterministic debugging of orchestrator sessions by recording
all events and LLM responses, then replaying them without API calls.

Example:
    from blackboard.replay import SessionRecorder, ReplayOrchestrator
    
    # Record a session
    recorder = SessionRecorder()
    orchestrator = Orchestrator(llm=llm, workers=workers, event_bus=recorder.event_bus)
    result = await orchestrator.run(goal="Write a poem")
    recorder.save("session.json")
    
    # Replay the session
    replay = ReplayOrchestrator.from_file("session.json", workers=workers)
    result = await replay.run()  # No API calls, deterministic output
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from .events import Event, EventBus, EventType
from .core import LLMClient, LLMResponse, LLMUsage

if TYPE_CHECKING:
    from .state import Blackboard
    from .core import Orchestrator

logger = logging.getLogger("blackboard.replay")


# =============================================================================
# Session Recording
# =============================================================================

@dataclass
class RecordedLLMCall:
    """A recorded LLM call for replay."""
    prompt: str
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedLLMCall":
        return cls(
            prompt=data["prompt"],
            response=data["response"],
            model=data.get("model", "unknown"),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class RecordedSession:
    """A complete recorded session."""
    goal: str
    events: List[Dict[str, Any]]
    llm_calls: List[RecordedLLMCall]
    final_status: str
    start_time: str
    end_time: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "events": self.events,
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "final_status": self.final_status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedSession":
        return cls(
            goal=data["goal"],
            events=data["events"],
            llm_calls=[RecordedLLMCall.from_dict(c) for c in data.get("llm_calls", [])],
            final_status=data.get("final_status", "unknown"),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            metadata=data.get("metadata", {})
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save session to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Session saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "RecordedSession":
        """Load session from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class SessionRecorder:
    """
    Records all orchestrator events and LLM calls for replay.
    
    Subscribes to the EventBus and captures everything needed to
    reproduce a session without making API calls.
    
    Example:
        recorder = SessionRecorder()
        
        # Option 1: Use recorder's event bus
        orchestrator = Orchestrator(..., event_bus=recorder.event_bus)
        
        # Option 2: Subscribe to existing bus
        recorder.attach(orchestrator.event_bus)
        
        result = await orchestrator.run(goal="...")
        recorder.save("session.json")
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or EventBus()
        self._events: List[Dict[str, Any]] = []
        self._llm_calls: List[RecordedLLMCall] = []
        self._goal: str = ""
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._final_status: str = "unknown"
        self._metadata: Dict[str, Any] = {}
        
        # Subscribe to all events
        self.event_bus.subscribe_all(self._on_event)
    
    def attach(self, event_bus: EventBus) -> None:
        """Attach to an existing event bus."""
        event_bus.subscribe_all(self._on_event)
    
    def _on_event(self, event: Event) -> None:
        """Record an event."""
        # Track session lifecycle
        if event.type == EventType.ORCHESTRATOR_STARTED:
            self._start_time = event.timestamp
            self._goal = event.data.get("goal", "")
        elif event.type == EventType.ORCHESTRATOR_COMPLETED:
            self._end_time = event.timestamp
            self._final_status = event.data.get("status", "unknown")
        
        # Store event
        self._events.append(event.to_dict())
    
    def record_llm_call(
        self,
        prompt: str,
        response: str,
        model: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        **metadata
    ) -> None:
        """Record an LLM call for replay."""
        self._llm_calls.append(RecordedLLMCall(
            prompt=prompt,
            response=response,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        ))
    
    def get_session(self) -> RecordedSession:
        """Get the recorded session."""
        return RecordedSession(
            goal=self._goal,
            events=self._events.copy(),
            llm_calls=self._llm_calls.copy(),
            final_status=self._final_status,
            start_time=self._start_time.isoformat() if self._start_time else "",
            end_time=self._end_time.isoformat() if self._end_time else "",
            metadata=self._metadata
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save recorded session to file."""
        self.get_session().save(path)
    
    def reset(self) -> None:
        """Clear all recorded data."""
        self._events.clear()
        self._llm_calls.clear()
        self._goal = ""
        self._start_time = None
        self._end_time = None
        self._final_status = "unknown"
        self._metadata.clear()


# =============================================================================
# Recording LLM Client Wrapper
# =============================================================================

class RecordingLLMClient(LLMClient):
    """
    LLM client wrapper that records all calls.
    
    Wraps any LLMClient to capture prompts and responses for replay.
    
    Example:
        original_llm = LiteLLMClient(model="gpt-4o")
        recorder = SessionRecorder()
        recording_llm = RecordingLLMClient(original_llm, recorder)
        
        orchestrator = Orchestrator(llm=recording_llm, ...)
    """
    
    def __init__(self, llm: LLMClient, recorder: SessionRecorder):
        self._llm = llm
        self._recorder = recorder
    
    def generate(self, prompt: str) -> LLMResponse:
        """Generate and record."""
        response = self._llm.generate(prompt)
        self._recorder.record_llm_call(
            prompt=prompt,
            response=response.content,
            model=response.usage.model if response.usage else "unknown",
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0
        )
        return response
    
    async def agenerate(self, prompt: str) -> LLMResponse:
        """Async generate and record."""
        response = await self._llm.agenerate(prompt)
        self._recorder.record_llm_call(
            prompt=prompt,
            response=response.content,
            model=response.usage.model if response.usage else "unknown",
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0
        )
        return response


# =============================================================================
# Replay LLM Client
# =============================================================================

class ReplayLLMClient(LLMClient):
    """
    LLM client that replays recorded responses.
    
    Returns recorded responses in order, enabling deterministic replay
    without making any API calls.
    
    Example:
        session = RecordedSession.load("session.json")
        replay_llm = ReplayLLMClient(session.llm_calls)
        
        orchestrator = Orchestrator(llm=replay_llm, workers=workers)
        result = await orchestrator.run(goal=session.goal)  # No API calls!
    """
    
    def __init__(
        self,
        recorded_calls: List[RecordedLLMCall],
        strict: bool = True,
        on_mismatch: Optional[Callable[[str, str], None]] = None
    ):
        self._calls = recorded_calls
        self._index = 0
        self._strict = strict
        self._on_mismatch = on_mismatch
    
    @property
    def calls_remaining(self) -> int:
        """Number of recorded calls remaining."""
        return len(self._calls) - self._index
    
    def _get_next_response(self, prompt: str) -> LLMResponse:
        """Get the next recorded response."""
        if self._index >= len(self._calls):
            if self._strict:
                raise RuntimeError(
                    f"Replay exhausted: no more recorded calls "
                    f"(requested: {len(self._calls) + 1}, available: {len(self._calls)})"
                )
            logger.warning("Replay exhausted, returning empty response")
            return LLMResponse(content="", usage=None)
        
        recorded = self._calls[self._index]
        self._index += 1
        
        # Optionally check prompt matches
        if self._on_mismatch and prompt != recorded.prompt:
            self._on_mismatch(recorded.prompt, prompt)
        
        return LLMResponse(
            content=recorded.response,
            usage=LLMUsage(
                input_tokens=recorded.input_tokens,
                output_tokens=recorded.output_tokens,
                model=recorded.model
            ),
            metadata={"replayed": True, "original_timestamp": recorded.timestamp}
        )
    
    def generate(self, prompt: str) -> LLMResponse:
        """Return next recorded response."""
        return self._get_next_response(prompt)
    
    async def agenerate(self, prompt: str) -> LLMResponse:
        """Return next recorded response (async)."""
        return self._get_next_response(prompt)
    
    def reset(self) -> None:
        """Reset to beginning of recorded calls."""
        self._index = 0


# =============================================================================
# Replay Orchestrator
# =============================================================================

class ReplayOrchestrator:
    """
    High-level replay interface for recorded sessions.
    
    Provides a simple way to replay a session with all the original
    workers but without making API calls.
    
    Example:
        # Load and replay
        replay = ReplayOrchestrator.from_file("session.json", workers=workers)
        result = await replay.run()
        
        # Compare with original
        print(f"Status: {result.status}")
        print(f"Artifacts: {len(result.artifacts)}")
    """
    
    def __init__(
        self,
        session: RecordedSession,
        workers: List[Any],
        verbose: bool = False,
        on_prompt_mismatch: Optional[Callable[[str, str], None]] = None
    ):
        self.session = session
        self.workers = workers
        self.verbose = verbose
        self._on_prompt_mismatch = on_prompt_mismatch
        
        # Create replay LLM
        self._llm = ReplayLLMClient(
            session.llm_calls,
            strict=True,
            on_mismatch=on_prompt_mismatch
        )
    
    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        workers: List[Any],
        **kwargs
    ) -> "ReplayOrchestrator":
        """Create replay orchestrator from saved session."""
        session = RecordedSession.load(path)
        return cls(session, workers, **kwargs)
    
    async def run(self, max_steps: Optional[int] = None) -> "Blackboard":
        """
        Replay the recorded session.
        
        Returns:
            The final Blackboard state (should match original)
        """
        from .core import Orchestrator
        
        # Create orchestrator with replay LLM
        orchestrator = Orchestrator(
            llm=self._llm,
            workers=self.workers,
            verbose=self.verbose
        )
        
        # Run with original goal
        steps = max_steps or 100  # High default since we have recorded calls
        result = await orchestrator.run(goal=self.session.goal, max_steps=steps)
        
        return result
    
    def get_call_at_step(self, step: int) -> Optional[RecordedLLMCall]:
        """Get the LLM call at a specific step (0-indexed)."""
        if 0 <= step < len(self.session.llm_calls):
            return self.session.llm_calls[step]
        return None
    
    def get_event_timeline(self) -> List[Dict[str, Any]]:
        """Get all events in chronological order."""
        return sorted(self.session.events, key=lambda e: e.get("timestamp", ""))


# =============================================================================
# Session Comparison
# =============================================================================

@dataclass
class SessionDiff:
    """Differences between two sessions."""
    status_match: bool
    artifact_count_match: bool
    step_count_match: bool
    differences: List[str] = field(default_factory=list)
    
    @property
    def is_identical(self) -> bool:
        return self.status_match and self.artifact_count_match and self.step_count_match


def compare_sessions(
    original: RecordedSession,
    replayed: "Blackboard"
) -> SessionDiff:
    """
    Compare an original session with a replayed result.
    
    Useful for validating that replay produces identical output.
    """
    differences = []
    
    # Count events by type in original
    original_steps = len([e for e in original.events if e.get("type") == "step_completed"])
    original_artifacts = len([e for e in original.events if e.get("type") == "artifact_created"])
    
    # Compare status
    status_match = replayed.status.value == original.final_status
    if not status_match:
        differences.append(f"Status: {original.final_status} -> {replayed.status.value}")
    
    # Compare artifact count
    artifact_count_match = len(replayed.artifacts) == original_artifacts
    if not artifact_count_match:
        differences.append(f"Artifacts: {original_artifacts} -> {len(replayed.artifacts)}")
    
    # Compare step count
    step_count_match = replayed.step_count == original_steps
    if not step_count_match:
        differences.append(f"Steps: {original_steps} -> {replayed.step_count}")
    
    return SessionDiff(
        status_match=status_match,
        artifact_count_match=artifact_count_match,
        step_count_match=step_count_match,
        differences=differences
    )
