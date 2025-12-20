"""
Session Manager

Manages active orchestrator sessions for the API server.

.. warning::
    The API is STATEFUL. Sessions are stored in memory by default.
    If the server restarts (deployment, crash), all active and paused
    sessions will be lost unless you configure a sessions_dir.
    
    For production deployments with HumanProxyWorker pause/resume flows,
    configure sessions_dir for auto-save/restore:
    
        manager = SessionManager(
            factory, 
            sessions_dir="./api_sessions"
        )
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Protocol, runtime_checkable
from uuid import uuid4

from blackboard.state import Blackboard, Status
from blackboard.events import Event, EventBus, EventType

logger = logging.getLogger("blackboard.serve.manager")


class RunStatus(str, Enum):
    """Status of a run session."""
    PENDING = "pending"      # Created but not started
    RUNNING = "running"      # Currently executing
    PAUSED = "paused"        # Paused waiting for input
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Finished with failure
    CANCELLED = "cancelled"  # Cancelled by user


@dataclass
class RunSession:
    """
    Represents a single run session.
    
    Tracks the state, events, and execution context for one
    orchestration run.
    """
    id: str
    goal: str
    status: RunStatus = RunStatus.PENDING
    state: Optional[Blackboard] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    max_events: int = 1000
    
    # Execution context
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    
    def add_event(self, event: Event) -> None:
        """Add an event to the session history."""
        event_dict = event.to_dict()
        self.events.append(event_dict)
        
        # Ring buffer
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Also put in queue for streaming
        try:
            self._event_queue.put_nowait(event_dict)
        except asyncio.QueueFull:
            pass  # Drop if queue is full
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "event_count": len(self.events),
        }
        
        if self.error:
            result["error"] = self.error
        
        if self.state:
            result["step_count"] = self.state.step_count
            result["artifacts_count"] = len(self.state.artifacts)
            result["feedback_count"] = len(self.state.feedback)
            
            if self.state.pending_input:
                result["pending_input"] = self.state.pending_input
            
            # Include last artifact preview
            last_artifact = self.state.get_last_artifact()
            if last_artifact:
                content = str(last_artifact.content)
                result["last_artifact"] = {
                    "id": last_artifact.id,
                    "type": last_artifact.type,
                    "creator": last_artifact.creator,
                    "preview": content[:500] + "..." if len(content) > 500 else content
                }
        
        return result
    
    def to_full_dict(self) -> Dict[str, Any]:
        """Convert to full API response with state details."""
        result = self.to_dict()
        
        if self.state:
            result["state"] = {
                "goal": self.state.goal,
                "status": self.state.status.value,
                "step_count": self.state.step_count,
                "artifacts": [
                    {
                        "id": a.id,
                        "type": a.type,
                        "creator": a.creator,
                        "content": str(a.content),
                        "version": a.version,
                        "created_at": a.created_at.isoformat(),
                        "metadata": a.metadata
                    }
                    for a in self.state.artifacts
                ],
                "feedback": [
                    {
                        "id": f.id,
                        "source": f.source,
                        "critique": f.critique,
                        "passed": f.passed,
                        "created_at": f.created_at.isoformat()
                    }
                    for f in self.state.feedback
                ],
                "pending_input": self.state.pending_input,
                "context_summary": self.state.context_summary
            }
        
        return result


@runtime_checkable
class PersistenceLayer(Protocol):
    """Protocol for session persistence backends."""
    async def save(self, state: Blackboard, session_id: str) -> None: ...
    async def load(self, session_id: str) -> Optional[Blackboard]: ...
    async def delete(self, session_id: str) -> None: ...
    async def list_sessions(self) -> List[str]: ...


class SessionManager:
    """
    Manages multiple run sessions.
    
    Handles session lifecycle, event routing, and background execution.
    
    .. warning::
        By default, sessions are stored IN MEMORY ONLY. If the server
        restarts, all sessions are lost. For production use with
        pause/resume flows (HumanProxyWorker), configure a persistence layer.
    
    Args:
        orchestrator_factory: Callable that creates an Orchestrator instance
        max_sessions: Maximum number of concurrent sessions
        session_ttl: Time-to-live for completed sessions in seconds
        persistence: Optional persistence layer for durable session storage
        sessions_dir: Directory for auto-saving session state (default: ./api_sessions)
        
    Example:
        def create_orchestrator():
            return Orchestrator(llm=my_llm, workers=[...])
        
        # Basic (in-memory only - sessions lost on restart)
        manager = SessionManager(create_orchestrator)
        
        # With file-based auto-save (survives restarts)
        manager = SessionManager(
            create_orchestrator,
            sessions_dir="./api_sessions"
        )
        
        session = await manager.create_run("Write a haiku")
    """
    
    def __init__(
        self,
        orchestrator_factory: Callable,
        max_sessions: int = 100,
        session_ttl: float = 3600.0,  # 1 hour
        persistence: Optional[PersistenceLayer] = None,
        sessions_dir: Optional[str] = None
    ):
        self.orchestrator_factory = orchestrator_factory
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self.persistence = persistence
        self.sessions_dir = Path(sessions_dir) if sessions_dir else None
        self._sessions: Dict[str, RunSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Create sessions directory if specified
        if self.sessions_dir:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Session auto-save enabled: {self.sessions_dir}")
    
    async def start(self) -> None:
        """Start the session manager (cleanup task)."""
        # Restore sessions from disk if sessions_dir is configured
        if self.sessions_dir:
            await self._restore_sessions()
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SessionManager started")
    
    async def _restore_sessions(self) -> None:
        """Restore paused sessions from disk."""
        if not self.sessions_dir or not self.sessions_dir.exists():
            return
        
        restored = 0
        for meta_file in self.sessions_dir.glob("*_meta.json"):
            try:
                session_id = meta_file.stem.replace("_meta", "")
                state_file = self.sessions_dir / f"{session_id}_state.json"
                
                # Load metadata
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                
                # Load state if exists
                state = None
                if state_file.exists():
                    state = Blackboard.load_from_json(str(state_file))
                
                # Recreate session
                session = RunSession(
                    id=session_id,
                    goal=meta.get("goal", ""),
                    status=RunStatus(meta.get("status", "pending")),
                    state=state,
                    created_at=datetime.fromisoformat(meta["created_at"]),
                    started_at=datetime.fromisoformat(meta["started_at"]) if meta.get("started_at") else None,
                    error=meta.get("error")
                )
                
                self._sessions[session_id] = session
                restored += 1
                
            except Exception as e:
                logger.warning(f"Failed to restore session from {meta_file}: {e}")
        
        if restored:
            logger.info(f"Restored {restored} sessions from {self.sessions_dir}")
    
    async def stop(self) -> None:
        """Stop the session manager and cancel all sessions."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running sessions
        for session in self._sessions.values():
            if session._task and not session._task.done():
                session._task.cancel()
        
        logger.info("SessionManager stopped")
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up old sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_old_sessions(self) -> None:
        """Remove sessions that have exceeded TTL."""
        now = datetime.now()
        to_remove = []
        
        for session_id, session in self._sessions.items():
            if session.completed_at:
                age = (now - session.completed_at).total_seconds()
                if age > self.session_ttl:
                    to_remove.append(session_id)
        
        for session_id in to_remove:
            del self._sessions[session_id]
            logger.debug(f"Cleaned up session {session_id}")
    
    def get_session(self, session_id: str) -> Optional[RunSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[RunSession]:
        """List sessions with pagination."""
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[offset:offset + limit]
    
    async def create_run(
        self,
        goal: str,
        max_steps: int = 20,
        start_immediately: bool = True
    ) -> RunSession:
        """
        Create a new run session.
        
        Args:
            goal: The objective for the orchestrator
            max_steps: Maximum steps for execution
            start_immediately: Whether to start execution right away
            
        Returns:
            The created RunSession
        """
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum number of sessions ({self.max_sessions}) reached")
        
        session_id = str(uuid4())
        session = RunSession(
            id=session_id,
            goal=goal
        )
        
        self._sessions[session_id] = session
        logger.info(f"Created session {session_id}: {goal[:50]}...")
        
        # Save session metadata
        await self._save_session_metadata(session)
        
        if start_immediately:
            await self._start_session(session, max_steps)
        
        return session
    
    async def _save_session_metadata(self, session: RunSession) -> None:
        """Save session metadata to disk for recovery."""
        if not self.sessions_dir:
            return
        
        try:
            meta_file = self.sessions_dir / f"{session.id}_meta.json"
            meta = {
                "id": session.id,
                "goal": session.goal,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "error": session.error
            }
            with open(meta_file, "w") as f:
                json.dump(meta, f)
        except Exception as e:
            logger.warning(f"Failed to save session metadata: {e}")
    
    async def _save_session_state(self, session: RunSession) -> None:
        """Save session state to disk."""
        if not self.sessions_dir or not session.state:
            return
        
        try:
            state_file = self.sessions_dir / f"{session.id}_state.json"
            session.state.save_to_json(str(state_file))
        except Exception as e:
            logger.warning(f"Failed to save session state: {e}")
    
    async def _start_session(self, session: RunSession, max_steps: int = 20) -> None:
        """Start execution for a session."""
        session.status = RunStatus.RUNNING
        session.started_at = datetime.now()
        
        # Create orchestrator
        orchestrator = self.orchestrator_factory()
        
        # Subscribe to events
        async def on_event(event: Event):
            session.add_event(event)
            
            # Update session status based on event
            if event.type == EventType.ORCHESTRATOR_PAUSED:
                session.status = RunStatus.PAUSED
        
        orchestrator.event_bus.subscribe_all_async(on_event)
        
        # Run in background task
        async def run_task():
            try:
                state = await orchestrator.run(goal=session.goal, max_steps=max_steps)
                session.state = state
                
                if state.status == Status.DONE:
                    session.status = RunStatus.COMPLETED
                elif state.status == Status.PAUSED:
                    session.status = RunStatus.PAUSED
                else:
                    session.status = RunStatus.FAILED
                    
            except asyncio.CancelledError:
                session.status = RunStatus.CANCELLED
                session.error = "Cancelled"
            except Exception as e:
                logger.error(f"Session {session.id} error: {e}")
                session.status = RunStatus.FAILED
                session.error = str(e)
            finally:
                if session.status != RunStatus.PAUSED:
                    session.completed_at = datetime.now()
                # Auto-save on pause or completion
                await self._save_session_metadata(session)
                if session.state:
                    await self._save_session_state(session)
        
        session._task = asyncio.create_task(run_task())
    
    async def resume_run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        max_steps: int = 20
    ) -> RunSession:
        """
        Resume a paused session with input.
        
        Args:
            session_id: The session to resume
            input_data: Data to inject (e.g., {"answer": "user's response"})
            max_steps: Maximum additional steps
            
        Returns:
            The updated session
        """
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session not found: {session_id}")
        
        if session.status != RunStatus.PAUSED:
            raise ValueError(f"Session is not paused (status: {session.status.value})")
        
        if not session.state:
            raise ValueError("Session has no state to resume")
        
        # Inject input
        if session.state.pending_input:
            session.state.pending_input.update(input_data)
        else:
            session.state.pending_input = input_data
        
        # Reset status to continue
        session.state.update_status(Status.GENERATING)
        
        logger.info(f"Resuming session {session_id}")
        
        # Restart execution
        session.completed_at = None
        await self._start_session_resume(session, max_steps)
        
        return session
    
    async def _start_session_resume(self, session: RunSession, max_steps: int = 20) -> None:
        """Resume execution for a paused session."""
        session.status = RunStatus.RUNNING
        
        # Create orchestrator
        orchestrator = self.orchestrator_factory()
        
        # Subscribe to events
        async def on_event(event: Event):
            session.add_event(event)
            if event.type == EventType.ORCHESTRATOR_PAUSED:
                session.status = RunStatus.PAUSED
        
        orchestrator.event_bus.subscribe_all_async(on_event)
        
        # Run with existing state
        async def run_task():
            try:
                state = await orchestrator.run(state=session.state, max_steps=max_steps)
                session.state = state
                
                if state.status == Status.DONE:
                    session.status = RunStatus.COMPLETED
                elif state.status == Status.PAUSED:
                    session.status = RunStatus.PAUSED
                else:
                    session.status = RunStatus.FAILED
                    
            except asyncio.CancelledError:
                session.status = RunStatus.CANCELLED
                session.error = "Cancelled"
            except Exception as e:
                logger.error(f"Session {session.id} error: {e}")
                session.status = RunStatus.FAILED
                session.error = str(e)
            finally:
                if session.status != RunStatus.PAUSED:
                    session.completed_at = datetime.now()
                # Auto-save on pause or completion
                await self._save_session_metadata(session)
                if session.state:
                    await self._save_session_state(session)
        
        session._task = asyncio.create_task(run_task())
    
    async def cancel_run(self, session_id: str) -> RunSession:
        """Cancel a running or paused session."""
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session not found: {session_id}")
        
        if session._task and not session._task.done():
            session._task.cancel()
            try:
                await session._task
            except asyncio.CancelledError:
                pass
        
        session.status = RunStatus.CANCELLED
        session.completed_at = datetime.now()
        
        logger.info(f"Cancelled session {session_id}")
        return session
