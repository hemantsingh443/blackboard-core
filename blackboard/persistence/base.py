"""
Persistence Layer Protocols and Base Classes

Defines the abstract interface for all persistence backends.
"""

import logging
from typing import Optional, Protocol, runtime_checkable, TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..state import Blackboard

logger = logging.getLogger("blackboard.persistence")


# =============================================================================
# Exceptions
# =============================================================================

class PersistenceError(Exception):
    """Base exception for persistence operations."""
    pass


class SessionNotFoundError(PersistenceError):
    """Raised when a session doesn't exist."""
    pass


class SessionConflictError(PersistenceError):
    """Raised when there's a version conflict during save (optimistic lock failure)."""
    pass


# =============================================================================
# Protocol
# =============================================================================

@runtime_checkable
class PersistenceLayer(Protocol):
    """
    Protocol for state persistence backends.
    
    Implementations should handle serialization, versioning, and atomic updates.
    All methods are async for compatibility with async backends (Redis, databases).
    
    Optimistic Locking:
        All implementations MUST check `state.version` on save. If the stored
        version is higher than the incoming version, raise `SessionConflictError`.
        This prevents lost updates from concurrent workers.
    
    Heartbeats:
        Backends MAY implement `update_heartbeat` for zombie session detection.
        If not implemented, the Orchestrator will skip heartbeat updates.
    
    Example:
        persistence = SQLitePersistence("./blackboard.db")
        await persistence.initialize()
        await persistence.save(state, session_id="user-123")
        state = await persistence.load(session_id="user-123")
    """
    
    async def save(
        self,
        state: "Blackboard",
        session_id: str,
        parent_session_id: Optional[str] = None
    ) -> None:
        """
        Save state to the backend.
        
        Args:
            state: Blackboard state to persist
            session_id: Unique identifier for this session
            parent_session_id: Optional parent session ID for fractal agents
            
        Raises:
            SessionConflictError: If version conflict detected (optimistic lock)
            PersistenceError: For other storage errors
        """
        ...
    
    async def load(self, session_id: str) -> "Blackboard":
        """
        Load state from the backend.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            The restored Blackboard state
            
        Raises:
            SessionNotFoundError: If session doesn't exist
            PersistenceError: For other storage errors
        """
        ...
    
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        ...
    
    async def delete(self, session_id: str) -> None:
        """Delete a session. No-op if doesn't exist."""
        ...
    
    async def list_sessions(self, parent_id: Optional[str] = None) -> List[str]:
        """
        List session IDs.
        
        Args:
            parent_id: If provided, list only child sessions of this parent
        """
        ...


# =============================================================================
# Optional Extensions (v2.0+)
# =============================================================================

class HeartbeatCapable(Protocol):
    """
    Extension protocol for backends that support active heartbeats.
    
    This enables zombie session detection by periodically updating
    a `heartbeat_at` timestamp while a session is actively running.
    """
    
    async def update_heartbeat(self, session_id: str) -> None:
        """
        Update the heartbeat timestamp for a running session.
        
        Called periodically (e.g., every 30 seconds) by the Orchestrator
        to signal that the process is still alive.
        """
        ...
    
    async def find_zombie_sessions(
        self,
        threshold_seconds: int = 180
    ) -> List[str]:
        """
        Find sessions that are marked as running but have stale heartbeats.
        
        Args:
            threshold_seconds: How long since last heartbeat to consider zombie
            
        Returns:
            List of session IDs that appear to be zombies
        """
        ...


class CheckpointCapable(Protocol):
    """
    Extension protocol for backends that support state checkpoints.
    
    This enables "time travel" debugging by storing full state snapshots
    at each step, allowing session forking and state reconstruction.
    
    Example:
        # Save checkpoint after each step
        await persistence.save_checkpoint(session_id, step_index, state)
        
        # Load state at a specific step for debugging
        historical_state = await persistence.load_state_at_step(session_id, 5)
        
        # List all checkpoints for a session
        checkpoints = await persistence.list_checkpoints(session_id)
    """
    
    async def save_checkpoint(
        self,
        session_id: str,
        step_index: int,
        state: "Blackboard"
    ) -> None:
        """
        Save a state checkpoint for a specific step.
        
        Args:
            session_id: Session identifier
            step_index: The step number (1-indexed)
            state: Full blackboard state to snapshot
        """
        ...
    
    async def load_state_at_step(
        self,
        session_id: str,
        step_index: int
    ) -> "Blackboard":
        """
        Load the state checkpoint at a specific step.
        
        Args:
            session_id: Session identifier
            step_index: The step number to load
            
        Returns:
            The Blackboard state as it was at that step
            
        Raises:
            SessionNotFoundError: If session doesn't exist
            PersistenceError: If checkpoint at step doesn't exist
        """
        ...
    
    async def list_checkpoints(self, session_id: str) -> List[int]:
        """
        List all available checkpoint step indices for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Sorted list of step indices with checkpoints
        """
        ...
