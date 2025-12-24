"""
In-Memory Persistence Backend

For testing and development. State is lost when the process exits.
"""

from typing import Optional, List, Dict, TYPE_CHECKING

from .base import (
    PersistenceError,
    SessionNotFoundError,
    SessionConflictError,
    logger,
)

if TYPE_CHECKING:
    from ..state import Blackboard


class InMemoryPersistence:
    """
    In-memory persistence for testing.
    
    State is lost when the process exits. Useful for unit tests
    where you don't want filesystem or network dependencies.
    
    Thread Safety:
        NOT thread-safe. Use only in single-threaded test environments.
        For async concurrency within a single thread, this is safe.
    
    Example:
        persistence = InMemoryPersistence()
        await persistence.save(state, "test-session")
        loaded = await persistence.load("test-session")
    """
    
    def __init__(self):
        self._store: Dict[str, str] = {}  # session_id -> JSON string
        self._parents: Dict[str, Optional[str]] = {}  # session_id -> parent_id
    
    async def save(
        self,
        state: "Blackboard",
        session_id: str,
        parent_session_id: Optional[str] = None
    ) -> None:
        """Save state to memory with optimistic locking."""
        from ..state import Blackboard
        
        # Optimistic locking check
        if session_id in self._store:
            existing = Blackboard.model_validate_json(self._store[session_id])
            if existing.version > state.version:
                raise SessionConflictError(
                    f"Version conflict: stored={existing.version}, incoming={state.version}"
                )
        
        # Increment version and store
        state.version += 1
        self._store[session_id] = state.model_dump_json()
        self._parents[session_id] = parent_session_id
    
    async def load(self, session_id: str) -> "Blackboard":
        """Load state from memory."""
        from ..state import Blackboard
        
        if session_id not in self._store:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        return Blackboard.model_validate_json(self._store[session_id])
    
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return session_id in self._store
    
    async def delete(self, session_id: str) -> None:
        """Delete a session."""
        self._store.pop(session_id, None)
        self._parents.pop(session_id, None)
    
    async def list_sessions(self, parent_id: Optional[str] = None) -> List[str]:
        """List all session IDs, optionally filtered by parent."""
        if parent_id is not None:
            return [
                sid for sid, pid in self._parents.items()
                if pid == parent_id
            ]
        return list(self._store.keys())
    
    def clear(self) -> None:
        """Clear all sessions (for testing)."""
        self._store.clear()
        self._parents.clear()
