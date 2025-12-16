"""
Persistence Layer for Blackboard State

Provides abstract persistence interface with multiple backend implementations
for distributed and serverless deployments.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Blackboard

logger = logging.getLogger("blackboard.persistence")


class PersistenceError(Exception):
    """Base exception for persistence operations."""
    pass


class SessionNotFoundError(PersistenceError):
    """Raised when a session doesn't exist."""
    pass


class SessionConflictError(PersistenceError):
    """Raised when there's a version conflict during save."""
    pass


@runtime_checkable
class PersistenceLayer(Protocol):
    """
    Protocol for state persistence backends.
    
    Implementations should handle serialization, versioning, and atomic updates.
    All methods are async for compatibility with async backends (Redis, databases).
    
    Example:
        persistence = RedisPersistence(redis_url="redis://localhost:6379")
        await persistence.save(state, session_id="user-123")
        state = await persistence.load(session_id="user-123")
    """
    
    async def save(self, state: "Blackboard", session_id: str) -> None:
        """
        Save state to the backend.
        
        Args:
            state: Blackboard state to persist
            session_id: Unique identifier for this session
            
        Raises:
            SessionConflictError: If version conflict detected
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
    
    async def list_sessions(self) -> list:
        """List all session IDs."""
        ...


class JSONFilePersistence:
    """
    File-based persistence using JSON files.
    
    Simple backend for local development and single-machine deployments.
    Uses optimistic locking via version field.
    
    Args:
        directory: Directory to store session files
        
    Example:
        persistence = JSONFilePersistence("./sessions")
        await persistence.save(state, "session-001")
    """
    
    def __init__(self, directory: str = "./sessions"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, session_id: str) -> Path:
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.directory / f"{safe_id}.json"
    
    async def save(self, state: "Blackboard", session_id: str) -> None:
        from .state import Blackboard
        
        path = self._get_path(session_id)
        
        # Optimistic locking check
        if path.exists():
            try:
                existing = await self.load(session_id)
                if existing.version > state.version:
                    raise SessionConflictError(
                        f"Version conflict: disk={existing.version}, memory={state.version}"
                    )
            except SessionNotFoundError:
                pass
        
        # Increment version and save
        state.version += 1
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(state.model_dump_json(indent=2))
            logger.debug(f"Saved session {session_id} (v{state.version})")
        except Exception as e:
            raise PersistenceError(f"Failed to save session: {e}") from e
    
    async def load(self, session_id: str) -> "Blackboard":
        from .state import Blackboard
        
        path = self._get_path(session_id)
        
        if not path.exists():
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Blackboard.model_validate(data)
        except json.JSONDecodeError as e:
            raise PersistenceError(f"Invalid JSON in session file: {e}") from e
        except Exception as e:
            raise PersistenceError(f"Failed to load session: {e}") from e
    
    async def exists(self, session_id: str) -> bool:
        return self._get_path(session_id).exists()
    
    async def delete(self, session_id: str) -> None:
        path = self._get_path(session_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted session {session_id}")
    
    async def list_sessions(self) -> list:
        return [p.stem for p in self.directory.glob("*.json")]


class RedisPersistence:
    """
    Redis-based persistence for distributed deployments.
    
    Provides atomic updates and works across multiple processes/containers.
    Requires redis-py: pip install blackboard-core[redis]
    
    Args:
        redis_url: Redis connection URL
        prefix: Key prefix for all sessions
        ttl: Optional TTL in seconds for sessions
        
    Example:
        persistence = RedisPersistence("redis://localhost:6379")
        await persistence.save(state, "session-001")
    """
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379",
        prefix: str = "blackboard:",
        ttl: Optional[int] = None
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl = ttl
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as redis
            except ImportError:
                raise ImportError(
                    "redis package required for RedisPersistence. "
                    "Install with: pip install blackboard-core[redis]"
                )
            self._client = redis.from_url(self.redis_url)
        return self._client
    
    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"
    
    async def save(self, state: "Blackboard", session_id: str) -> None:
        client = await self._get_client()
        key = self._key(session_id)
        
        # Optimistic locking with WATCH
        async with client.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                
                # Check existing version
                existing_data = await client.get(key)
                if existing_data:
                    from .state import Blackboard
                    existing = Blackboard.model_validate_json(existing_data)
                    if existing.version > state.version:
                        raise SessionConflictError(
                            f"Version conflict: redis={existing.version}, memory={state.version}"
                        )
                
                # Increment and save
                state.version += 1
                
                pipe.multi()
                if self.ttl:
                    await pipe.setex(key, self.ttl, state.model_dump_json())
                else:
                    await pipe.set(key, state.model_dump_json())
                await pipe.execute()
                
                logger.debug(f"Saved session {session_id} to Redis (v{state.version})")
                
            except Exception as e:
                if "WatchError" in type(e).__name__:
                    raise SessionConflictError("Concurrent modification detected") from e
                raise PersistenceError(f"Redis save failed: {e}") from e
    
    async def load(self, session_id: str) -> "Blackboard":
        from .state import Blackboard
        
        client = await self._get_client()
        key = self._key(session_id)
        
        data = await client.get(key)
        if data is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        try:
            return Blackboard.model_validate_json(data)
        except Exception as e:
            raise PersistenceError(f"Failed to deserialize session: {e}") from e
    
    async def exists(self, session_id: str) -> bool:
        client = await self._get_client()
        return await client.exists(self._key(session_id)) > 0
    
    async def delete(self, session_id: str) -> None:
        client = await self._get_client()
        await client.delete(self._key(session_id))
        logger.debug(f"Deleted session {session_id} from Redis")
    
    async def list_sessions(self) -> list:
        client = await self._get_client()
        keys = await client.keys(f"{self.prefix}*")
        return [k.decode().replace(self.prefix, "") for k in keys]
    
    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


class InMemoryPersistence:
    """
    In-memory persistence for testing.
    
    State is lost when the process exits.
    """
    
    def __init__(self):
        self._store: dict = {}
    
    async def save(self, state: "Blackboard", session_id: str) -> None:
        from .state import Blackboard
        
        if session_id in self._store:
            existing = Blackboard.model_validate_json(self._store[session_id])
            if existing.version > state.version:
                raise SessionConflictError(
                    f"Version conflict: stored={existing.version}, memory={state.version}"
                )
        
        state.version += 1
        self._store[session_id] = state.model_dump_json()
    
    async def load(self, session_id: str) -> "Blackboard":
        from .state import Blackboard
        
        if session_id not in self._store:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        return Blackboard.model_validate_json(self._store[session_id])
    
    async def exists(self, session_id: str) -> bool:
        return session_id in self._store
    
    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
    
    async def list_sessions(self) -> list:
        return list(self._store.keys())
    
    def clear(self) -> None:
        """Clear all sessions (for testing)."""
        self._store.clear()
