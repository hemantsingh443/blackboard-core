"""
Redis Persistence Backend

High-throughput persistence for distributed deployments.
"""

import logging
from typing import Optional, List, TYPE_CHECKING

from .base import (
    PersistenceError,
    SessionNotFoundError,
    SessionConflictError,
)

if TYPE_CHECKING:
    from ..state import Blackboard

logger = logging.getLogger("blackboard.persistence.redis")


class RedisPersistence:
    """
    Redis-based persistence for distributed deployments.
    
    Provides atomic updates and works across multiple processes/containers.
    Requires redis-py: pip install blackboard-core[redis]
    
    Note:
        Redis persistence does NOT support parent-child session tracking
        or event logging. Use SQLitePersistence or PostgresPersistence
        for fractal agent observability features.
    
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
    
    async def save(
        self,
        state: "Blackboard",
        session_id: str,
        parent_session_id: Optional[str] = None
    ) -> None:
        """Save state to Redis with optimistic locking via WATCH."""
        from ..state import Blackboard
        
        client = await self._get_client()
        key = self._key(session_id)
        
        # Note: parent_session_id is ignored in RedisPersistence
        if parent_session_id:
            logger.warning(
                "RedisPersistence does not support parent_session_id. "
                "Use SQLitePersistence or PostgresPersistence for fractal agents."
            )
        
        # Optimistic locking with WATCH
        async with client.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                
                # Check existing version
                existing_data = await client.get(key)
                if existing_data:
                    existing = Blackboard.model_validate_json(existing_data)
                    if existing.version > state.version:
                        raise SessionConflictError(
                            f"Version conflict: redis={existing.version}, incoming={state.version}"
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
                if isinstance(e, SessionConflictError):
                    raise
                raise PersistenceError(f"Redis save failed: {e}") from e
    
    async def load(self, session_id: str) -> "Blackboard":
        """Load state from Redis."""
        from ..state import Blackboard
        
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
        """Check if session exists."""
        client = await self._get_client()
        return await client.exists(self._key(session_id)) > 0
    
    async def delete(self, session_id: str) -> None:
        """Delete a session."""
        client = await self._get_client()
        await client.delete(self._key(session_id))
        logger.debug(f"Deleted session {session_id} from Redis")
    
    async def list_sessions(self, parent_id: Optional[str] = None) -> List[str]:
        """
        List session IDs.
        
        Note: parent_id filtering is not supported in Redis backend.
        """
        if parent_id is not None:
            logger.warning(
                "RedisPersistence does not support parent_id filtering. "
                "Use SQLitePersistence or PostgresPersistence."
            )
            return []
        
        client = await self._get_client()
        keys = await client.keys(f"{self.prefix}*")
        return [k.decode().replace(self.prefix, "") for k in keys]
    
    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
