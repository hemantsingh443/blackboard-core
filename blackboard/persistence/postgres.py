"""
PostgreSQL Persistence Backend

Production-grade persistence for distributed deployments using asyncpg.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, TYPE_CHECKING

from .base import (
    PersistenceError,
    SessionNotFoundError,
    SessionConflictError,
    HeartbeatCapable,
    CheckpointCapable,
)

if TYPE_CHECKING:
    from ..state import Blackboard

logger = logging.getLogger("blackboard.persistence.postgres")


class PostgresPersistence(HeartbeatCapable, CheckpointCapable):
    """
    PostgreSQL-based persistence for distributed production deployments.
    
    Provides atomic transactions, connection pooling, and full feature parity
    with SQLitePersistence including heartbeat support and event logging.
    
    Requires: pip install asyncpg
    
    Args:
        dsn: PostgreSQL connection string (e.g., "postgresql://user:pass@host/db")
        pool_min_size: Minimum connections in pool (default: 2)
        pool_max_size: Maximum connections in pool (default: 10)
        
    Features:
        - Connection pooling for high throughput
        - JSONB storage for efficient queries
        - Full parent-child session tracking
        - Heartbeat support for zombie detection
        - Event logging for observability
        
    Example:
        persistence = PostgresPersistence("postgresql://user:pass@localhost/blackboard")
        await persistence.initialize()
        await persistence.save(state, "session-001")
    """
    
    # SQL Schema for PostgreSQL with checkpoint support
    SCHEMA = '''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        parent_session_id TEXT REFERENCES sessions(id),
        goal TEXT NOT NULL,
        status TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        heartbeat_at TIMESTAMPTZ DEFAULT NOW(),
        data JSONB NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        parent_event_id TEXT REFERENCES events(id),
        step_index INTEGER,
        event_type TEXT NOT NULL,
        source TEXT,
        payload JSONB,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    
    CREATE TABLE IF NOT EXISTS session_checkpoints (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        step_index INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        data JSONB NOT NULL,
        UNIQUE(session_id, step_index)
    );
    
    CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(heartbeat_at);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
    CREATE INDEX IF NOT EXISTS idx_checkpoints_session_step ON session_checkpoints(session_id, step_index);
    '''
    
    def __init__(
        self,
        dsn: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10
    ):
        self.dsn = dsn
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool = None
        self._initialized = False
    
    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg package required for PostgresPersistence. "
                    "Install with: pip install blackboard-core[postgres]"
                )
            
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size
            )
            logger.info(f"Created PostgreSQL connection pool (size: {self.pool_min_size}-{self.pool_max_size})")
        
        return self._pool
    
    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(self.SCHEMA)
        
        self._initialized = True
        logger.info("PostgreSQL schema initialized")
    
    async def save(
        self,
        state: "Blackboard",
        session_id: str,
        parent_session_id: Optional[str] = None
    ) -> None:
        """
        Save state to PostgreSQL with optimistic locking.
        
        Uses UPDATE ... WHERE version = X for atomic version checking.
        """
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Check existing version
                row = await conn.fetchrow(
                    "SELECT version FROM sessions WHERE id = $1",
                    session_id
                )
                
                if row is not None:
                    existing_version = row["version"]
                    if existing_version > state.version:
                        raise SessionConflictError(
                            f"Version conflict: db={existing_version}, incoming={state.version}"
                        )
                
                # Increment version
                state.version += 1
                now = datetime.now()
                
                # Upsert using ON CONFLICT
                await conn.execute(
                    '''
                    INSERT INTO sessions (id, parent_session_id, goal, status, version, created_at, updated_at, heartbeat_at, data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        version = EXCLUDED.version,
                        updated_at = EXCLUDED.updated_at,
                        heartbeat_at = EXCLUDED.heartbeat_at,
                        data = EXCLUDED.data
                    ''',
                    session_id,
                    parent_session_id,
                    state.goal,
                    state.status.value if hasattr(state.status, 'value') else str(state.status),
                    state.version,
                    now,
                    now,
                    now,
                    state.model_dump_json()
                )
        
        logger.debug(f"Saved session {session_id} (v{state.version})")
    
    async def load(self, session_id: str) -> "Blackboard":
        """Load state from PostgreSQL."""
        from ..state import Blackboard
        
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM sessions WHERE id = $1",
                session_id
            )
        
        if row is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        try:
            return Blackboard.model_validate_json(row["data"])
        except Exception as e:
            raise PersistenceError(f"Failed to deserialize session: {e}") from e
    
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM sessions WHERE id = $1",
                session_id
            )
        
        return row is not None
    
    async def delete(self, session_id: str) -> None:
        """Delete a session (cascades to events)."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)
        
        logger.debug(f"Deleted session {session_id}")
    
    async def list_sessions(self, parent_id: Optional[str] = None) -> List[str]:
        """List session IDs, optionally filtered by parent."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            if parent_id:
                rows = await conn.fetch(
                    "SELECT id FROM sessions WHERE parent_session_id = $1 ORDER BY created_at DESC",
                    parent_id
                )
            else:
                rows = await conn.fetch(
                    "SELECT id FROM sessions ORDER BY created_at DESC"
                )
        
        return [row["id"] for row in rows]
    
    # =========================================================================
    # Heartbeat Support (HeartbeatCapable)
    # =========================================================================
    
    async def update_heartbeat(self, session_id: str) -> None:
        """Update the heartbeat timestamp for a running session."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET heartbeat_at = NOW() WHERE id = $1",
                session_id
            )
    
    async def find_zombie_sessions(
        self,
        threshold_seconds: int = 180
    ) -> List[str]:
        """Find sessions with stale heartbeats."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                '''
                SELECT id FROM sessions 
                WHERE status IN ('running', 'generating', 'planning', 'critiquing', 'refining')
                  AND heartbeat_at < NOW() - INTERVAL '$1 seconds'
                ORDER BY heartbeat_at ASC
                ''',
                threshold_seconds
            )
        
        return [row["id"] for row in rows]
    
    # =========================================================================
    # Event Logging
    # =========================================================================
    
    async def log_event(
        self,
        session_id: str,
        event_type: str,
        payload: Optional[dict] = None,
        source: Optional[str] = None,
        step_index: Optional[int] = None,
        parent_event_id: Optional[str] = None
    ) -> str:
        """Log an event to the events table."""
        import uuid
        
        await self.initialize()
        pool = await self._get_pool()
        
        event_id = str(uuid.uuid4())
        
        async with pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO events (id, session_id, parent_event_id, step_index, event_type, source, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ''',
                event_id,
                session_id,
                parent_event_id,
                step_index,
                event_type,
                source,
                json.dumps(payload) if payload else None
            )
        
        return event_id
    
    async def get_events(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[dict]:
        """Retrieve events for a session."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            if event_type:
                rows = await conn.fetch(
                    '''
                    SELECT * FROM events 
                    WHERE session_id = $1 AND event_type = $2
                    ORDER BY timestamp ASC
                    LIMIT $3 OFFSET $4
                    ''',
                    session_id, event_type, limit, offset
                )
            else:
                rows = await conn.fetch(
                    '''
                    SELECT * FROM events 
                    WHERE session_id = $1
                    ORDER BY timestamp ASC
                    LIMIT $2 OFFSET $3
                    ''',
                    session_id, limit, offset
                )
        
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "parent_event_id": row["parent_event_id"],
                "step_index": row["step_index"],
                "event_type": row["event_type"],
                "source": row["source"],
                "payload": json.loads(row["payload"]) if row["payload"] else None,
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
            }
            for row in rows
        ]
    
    # =========================================================================
    # Checkpoint Support (CheckpointCapable) - Time Travel Debugging
    # =========================================================================
    
    async def save_checkpoint(
        self,
        session_id: str,
        step_index: int,
        state: "Blackboard"
    ) -> None:
        """
        Save a state checkpoint for a specific step.
        
        Uses UPSERT to handle re-runs of the same step (e.g., after a crash).
        """
        import uuid
        
        await self.initialize()
        pool = await self._get_pool()
        
        checkpoint_id = str(uuid.uuid4())
        
        async with pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO session_checkpoints (id, session_id, step_index, data)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (session_id, step_index) DO UPDATE SET
                    data = EXCLUDED.data,
                    created_at = NOW()
                ''',
                checkpoint_id,
                session_id,
                step_index,
                state.model_dump_json()
            )
        
        logger.debug(f"Saved checkpoint for session {session_id} at step {step_index}")
    
    async def load_state_at_step(
        self,
        session_id: str,
        step_index: int
    ) -> "Blackboard":
        """
        Load the state checkpoint at a specific step.
        """
        from ..state import Blackboard
        
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM session_checkpoints WHERE session_id = $1 AND step_index = $2",
                session_id, step_index
            )
        
        if row is None:
            if not await self.exists(session_id):
                raise SessionNotFoundError(f"Session not found: {session_id}")
            raise PersistenceError(f"No checkpoint at step {step_index} for session {session_id}")
        
        try:
            return Blackboard.model_validate_json(row["data"])
        except Exception as e:
            raise PersistenceError(f"Failed to deserialize checkpoint: {e}") from e
    
    async def list_checkpoints(self, session_id: str) -> List[int]:
        """List all available checkpoint step indices for a session."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT step_index FROM session_checkpoints WHERE session_id = $1 ORDER BY step_index ASC",
                session_id
            )
        
        return [row["step_index"] for row in rows]
    
    async def delete_checkpoints(self, session_id: str) -> int:
        """Delete all checkpoints for a session."""
        await self.initialize()
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM session_checkpoints WHERE session_id = $1",
                session_id
            )
        
        # Parse the result string "DELETE N" to get count
        deleted = int(result.split(" ")[1]) if result else 0
        logger.debug(f"Deleted {deleted} checkpoints for session {session_id}")
        return deleted
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL connection pool")
