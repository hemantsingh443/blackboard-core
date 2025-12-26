"""
SQLite Persistence Backend

SQLite persistence with atomic writes, WAL mode, and heartbeat support.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
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

logger = logging.getLogger("blackboard.persistence.sqlite")


class SQLitePersistence(HeartbeatCapable, CheckpointCapable):
    """
    SQLite-based persistence for production deployments.
    
    Provides atomic writes, structured queries, and WAL mode for concurrency.
    Designed to support hierarchical/fractal agent architectures and zombie detection.
    
    Requires: pip install aiosqlite
    
    Args:
        db_path: Path to SQLite database file (default: ./blackboard.db)
        shared_connection: If provided, share connection with parent (for sub-agents)
        
    Features:
        - WAL mode for concurrent reads/writes
        - Structured events table for time-travel debugging
        - Parent session tracking for fractal agents
        - Optimistic locking via version field
        - Heartbeat support for zombie detection (v2.0+)
        
    Example:
        persistence = SQLitePersistence("./data/blackboard.db")
        await persistence.initialize()
        await persistence.save(state, "session-001")
    """
    
    # SQL Schema with heartbeat and checkpoint support
    SCHEMA = '''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        parent_session_id TEXT,
        goal TEXT NOT NULL,
        status TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        heartbeat_at TEXT,
        data JSON NOT NULL,
        FOREIGN KEY(parent_session_id) REFERENCES sessions(id)
    );
    
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        parent_event_id TEXT,
        step_index INTEGER,
        event_type TEXT NOT NULL,
        source TEXT,
        payload JSON,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(id),
        FOREIGN KEY(parent_event_id) REFERENCES events(id)
    );
    
    CREATE TABLE IF NOT EXISTS session_checkpoints (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        step_index INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        data JSON NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
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
        db_path: str = "./blackboard.db",
        shared_connection: Optional["SQLitePersistence"] = None
    ):
        self.db_path = db_path
        self._connection = None
        self._shared = shared_connection
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def _get_connection(self):
        """Get or create database connection."""
        # If sharing connection with parent, use that
        if self._shared is not None:
            return await self._shared._get_connection()
        
        if self._connection is None:
            try:
                import aiosqlite
            except ImportError:
                raise ImportError(
                    "aiosqlite package required for SQLitePersistence. "
                    "Install with: pip install aiosqlite"
                )
            
            # Create directory if needed
            db_dir = Path(self.db_path).parent
            if db_dir and str(db_dir) != "." and not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)
            
            self._connection = await aiosqlite.connect(self.db_path)
            
            # Enable WAL mode for concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")
            await self._connection.execute("PRAGMA foreign_keys=ON")
            
            # Row factory for dict-like access
            self._connection.row_factory = aiosqlite.Row
            
            logger.debug(f"Opened SQLite connection: {self.db_path}")
        
        return self._connection
    
    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            conn = await self._get_connection()
            await conn.executescript(self.SCHEMA)
            await conn.commit()
            self._initialized = True
            logger.info(f"SQLite schema initialized: {self.db_path}")
    
    async def save(
        self,
        state: "Blackboard",
        session_id: str,
        parent_session_id: Optional[str] = None
    ) -> None:
        """
        Save state to SQLite with optimistic locking.
        
        Args:
            state: Blackboard state to persist
            session_id: Unique session identifier
            parent_session_id: Optional parent session (for sub-agents)
        """
        await self.initialize()
        conn = await self._get_connection()
        
        async with self._lock:
            # Check for version conflict (optimistic locking)
            cursor = await conn.execute(
                "SELECT version FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            
            if row is not None:
                existing_version = row["version"]
                if existing_version > state.version:
                    raise SessionConflictError(
                        f"Version conflict: db={existing_version}, incoming={state.version}"
                    )
            
            # Increment version
            state.version += 1
            now = datetime.now().isoformat()
            
            # Upsert session
            await conn.execute(
                '''
                INSERT INTO sessions (id, parent_session_id, goal, status, version, created_at, updated_at, heartbeat_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    version = excluded.version,
                    updated_at = excluded.updated_at,
                    heartbeat_at = excluded.heartbeat_at,
                    data = excluded.data
                ''',
                (
                    session_id,
                    parent_session_id,
                    state.goal,
                    state.status.value if hasattr(state.status, 'value') else str(state.status),
                    state.version,
                    now,
                    now,
                    now,  # Initial heartbeat
                    state.model_dump_json()
                )
            )
            await conn.commit()
            logger.debug(f"Saved session {session_id} (v{state.version})")
    
    async def load(self, session_id: str) -> "Blackboard":
        """Load state from SQLite."""
        from ..state import Blackboard
        
        await self.initialize()
        conn = await self._get_connection()
        
        cursor = await conn.execute(
            "SELECT data FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()
        
        if row is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        try:
            return Blackboard.model_validate_json(row["data"])
        except Exception as e:
            raise PersistenceError(f"Failed to deserialize session: {e}") from e
    
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        await self.initialize()
        conn = await self._get_connection()
        
        cursor = await conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
            (session_id,)
        )
        return await cursor.fetchone() is not None
    
    async def delete(self, session_id: str) -> None:
        """Delete a session and its events."""
        await self.initialize()
        conn = await self._get_connection()
        
        async with self._lock:
            # Delete events first (foreign key)
            await conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await conn.commit()
            logger.debug(f"Deleted session {session_id}")
    
    async def list_sessions(self, parent_id: Optional[str] = None) -> List[str]:
        """
        List session IDs.
        
        Args:
            parent_id: If provided, list only child sessions of this parent
        """
        await self.initialize()
        conn = await self._get_connection()
        
        if parent_id:
            cursor = await conn.execute(
                "SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY created_at DESC",
                (parent_id,)
            )
        else:
            cursor = await conn.execute(
                "SELECT id FROM sessions ORDER BY created_at DESC"
            )
        
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]
    
    # =========================================================================
    # Heartbeat Support (HeartbeatCapable)
    # =========================================================================
    
    async def update_heartbeat(self, session_id: str) -> None:
        """
        Update the heartbeat timestamp for a running session.
        
        Called periodically by the Orchestrator to signal liveness.
        """
        await self.initialize()
        conn = await self._get_connection()
        
        now = datetime.now().isoformat()
        await conn.execute(
            "UPDATE sessions SET heartbeat_at = ? WHERE id = ?",
            (now, session_id)
        )
        await conn.commit()
    
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
        await self.initialize()
        conn = await self._get_connection()
        
        # Calculate cutoff time
        cutoff = (datetime.now() - timedelta(seconds=threshold_seconds)).isoformat()
        
        cursor = await conn.execute(
            '''
            SELECT id FROM sessions 
            WHERE status IN ('running', 'generating', 'planning', 'critiquing', 'refining')
              AND heartbeat_at < ?
            ORDER BY heartbeat_at ASC
            ''',
            (cutoff,)
        )
        
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]
    
    # =========================================================================
    # Event Logging (for fractal agent observability)
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
        """
        Log an event to the events table.
        
        Returns the generated event ID.
        """
        import uuid
        
        await self.initialize()
        conn = await self._get_connection()
        
        event_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        await conn.execute(
            '''
            INSERT INTO events (id, session_id, parent_event_id, step_index, event_type, source, payload, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                event_id,
                session_id,
                parent_event_id,
                step_index,
                event_type,
                source,
                json.dumps(payload) if payload else None,
                now
            )
        )
        await conn.commit()
        return event_id
    
    async def get_events(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[dict]:
        """
        Retrieve events for a session.
        
        Args:
            session_id: Session to get events for
            event_type: Optional filter by event type
            limit: Max events to return
            offset: Pagination offset
        """
        await self.initialize()
        conn = await self._get_connection()
        
        if event_type:
            cursor = await conn.execute(
                '''
                SELECT * FROM events 
                WHERE session_id = ? AND event_type = ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
                ''',
                (session_id, event_type, limit, offset)
            )
        else:
            cursor = await conn.execute(
                '''
                SELECT * FROM events 
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
                ''',
                (session_id, limit, offset)
            )
        
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "parent_event_id": row["parent_event_id"],
                "step_index": row["step_index"],
                "event_type": row["event_type"],
                "source": row["source"],
                "payload": json.loads(row["payload"]) if row["payload"] else None,
                "timestamp": row["timestamp"]
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
        conn = await self._get_connection()
        
        checkpoint_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        async with self._lock:
            await conn.execute(
                '''
                INSERT INTO session_checkpoints (id, session_id, step_index, created_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, step_index) DO UPDATE SET
                    data = excluded.data,
                    created_at = excluded.created_at
                ''',
                (
                    checkpoint_id,
                    session_id,
                    step_index,
                    now,
                    state.model_dump_json()
                )
            )
            await conn.commit()
            logger.debug(f"Saved checkpoint for session {session_id} at step {step_index}")
    
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
        from ..state import Blackboard
        
        await self.initialize()
        conn = await self._get_connection()
        
        cursor = await conn.execute(
            "SELECT data FROM session_checkpoints WHERE session_id = ? AND step_index = ?",
            (session_id, step_index)
        )
        row = await cursor.fetchone()
        
        if row is None:
            # Check if session exists at all
            if not await self.exists(session_id):
                raise SessionNotFoundError(f"Session not found: {session_id}")
            raise PersistenceError(f"No checkpoint at step {step_index} for session {session_id}")
        
        try:
            return Blackboard.model_validate_json(row["data"])
        except Exception as e:
            raise PersistenceError(f"Failed to deserialize checkpoint: {e}") from e
    
    async def list_checkpoints(self, session_id: str) -> List[int]:
        """
        List all available checkpoint step indices for a session.
        
        Returns:
            Sorted list of step indices with checkpoints
        """
        await self.initialize()
        conn = await self._get_connection()
        
        cursor = await conn.execute(
            "SELECT step_index FROM session_checkpoints WHERE session_id = ? ORDER BY step_index ASC",
            (session_id,)
        )
        rows = await cursor.fetchall()
        return [row["step_index"] for row in rows]
    
    async def delete_checkpoints(self, session_id: str) -> int:
        """
        Delete all checkpoints for a session.
        
        Returns:
            Number of checkpoints deleted
        """
        await self.initialize()
        conn = await self._get_connection()
        
        async with self._lock:
            cursor = await conn.execute(
                "DELETE FROM session_checkpoints WHERE session_id = ?",
                (session_id,)
            )
            await conn.commit()
            deleted = cursor.rowcount
            logger.debug(f"Deleted {deleted} checkpoints for session {session_id}")
            return deleted
    
    async def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None and self._shared is None:
            await self._connection.close()
            self._connection = None
            logger.debug(f"Closed SQLite connection: {self.db_path}")
