"""
Persistence Layer for Blackboard State

Provides abstract persistence interface with multiple backend implementations
for distributed and serverless deployments.
"""

import asyncio
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
    
    async def save(self, state: "Blackboard", session_id: str, parent_session_id: Optional[str] = None) -> None:
        """
        Save state to the backend.
        
        Args:
            state: Blackboard state to persist
            session_id: Unique identifier for this session
            parent_session_id: Optional parent session ID for fractal agents (v1.6+)
            
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
    
    .. deprecated:: 1.5.1
        Use SQLitePersistence for production. JSON files are retained for
        debugging and git-diffable state inspection.
    
    Args:
        directory: Directory to store session files
        
    Example:
        persistence = JSONFilePersistence("./sessions")
        await persistence.save(state, "session-001")
    """
    
    def __init__(self, directory: str = "./sessions"):
        import warnings
        warnings.warn(
            "JSONFilePersistence is deprecated for production use. "
            "Use SQLitePersistence instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, session_id: str) -> Path:
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.directory / f"{safe_id}.json"
    
    async def save(self, state: "Blackboard", session_id: str, parent_session_id: Optional[str] = None) -> None:
        from .state import Blackboard
        # Note: parent_session_id is ignored in JSONFilePersistence (no hierarchy tracking)
        
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
        
        def _write_file():
            with open(path, 'w', encoding='utf-8') as f:
                f.write(state.model_dump_json(indent=2))
        
        try:
            await asyncio.to_thread(_write_file)
            logger.debug(f"Saved session {session_id} (v{state.version})")
        except Exception as e:
            raise PersistenceError(f"Failed to save session: {e}") from e
    
    async def load(self, session_id: str) -> "Blackboard":
        from .state import Blackboard
        
        path = self._get_path(session_id)
        
        if not path.exists():
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        def _read_file():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        try:
            data = await asyncio.to_thread(_read_file)
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


class SQLitePersistence:
    """
    SQLite-based persistence for production deployments.
    
    Provides atomic writes, structured queries, and WAL mode for concurrency.
    Designed to support hierarchical/fractal agent architectures.
    
    Requires: pip install blackboard-core[sqlite]
    
    Args:
        db_path: Path to SQLite database file (default: ./blackboard.db)
        shared: If True, this instance shares connection with parent (for sub-agents)
        
    Features:
        - WAL mode for concurrent reads/writes
        - Structured events table for time-travel debugging
        - Parent session tracking for fractal agents
        - Optimistic locking via version field
        
    Example:
        persistence = SQLitePersistence("./data/blackboard.db")
        await persistence.initialize()
        await persistence.save(state, "session-001")
        
        # For sub-agents, share the connection:
        child_persistence = SQLitePersistence(shared_connection=persistence)
    """
    
    # SQL Schema
    SCHEMA = '''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        parent_session_id TEXT,
        goal TEXT NOT NULL,
        status TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
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
    
    CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
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
                    "Install with: pip install blackboard-core[sqlite]"
                )
            
            # Create directory if needed
            db_dir = Path(self.db_path).parent
            if db_dir and not db_dir.exists():
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
    
    async def save(self, state: "Blackboard", session_id: str, parent_session_id: Optional[str] = None) -> None:
        """
        Save state to SQLite.
        
        Args:
            state: Blackboard state to persist
            session_id: Unique session identifier
            parent_session_id: Optional parent session (for sub-agents)
        """
        from .state import Blackboard
        from datetime import datetime
        
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
                        f"Version conflict: db={existing_version}, memory={state.version}"
                    )
            
            # Increment version
            state.version += 1
            now = datetime.now().isoformat()
            
            # Upsert session
            await conn.execute(
                '''
                INSERT INTO sessions (id, parent_session_id, goal, status, version, created_at, updated_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    version = excluded.version,
                    updated_at = excluded.updated_at,
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
                    state.model_dump_json()
                )
            )
            await conn.commit()
            logger.debug(f"Saved session {session_id} (v{state.version})")
    
    async def load(self, session_id: str) -> "Blackboard":
        """Load state from SQLite."""
        from .state import Blackboard
        
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
    
    async def list_sessions(self, parent_id: Optional[str] = None) -> list:
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
    
    # ==========================================================================
    # Event Logging (for fractal agent observability)
    # ==========================================================================
    
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
        from datetime import datetime
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
    ) -> list:
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
    
    async def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None and self._shared is None:
            await self._connection.close()
            self._connection = None
            logger.debug(f"Closed SQLite connection: {self.db_path}")





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
    
    async def save(self, state: "Blackboard", session_id: str, parent_session_id: Optional[str] = None) -> None:
        client = await self._get_client()
        # Note: parent_session_id is ignored in RedisPersistence (no hierarchy tracking)
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
    
    async def save(self, state: "Blackboard", session_id: str, parent_session_id: Optional[str] = None) -> None:
        from .state import Blackboard
        
        if session_id in self._store:
            existing = Blackboard.model_validate_json(self._store[session_id])
            if existing.version > state.version:
                raise SessionConflictError(
                    f"Version conflict: stored={existing.version}, memory={state.version}"
                )
        
        state.version += 1
        self._store[session_id] = state.model_dump_json()
        # Note: parent_session_id is ignored in InMemoryPersistence (no hierarchy tracking)
    
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
