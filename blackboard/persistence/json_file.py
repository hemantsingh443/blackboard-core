"""
JSON File Persistence Backend (DEPRECATED)

Retained for backward compatibility. Use SQLitePersistence for production.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

from .base import (
    PersistenceError,
    SessionNotFoundError,
    SessionConflictError,
)

if TYPE_CHECKING:
    from ..state import Blackboard

logger = logging.getLogger("blackboard.persistence.json")


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
    
    async def save(
        self,
        state: "Blackboard",
        session_id: str,
        parent_session_id: Optional[str] = None
    ) -> None:
        from ..state import Blackboard
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
        from ..state import Blackboard
        
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
    
    async def list_sessions(self, parent_id: Optional[str] = None) -> List[str]:
        # parent_id is ignored for JSONFilePersistence
        return [p.stem for p in self.directory.glob("*.json")]
