"""
Persistence Layer for Blackboard State

Provides abstract persistence interface with multiple backend implementations
for distributed and serverless deployments.

Backends:
- InMemoryPersistence: For testing (no durability)
- SQLitePersistence: Single-node production (WAL mode)
- PostgresPersistence: Distributed production (asyncpg)
- RedisPersistence: High-throughput hot storage
- JSONFilePersistence: DEPRECATED - for debugging only
"""

from .base import (
    PersistenceLayer,
    PersistenceError,
    SessionNotFoundError,
    SessionConflictError,
    HeartbeatCapable,
    CheckpointCapable,
)
from .memory import InMemoryPersistence
from .sqlite import SQLitePersistence
from .redis import RedisPersistence
from .json_file import JSONFilePersistence

# Postgres is optional - requires asyncpg
try:
    from .postgres import PostgresPersistence
    _HAS_POSTGRES = True
except ImportError:
    PostgresPersistence = None  # type: ignore
    _HAS_POSTGRES = False


__all__ = [
    # Protocols
    "PersistenceLayer",
    "HeartbeatCapable",
    "CheckpointCapable",
    # Errors
    "PersistenceError",
    "SessionNotFoundError",
    "SessionConflictError",
    # Backends
    "InMemoryPersistence",
    "SQLitePersistence",
    "RedisPersistence",
    "PostgresPersistence",
    "JSONFilePersistence",  # Deprecated
]
