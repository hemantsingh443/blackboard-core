"""
Long-Term Memory System

Vector-based memory for RAG (Retrieval Augmented Generation) support.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

from .embeddings import (
    EmbeddingModel, TFIDFEmbedder, NoOpEmbedder, 
    cosine_similarity, get_default_embedder
)


@dataclass
class MemoryEntry:
    """
    A single memory entry.
    
    Attributes:
        id: Unique identifier
        content: The text content
        metadata: Additional metadata (source, type, etc.)
        embedding: Vector embedding (if computed)
        timestamp: When this memory was created
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now()
        )


@dataclass
class SearchResult:
    """A memory search result with relevance score."""
    entry: MemoryEntry
    score: float  # Similarity score (0-1, higher is better)


class Memory(ABC):
    """
    Abstract interface for long-term memory.
    
    All methods are async to support non-blocking I/O operations
    with external embedders and vector databases.
    
    Implement this to create custom memory backends.
    """
    
    @abstractmethod
    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Add a memory entry."""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for relevant memories."""
        pass
    
    @abstractmethod
    async def get(self, id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """Clear all memories. Returns count of deleted entries."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Return total number of memory entries."""
        pass
    
    async def get_all(self) -> List[MemoryEntry]:
        """Get all memories (optional, may not scale)."""
        raise NotImplementedError("get_all not implemented for this memory backend")


class SimpleVectorMemory(Memory):
    """
    A simple in-memory vector store using cosine similarity.
    
    .. warning::
        **Development/Testing Only**: Uses O(N) linear search.
        For production with >10k entries, use a dedicated vector DB
        (FAISS, Chroma, pgvector) via a custom Memory implementation.
    
    Accepts any EmbeddingModel implementation for flexibility:
    - TFIDFEmbedder (default): Lightweight, no dependencies
    - LocalEmbedder: High quality, uses sentence-transformers
    - OpenAIEmbedder: Best quality, requires API key
    
    Example:
        # Default (TF-IDF)
        memory = SimpleVectorMemory()
        
        # With semantic embeddings
        from blackboard import LocalEmbedder
        memory = SimpleVectorMemory(embedder=LocalEmbedder())
        
        # With OpenAI
        from blackboard import OpenAIEmbedder
        memory = SimpleVectorMemory(embedder=OpenAIEmbedder(api_key="sk-..."))
    """
    
    def __init__(
        self,
        embedder: Optional[EmbeddingModel] = None,
        persist_path: Optional[str] = None,
        _suppress_warning: bool = False
    ):
        """
        Initialize the memory store.
        
        Args:
            embedder: Embedding model (default: TFIDFEmbedder)
            persist_path: Optional path to persist memories to disk
            _suppress_warning: Set to True to suppress the O(N) search warning
        """
        import warnings
        
        if not _suppress_warning:
            warnings.warn(
                "SimpleVectorMemory uses O(N) linear search. "
                "For production with >10k entries, use ChromaMemory from "
                "blackboard.vectordb or HybridSearchMemory. "
                "SQLite persistence solves RAM, not search speed.",
                UserWarning,
                stacklevel=2
            )
        
        self._memories: Dict[str, MemoryEntry] = {}
        self._embedder = embedder or TFIDFEmbedder()
        self.persist_path = persist_path
        
        if persist_path and Path(persist_path).exists():
            self._load()
    
    @property
    def embedder(self) -> EmbeddingModel:
        """Get the current embedder."""
        return self._embedder
    
    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Add a memory with auto-generated embedding."""
        entry = MemoryEntry(content=content, metadata=metadata or {})
        
        # Use async embedder if available, otherwise run in thread pool
        if hasattr(self._embedder, 'embed_query_async'):
            entry.embedding = await self._embedder.embed_query_async(content)
        else:
            entry.embedding = await asyncio.to_thread(self._embedder.embed_query, content)
        
        self._memories[entry.id] = entry
        
        if self.persist_path:
            await asyncio.to_thread(self._save)
        
        return entry
    
    async def add_many(self, contents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[MemoryEntry]:
        """Add multiple memories efficiently (batch embedding)."""
        if metadata is None:
            metadata = [{}] * len(contents)
        
        # Batch embed - use async if available
        if hasattr(self._embedder, 'embed_documents_async'):
            embeddings = await self._embedder.embed_documents_async(contents)
        else:
            embeddings = await asyncio.to_thread(self._embedder.embed_documents, contents)
        
        entries = []
        for content, emb, meta in zip(contents, embeddings, metadata):
            entry = MemoryEntry(content=content, metadata=meta)
            entry.embedding = emb
            self._memories[entry.id] = entry
            entries.append(entry)
        
        if self.persist_path:
            await asyncio.to_thread(self._save)
        
        return entries
    
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for similar memories using cosine similarity."""
        if not self._memories:
            return []
        
        # Use async embedder if available, otherwise run in thread pool
        if hasattr(self._embedder, 'embed_query_async'):
            query_embedding = await self._embedder.embed_query_async(query)
        else:
            query_embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        
        results = []
        for entry in self._memories.values():
            if entry.embedding:
                score = cosine_similarity(query_embedding, entry.embedding)
                results.append(SearchResult(entry=entry, score=score))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    async def get(self, id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        return self._memories.get(id)
    
    async def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        if id in self._memories:
            del self._memories[id]
            if self.persist_path:
                await asyncio.to_thread(self._save)
            return True
        return False
    
    async def clear(self) -> int:
        """Clear all memories."""
        count = len(self._memories)
        self._memories.clear()
        if self.persist_path:
            await asyncio.to_thread(self._save)
        return count
    
    async def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        return list(self._memories.values())
    
    async def count(self) -> int:
        """Return total number of memory entries."""
        return len(self._memories)
    
    def _save(self) -> None:
        """Persist memories to disk."""
        if not self.persist_path:
            return
        
        data = {
            "memories": [e.to_dict() for e in self._memories.values()],
            "embedder_type": type(self._embedder).__name__
        }
        
        Path(self.persist_path).write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load memories from disk."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        
        data = json.loads(Path(self.persist_path).read_text())
        
        for entry_data in data.get("memories", []):
            entry = MemoryEntry.from_dict(entry_data)
            self._memories[entry.id] = entry


# =============================================================================
# Memory Worker
# =============================================================================

from .protocols import Worker, WorkerOutput, WorkerInput
from .state import Artifact, Feedback

class MemoryInput(WorkerInput):
    """Input schema for memory operations."""
    operation: str = "search"  # "search", "add", "delete"
    query: str = ""  # For search
    content: str = ""  # For add
    memory_id: str = ""  # For delete
    limit: int = 5


class MemoryWorker(Worker):
    """
    A built-in worker for memory operations.
    
    Operations:
    - search: Find relevant memories
    - add: Store new information
    - delete: Remove a memory
    
    Example LLM call:
        {"action": "call", "worker": "Memory", "instructions": "Find user's Python preferences"}
    """
    
    name = "Memory"
    description = "Long-term memory system. Can search, add, or delete memories."
    input_schema = MemoryInput
    
    def __init__(self, memory: Memory):
        self.memory = memory
    
    async def run(self, state, inputs: Optional[MemoryInput] = None) -> WorkerOutput:
        if inputs is None:
            inputs = MemoryInput()
        
        # Operation must be explicitly set - no heuristic guessing
        operation = inputs.operation
        
        if operation == "search":
            query = inputs.query or inputs.instructions
            results = await self.memory.search(query, limit=inputs.limit)
            
            if not results:
                content = "No relevant memories found."
            else:
                content = "Retrieved memories:\n"
                for i, r in enumerate(results, 1):
                    content += f"\n{i}. [{r.score:.2f}] {r.entry.content}"
            
            return WorkerOutput(
                artifact=Artifact(
                    type="memory_search",
                    content=content,
                    creator=self.name,
                    metadata={
                        "query": query,
                        "results_count": len(results),
                        "results": [{"id": r.entry.id, "score": r.score} for r in results]
                    }
                )
            )
        
        elif operation == "add":
            content = inputs.content or inputs.instructions
            entry = await self.memory.add(content, {"source": "worker"})
            
            return WorkerOutput(
                artifact=Artifact(
                    type="memory_added",
                    content=f"Stored memory: {content[:100]}...",
                    creator=self.name,
                    metadata={"memory_id": entry.id}
                )
            )
        
        elif operation == "delete":
            success = await self.memory.delete(inputs.memory_id)
            
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    critique=f"Memory {'deleted' if success else 'not found'}: {inputs.memory_id}",
                    passed=success
                )
            )
        
        return WorkerOutput(
            feedback=Feedback(
                source=self.name,
                critique=f"Unknown operation: {operation}",
                passed=False
            )
        )
