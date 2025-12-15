"""
Long-Term Memory System

Vector-based memory for RAG (Retrieval Augmented Generation) support.
"""

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
    
    Implement this to create custom memory backends.
    """
    
    @abstractmethod
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Add a memory entry."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for relevant memories."""
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Clear all memories. Returns count of deleted entries."""
        pass
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories (optional, may not scale)."""
        raise NotImplementedError("get_all not implemented for this memory backend")


class SimpleVectorMemory(Memory):
    """
    A simple in-memory vector store using cosine similarity.
    
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
        persist_path: Optional[str] = None
    ):
        """
        Initialize the memory store.
        
        Args:
            embedder: Embedding model (default: TFIDFEmbedder)
            persist_path: Optional path to persist memories to disk
        """
        self._memories: Dict[str, MemoryEntry] = {}
        self._embedder = embedder or TFIDFEmbedder()
        self.persist_path = persist_path
        
        if persist_path and Path(persist_path).exists():
            self._load()
    
    @property
    def embedder(self) -> EmbeddingModel:
        """Get the current embedder."""
        return self._embedder
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Add a memory with auto-generated embedding."""
        entry = MemoryEntry(content=content, metadata=metadata or {})
        entry.embedding = self._embedder.embed_query(content)
        self._memories[entry.id] = entry
        
        if self.persist_path:
            self._save()
        
        return entry
    
    def add_many(self, contents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[MemoryEntry]:
        """Add multiple memories efficiently (batch embedding)."""
        if metadata is None:
            metadata = [{}] * len(contents)
        
        # Batch embed
        embeddings = self._embedder.embed_documents(contents)
        
        entries = []
        for content, emb, meta in zip(contents, embeddings, metadata):
            entry = MemoryEntry(content=content, metadata=meta)
            entry.embedding = emb
            self._memories[entry.id] = entry
            entries.append(entry)
        
        if self.persist_path:
            self._save()
        
        return entries
    
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for similar memories using cosine similarity."""
        if not self._memories:
            return []
        
        query_embedding = self._embedder.embed_query(query)
        
        results = []
        for entry in self._memories.values():
            if entry.embedding:
                score = cosine_similarity(query_embedding, entry.embedding)
                results.append(SearchResult(entry=entry, score=score))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    def get(self, id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        return self._memories.get(id)
    
    def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        if id in self._memories:
            del self._memories[id]
            if self.persist_path:
                self._save()
            return True
        return False
    
    def clear(self) -> int:
        """Clear all memories."""
        count = len(self._memories)
        self._memories.clear()
        if self.persist_path:
            self._save()
        return count
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        return list(self._memories.values())
    
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
        
        # Determine operation from inputs or instructions
        operation = inputs.operation
        
        # Parse operation from instructions if not explicit
        if inputs.instructions:
            instr_lower = inputs.instructions.lower()
            if "remember" in instr_lower or "store" in instr_lower or "save" in instr_lower:
                operation = "add"
            elif "forget" in instr_lower or "delete" in instr_lower:
                operation = "delete"
            else:
                operation = "search"
        
        if operation == "search":
            query = inputs.query or inputs.instructions
            results = self.memory.search(query, limit=inputs.limit)
            
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
            entry = self.memory.add(content, {"source": "worker"})
            
            return WorkerOutput(
                artifact=Artifact(
                    type="memory_added",
                    content=f"Stored memory: {content[:100]}...",
                    creator=self.name,
                    metadata={"memory_id": entry.id}
                )
            )
        
        elif operation == "delete":
            success = self.memory.delete(inputs.memory_id)
            
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
