"""
Long-Term Memory System

Vector-based memory for RAG (Retrieval Augmented Generation) support.
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib


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
    
    Uses basic TF-IDF-like embeddings. For production, use
    OpenAI embeddings or a proper vector database.
    
    Example:
        memory = SimpleVectorMemory()
        memory.add("User prefers Python 3.11", {"source": "conversation"})
        
        results = memory.search("What Python version?")
        for r in results:
            print(f"{r.score:.2f}: {r.entry.content}")
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the memory store.
        
        Args:
            persist_path: Optional path to persist memories to disk
        """
        self._memories: Dict[str, MemoryEntry] = {}
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self.persist_path = persist_path
        
        if persist_path and Path(persist_path).exists():
            self._load()
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Add a memory with auto-generated embedding."""
        entry = MemoryEntry(content=content, metadata=metadata or {})
        entry.embedding = self._compute_embedding(content)
        self._memories[entry.id] = entry
        self._update_idf()
        
        if self.persist_path:
            self._save()
        
        return entry
    
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for similar memories using cosine similarity."""
        if not self._memories:
            return []
        
        query_embedding = self._compute_embedding(query)
        
        results = []
        for entry in self._memories.values():
            if entry.embedding:
                score = self._cosine_similarity(query_embedding, entry.embedding)
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
        self._vocabulary.clear()
        self._idf.clear()
        if self.persist_path:
            self._save()
        return count
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        return list(self._memories.values())
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute a simple TF-IDF-like embedding."""
        tokens = self._tokenize(text)
        
        # Update vocabulary
        for token in tokens:
            if token not in self._vocabulary:
                self._vocabulary[token] = len(self._vocabulary)
        
        # Compute term frequency
        tf: Dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        # Normalize TF
        max_tf = max(tf.values()) if tf else 1
        for token in tf:
            tf[token] = tf[token] / max_tf
        
        # Create embedding vector
        embedding = [0.0] * len(self._vocabulary)
        for token, freq in tf.items():
            idx = self._vocabulary[token]
            idf = self._idf.get(token, 1.0)
            embedding[idx] = freq * idf
        
        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _update_idf(self) -> None:
        """Update IDF scores."""
        n_docs = len(self._memories)
        if n_docs == 0:
            return
        
        # Count document frequency for each term
        df: Dict[str, int] = {}
        for entry in self._memories.values():
            tokens = set(self._tokenize(entry.content))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        
        # Compute IDF
        for token, count in df.items():
            self._idf[token] = math.log(n_docs / (1 + count))
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        # Pad shorter vector
        max_len = max(len(a), len(b))
        a = a + [0.0] * (max_len - len(a))
        b = b + [0.0] * (max_len - len(b))
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _save(self) -> None:
        """Persist memories to disk."""
        if not self.persist_path:
            return
        
        data = {
            "memories": [e.to_dict() for e in self._memories.values()],
            "vocabulary": self._vocabulary,
            "idf": self._idf
        }
        
        Path(self.persist_path).write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load memories from disk."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        
        data = json.loads(Path(self.persist_path).read_text())
        
        self._vocabulary = data.get("vocabulary", {})
        self._idf = data.get("idf", {})
        
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
