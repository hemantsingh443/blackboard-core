"""
Production-Grade Vector Database Backends

Provides scalable vector storage for memory systems using external databases.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .memory import Memory, MemoryEntry, SearchResult

logger = logging.getLogger("blackboard.vectordb")


@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol for vector database backends.
    
    Implement this for custom vector database integrations.
    """
    
    async def add(self, entries: List[MemoryEntry]) -> None:
        """Add entries to the store."""
        ...
    
    async def search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar entries."""
        ...
    
    async def delete(self, ids: List[str]) -> None:
        """Delete entries by ID."""
        ...
    
    async def clear(self) -> None:
        """Clear all entries."""
        ...


class ChromaMemory(Memory):
    """
    ChromaDB-backed memory for production use.
    
    Provides persistent, scalable vector storage with efficient similarity search.
    Requires chromadb: pip install blackboard-core[chroma]
    
    Args:
        collection_name: Name of the Chroma collection
        persist_directory: Directory for persistent storage (None = in-memory)
        embedding_function: Optional custom embedding function
        
    Example:
        memory = ChromaMemory(
            collection_name="agent_memory",
            persist_directory="./chroma_data"
        )
        
        await memory.store(MemoryEntry(
            content="Important fact",
            metadata={"source": "user"}
        ))
        
        results = await memory.search("fact", k=5)
    """
    
    def __init__(
        self,
        collection_name: str = "blackboard_memory",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._embedding_function = embedding_function
        self._client = None
        self._collection = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError(
                    "chromadb package required for ChromaMemory. "
                    "Install with: pip install blackboard-core[chroma]"
                )
            
            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.Client()
        
        return self._client
    
    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            
            kwargs = {"name": self.collection_name}
            if self._embedding_function:
                kwargs["embedding_function"] = self._embedding_function
            
            self._collection = client.get_or_create_collection(**kwargs)
        
        return self._collection
    
    async def store(self, entry: MemoryEntry) -> str:
        collection = self._get_collection()
        
        def _add():
            collection.add(
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[entry.metadata or {}]
            )
        
        await asyncio.to_thread(_add)
        
        logger.debug(f"Stored entry {entry.id} in ChromaDB")
        return entry.id
    
    async def search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        collection = self._get_collection()
        
        kwargs = {
            "query_texts": [query],
            "n_results": k
        }
        if filter:
            kwargs["where"] = filter
        
        def _query():
            return collection.query(**kwargs)
        
        results = await asyncio.to_thread(_query)
        
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, id in enumerate(results['ids'][0]):
                search_results.append(SearchResult(
                    entry=MemoryEntry(
                        id=id,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    ),
                    score=1 - (results['distances'][0][i] if results['distances'] else 0)
                ))
        
        return search_results
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        collection = self._get_collection()
        
        def _get():
            return collection.get(ids=[entry_id])
        
        results = await asyncio.to_thread(_get)
        
        if results['ids']:
            return MemoryEntry(
                id=results['ids'][0],
                content=results['documents'][0],
                metadata=results['metadatas'][0] if results['metadatas'] else {}
            )
        
        return None
    
    async def delete(self, entry_id: str) -> bool:
        collection = self._get_collection()
        
        def _delete():
            collection.delete(ids=[entry_id])
        
        try:
            await asyncio.to_thread(_delete)
            return True
        except:
            return False
    
    async def clear(self) -> None:
        """Clear all entries from the collection."""
        client = self._get_client()
        
        def _delete_collection():
            client.delete_collection(self.collection_name)
        
        try:
            await asyncio.to_thread(_delete_collection)
            self._collection = None
            logger.info(f"Cleared ChromaDB collection: {self.collection_name}")
        except:
            pass
    
    async def count(self) -> int:
        """Get total number of entries."""
        collection = self._get_collection()
        return await asyncio.to_thread(collection.count)


class HybridSearchMemory(Memory):
    """
    Combines semantic search with keyword search (BM25).
    
    Uses both vector similarity and keyword matching for better retrieval.
    Requires rank-bm25: pip install rank-bm25
    
    Args:
        vector_memory: Underlying vector memory
        alpha: Weight for semantic vs keyword (0=keyword only, 1=semantic only)
        
    Example:
        base_memory = ChromaMemory(collection_name="docs")
        hybrid = HybridSearchMemory(base_memory, alpha=0.7)
        
        results = await hybrid.search("python async")
    """
    
    def __init__(self, vector_memory: Memory, alpha: float = 0.7):
        self.vector_memory = vector_memory
        self.alpha = alpha
        self._documents: List[MemoryEntry] = []
        self._tokenized_docs: List[List[str]] = []  # Cache tokenized documents
        self._bm25 = None
    
    def _build_bm25_index(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 package required for HybridSearchMemory. "
                "Install with: pip install rank-bm25"
            )
        
        # Use cached tokenization - only build BM25 from cache
        self._bm25 = BM25Okapi(self._tokenized_docs)
    
    async def store(self, entry: MemoryEntry) -> str:
        # Store in vector memory
        result = await self.vector_memory.store(entry)
        
        # Add to document list and tokenize incrementally (O(1) per add)
        self._documents.append(entry)
        tokens = entry.content.lower().split()
        self._tokenized_docs.append(tokens)
        self._bm25 = None  # Invalidate BM25 (rebuild is now fast with cached tokens)
        
        return result
    
    async def search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        # Get semantic results
        semantic_results = await self.vector_memory.search(query, k=k*2, filter=filter)
        
        # Build BM25 index if needed (CPU-bound, run in thread)
        if self._bm25 is None and self._documents:
            await asyncio.to_thread(self._build_bm25_index)
        
        if self._bm25 and self._documents:
            tokenized_query = query.lower().split()
            
            # BM25 scoring is CPU-bound
            bm25_scores = await asyncio.to_thread(self._bm25.get_scores, tokenized_query)
            
            # Normalize BM25 scores
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            
            # Combine scores
            combined = {}
            for result in semantic_results:
                combined[result.entry.id] = {
                    "entry": result.entry,
                    "semantic": result.score,
                    "bm25": 0
                }
            
            for i, score in enumerate(bm25_scores):
                doc = self._documents[i]
                if doc.id in combined:
                    combined[doc.id]["bm25"] = score / max_bm25
                else:
                    combined[doc.id] = {
                        "entry": doc,
                        "semantic": 0,
                        "bm25": score / max_bm25
                    }
            
            # Calculate final scores
            final_results = []
            for id, data in combined.items():
                final_score = (
                    self.alpha * data["semantic"] + 
                    (1 - self.alpha) * data["bm25"]
                )
                final_results.append(SearchResult(
                    entry=data["entry"],
                    score=final_score
                ))
            
            # Sort and return top k
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        
        return semantic_results[:k]
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        return await self.vector_memory.retrieve(entry_id)
    
    async def delete(self, entry_id: str) -> bool:
        result = await self.vector_memory.delete(entry_id)
        
        # Find index of document to remove
        for i, doc in enumerate(self._documents):
            if doc.id == entry_id:
                self._documents.pop(i)
                self._tokenized_docs.pop(i)  # Keep cache in sync
                break
        
        self._bm25 = None
        return result
    
    async def clear(self) -> None:
        await self.vector_memory.clear()
        self._documents.clear()
        self._tokenized_docs.clear()  # Clear cache
        self._bm25 = None
    
    async def count(self) -> int:
        """Return total number of memory entries."""
        return await self.vector_memory.count()
