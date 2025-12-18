"""
Embedding Models for Semantic Search

Provides pluggable embedding backends via the Adapter pattern.
"""

import math
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """
    Protocol for embedding models.
    
    Implement this to create custom embedding backends.
    
    Example:
        class MyEmbedder:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self.embed_query(t) for t in texts]
            
            def embed_query(self, text: str) -> List[float]:
                # Your embedding logic
                return [0.1, 0.2, 0.3, ...]
    """
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        ...
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        ...


class NoOpEmbedder:
    """
    A no-op embedder that returns zero vectors.
    
    Useful for testing or when semantic search is not needed.
    The vectors are deterministic based on text hash for consistency.
    
    Example:
        embedder = NoOpEmbedder(dimension=384)
        memory = SimpleVectorMemory(embedder=embedder)
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the no-op embedder.
        
        Args:
            dimension: Size of embedding vectors
        """
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return [self.embed_query(t) for t in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate a deterministic pseudo-random vector from text.
        
        Uses hash-based seeding for reproducibility.
        """
        import hashlib
        
        # Create a deterministic seed from text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        
        # Generate pseudo-random but deterministic values
        values = []
        for i in range(self.dimension):
            # Simple LCG-like generator
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            values.append((seed / 0x7fffffff) * 2 - 1)  # Range [-1, 1]
        
        # Normalize
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]
        
        return values


class TFIDFEmbedder:
    """
    A simple TF-IDF-based embedder.
    
    Lightweight and requires no external dependencies.
    Good for prototyping but not production semantic search.
    
    Note: Vocabulary grows with usage, causing embedding drift.
    For production, use LocalEmbedder or OpenAIEmbedder.
    """
    
    def __init__(self):
        self._vocabulary: dict = {}
        self._idf: dict = {}
        self._documents: List[str] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents and update vocabulary."""
        # Add to document collection
        self._documents.extend(texts)
        self._update_idf()
        
        return [self.embed_query(t) for t in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using current vocabulary."""
        tokens = self._tokenize(text)
        
        # Update vocabulary with new tokens
        for token in tokens:
            if token not in self._vocabulary:
                self._vocabulary[token] = len(self._vocabulary)
        
        # Compute term frequency
        tf: dict = {}
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
        n_docs = len(self._documents)
        if n_docs == 0:
            return
        
        # Count document frequency
        df: dict = {}
        for doc in self._documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        
        # Compute IDF
        for token, count in df.items():
            self._idf[token] = math.log(n_docs / (1 + count))


class LocalEmbedder:
    """
    Local embeddings using sentence-transformers.
    
    Requires: pip install sentence-transformers
    
    Fast, free, runs locally. Adds ~100MB dependency.
    
    Example:
        embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        memory = SimpleVectorMemory(embedder=embedder)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence-transformers model.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for LocalEmbedder. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class OpenAIEmbedder:
    """
    Embeddings via OpenAI API.
    
    Requires: pip install openai
    
    High quality embeddings. Requires API key.
    
    Example:
        embedder = OpenAIEmbedder(api_key="sk-...")
        memory = SimpleVectorMemory(embedder=embedder)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None
    ):
        """
        Initialize the OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Embedding model to use
            dimensions: Optional dimension override (for text-embedding-3-*)
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._client = None
    
    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                import openai
                if self.api_key:
                    self._client = openai.OpenAI(api_key=self.api_key)
                else:
                    self._client = openai.OpenAI()  # Uses env var
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAIEmbedder. "
                    "Install with: pip install openai"
                )
        return self._client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        client = self._get_client()
        
        kwargs = {"input": texts, "model": self.model}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        
        response = client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Handles vectors of different lengths by padding.
    """
    # Pad shorter vector
    max_len = max(len(a), len(b))
    a = list(a) + [0.0] * (max_len - len(a))
    b = list(b) + [0.0] * (max_len - len(b))
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def get_default_embedder() -> EmbeddingModel:
    """
    Get the best available embedder.
    
    Priority:
    1. LocalEmbedder (if sentence-transformers available)
    2. TFIDFEmbedder (fallback)
    """
    try:
        import sentence_transformers
        return LocalEmbedder()
    except ImportError:
        return TFIDFEmbedder()


# =============================================================================
# Async Embedders (for non-blocking I/O)
# =============================================================================

class AsyncOpenAIEmbedder:
    """
    Async embeddings via OpenAI API.
    
    Uses AsyncOpenAI client for non-blocking operations.
    Recommended for web servers and async applications.
    
    Requires: pip install openai
    
    Example:
        embedder = AsyncOpenAIEmbedder(api_key="sk-...")
        embedding = await embedder.embed_query_async("hello")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None
    ):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._client = None
    
    def _get_client(self):
        """Lazy-load the async OpenAI client."""
        if self._client is None:
            try:
                import openai
                if self.api_key:
                    self._client = openai.AsyncOpenAI(api_key=self.api_key)
                else:
                    self._client = openai.AsyncOpenAI()
            except ImportError:
                raise ImportError(
                    "openai is required for AsyncOpenAIEmbedder. "
                    "Install with: pip install openai"
                )
        return self._client
    
    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents asynchronously."""
        client = self._get_client()
        
        kwargs = {"input": texts, "model": self.model}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        
        response = await client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
    
    async def embed_query_async(self, text: str) -> List[float]:
        """Embed a single query asynchronously."""
        results = await self.embed_documents_async([text])
        return results[0]
    
    # Sync fallback for EmbeddingModel protocol compatibility
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Sync wrapper. Safe inside existing event loops (uses thread pool)."""
        import asyncio
        import concurrent.futures
        try:
            asyncio.get_running_loop()
            # Inside existing loop - run in thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.embed_documents_async(texts))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.embed_documents_async(texts))
    
    def embed_query(self, text: str) -> List[float]:
        """Sync wrapper. Safe inside existing event loops (uses thread pool)."""
        import asyncio
        import concurrent.futures
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.embed_query_async(text))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.embed_query_async(text))


def run_embedder_async(embedder: EmbeddingModel, text: str):
    """
    Run a sync embedder in a thread pool (non-blocking).
    
    Use this to avoid blocking the event loop with CPU-bound embedders.
    
    Example:
        embedding = await run_embedder_async(LocalEmbedder(), "hello")
    """
    import asyncio
    return asyncio.to_thread(embedder.embed_query, text)

