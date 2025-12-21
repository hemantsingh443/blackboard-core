"""
Graph Memory - Knowledge Graph Enhanced Memory

Provides GraphRAG capabilities for deep context retrieval using a hybrid approach:
1. Vector search to find entry nodes
2. Graph traversal to discover related context

This combines the strengths of semantic search (finding relevant starting points)
with knowledge graphs (discovering relationships and structured context).

Example:
    from blackboard.graph import GraphMemory
    from blackboard.vectordb import ChromaMemory
    
    # Create hybrid graph memory
    vector_memory = ChromaMemory(collection_name="entities")
    graph_memory = GraphMemory(vector_memory=vector_memory)
    
    # Store facts (extracts entities and relations)
    await graph_memory.store(MemoryEntry(
        content="Alice works at Google. Google is headquartered in Mountain View."
    ))
    
    # Search with graph traversal
    results = await graph_memory.search("Where is Alice based?")
    # Returns: Alice -> works_at -> Google -> headquartered_in -> Mountain View

Requirements:
    pip install networkx
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from .memory import MemoryEntry, SearchResult

logger = logging.getLogger("blackboard.graph")


# =============================================================================
# Graph Store Protocol
# =============================================================================

@runtime_checkable
class GraphStore(Protocol):
    """
    Protocol for graph storage backends.
    
    Implement this for custom graph database integrations (Neo4j, etc.).
    """
    
    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node with associated data."""
        ...
    
    def add_edge(
        self, 
        source: str, 
        target: str, 
        relation: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a directed edge between nodes."""
        ...
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        ...
    
    def get_neighbors(
        self, 
        node_id: str, 
        direction: str = "both",
        max_hops: int = 1
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get neighboring nodes (node_id, relation, node_data)."""
        ...
    
    def search_nodes(self, query: str) -> List[str]:
        """Search for nodes by pattern/keyword."""
        ...
    
    def save(self, path: str) -> None:
        """Persist graph to file."""
        ...
    
    def load(self, path: str) -> None:
        """Load graph from file."""
        ...


# =============================================================================
# Triple Extraction
# =============================================================================

@dataclass
class Triple:
    """A subject-predicate-object triple."""
    subject: str
    predicate: str
    object: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"({self.subject}) -[{self.predicate}]-> ({self.object})"


def extract_triples_simple(text: str) -> List[Triple]:
    """
    Simple rule-based triple extraction.
    
    This is a basic fallback when no LLM is available.
    For production, use LLM-based extraction.
    """
    triples = []
    
    # Common patterns
    patterns = [
        (r"(\w+) works at (\w+)", "works_at"),
        (r"(\w+) is located in (\w+)", "located_in"),
        (r"(\w+) headquartered in (\w+)", "headquartered_in"),
        (r"(\w+) is a (\w+)", "is_a"),
        (r"(\w+) created (\w+)", "created"),
        (r"(\w+) uses (\w+)", "uses"),
    ]
    
    import re
    for pattern, predicate in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            triples.append(Triple(
                subject=match.group(1),
                predicate=predicate,
                object=match.group(2)
            ))
    
    return triples


async def extract_triples_llm(
    text: str,
    llm: Any,
    max_triples: int = 10
) -> List[Triple]:
    """
    Extract triples using an LLM.
    
    Args:
        text: Text to extract triples from
        llm: LLM client with generate() method
        max_triples: Maximum number of triples to extract
        
    Returns:
        List of extracted triples
    """
    prompt = f"""Extract knowledge graph triples from the following text.
Return a JSON array of triples in format: [{{"subject": "...", "predicate": "...", "object": "..."}}]

Rules:
1. Extract factual relationships only
2. Use simple, lowercase predicates (works_at, located_in, is_a, etc.)
3. Normalize entity names (proper capitalization)
4. Maximum {max_triples} triples

TEXT:
{text}

JSON TRIPLES:"""

    try:
        result = llm.generate(prompt)
        if asyncio.iscoroutine(result):
            response = await result
        else:
            response = result
        
        # Parse response
        content = response if isinstance(response, str) else response.content
        
        # Extract JSON array
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return [
                Triple(
                    subject=t.get("subject", ""),
                    predicate=t.get("predicate", ""),
                    object=t.get("object", "")
                )
                for t in data
                if t.get("subject") and t.get("object")
            ]
    except Exception as e:
        logger.warning(f"LLM triple extraction failed: {e}")
    
    return []


# =============================================================================
# NetworkX Graph Store
# =============================================================================

class NetworkXStore:
    """
    Graph store backed by NetworkX.
    
    Provides in-memory graph storage with optional file persistence.
    Suitable for development and small-to-medium graphs.
    
    For production with large graphs, consider Neo4j.
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the NetworkX store.
        
        Args:
            persist_path: Optional path for auto-persistence
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx package required for GraphMemory. "
                "Install with: pip install networkx"
            )
        
        self._nx = nx
        self._graph: "nx.DiGraph" = nx.DiGraph()
        self.persist_path = persist_path
        
        # Load existing graph if available
        if persist_path and Path(persist_path).exists():
            self.load(persist_path)
    
    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node with associated data."""
        normalized_id = node_id.lower().strip()
        self._graph.add_node(normalized_id, **data)
    
    def add_edge(
        self, 
        source: str, 
        target: str, 
        relation: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a directed edge between nodes."""
        source_norm = source.lower().strip()
        target_norm = target.lower().strip()
        
        # Ensure nodes exist
        if source_norm not in self._graph:
            self._graph.add_node(source_norm, label=source)
        if target_norm not in self._graph:
            self._graph.add_node(target_norm, label=target)
        
        edge_data = data or {}
        edge_data["relation"] = relation
        self._graph.add_edge(source_norm, target_norm, **edge_data)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        normalized_id = node_id.lower().strip()
        if normalized_id in self._graph:
            return dict(self._graph.nodes[normalized_id])
        return None
    
    def get_neighbors(
        self, 
        node_id: str, 
        direction: str = "both",
        max_hops: int = 1
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get neighboring nodes with BFS traversal.
        
        Args:
            node_id: Starting node
            direction: "out", "in", or "both"
            max_hops: Maximum traversal depth
            
        Returns:
            List of (node_id, relation, node_data) tuples
        """
        normalized_id = node_id.lower().strip()
        
        if normalized_id not in self._graph:
            return []
        
        neighbors = []
        visited: Set[str] = {normalized_id}
        queue = [(normalized_id, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= max_hops:
                continue
            
            # Get edges based on direction
            edges = []
            if direction in ("out", "both"):
                edges.extend([
                    (current, succ, self._graph.edges[current, succ])
                    for succ in self._graph.successors(current)
                ])
            if direction in ("in", "both"):
                edges.extend([
                    (pred, current, self._graph.edges[pred, current])
                    for pred in self._graph.predecessors(current)
                ])
            
            for source, target, edge_data in edges:
                neighbor = target if source == current else source
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    node_data = dict(self._graph.nodes[neighbor])
                    relation = edge_data.get("relation", "related_to")
                    neighbors.append((neighbor, relation, node_data))
                    queue.append((neighbor, depth + 1))
        
        return neighbors
    
    def search_nodes(self, query: str) -> List[str]:
        """Search for nodes by keyword matching."""
        query_lower = query.lower()
        matches = []
        
        for node_id in self._graph.nodes():
            if query_lower in node_id:
                matches.append(node_id)
            else:
                node_data = self._graph.nodes[node_id]
                label = node_data.get("label", "")
                if query_lower in label.lower():
                    matches.append(node_id)
        
        return matches
    
    def save(self, path: str) -> None:
        """Save graph to file."""
        with open(path, 'wb') as f:
            pickle.dump(self._graph, f)
        logger.debug(f"Graph saved to {path}")
    
    def load(self, path: str) -> None:
        """Load graph from file."""
        try:
            with open(path, 'rb') as f:
                self._graph = pickle.load(f)
            logger.debug(f"Graph loaded from {path} ({len(self._graph.nodes())} nodes)")
        except Exception as e:
            logger.warning(f"Failed to load graph: {e}")
    
    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self._graph.nodes())
    
    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return len(self._graph.edges())


# =============================================================================
# Graph Memory
# =============================================================================

class GraphMemory:
    """
    Knowledge Graph enhanced memory with hybrid search.
    
    Combines vector similarity search for finding entry points with
    graph traversal for discovering related context.
    
    Architecture:
    1. On store: Extract triples, store in graph + vector memory
    2. On search: Find entry nodes via vector search, traverse graph for context
    
    Args:
        vector_memory: Underlying vector memory for semantic search
        graph_store: Graph storage backend (default: NetworkX)
        llm: Optional LLM for triple extraction (falls back to rules)
        max_hops: Maximum graph traversal depth
        persist_path: Optional path for graph persistence
        
    Example:
        from blackboard.vectordb import ChromaMemory
        
        vector_mem = ChromaMemory("entities")
        graph_mem = GraphMemory(
            vector_memory=vector_mem,
            max_hops=2
        )
        
        await graph_mem.store(MemoryEntry(
            content="Alice works at Google. Google is in Mountain View."
        ))
        
        results = await graph_mem.search("Where does Alice work?")
    """
    
    def __init__(
        self,
        vector_memory,
        graph_store: Optional[GraphStore] = None,
        llm: Optional[Any] = None,
        max_hops: int = 2,
        persist_path: Optional[str] = None
    ):
        self.vector_memory = vector_memory
        self.graph_store = graph_store or NetworkXStore(persist_path)
        self.llm = llm
        self.max_hops = max_hops
        self.persist_path = persist_path
    
    async def store(self, entry: MemoryEntry) -> str:
        """
        Store entry with automatic triple extraction.
        
        1. Extracts triples from content
        2. Adds nodes and edges to graph
        3. Stores original entry in vector memory
        """
        # Store in vector memory first
        entry_id = await self.vector_memory.store(entry)
        
        # Extract triples
        if self.llm:
            triples = await extract_triples_llm(entry.content, self.llm)
        else:
            triples = extract_triples_simple(entry.content)
        
        # Add to graph
        for triple in triples:
            # Add nodes
            self.graph_store.add_node(triple.subject, {
                "label": triple.subject,
                "source_entry": entry_id
            })
            self.graph_store.add_node(triple.object, {
                "label": triple.object,
                "source_entry": entry_id
            })
            
            # Add edge
            self.graph_store.add_edge(
                triple.subject,
                triple.object,
                triple.predicate,
                {"source_entry": entry_id}
            )
        
        # Persist if configured
        if self.persist_path:
            self.graph_store.save(self.persist_path)
        
        logger.debug(f"Stored entry {entry_id} with {len(triples)} triples")
        return entry_id
    
    async def search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Hybrid search: vector lookup + graph traversal.
        
        1. Find entry nodes via vector similarity search
        2. Traverse graph from entry nodes to find related context
        3. Combine and rank results
        """
        # Step 1: Vector search for entry points
        vector_results = await self.vector_memory.search(query, k=k, filter=filter)
        
        # Step 2: Find graph nodes related to vector results
        graph_context = []
        seen_entries: Set[str] = set()
        
        for result in vector_results:
            seen_entries.add(result.entry.id)
            
            # Extract potential entity names from the entry
            # Simple approach: use keywords from content
            keywords = self._extract_keywords(result.entry.content)
            
            for keyword in keywords:
                # Find matching nodes
                matching_nodes = self.graph_store.search_nodes(keyword)
                
                for node_id in matching_nodes:
                    # Traverse from this node
                    neighbors = self.graph_store.get_neighbors(
                        node_id,
                        direction="both",
                        max_hops=self.max_hops
                    )
                    
                    for neighbor_id, relation, node_data in neighbors:
                        source_entry_id = node_data.get("source_entry")
                        if source_entry_id and source_entry_id not in seen_entries:
                            seen_entries.add(source_entry_id)
                            
                            # Retrieve the full entry
                            entry = await self.vector_memory.retrieve(source_entry_id)
                            if entry:
                                graph_context.append(SearchResult(
                                    entry=entry,
                                    score=0.5,  # Graph-discovered entries get base score
                                    metadata={
                                        "discovery": "graph_traversal",
                                        "via_node": node_id,
                                        "relation": relation
                                    }
                                ))
        
        # Step 3: Combine results
        all_results = list(vector_results) + graph_context
        
        # Sort by score (vector results first, then graph)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:k]
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract potential entity keywords from text."""
        import re
        
        # Find capitalized words (likely entities)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Deduplicate while preserving order
        seen = set()
        keywords = []
        for word in words:
            if word.lower() not in seen:
                seen.add(word.lower())
                keywords.append(word)
        
        return keywords[:max_keywords]
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry by ID from vector memory."""
        return await self.vector_memory.retrieve(entry_id)
    
    async def delete(self, entry_id: str) -> bool:
        """Delete entry from vector memory (graph nodes persist)."""
        return await self.vector_memory.delete(entry_id)
    
    async def clear(self) -> None:
        """Clear vector memory (graph is reset separately)."""
        await self.vector_memory.clear()
    
    async def count(self) -> int:
        """Return count from vector memory."""
        return await self.vector_memory.count()
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        if isinstance(self.graph_store, NetworkXStore):
            return {
                "nodes": self.graph_store.node_count,
                "edges": self.graph_store.edge_count
            }
        return {}
