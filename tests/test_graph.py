"""
Tests for Graph Memory System

Tests the graph.py module including NetworkXStore, GraphMemory, and triple extraction.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from blackboard.memory import MemoryEntry, SearchResult
from blackboard.graph import Triple, extract_triples_simple

# Check if networkx is available
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Skip all tests in this module if networkx is not installed
pytestmark = pytest.mark.skipif(
    not HAS_NETWORKX, 
    reason="networkx not installed (optional dependency)"
)

# Import store/memory only if networkx is available (to avoid import errors)
if HAS_NETWORKX:
    from blackboard.graph import NetworkXStore, GraphMemory


# =============================================================================
# Triple Extraction Tests
# =============================================================================

class TestTripleExtraction:
    """Tests for triple extraction utilities."""
    
    def test_triple_creation(self):
        """Test Triple dataclass."""
        triple = Triple(
            subject="Alice",
            predicate="works_at",
            object="Google"
        )
        
        assert triple.subject == "Alice"
        assert triple.predicate == "works_at"
        assert triple.object == "Google"
        assert "Alice" in str(triple)
        assert "works_at" in str(triple)
    
    def test_extract_triples_simple_works_at(self):
        """Test simple extraction of 'works at' pattern."""
        text = "Alice works at Google"
        triples = extract_triples_simple(text)
        
        assert len(triples) >= 1
        assert any(t.predicate == "works_at" for t in triples)
    
    def test_extract_triples_simple_located_in(self):
        """Test simple extraction of 'located in' pattern."""
        text = "Google is located in California"
        triples = extract_triples_simple(text)
        
        assert len(triples) >= 1
        assert any(t.predicate == "located_in" for t in triples)
    
    def test_extract_triples_simple_no_match(self):
        """Test extraction when no patterns match."""
        text = "The quick brown fox jumps over the lazy dog"
        triples = extract_triples_simple(text)
        
        assert len(triples) == 0


# =============================================================================
# NetworkX Store Tests
# =============================================================================

class TestNetworkXStore:
    """Tests for NetworkX graph store."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh NetworkX store."""
        return NetworkXStore()
    
    def test_add_node(self, store):
        """Test adding nodes."""
        store.add_node("alice", {"label": "Alice", "type": "person"})
        
        node = store.get_node("alice")
        assert node is not None
        assert node["label"] == "Alice"
        assert node["type"] == "person"
    
    def test_add_node_normalizes_id(self, store):
        """Test that node IDs are normalized."""
        store.add_node("Alice", {"label": "Alice"})
        
        # Should be retrievable with lowercase
        node = store.get_node("alice")
        assert node is not None
    
    def test_add_edge(self, store):
        """Test adding edges."""
        store.add_edge("alice", "google", "works_at")
        
        # Both nodes should exist
        assert store.get_node("alice") is not None
        assert store.get_node("google") is not None
        assert store.edge_count == 1
    
    def test_get_neighbors_out(self, store):
        """Test getting outgoing neighbors."""
        store.add_edge("alice", "google", "works_at")
        store.add_edge("alice", "python", "knows")
        
        neighbors = store.get_neighbors("alice", direction="out")
        
        assert len(neighbors) == 2
        neighbor_ids = [n[0] for n in neighbors]
        assert "google" in neighbor_ids
        assert "python" in neighbor_ids
    
    def test_get_neighbors_in(self, store):
        """Test getting incoming neighbors."""
        store.add_edge("alice", "google", "works_at")
        
        neighbors = store.get_neighbors("google", direction="in")
        
        assert len(neighbors) == 1
        assert neighbors[0][0] == "alice"
        assert neighbors[0][1] == "works_at"
    
    def test_get_neighbors_multi_hop(self, store):
        """Test multi-hop traversal."""
        store.add_edge("alice", "google", "works_at")
        store.add_edge("google", "california", "located_in")
        
        # 1 hop should only get google
        neighbors_1 = store.get_neighbors("alice", direction="out", max_hops=1)
        assert len(neighbors_1) == 1
        
        # 2 hops should get google and california
        neighbors_2 = store.get_neighbors("alice", direction="out", max_hops=2)
        assert len(neighbors_2) == 2
    
    def test_search_nodes(self, store):
        """Test node search."""
        store.add_node("google", {"label": "Google Inc"})
        store.add_node("microsoft", {"label": "Microsoft Corp"})
        
        # Search by ID
        results = store.search_nodes("google")
        assert "google" in results
        
        # Search by label (partial match)
        results = store.search_nodes("corp")
        assert "microsoft" in results
    
    def test_persistence(self, store, tmp_path):
        """Test save and load."""
        store.add_edge("alice", "google", "works_at")
        
        path = str(tmp_path / "graph.pkl")
        store.save(path)
        
        # Create new store and load
        new_store = NetworkXStore()
        new_store.load(path)
        
        assert new_store.node_count == store.node_count
        assert new_store.edge_count == store.edge_count
    
    def test_node_count(self, store):
        """Test node counting."""
        assert store.node_count == 0
        
        store.add_node("a", {})
        store.add_node("b", {})
        
        assert store.node_count == 2
    
    def test_edge_count(self, store):
        """Test edge counting."""
        assert store.edge_count == 0
        
        store.add_edge("a", "b", "relates")
        
        assert store.edge_count == 1


# =============================================================================
# Graph Memory Tests
# =============================================================================

class TestGraphMemory:
    """Tests for GraphMemory class."""
    
    @pytest.fixture
    def mock_vector_memory(self):
        """Create a mock vector memory."""
        memory = AsyncMock()
        memory.store = AsyncMock(return_value="entry-123")
        memory.search = AsyncMock(return_value=[])
        memory.retrieve = AsyncMock(return_value=None)
        memory.delete = AsyncMock(return_value=True)
        memory.clear = AsyncMock()
        memory.count = AsyncMock(return_value=0)
        return memory
    
    @pytest.fixture
    def graph_memory(self, mock_vector_memory):
        """Create a GraphMemory instance with mocked vector memory."""
        return GraphMemory(
            vector_memory=mock_vector_memory,
            max_hops=2
        )
    
    @pytest.mark.asyncio
    async def test_store_extracts_triples(self, graph_memory, mock_vector_memory):
        """Test that store extracts triples and adds to graph."""
        entry = MemoryEntry(content="Alice works at Google")
        
        entry_id = await graph_memory.store(entry)
        
        # Should store in vector memory
        mock_vector_memory.store.assert_called_once()
        assert entry_id == "entry-123"
        
        # Should have nodes in graph
        stats = graph_memory.get_graph_stats()
        assert stats["nodes"] >= 2
        assert stats["edges"] >= 1
    
    @pytest.mark.asyncio
    async def test_store_with_no_triples(self, graph_memory, mock_vector_memory):
        """Test store with text that has no extractable triples."""
        entry = MemoryEntry(content="Hello world")
        
        entry_id = await graph_memory.store(entry)
        
        # Should still store in vector memory
        mock_vector_memory.store.assert_called_once()
        assert entry_id == "entry-123"
    
    @pytest.mark.asyncio
    async def test_search_uses_vector_memory(self, graph_memory, mock_vector_memory):
        """Test that search queries vector memory first."""
        mock_vector_memory.search.return_value = [
            SearchResult(
                entry=MemoryEntry(id="v1", content="Test content"),
                score=0.9
            )
        ]
        
        results = await graph_memory.search("test query")
        
        mock_vector_memory.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_delegates(self, graph_memory, mock_vector_memory):
        """Test that retrieve delegates to vector memory."""
        await graph_memory.retrieve("entry-123")
        mock_vector_memory.retrieve.assert_called_once_with("entry-123")
    
    @pytest.mark.asyncio
    async def test_delete_delegates(self, graph_memory, mock_vector_memory):
        """Test that delete delegates to vector memory."""
        await graph_memory.delete("entry-123")
        mock_vector_memory.delete.assert_called_once_with("entry-123")
    
    @pytest.mark.asyncio
    async def test_clear_delegates(self, graph_memory, mock_vector_memory):
        """Test that clear delegates to vector memory."""
        await graph_memory.clear()
        mock_vector_memory.clear.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_count_delegates(self, graph_memory, mock_vector_memory):
        """Test that count delegates to vector memory."""
        await graph_memory.count()
        mock_vector_memory.count.assert_called_once()
    
    def test_get_graph_stats(self, graph_memory):
        """Test graph statistics."""
        stats = graph_memory.get_graph_stats()
        
        assert "nodes" in stats
        assert "edges" in stats
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
    
    def test_extract_keywords(self, graph_memory):
        """Test keyword extraction."""
        text = "Alice from Google visited Microsoft headquarters"
        keywords = graph_memory._extract_keywords(text)
        
        assert "Alice" in keywords
        assert "Google" in keywords
        assert "Microsoft" in keywords
