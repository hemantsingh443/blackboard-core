"""
Tests for SDK Fixes

Tests for:
1. Event Loop Safety (_run_sync)
2. Telemetry Memory Leak (UsageTracker max_records)
3. MCP Performance (persistent session)
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Test 1: Event Loop Safety
# =============================================================================

def test_run_sync_outside_event_loop():
    """Test _run_sync works when no event loop is running."""
    from blackboard.core import _run_sync
    
    async def simple_coro():
        return 42
    
    result = _run_sync(simple_coro())
    assert result == 42


def test_run_sync_inside_event_loop():
    """Test _run_sync works when called from inside an event loop."""
    from blackboard.core import _run_sync
    
    async def inner_coro():
        return "hello"
    
    async def outer_coro():
        # This simulates calling run_sync from inside an async context
        # (e.g., FastAPI, Jupyter)
        result = _run_sync(inner_coro())
        return result
    
    result = asyncio.run(outer_coro())
    assert result == "hello"


# =============================================================================
# Test 2: Telemetry Memory Leak
# =============================================================================

def test_usage_tracker_max_records():
    """Test UsageTracker evicts old records when limit is reached."""
    from blackboard.usage import UsageTracker, LLMUsage
    
    tracker = UsageTracker(max_records=5)
    
    # Add 10 records
    for i in range(10):
        tracker.record(LLMUsage(input_tokens=i, output_tokens=i), context=f"call_{i}")
    
    # Should only have 5 records in memory
    assert len(tracker._records) == 5
    # But call_count should show all 10
    assert tracker.call_count == 10
    # Evicted count should be 5
    assert tracker._evicted_count == 5


def test_usage_tracker_on_flush_callback():
    """Test on_flush callback is called when records are evicted."""
    from blackboard.usage import UsageTracker, LLMUsage
    
    flushed_records = []
    def on_flush(records):
        flushed_records.extend(records)
    
    tracker = UsageTracker(max_records=3, on_flush=on_flush)
    
    # Add 5 records
    for i in range(5):
        tracker.record(LLMUsage(input_tokens=i, output_tokens=i))
    
    # Should have flushed 2 records
    assert len(flushed_records) == 2


def test_usage_tracker_total_tokens_includes_evicted():
    """Test total_tokens includes evicted records."""
    from blackboard.usage import UsageTracker, LLMUsage
    
    tracker = UsageTracker(max_records=2)
    
    # Add 4 records with 10 tokens each
    for _ in range(4):
        tracker.record(LLMUsage(input_tokens=5, output_tokens=5))
    
    # Total should be 40 (4 * 10), not just 20 (2 * 10)
    assert tracker.total_tokens == 40
    assert tracker.total_cost == 0.0  # No cost configured


def test_usage_tracker_unlimited():
    """Test UsageTracker with max_records=None (unlimited)."""
    from blackboard.usage import UsageTracker, LLMUsage
    
    tracker = UsageTracker(max_records=None)
    
    # Add many records
    for i in range(100):
        tracker.record(LLMUsage(input_tokens=1, output_tokens=1))
    
    # All should be in memory
    assert len(tracker._records) == 100
    assert tracker._evicted_count == 0


# =============================================================================
# Test 3: MCP Performance (Persistent Session)
# =============================================================================

def test_mcp_server_worker_session_management():
    """Test MCPServerWorker session attributes exist."""
    from blackboard.mcp import MCPServerWorker
    
    worker = MCPServerWorker(
        name="Test",
        command="echo",
        args=["hello"]
    )
    
    # Check session management attributes
    assert hasattr(worker, 'is_connected')
    assert hasattr(worker, 'connect')
    assert hasattr(worker, 'disconnect')
    assert hasattr(worker, 'call_tool')
    assert hasattr(worker, '__aenter__')
    assert hasattr(worker, '__aexit__')
    
    # Initially not connected
    assert worker.is_connected == False


def test_mcp_server_worker_context_manager():
    """Test MCPServerWorker async context manager exists."""
    from blackboard.mcp import MCPServerWorker
    import inspect
    
    # Verify __aenter__ and __aexit__ are async
    assert inspect.iscoroutinefunction(MCPServerWorker.__aenter__)
    assert inspect.iscoroutinefunction(MCPServerWorker.__aexit__)


# =============================================================================
# Test 4: Context Window Management
# =============================================================================

def test_context_string_token_limit():
    """Test to_context_string respects token limit."""
    from blackboard.state import Blackboard, Artifact
    
    bb = Blackboard(goal="Test goal")
    # Add a large artifact (10k chars)
    bb.add_artifact(Artifact(type="code", content="x" * 10000, creator="Worker"))
    
    # With small token limit, should truncate
    ctx = bb.to_context_string(max_tokens=500)
    
    # Should be within budget (500 * 4 = 2000 chars)
    assert len(ctx) < 2500  # Some buffer for overhead


def test_context_string_smart_truncation():
    """Test large artifacts get head/tail preview."""
    from blackboard.state import Blackboard, Artifact
    
    bb = Blackboard(goal="Test")
    # Add artifact with distinct head and tail
    content = "HEAD" + "x" * 9000 + "TAIL"
    bb.add_artifact(Artifact(type="code", content=content, creator="Worker"))
    
    ctx = bb.to_context_string(max_tokens=2000)
    
    # Should contain the "chars omitted" marker
    assert "chars omitted" in ctx
    # Should show head
    assert "HEAD" in ctx
    # Should show tail  
    assert "TAIL" in ctx


def test_context_string_prioritizes_feedback():
    """Test feedback is included before artifacts."""
    from blackboard.state import Blackboard, Artifact, Feedback
    
    bb = Blackboard(goal="Test")
    bb.add_artifact(Artifact(type="code", content="x" * 5000, creator="Worker"))
    bb.add_feedback(Feedback(source="Critic", critique="This is broken!", passed=False))
    
    ctx = bb.to_context_string(max_tokens=1000)
    
    # Feedback should be present even with tight budget
    assert "FAILED" in ctx
    assert "This is broken!" in ctx


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
