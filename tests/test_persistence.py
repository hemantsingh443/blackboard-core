"""Tests for state persistence and history management."""

import pytest
import tempfile
from pathlib import Path

from blackboard import Blackboard, Artifact, Feedback, Status


class TestPersistence:
    """Tests for save/load functionality."""
    
    def test_save_to_json(self, tmp_path):
        """Test saving state to JSON."""
        state = Blackboard(goal="Test goal")
        state.add_artifact(Artifact(type="text", content="Hello", creator="Test"))
        state.add_feedback(Feedback(source="Critic", critique="Good", passed=True))
        
        path = tmp_path / "state.json"
        state.save_to_json(path)
        
        assert path.exists()
        content = path.read_text()
        assert "Test goal" in content
        assert "Hello" in content
    
    def test_load_from_json(self, tmp_path):
        """Test loading state from JSON."""
        # Create and save state
        original = Blackboard(goal="Test goal")
        original.add_artifact(Artifact(type="text", content="Hello", creator="Test"))
        original.step_count = 5
        
        path = tmp_path / "state.json"
        original.save_to_json(path)
        
        # Load and verify
        loaded = Blackboard.load_from_json(path)
        
        assert loaded.goal == "Test goal"
        assert loaded.step_count == 5
        assert len(loaded.artifacts) == 1
        assert loaded.artifacts[0].content == "Hello"
    
    def test_round_trip_complex_state(self, tmp_path):
        """Test save/load with complex nested state."""
        state = Blackboard(goal="Complex test")
        state.status = Status.GENERATING
        state.metadata["key"] = {"nested": [1, 2, 3]}
        
        for i in range(3):
            state.add_artifact(Artifact(
                type="code",
                content=f"def func_{i}(): pass",
                creator=f"Worker{i}"
            ))
        
        state.add_feedback(Feedback(source="Critic", critique="Good", passed=True))
        state.add_feedback(Feedback(source="Critic", critique="Bad", passed=False))
        
        path = tmp_path / "complex.json"
        state.save_to_json(path)
        loaded = Blackboard.load_from_json(path)
        
        assert loaded.status == Status.GENERATING
        assert loaded.metadata["key"]["nested"] == [1, 2, 3]
        assert len(loaded.artifacts) == 3
        assert len(loaded.feedback) == 2
    
    def test_to_dict_from_dict(self):
        """Test dictionary serialization."""
        state = Blackboard(goal="Dict test")
        state.add_artifact(Artifact(type="text", content="Test", creator="A"))
        
        data = state.to_dict()
        restored = Blackboard.from_dict(data)
        
        assert restored.goal == "Dict test"
        assert len(restored.artifacts) == 1


class TestHistoryManagement:
    """Tests for sliding window context generation."""
    
    def test_context_string_with_limits(self):
        """Test that context respects max_artifacts limit."""
        state = Blackboard(goal="Test")
        
        # Add 5 artifacts
        for i in range(5):
            state.add_artifact(Artifact(
                type="text",
                content=f"Content {i}",
                creator=f"Worker{i}"
            ))
        
        # Request only last 2
        context = state.to_context_string(max_artifacts=2)
        
        assert "Content 3" in context
        assert "Content 4" in context
        assert "Content 0" not in context
        assert "5 total, showing last 2" in context
    
    def test_context_string_with_feedback_limit(self):
        """Test that context respects max_feedback limit."""
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="text", content="Test", creator="A"))
        
        # Add 5 feedback entries
        for i in range(5):
            state.add_feedback(Feedback(
                source=f"Critic{i}",
                critique=f"Feedback {i}",
                passed=(i % 2 == 0)
            ))
        
        context = state.to_context_string(max_feedback=2)
        
        assert "Feedback 3" in context
        assert "Feedback 4" in context
        assert "Feedback 0" not in context
    
    def test_content_length_truncation(self):
        """Test that long content is truncated via token limits."""
        state = Blackboard(goal="Test")
        long_content = "x" * 1000
        state.add_artifact(Artifact(type="text", content=long_content, creator="A"))
        
        # Use token-based truncation (100 tokens * 4 chars = 400 chars max)
        context = state.to_context_string(max_tokens=100)
        
        # Should be truncated (either with ... or head/tail preview)
        assert len(context) < 2000  # Reasonable limit
    
    def test_context_summary(self):
        """Test brief context summary."""
        state = Blackboard(goal="Test")
        state.step_count = 10
        state.add_artifact(Artifact(type="text", content="Test", creator="A"))
        state.add_feedback(Feedback(source="C", critique="Good", passed=True))
        
        summary = state.get_context_summary()
        
        assert "Steps: 10" in summary
        assert "Artifacts: 1" in summary
        assert "Passed" in summary


# =============================================================================
# SQLitePersistence Tests
# =============================================================================

# Skip all SQLite tests if aiosqlite is not installed (optional dependency)
aiosqlite = pytest.importorskip("aiosqlite", reason="aiosqlite required for SQLite tests")


class TestSQLitePersistence:
    """Tests for SQLitePersistence backend."""
    
    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test.db")
    
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, db_path):
        """Test that initialize creates the schema."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        # Check tables exist
        conn = await persistence._get_connection()
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        
        assert "sessions" in tables
        assert "events" in tables
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_save_and_load(self, db_path):
        """Test basic save and load functionality."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        
        state = Blackboard(goal="Test goal")
        state.add_artifact(Artifact(type="text", content="Hello", creator="Test"))
        
        await persistence.save(state, "session-001")
        loaded = await persistence.load("session-001")
        
        assert loaded.goal == "Test goal"
        assert len(loaded.artifacts) == 1
        assert loaded.artifacts[0].content == "Hello"
        assert loaded.version == 2  # Version starts at 1, incremented to 2 on save
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_version_conflict(self, db_path):
        """Test that version conflicts are detected."""
        from blackboard.persistence import SQLitePersistence, SessionConflictError
        
        persistence = SQLitePersistence(db_path)
        
        # Save initial state
        state1 = Blackboard(goal="Original")
        await persistence.save(state1, "session-001")
        
        # Load and modify
        state2 = await persistence.load("session-001")
        state2.goal = "Modified by worker 1"
        
        # Load again (simulating concurrent access)
        state3 = await persistence.load("session-001")
        state3.goal = "Modified by worker 2"
        
        # Save state2 first (increments version to 2)
        await persistence.save(state2, "session-001")
        
        # Now state3 has version 1 but DB has version 2 -> conflict
        with pytest.raises(SessionConflictError):
            await persistence.save(state3, "session-001")
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_exists_and_delete(self, db_path):
        """Test exists and delete operations."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        
        state = Blackboard(goal="Test")
        await persistence.save(state, "session-001")
        
        assert await persistence.exists("session-001") is True
        assert await persistence.exists("nonexistent") is False
        
        await persistence.delete("session-001")
        assert await persistence.exists("session-001") is False
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, db_path):
        """Test listing sessions."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        
        await persistence.save(Blackboard(goal="A"), "session-a")
        await persistence.save(Blackboard(goal="B"), "session-b")
        await persistence.save(Blackboard(goal="C"), "session-c")
        
        sessions = await persistence.list_sessions()
        
        assert len(sessions) == 3
        assert "session-a" in sessions
        assert "session-b" in sessions
        assert "session-c" in sessions
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_parent_session_tracking(self, db_path):
        """Test parent-child session relationships."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        
        # Create parent session
        parent_state = Blackboard(goal="Parent task")
        await persistence.save(parent_state, "parent-001")
        
        # Create child sessions
        child1 = Blackboard(goal="Child 1")
        child2 = Blackboard(goal="Child 2")
        await persistence.save(child1, "child-001", parent_session_id="parent-001")
        await persistence.save(child2, "child-002", parent_session_id="parent-001")
        
        # List children of parent
        children = await persistence.list_sessions(parent_id="parent-001")
        
        assert len(children) == 2
        assert "child-001" in children
        assert "child-002" in children
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_event_logging(self, db_path):
        """Test event logging functionality."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        
        # Create session
        state = Blackboard(goal="Test")
        await persistence.save(state, "session-001")
        
        # Log events
        event1_id = await persistence.log_event(
            session_id="session-001",
            event_type="worker_called",
            source="Orchestrator",
            step_index=1,
            payload={"worker": "Writer", "inputs": {"task": "Generate text"}}
        )
        
        event2_id = await persistence.log_event(
            session_id="session-001",
            event_type="worker_completed",
            source="Writer",
            step_index=1,
            parent_event_id=event1_id,
            payload={"result": "Hello, world!"}
        )
        
        # Retrieve events
        events = await persistence.get_events("session-001")
        
        assert len(events) == 2
        assert events[0]["event_type"] == "worker_called"
        assert events[0]["step_index"] == 1
        assert events[1]["parent_event_id"] == event1_id
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, db_path):
        """Test filtering events by type."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        
        state = Blackboard(goal="Test")
        await persistence.save(state, "session-001")
        
        # Log mixed events
        await persistence.log_event("session-001", "step_started", payload={"step": 1})
        await persistence.log_event("session-001", "worker_called", payload={"worker": "A"})
        await persistence.log_event("session-001", "step_started", payload={"step": 2})
        await persistence.log_event("session-001", "worker_called", payload={"worker": "B"})
        
        # Filter by type
        step_events = await persistence.get_events("session-001", event_type="step_started")
        worker_events = await persistence.get_events("session-001", event_type="worker_called")
        
        assert len(step_events) == 2
        assert len(worker_events) == 2
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_shared_connection(self, db_path):
        """Test shared connection between parent and child persistence."""
        from blackboard.persistence import SQLitePersistence
        
        parent_persistence = SQLitePersistence(db_path)
        await parent_persistence.initialize()
        
        # Child shares connection
        child_persistence = SQLitePersistence(shared_connection=parent_persistence)
        
        # Both should work with same connection
        await parent_persistence.save(Blackboard(goal="Parent"), "parent")
        await child_persistence.save(Blackboard(goal="Child"), "child")
        
        # Verify both sessions exist
        assert await parent_persistence.exists("parent")
        assert await parent_persistence.exists("child")
        assert await child_persistence.exists("parent")
        assert await child_persistence.exists("child")
        
        # Only close parent (child shares connection)
        await parent_persistence.close()
    
    @pytest.mark.asyncio
    async def test_session_not_found(self, db_path):
        """Test SessionNotFoundError is raised for missing sessions."""
        from blackboard.persistence import SQLitePersistence, SessionNotFoundError
        
        persistence = SQLitePersistence(db_path)
        
        with pytest.raises(SessionNotFoundError):
            await persistence.load("nonexistent")
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, db_path):
        """Test that WAL mode is enabled for concurrency."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        conn = await persistence._get_connection()
        cursor = await conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        
        assert row[0].lower() == "wal"
        
        await persistence.close()

