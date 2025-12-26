"""Tests for time-travel debugging via checkpoints."""

import pytest
import tempfile
from pathlib import Path

from blackboard import Blackboard, Artifact, Feedback, Status


# Skip all tests if aiosqlite is not installed (optional dependency)
aiosqlite = pytest.importorskip("aiosqlite", reason="aiosqlite required for checkpoint tests")


class TestCheckpointPersistence:
    """Tests for checkpoint save/load functionality."""
    
    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test_checkpoints.db")
    
    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, db_path):
        """Test basic checkpoint save and load."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        # Create session and checkpoints at different steps
        state1 = Blackboard(goal="Test checkpoints")
        state1.step_count = 1
        state1.add_artifact(Artifact(type="text", content="Step 1", creator="Writer"))
        
        await persistence.save(state1, "session-001")
        await persistence.save_checkpoint("session-001", 1, state1)
        
        # Advance state and save another checkpoint
        state1.step_count = 2
        state1.add_artifact(Artifact(type="text", content="Step 2", creator="Writer"))
        await persistence.save_checkpoint("session-001", 2, state1)
        
        # Load checkpoint at step 1
        restored = await persistence.load_state_at_step("session-001", 1)
        
        assert restored.step_count == 1
        assert len(restored.artifacts) == 1
        assert restored.artifacts[0].content == "Step 1"
        
        # Load checkpoint at step 2
        restored2 = await persistence.load_state_at_step("session-001", 2)
        
        assert restored2.step_count == 2
        assert len(restored2.artifacts) == 2
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(self, db_path):
        """Test listing checkpoints for a session."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        state = Blackboard(goal="Test")
        await persistence.save(state, "session-001")
        
        # Create checkpoints at various steps
        for step in [1, 3, 5, 7]:
            state.step_count = step
            await persistence.save_checkpoint("session-001", step, state)
        
        checkpoints = await persistence.list_checkpoints("session-001")
        
        assert checkpoints == [1, 3, 5, 7]
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_checkpoint_not_found(self, db_path):
        """Test error when checkpoint doesn't exist."""
        from blackboard.persistence import SQLitePersistence, PersistenceError
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        state = Blackboard(goal="Test")
        await persistence.save(state, "session-001")
        await persistence.save_checkpoint("session-001", 1, state)
        
        # Try to load non-existent checkpoint
        with pytest.raises(PersistenceError) as exc_info:
            await persistence.load_state_at_step("session-001", 99)
        
        assert "No checkpoint at step 99" in str(exc_info.value)
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_checkpoint_upsert(self, db_path):
        """Test that checkpoints are upserted on re-run."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        state = Blackboard(goal="Test")
        await persistence.save(state, "session-001")
        
        # Save checkpoint at step 1
        state.step_count = 1
        state.add_artifact(Artifact(type="text", content="Original", creator="W"))
        await persistence.save_checkpoint("session-001", 1, state)
        
        # Upsert with new data at same step
        state2 = Blackboard(goal="Test")
        state2.step_count = 1
        state2.add_artifact(Artifact(type="text", content="Updated", creator="W"))
        await persistence.save_checkpoint("session-001", 1, state2)
        
        # Should get updated version
        restored = await persistence.load_state_at_step("session-001", 1)
        assert restored.artifacts[0].content == "Updated"
        
        # Should only have one checkpoint at step 1
        checkpoints = await persistence.list_checkpoints("session-001")
        assert checkpoints == [1]
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_delete_checkpoints(self, db_path):
        """Test deleting all checkpoints for a session."""
        from blackboard.persistence import SQLitePersistence
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        state = Blackboard(goal="Test")
        await persistence.save(state, "session-001")
        
        for step in [1, 2, 3]:
            state.step_count = step
            await persistence.save_checkpoint("session-001", step, state)
        
        assert len(await persistence.list_checkpoints("session-001")) == 3
        
        deleted = await persistence.delete_checkpoints("session-001")
        assert deleted == 3
        
        assert await persistence.list_checkpoints("session-001") == []
        
        await persistence.close()


class TestForkSession:
    """Tests for Orchestrator.fork_session functionality."""
    
    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test_fork.db")
    
    @pytest.mark.asyncio
    async def test_fork_session(self, db_path):
        """Test forking a session at a specific step."""
        from blackboard.persistence import SQLitePersistence
        from blackboard import Orchestrator, Worker, WorkerOutput
        
        class MockLLM:
            def generate(self, prompt):
                return '{"action": "done", "reasoning": "complete"}'
        
        class MockWorker(Worker):
            name = "Test"
            description = "Test worker"
            async def run(self, state, inputs=None):
                return WorkerOutput()
        
        persistence = SQLitePersistence(db_path)
        await persistence.initialize()
        
        # Create initial session with checkpoints
        state = Blackboard(goal="Original session")
        state.metadata["session_id"] = "session-001"
        state.step_count = 1
        state.add_artifact(Artifact(type="text", content="Step 1 artifact", creator="W"))
        await persistence.save(state, "session-001")
        await persistence.save_checkpoint("session-001", 1, state)
        
        state.step_count = 2
        state.add_artifact(Artifact(type="text", content="Step 2 artifact", creator="W"))
        await persistence.save(state, "session-001")
        await persistence.save_checkpoint("session-001", 2, state)
        
        # Create orchestrator with persistence
        orchestrator = Orchestrator(llm=MockLLM(), workers=[MockWorker()])
        orchestrator.set_persistence(persistence)
        
        # Fork at step 1
        fork_id = await orchestrator.fork_session("session-001", 1, fork_suffix="test")
        
        assert "session-001_fork_test" == fork_id
        
        # Verify forked session
        forked_state = await persistence.load(fork_id)
        assert forked_state.step_count == 1
        assert len(forked_state.artifacts) == 1
        assert forked_state.metadata["forked_from"] == "session-001"
        assert forked_state.metadata["forked_at_step"] == 1
        
        await persistence.close()
    
    @pytest.mark.asyncio
    async def test_fork_requires_persistence(self):
        """Test that fork_session requires persistence layer."""
        from blackboard import Orchestrator, Worker, WorkerOutput
        
        class MockLLM:
            def generate(self, prompt):
                return '{"action": "done"}'
        
        class MockWorker(Worker):
            name = "Test"
            description = "Test"
            async def run(self, state, inputs=None):
                return WorkerOutput()
        
        orchestrator = Orchestrator(llm=MockLLM(), workers=[MockWorker()])
        
        with pytest.raises(ValueError) as exc:
            await orchestrator.fork_session("session-001", 1)
        
        assert "Persistence layer required" in str(exc.value)
