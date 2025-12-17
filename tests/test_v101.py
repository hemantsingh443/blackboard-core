"""
Tests for v1.0.1 features.

Tests persistence, sandbox, hierarchy, streaming, vectordb, evals, and dynamic loading.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path

from blackboard import Blackboard, Status, Artifact, Worker, WorkerOutput


# =============================================================================
# Persistence Tests
# =============================================================================

class TestPersistence:
    """Tests for persistence layer."""
    
    def test_json_file_persistence(self):
        """Test JSONFilePersistence save and load."""
        from blackboard.persistence import JSONFilePersistence, SessionNotFoundError
        
        async def test():
            with tempfile.TemporaryDirectory() as tmpdir:
                persistence = JSONFilePersistence(tmpdir)
                
                # Create and save state
                state = Blackboard(goal="Test goal")
                state.add_artifact(Artifact(type="text", content="Hello", creator="Test"))
                
                await persistence.save(state, "session-001")
                
                # Check exists
                assert await persistence.exists("session-001")
                
                # Load state
                loaded = await persistence.load("session-001")
                assert loaded.goal == "Test goal"
                assert len(loaded.artifacts) == 1
                assert loaded.version == 2  # Incremented on save
                
                # List sessions
                sessions = await persistence.list_sessions()
                assert "session-001" in sessions
                
                # Delete
                await persistence.delete("session-001")
                assert not await persistence.exists("session-001")
        
        asyncio.run(test())
    
    def test_in_memory_persistence(self):
        """Test InMemoryPersistence."""
        from blackboard.persistence import InMemoryPersistence
        
        async def test():
            persistence = InMemoryPersistence()
            
            state = Blackboard(goal="Test")
            await persistence.save(state, "test-1")
            
            loaded = await persistence.load("test-1")
            assert loaded.goal == "Test"
            
            persistence.clear()
            assert not await persistence.exists("test-1")
        
        asyncio.run(test())
    
    def test_version_conflict(self):
        """Test optimistic locking conflict detection."""
        from blackboard.persistence import InMemoryPersistence, SessionConflictError
        
        async def test():
            persistence = InMemoryPersistence()
            
            state1 = Blackboard(goal="Test")
            await persistence.save(state1, "shared")  # v2
            
            # Simulate another process saving
            state2 = await persistence.load("shared")
            await persistence.save(state2, "shared")  # v3
            
            # Now state1 has v2, disk has v3 - should conflict
            state1.version = 1  # Reset to trigger conflict
            with pytest.raises(SessionConflictError):
                await persistence.save(state1, "shared")
        
        asyncio.run(test())


# =============================================================================
# Sandbox Tests
# =============================================================================

class TestSandbox:
    """Tests for sandbox code execution."""
    
    def test_insecure_local_executor_success(self):
        """Test successful code execution."""
        from blackboard.sandbox import InsecureLocalExecutor
        
        async def test():
            executor = InsecureLocalExecutor(timeout=5)
            result = await executor.execute("print('hello world')")
            
            assert result.success
            assert "hello world" in result.stdout
        
        asyncio.run(test())
    
    def test_insecure_local_executor_error(self):
        """Test code with error."""
        from blackboard.sandbox import InsecureLocalExecutor
        
        async def test():
            executor = InsecureLocalExecutor(timeout=5)
            result = await executor.execute("1/0")
            
            assert not result.success
            assert "ZeroDivisionError" in result.stderr
        
        asyncio.run(test())
    
    def test_insecure_local_executor_timeout(self):
        """Test timeout handling."""
        from blackboard.sandbox import InsecureLocalExecutor, SandboxTimeoutError
        
        async def test():
            executor = InsecureLocalExecutor(timeout=1)
            
            with pytest.raises(SandboxTimeoutError):
                await executor.execute("import time; time.sleep(10)")
        
        asyncio.run(test())
    
    def test_noop_sandbox(self):
        """Test NoOpSandbox for trusted code."""
        from blackboard.sandbox import NoOpSandbox
        
        async def test():
            sandbox = NoOpSandbox()
            result = await sandbox.execute("x = 2 + 2; print(x)")
            
            assert result.success
            assert "4" in result.stdout
        
        asyncio.run(test())


# =============================================================================
# Hierarchy Tests
# =============================================================================

class TestHierarchy:
    """Tests for hierarchical orchestration."""
    
    def test_sub_goal_input(self):
        """Test SubGoalInput model."""
        from blackboard.hierarchy import SubGoalInput
        
        input = SubGoalInput(sub_goal="Research AI safety", max_steps=5)
        assert input.sub_goal == "Research AI safety"
        assert input.max_steps == 5


# =============================================================================
# Streaming Tests
# =============================================================================

class TestStreaming:
    """Tests for streaming support."""
    
    def test_buffered_stream(self):
        """Test BufferedStream for token collection."""
        from blackboard.streaming import BufferedStream
        
        async def test():
            buffer = BufferedStream()
            
            # Producer
            async def produce():
                for token in ["Hello", " ", "world", "!"]:
                    await buffer.add(token)
                await buffer.close()
            
            # Start producer
            asyncio.create_task(produce())
            
            # Consumer
            tokens = []
            async for token in buffer:
                tokens.append(token)
            
            assert "".join(tokens) == "Hello world!"
        
        asyncio.run(test())
    
    def test_wrap_non_streaming(self):
        """Test wrapping non-streaming LLM."""
        from blackboard.streaming import wrap_non_streaming
        
        async def test():
            def fake_llm(prompt):
                return "Response to: " + prompt
            
            streaming = wrap_non_streaming(fake_llm)
            
            tokens = []
            async for token in streaming("Hello"):
                tokens.append(token)
            
            assert tokens == ["Response to: Hello"]
        
        asyncio.run(test())


# =============================================================================
# Evals Tests
# =============================================================================

class TestEvals:
    """Tests for evaluation framework."""
    
    def test_rule_based_judge(self):
        """Test RuleBasedJudge scoring."""
        from blackboard.evals import RuleBasedJudge
        
        async def test():
            judge = RuleBasedJudge([
                ("has_artifacts", lambda bb: len(bb.artifacts) > 0),
                ("is_done", lambda bb: bb.status == Status.DONE),
            ], threshold=0.5)
            
            # Test passing case
            state = Blackboard(goal="Test", status=Status.DONE)
            state.add_artifact(Artifact(type="text", content="Result", creator="Test"))
            
            result = await judge.score(state, [], "Test goal")
            assert result["passed"]
            assert result["score"] == 1.0
            
            # Test failing case
            empty_state = Blackboard(goal="Test")
            result = await judge.score(empty_state, [], "Test goal")
            assert not result["passed"]
        
        asyncio.run(test())
    
    def test_eval_case_and_result(self):
        """Test EvalCase and EvalResult dataclasses."""
        from blackboard.evals import EvalCase, EvalResult
        
        case = EvalCase(
            id="test-1",
            goal="Write a poem",
            expected_criteria=["Has rhymes"]
        )
        assert case.id == "test-1"
        
        result = EvalResult(
            case_id="test-1",
            goal="Write a poem",
            success=True,
            score=0.9
        )
        assert result.success


# =============================================================================
# Dynamic Loading Tests
# =============================================================================

class TestDynamicLoading:
    """Tests for WorkerFactory and LazyWorkerRegistry."""
    
    def test_lazy_worker_registry(self):
        """Test LazyWorkerRegistry loading workers on demand."""
        from blackboard.protocols import LazyWorkerRegistry, WorkerFactory
        
        class DummyWorker(Worker):
            name = "Dummy"
            description = "A dummy worker"
            
            async def run(self, state, inputs=None):
                return WorkerOutput()
        
        class TestFactory:
            def get_worker(self, name):
                if name == "Dummy":
                    return DummyWorker()
                return None
            
            def list_available(self):
                return ["Dummy"]
            
            def get_description(self, name):
                return "A dummy worker"
        
        registry = LazyWorkerRegistry(factory=TestFactory())
        
        # Check worker is listed but not loaded
        assert "Dummy" in registry.list_workers()
        assert "Dummy" in registry
        
        # Now load it
        worker = registry.get("Dummy")
        assert worker is not None
        assert worker.name == "Dummy"


# =============================================================================
# Pause/Resume Tests
# =============================================================================

class TestPauseResume:
    """Tests for pause/resume functionality."""
    
    def test_pending_input_field(self):
        """Test pending_input field in Blackboard."""
        state = Blackboard(goal="Test")
        assert state.pending_input is None
        
        state.pending_input = {"approved": True}
        assert state.pending_input["approved"] is True
    
    def test_pause_status(self):
        """Test PAUSED status."""
        state = Blackboard(goal="Test")
        state.update_status(Status.PAUSED)
        assert state.status == Status.PAUSED
