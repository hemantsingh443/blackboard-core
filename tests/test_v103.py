"""
Tests for v1.0.3 features.

- Memory.count() method
- ConsoleLoggingMiddleware
"""

import pytest
from blackboard import Orchestrator, Worker, WorkerOutput, Artifact, Blackboard
from blackboard.memory import SimpleVectorMemory
from blackboard.middleware import ConsoleLoggingMiddleware


class TestMemoryCount:
    """Tests for Memory.count() method."""
    
    @pytest.mark.asyncio
    async def test_simple_vector_memory_count(self):
        """Test count() on SimpleVectorMemory."""
        memory = SimpleVectorMemory()
        
        assert await memory.count() == 0
        
        await memory.add("First memory")
        assert await memory.count() == 1
        
        await memory.add("Second memory")
        assert await memory.count() == 2
        
        await memory.clear()
        assert await memory.count() == 0
    
    @pytest.mark.asyncio
    async def test_count_after_delete(self):
        """Test count() after deleting entries."""
        memory = SimpleVectorMemory()
        
        entry1 = await memory.add("Memory 1")
        entry2 = await memory.add("Memory 2")
        
        assert await memory.count() == 2
        
        await memory.delete(entry1.id)
        assert await memory.count() == 1


class TestConsoleLoggingMiddleware:
    """Tests for ConsoleLoggingMiddleware."""
    
    def test_middleware_initialization(self):
        """Test ConsoleLoggingMiddleware can be instantiated."""
        middleware = ConsoleLoggingMiddleware()
        assert middleware.use_colors is True
        assert middleware.show_state is False
        
        middleware_no_colors = ConsoleLoggingMiddleware(use_colors=False)
        assert middleware_no_colors.use_colors is False
    
    def test_color_formatting(self):
        """Test color formatting helper."""
        middleware = ConsoleLoggingMiddleware(use_colors=True)
        
        colored = middleware._c("green", "test")
        assert "\033[32m" in colored  # Green ANSI code
        assert "test" in colored
        assert "\033[0m" in colored  # Reset code
        
        middleware_no_colors = ConsoleLoggingMiddleware(use_colors=False)
        plain = middleware_no_colors._c("green", "test")
        assert plain == "test"  # No ANSI codes
    
    @pytest.mark.asyncio
    async def test_middleware_in_orchestrator(self):
        """Test ConsoleLoggingMiddleware runs without crashing."""
        class DummyLLM:
            def generate(self, prompt):
                return '{"action": "done", "reason": "Complete"}'
        
        class DummyWorker(Worker):
            name = "Dummy"
            description = "Does nothing"
            
            async def run(self, state, inputs=None):
                return WorkerOutput(
                    artifact=Artifact(type="test", content="done", creator=self.name)
                )
        
        middleware = ConsoleLoggingMiddleware(use_colors=False)
        
        orchestrator = Orchestrator(
            llm=DummyLLM(),
            workers=[DummyWorker()],
            middleware=[middleware]
        )
        
        # Should not crash
        result = await orchestrator.run(goal="Test goal", max_steps=1)
        assert result is not None
