"""Tests for functional worker decorators and LiteLLM integration."""

import pytest
from typing import Optional

from blackboard import (
    Blackboard, Artifact, Feedback, Status,
    Worker, WorkerOutput, WorkerInput,
    worker, critic
)


class TestWorkerDecorator:
    """Tests for the @worker decorator."""
    
    def test_simple_worker(self):
        """Test creating a simple worker from a function."""
        @worker(name="Adder", description="Adds two numbers")
        def add(a: int, b: int) -> int:
            return a + b
        
        assert isinstance(add, Worker)
        assert add.name == "Adder"
        assert add.description == "Adds two numbers"
    
    @pytest.mark.asyncio
    async def test_worker_execution(self):
        """Test executing a decorated worker."""
        @worker(name="Greeter", description="Says hello")
        def greet(name: str = "World") -> str:
            return f"Hello, {name}!"
        
        state = Blackboard(goal="Greet someone")
        
        # Create inputs with name
        inputs = WorkerInput(instructions="greet")
        inputs.name = "Alice"
        
        output = await greet.run(state, inputs)
        
        assert output.has_artifact()
        assert "Hello" in output.artifact.content
        assert output.artifact.creator == "Greeter"
    
    @pytest.mark.asyncio
    async def test_worker_with_state(self):
        """Test worker that accesses state."""
        @worker(name="Counter", description="Counts artifacts")
        def count_artifacts(state: Blackboard) -> str:
            return f"Found {len(state.artifacts)} artifacts"
        
        state = Blackboard(goal="Count")
        state.add_artifact(Artifact(type="test", content="1", creator="Test"))
        state.add_artifact(Artifact(type="test", content="2", creator="Test"))
        
        output = await count_artifacts.run(state)
        
        assert "Found 2 artifacts" in output.artifact.content
    
    @pytest.mark.asyncio
    async def test_async_worker(self):
        """Test async decorated worker."""
        import asyncio
        
        @worker(name="AsyncWorker", description="Async test")
        async def async_work(state: Blackboard) -> str:
            await asyncio.sleep(0.01)
            return "Async result"
        
        state = Blackboard(goal="Test")
        output = await async_work.run(state)
        
        assert output.artifact.content == "Async result"
    
    def test_worker_artifact_type(self):
        """Test custom artifact type."""
        @worker(name="Coder", description="Writes code", artifact_type="code")
        def write_code() -> str:
            return "def hello(): pass"
        
        assert write_code.name == "Coder"
    
    def test_worker_parallel_safe(self):
        """Test parallel_safe setting."""
        @worker(name="Safe", description="Safe worker", parallel_safe=True)
        def safe_work() -> str:
            return "result"
        
        assert safe_work.parallel_safe is True


class TestCriticDecorator:
    """Tests for the @critic decorator."""
    
    @pytest.mark.asyncio
    async def test_critic_pass(self):
        """Test critic that passes."""
        @critic(name="Approver", description="Always approves")
        def approve(state: Blackboard) -> tuple[bool, str]:
            return True, "Looks good!"
        
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="test", content="content", creator="Test"))
        
        output = await approve.run(state)
        
        assert output.has_feedback()
        assert output.feedback.passed is True
        assert output.feedback.critique == "Looks good!"
    
    @pytest.mark.asyncio
    async def test_critic_fail(self):
        """Test critic that fails."""
        @critic(name="Rejector", description="Always rejects")
        def reject(state: Blackboard) -> tuple[bool, str]:
            return False, "Needs improvement"
        
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="test", content="bad", creator="Test"))
        
        output = await reject.run(state)
        
        assert output.feedback.passed is False
        assert "improvement" in output.feedback.critique
    
    @pytest.mark.asyncio
    async def test_critic_bool_only(self):
        """Test critic returning just a bool."""
        @critic(name="SimpleCritic", description="Simple critic")
        def simple_check() -> bool:
            return True
        
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="test", content="test", creator="Test"))
        
        output = await simple_check.run(state)
        
        assert output.feedback.passed is True
        assert output.feedback.critique == "Approved"


class TestLiteLLMClient:
    """Tests for LiteLLM client (mocked)."""
    
    def test_import(self):
        """Test that LiteLLMClient can be imported."""
        from blackboard.llm import LiteLLMClient, create_llm
        
        # Just verify imports work - actual LLM calls need API keys
        assert LiteLLMClient is not None
        assert create_llm is not None
    
    def test_client_creation(self):
        """Test creating a LiteLLMClient."""
        from blackboard.llm import LiteLLMClient
        
        client = LiteLLMClient(
            model="gpt-4o",
            fallback_models=["gpt-4o-mini"],
            temperature=0.5
        )
        
        assert client.model == "gpt-4o"
        assert client.fallback_models == ["gpt-4o-mini"]
        assert client.temperature == 0.5
    
    def test_repr(self):
        """Test string representation."""
        from blackboard.llm import LiteLLMClient
        
        client = LiteLLMClient(model="claude-3-5-sonnet-20241022")
        assert "claude-3-5-sonnet-20241022" in repr(client)
    
    def test_tool_calling_protocol(self):
        """Test that LiteLLMClient implements ToolCallingLLMClient."""
        from blackboard.llm import LiteLLMClient
        from blackboard.tools import ToolCallingLLMClient
        
        client = LiteLLMClient(model="gpt-4o")
        
        # Verify it passes isinstance check
        assert isinstance(client, ToolCallingLLMClient)
        
        # Verify method exists
        assert hasattr(client, 'generate_with_tools')


class TestTUI:
    """Tests for Terminal UI (with graceful fallback)."""
    
    def test_import(self):
        """Test that TUI can be imported."""
        from blackboard.tui import BlackboardTUI, watch
        
        assert BlackboardTUI is not None
        assert watch is not None
    
    def test_tui_creation_without_rich(self):
        """Test TUI creation (works even without rich installed)."""
        from blackboard.tui import BlackboardTUI
        
        # Should not raise even if rich is not installed
        tui = BlackboardTUI(event_bus=None)
        assert tui is not None
    
    def test_render_state(self):
        """Test rendering state (requires rich)."""
        try:
            from rich.console import Console
            from blackboard.tui import BlackboardTUI
            
            tui = BlackboardTUI()
            if tui._rich_available:
                state = Blackboard(goal="Test goal")
                panel = tui.render_state(state)
                assert panel is not None
        except ImportError:
            pytest.skip("rich not installed")
