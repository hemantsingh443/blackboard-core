"""Tests for the Agent (fractal agent) implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestAgentClass:
    """Tests for the Agent class."""
    
    def test_agent_is_worker(self):
        """Test that Agent implements Worker protocol."""
        from blackboard.core import Agent
        from blackboard.protocols import Worker
        
        # Create a mock LLM
        mock_llm = MagicMock()
        
        agent = Agent(
            name="TestAgent",
            description="A test agent",
            llm=mock_llm,
            workers=[]
        )
        
        # Check it has Worker attributes
        assert hasattr(agent, "name")
        assert hasattr(agent, "description")
        assert hasattr(agent, "run")
        assert agent.name == "TestAgent"
    
    def test_agent_has_recursion_depth(self):
        """Test that Agent respects recursion depth."""
        from blackboard.core import Agent
        from blackboard.config import BlackboardConfig
        
        mock_llm = MagicMock()
        config = BlackboardConfig(max_recursion_depth=2)
        
        agent = Agent(
            name="TestAgent",
            description="Test",
            llm=mock_llm,
            workers=[],
            config=config,
            current_depth=0
        )
        
        assert agent.current_depth == 0
        assert agent.config.max_recursion_depth == 2
    
    @pytest.mark.asyncio
    async def test_recursion_depth_exceeded_error(self):
        """Test that RecursionDepthExceededError is raised when depth exceeded."""
        from blackboard.core import Agent, RecursionDepthExceededError
        from blackboard.config import BlackboardConfig
        from blackboard.state import Blackboard
        from blackboard.protocols import WorkerInput
        
        mock_llm = MagicMock()
        config = BlackboardConfig(max_recursion_depth=1)
        
        # Agent at depth 1 when max is 1 should fail
        agent = Agent(
            name="TestAgent",
            description="Test",
            llm=mock_llm,
            workers=[],
            config=config,
            current_depth=1  # Already at max depth
        )
        
        state = Blackboard(goal="Test")
        inputs = WorkerInput(instructions="Do something")
        
        with pytest.raises(RecursionDepthExceededError) as exc_info:
            await agent.run(state, inputs)
        
        assert "exceeded max recursion depth" in str(exc_info.value).lower()
    
    def test_for_child_agent_config(self):
        """Test that for_child_agent creates proper config."""
        from blackboard.config import BlackboardConfig
        
        parent = BlackboardConfig(
            max_recursion_depth=3,
            allow_unsafe_execution=True,
            max_steps=50
        )
        
        child = parent.for_child_agent()
        
        assert child.max_recursion_depth == 2  # Decremented
        assert child.allow_unsafe_execution is True  # Inherited
        assert child.max_steps == 50  # Inherited
    
    def test_agent_get_schema_json(self):
        """Test that Agent provides proper JSON schema."""
        from blackboard.core import Agent
        
        mock_llm = MagicMock()
        agent = Agent(
            name="ResearchAgent",
            description="Researches topics",
            llm=mock_llm,
            workers=[]
        )
        
        schema = agent.get_schema_json()
        
        assert schema is not None
        assert "properties" in schema
        assert "instructions" in schema["properties"]
        assert "ResearchAgent" in schema["properties"]["instructions"]["description"]


class TestWorkerOutputTraceId:
    """Tests for WorkerOutput trace_id field."""
    
    def test_worker_output_has_trace_id(self):
        """Test that WorkerOutput has trace_id field."""
        from blackboard.protocols import WorkerOutput
        
        output = WorkerOutput(trace_id="session-123")
        
        assert output.trace_id == "session-123"
        assert output.has_trace() is True
    
    def test_worker_output_trace_id_optional(self):
        """Test that trace_id is optional."""
        from blackboard.protocols import WorkerOutput
        
        output = WorkerOutput()
        
        assert output.trace_id is None
        assert output.has_trace() is False
    
    def test_worker_output_with_artifact_and_trace(self):
        """Test WorkerOutput with both artifact and trace."""
        from blackboard.protocols import WorkerOutput
        from blackboard.state import Artifact
        
        artifact = Artifact(type="text", content="Hello", creator="Agent")
        output = WorkerOutput(
            artifact=artifact,
            trace_id="agent-abc123"
        )
        
        assert output.has_artifact() is True
        assert output.has_trace() is True
        assert output.artifact.content == "Hello"
        assert output.trace_id == "agent-abc123"
