"""
Tests for the functional worker decorators.

Verifies:
- Automatic schema generation from type hints
- Docstring extraction for descriptions
- State injection
- Async function support
- Critic decorator
"""

import pytest
import asyncio
from typing import Optional

from blackboard.decorators import worker, critic, FunctionalWorker, CriticWorker
from blackboard.protocols import WorkerInput, WorkerOutput
from blackboard.state import Blackboard, Artifact, Feedback


class TestWorkerDecorator:
    """Tests for the @worker decorator."""
    
    def test_basic_worker_creation(self):
        """Worker created from simple function."""
        @worker
        def greet(name: str) -> str:
            """Greets a person."""
            return f"Hello, {name}!"
        
        assert isinstance(greet, FunctionalWorker)
        assert greet.name == "Greet"
        assert greet.description == "Greets a person."
        assert greet.input_schema is not None
    
    def test_worker_with_explicit_name(self):
        """Worker respects explicit name parameter."""
        @worker(name="CustomGreeter")
        def greet(name: str) -> str:
            """Says hello."""
            return f"Hi, {name}!"
        
        assert greet.name == "CustomGreeter"
    
    def test_worker_with_explicit_description(self):
        """Worker respects explicit description parameter."""
        @worker(description="A custom description")
        def do_something() -> str:
            """This docstring is ignored."""
            return "done"
        
        assert do_something.description == "A custom description"
    
    def test_schema_generation_with_defaults(self):
        """Schema correctly handles default values."""
        @worker
        def process(required: str, optional: int = 10) -> str:
            return f"{required}: {optional}"
        
        schema = process.input_schema
        assert schema is not None
        
        # Verify schema fields
        fields = schema.model_fields
        assert "required" in fields
        assert "optional" in fields
        assert fields["optional"].default == 10
    
    def test_schema_generation_excludes_state(self):
        """Schema does not include 'state' parameter."""
        @worker
        def analyze(topic: str, state: Blackboard) -> str:
            return f"Analyzing {topic}"
        
        schema = analyze.input_schema
        fields = schema.model_fields
        
        assert "topic" in fields
        assert "state" not in fields
    
    @pytest.mark.asyncio
    async def test_worker_execution_simple(self):
        """Worker executes and returns artifact."""
        @worker
        def add(a: int, b: int) -> int:
            return a + b
        
        state = Blackboard(goal="test")
        
        # Create input
        schema = add.input_schema
        inputs = schema(a=5, b=3)
        
        result = await add.run(state, inputs)
        
        assert isinstance(result, WorkerOutput)
        assert result.artifact is not None
        assert result.artifact.content == "8"
    
    @pytest.mark.asyncio
    async def test_worker_with_state_injection(self):
        """Worker receives state when requested."""
        @worker
        def summarize(state: Blackboard) -> str:
            return f"Goal: {state.goal}"
        
        state = Blackboard(goal="Write a poem")
        result = await summarize.run(state, None)
        
        assert result.artifact.content == "Goal: Write a poem"
    
    @pytest.mark.asyncio
    async def test_worker_mixed_params(self):
        """Worker handles both user inputs and state."""
        @worker
        def analyze(topic: str, depth: int, state: Blackboard) -> str:
            return f"Analyzing {topic} at depth {depth} for goal: {state.goal}"
        
        state = Blackboard(goal="Research AI")
        schema = analyze.input_schema
        inputs = schema(topic="agents", depth=3)
        
        result = await analyze.run(state, inputs)
        
        assert "agents" in result.artifact.content
        assert "depth 3" in result.artifact.content
        assert "Research AI" in result.artifact.content
    
    @pytest.mark.asyncio
    async def test_async_worker(self):
        """Async functions work correctly."""
        @worker
        async def slow_process(data: str) -> str:
            await asyncio.sleep(0.01)
            return f"Processed: {data}"
        
        state = Blackboard(goal="test")
        schema = slow_process.input_schema
        inputs = schema(data="hello")
        
        result = await slow_process.run(state, inputs)
        
        assert result.artifact.content == "Processed: hello"
    
    def test_worker_returns_worker_output(self):
        """Worker can return WorkerOutput directly."""
        @worker
        def custom_output() -> WorkerOutput:
            return WorkerOutput(
                artifact=Artifact(type="custom", content="Custom!", creator="test"),
                metadata={"key": "value"}
            )
        
        # Schema should still be created (empty in this case)
        assert custom_output.input_schema is not None
    
    @pytest.mark.asyncio
    async def test_worker_returns_artifact(self):
        """Worker can return Artifact directly."""
        @worker
        def make_artifact(name: str) -> Artifact:
            return Artifact(type="doc", content=f"Doc: {name}", creator="test")
        
        state = Blackboard(goal="test")
        schema = make_artifact.input_schema
        inputs = schema(name="readme")
        
        result = await make_artifact.run(state, inputs)
        
        assert result.artifact.type == "doc"
        assert result.artifact.content == "Doc: readme"


class TestCriticDecorator:
    """Tests for the @critic decorator."""
    
    def test_critic_creation(self):
        """Critic worker created correctly."""
        @critic(name="Reviewer", description="Reviews stuff")
        def review(state: Blackboard) -> bool:
            return True
        
        assert isinstance(review, CriticWorker)
        assert review.name == "Reviewer"
        assert review.description == "Reviews stuff"
    
    @pytest.mark.asyncio
    async def test_critic_returns_bool(self):
        """Critic handles simple bool return."""
        @critic(name="SimpleReviewer", description="Simple review")
        def simple_review(state: Blackboard) -> bool:
            return True
        
        state = Blackboard(goal="test")
        result = await simple_review.run(state, None)
        
        assert result.feedback is not None
        assert result.feedback.passed is True
        assert result.feedback.critique == "Approved"
    
    @pytest.mark.asyncio
    async def test_critic_returns_tuple(self):
        """Critic handles (bool, str) return."""
        @critic(name="DetailedReviewer", description="Detailed review")
        def detailed_review(state: Blackboard) -> tuple:
            return False, "Missing tests"
        
        state = Blackboard(goal="test")
        result = await detailed_review.run(state, None)
        
        assert result.feedback.passed is False
        assert result.feedback.critique == "Missing tests"
    
    @pytest.mark.asyncio
    async def test_critic_links_to_artifact(self):
        """Feedback links to the last artifact."""
        @critic(name="ArtifactReviewer", description="Reviews artifacts")
        def review_artifact(state: Blackboard) -> bool:
            return True
        
        state = Blackboard(goal="test")
        artifact = Artifact(type="code", content="print('hi')", creator="Writer")
        state.add_artifact(artifact)
        
        result = await review_artifact.run(state, None)
        
        assert result.feedback.artifact_id == artifact.id
    
    @pytest.mark.asyncio
    async def test_async_critic(self):
        """Async critic functions work."""
        @critic(name="AsyncReviewer", description="Async review")
        async def async_review(state: Blackboard) -> tuple:
            await asyncio.sleep(0.01)
            return True, "All good!"
        
        state = Blackboard(goal="test")
        result = await async_review.run(state, None)
        
        assert result.feedback.passed is True
        assert result.feedback.critique == "All good!"


class TestEdgeCases:
    """Edge case and error handling tests."""
    
    def test_worker_no_params(self):
        """Worker with no parameters works."""
        @worker
        def no_params() -> str:
            """Does nothing special."""
            return "nothing"
        
        assert no_params.input_schema is not None
    
    def test_worker_fallback_description(self):
        """Worker without docstring gets fallback description."""
        @worker
        def no_docstring(x: int) -> int:
            return x * 2
        
        assert "NoDocstring" in no_docstring.description or "Worker" in no_docstring.description
    
    @pytest.mark.asyncio
    async def test_worker_none_return(self):
        """Worker handles None return gracefully."""
        @worker
        def returns_none() -> None:
            pass
        
        state = Blackboard(goal="test")
        result = await returns_none.run(state, None)
        
        assert result.artifact.content == "Done"
