"""Tests for the Orchestrator."""

import pytest
from typing import List

from blackboard import (
    Orchestrator, Worker, WorkerOutput, 
    Artifact, Feedback, Blackboard, Status
)


class MockLLM:
    """A mock LLM that returns predefined responses."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_index = 0
        self.prompts = []
    
    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return '{"action": "fail", "reasoning": "No more mock responses"}'


class SimpleWriter(Worker):
    """A simple writer worker for testing."""
    name = "Writer"
    description = "Writes text content"
    
    async def run(self, state: Blackboard) -> WorkerOutput:
        instructions = state.metadata.get("current_instructions", "")
        content = f"Generated content: {instructions}" if instructions else "Default content"
        return WorkerOutput(
            artifact=Artifact(type="text", content=content, creator=self.name)
        )


class SimpleReviewer(Worker):
    """A simple reviewer worker for testing."""
    name = "Reviewer"
    description = "Reviews text content and provides feedback"
    
    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass
    
    async def run(self, state: Blackboard) -> WorkerOutput:
        last_artifact = state.get_last_artifact()
        return WorkerOutput(
            feedback=Feedback(
                source=self.name,
                critique="Approved!" if self.should_pass else "Needs work",
                passed=self.should_pass,
                artifact_id=last_artifact.id if last_artifact else None
            )
        )


class TestOrchestrator:
    """Tests for the Orchestrator class."""
    
    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        llm = MockLLM([])
        workers = [SimpleWriter(), SimpleReviewer()]
        
        orch = Orchestrator(llm=llm, workers=workers)
        
        assert "Writer" in orch.registry
        assert "Reviewer" in orch.registry
    
    @pytest.mark.asyncio
    async def test_simple_success_flow(self):
        """Test a simple successful write -> review -> done flow."""
        llm = MockLLM([
            '{"action": "call", "worker": "Writer", "instructions": "Write hello", "reasoning": "Start"}',
            '{"action": "call", "worker": "Reviewer", "instructions": "Check it", "reasoning": "Review"}',
            '{"action": "done", "reasoning": "Passed review"}'
        ])
        
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWriter(), SimpleReviewer(should_pass=True)]
        )
        
        result = await orch.run(goal="Write a greeting", max_steps=5)
        
        assert result.status == Status.DONE
        assert len(result.artifacts) == 1
        assert len(result.feedback) == 1
        assert result.feedback[0].passed is True
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that the orchestrator handles worker failures."""
        llm = MockLLM([
            '{"action": "call", "worker": "Writer", "instructions": "First try"}',
            '{"action": "call", "worker": "Reviewer", "instructions": "Check"}',
            '{"action": "call", "worker": "Writer", "instructions": "Fix based on feedback"}',
            '{"action": "done", "reasoning": "Good enough"}'
        ])
        
        # First review fails, but we don't re-review in this simple test
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWriter(), SimpleReviewer(should_pass=False)]
        )
        
        result = await orch.run(goal="Write something good", max_steps=5)
        
        # Should have 2 artifacts (first try + retry)
        assert len(result.artifacts) == 2
    
    @pytest.mark.asyncio
    async def test_max_steps_reached(self):
        """Test that orchestrator stops at max steps."""
        llm = MockLLM([
            '{"action": "call", "worker": "Writer", "instructions": "Write"}',
            '{"action": "call", "worker": "Writer", "instructions": "Write more"}',
            '{"action": "call", "worker": "Writer", "instructions": "Keep writing"}',
        ])
        
        orch = Orchestrator(llm=llm, workers=[SimpleWriter()])
        result = await orch.run(goal="Test", max_steps=3)
        
        assert result.status == Status.FAILED
        assert result.step_count == 3
    
    @pytest.mark.asyncio
    async def test_unknown_worker(self):
        """Test handling of unknown worker names."""
        llm = MockLLM([
            '{"action": "call", "worker": "NonExistent", "instructions": "Do something"}',
            '{"action": "done", "reasoning": "Give up"}'
        ])
        
        orch = Orchestrator(llm=llm, workers=[SimpleWriter()])
        result = await orch.run(goal="Test", max_steps=3)
        
        # Should continue despite unknown worker
        assert result.status == Status.DONE
    
    @pytest.mark.asyncio
    async def test_step_callback(self):
        """Test that step callback is called."""
        llm = MockLLM([
            '{"action": "call", "worker": "Writer", "instructions": "Write"}',
            '{"action": "done", "reasoning": "Done"}'
        ])
        
        steps_seen = []
        
        def on_step(step, state, decision):
            steps_seen.append((step, decision.action))
        
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWriter()],
            on_step=on_step
        )
        
        await orch.run(goal="Test", max_steps=5)
        
        assert len(steps_seen) == 2
        assert steps_seen[0][1] == "call"
        assert steps_seen[1][1] == "done"
    
    def test_run_sync(self):
        """Test the synchronous wrapper."""
        llm = MockLLM(['{"action": "done", "reasoning": "Quick"}'])
        orch = Orchestrator(llm=llm, workers=[SimpleWriter()])
        
        # Use sync wrapper
        result = orch.run_sync(goal="Quick test", max_steps=1)
        assert result.status == Status.DONE


class TestLLMProtocol:
    """Test that various LLM implementations work."""
    
    def test_mock_llm_satisfies_protocol(self):
        """Test that MockLLM works with the orchestrator."""
        from blackboard.core import LLMClient
        
        llm = MockLLM(['{"action": "done"}'])
        assert isinstance(llm, LLMClient)
    
    @pytest.mark.asyncio
    async def test_callable_as_llm(self):
        """Test using a simple class as LLM client."""
        
        class SimpleLLM:
            def generate(self, prompt: str) -> str:
                return '{"action": "done", "reasoning": "Simple"}'
        
        orch = Orchestrator(llm=SimpleLLM(), workers=[SimpleWriter()])
        result = await orch.run(goal="Quick test", max_steps=1)
        
        assert result.status == Status.DONE
