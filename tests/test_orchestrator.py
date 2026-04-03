"""Tests for the Orchestrator."""

import pytest
from typing import List, Optional

from blackboard import (
    Orchestrator, Worker, WorkerOutput, WorkerInput,
    Artifact, Feedback, Blackboard, Status
)
from blackboard.flow import Blueprint, Step, SequentialPipeline
from blackboard.tools import ToolCall
from blackboard.usage import LLMUsage


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


class MockToolCallingLLM:
    """A mock tool-calling LLM that records the exposed tools per step."""

    def __init__(self, tool_responses, json_responses=None, fail_tool_calls: bool = False):
        self.tool_responses = tool_responses
        self.json_responses = json_responses or []
        self.fail_tool_calls = fail_tool_calls
        self.tool_call_index = 0
        self.json_call_index = 0
        self.tool_history = []
        self.prompts = []
        self.last_usage = None

    def generate_with_tools(self, prompt, tools):
        self.prompts.append(prompt)
        self.tool_history.append([tool.name for tool in tools])

        if self.fail_tool_calls:
            raise RuntimeError("native tool calling unavailable")

        self.last_usage = LLMUsage(
            input_tokens=40,
            output_tokens=10,
            model="tool-mock"
        )
        response = self.tool_responses[self.tool_call_index]
        self.tool_call_index += 1
        return response

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = self.json_responses[self.json_call_index]
        self.json_call_index += 1
        return response


class SimpleWriter(Worker):
    """A simple writer worker for testing."""
    name = "Writer"
    description = "Writes text content"
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        instructions = inputs.instructions if inputs else ""
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
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
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

    @pytest.mark.asyncio
    async def test_blueprint_advances_after_single_success_and_auto_completes(self):
        """Blueprint phases should advance after one successful worker call."""
        writer = SimpleWriter()
        reviewer = SimpleReviewer(should_pass=True)
        llm = MockLLM([
            '{"action": "call", "worker": "Writer", "instructions": "Draft"}',
            '{"action": "call", "worker": "Reviewer", "instructions": "Review"}',
        ])

        orchestrator = Orchestrator(llm=llm, workers=[writer, reviewer])
        blueprint = SequentialPipeline([writer, reviewer])

        result = await orchestrator.run(
            goal="Write and review",
            max_steps=5,
            blueprint=blueprint
        )

        assert result.status == Status.DONE
        assert result.metadata["_blueprint_step_index"] == len(blueprint.steps)
        assert llm.call_index == 2
        assert len(result.artifacts) == 1
        assert len(result.feedback) == 1

    @pytest.mark.asyncio
    async def test_blueprint_blocks_early_done_and_reasks_once(self):
        """Strict blueprints should reject early done decisions and re-ask."""
        llm = MockLLM([
            '{"action": "done", "reasoning": "Looks complete"}',
            '{"action": "call", "worker": "Writer", "instructions": "Write the draft"}',
        ])

        strict_blueprint = Blueprint(
            name="Strict Draft",
            allow_skip_to_done=False,
            steps=[
                Step(
                    name="draft",
                    description="Create the first draft",
                    allowed_workers=["Writer"],
                    instructions="Use Writer to produce the draft.",
                    max_iterations=1
                )
            ]
        )

        orchestrator = Orchestrator(llm=llm, workers=[SimpleWriter()])
        result = await orchestrator.run(
            goal="Produce a draft",
            max_steps=3,
            blueprint=strict_blueprint
        )

        assert result.status == Status.DONE
        assert llm.call_index == 2
        assert "Correction" in llm.prompts[1]
        assert "You may NOT mark the task done" in llm.prompts[0]
        assert len(result.artifacts) == 1

    @pytest.mark.asyncio
    async def test_tool_calling_filters_workers_by_blueprint_phase(self):
        """Tool calling should only expose workers allowed in the current phase."""
        writer = SimpleWriter()
        reviewer = SimpleReviewer(should_pass=True)
        llm = MockToolCallingLLM([
            [ToolCall(id="call-1", name="Writer", arguments={"instructions": "Draft"})],
            [ToolCall(id="call-2", name="Reviewer", arguments={"instructions": "Review"})],
        ])

        blueprint = Blueprint(
            name="Strict Pipeline",
            allow_skip_to_done=False,
            steps=[
                Step(name="draft", allowed_workers=["Writer"], max_iterations=1),
                Step(name="review", allowed_workers=["Reviewer"], max_iterations=1),
            ]
        )

        orchestrator = Orchestrator(llm=llm, workers=[writer, reviewer])
        result = await orchestrator.run(
            goal="Write then review",
            max_steps=4,
            blueprint=blueprint
        )

        assert result.status == Status.DONE
        assert llm.tool_history[0] == ["Writer", "mark_failed"]
        assert llm.tool_history[1] == ["Reviewer", "mark_failed"]

    @pytest.mark.asyncio
    async def test_tool_calling_fallback_preserves_blueprint_constraints(self):
        """JSON fallback should keep the active blueprint restrictions."""
        writer = SimpleWriter()

        class BlockedWorker(Worker):
            name = "BlockedWorker"
            description = "Should never be exposed in this phase"

            async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
                return WorkerOutput(
                    artifact=Artifact(type="text", content="blocked", creator=self.name)
                )

        llm = MockToolCallingLLM(
            tool_responses=[],
            json_responses=[
                '{"action": "call", "worker": "Writer", "instructions": "Draft"}'
            ],
            fail_tool_calls=True
        )

        strict_blueprint = Blueprint(
            name="Fallback Pipeline",
            allow_skip_to_done=False,
            steps=[
                Step(
                    name="draft",
                    allowed_workers=["Writer"],
                    instructions="Only Writer may run in this phase.",
                    max_iterations=1
                )
            ]
        )

        orchestrator = Orchestrator(llm=llm, workers=[writer, BlockedWorker()])
        result = await orchestrator.run(
            goal="Draft with fallback",
            max_steps=3,
            blueprint=strict_blueprint
        )

        assert result.status == Status.DONE
        assert llm.json_call_index == 1
        assert "Writer" in llm.prompts[-1]
        assert "BlockedWorker" not in llm.prompts[-1]
        assert "Only Writer may run in this phase." in llm.prompts[-1]
    
    def test_run_sync(self):
        """Test the synchronous wrapper."""
        llm = MockLLM(['{"action": "done", "reasoning": "Quick"}'])
        orch = Orchestrator(llm=llm, workers=[SimpleWriter()])
        
        # Use sync wrapper
        result = orch.run_sync(goal="Quick test", max_steps=1)
        assert result.status == Status.DONE


class TestParallelExecution:
    """Tests for parallel worker execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_worker_call(self):
        """Test parallel worker execution."""
        
        class ResearcherA(Worker):
            name = "ResearcherA"
            description = "Researches topic A"
            parallel_safe = True  # Opt-in to parallel execution
            
            async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
                return WorkerOutput(
                    artifact=Artifact(type="research", content="Topic A findings", creator=self.name)
                )
        
        class ResearcherB(Worker):
            name = "ResearcherB"
            description = "Researches topic B"
            parallel_safe = True  # Opt-in to parallel execution
            
            async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
                return WorkerOutput(
                    artifact=Artifact(type="research", content="Topic B findings", creator=self.name)
                )
        
        llm = MockLLM([
            '{"action": "call_independent", "calls": [{"worker": "ResearcherA", "instructions": "Research A"}, {"worker": "ResearcherB", "instructions": "Research B"}], "reasoning": "Independent tasks"}',
            '{"action": "done", "reasoning": "Both done"}'
        ])
        
        orch = Orchestrator(
            llm=llm,
            workers=[ResearcherA(), ResearcherB()],
            enable_parallel=True
        )
        
        result = await orch.run(goal="Research topics", max_steps=5)
        
        assert result.status == Status.DONE
        # Both artifacts should be created
        assert len(result.artifacts) == 2
    
    @pytest.mark.asyncio
    async def test_parallel_disabled_falls_back_to_sequential(self):
        """Test that parallel execution can be disabled."""
        
        class SimpleWorker(Worker):
            name = "Simple"
            description = "Simple worker"
            execution_count = 0
            
            async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
                SimpleWorker.execution_count += 1
                return WorkerOutput(
                    artifact=Artifact(type="text", content=f"Run {SimpleWorker.execution_count}", creator=self.name)
                )
        
        SimpleWorker.execution_count = 0
        
        llm = MockLLM([
            '{"action": "call", "worker": "Simple", "instructions": "Work"}',
            '{"action": "done", "reasoning": "Done"}'
        ])
        
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWorker()],
            enable_parallel=False
        )
        
        result = await orch.run(goal="Test", max_steps=5)
        
        assert result.status == Status.DONE
        assert SimpleWorker.execution_count == 1

    @pytest.mark.asyncio
    async def test_parallel_batch_advances_blueprint_once(self):
        """A successful parallel batch should count as one blueprint iteration."""

        class ResearcherA(Worker):
            name = "ResearcherA"
            description = "Researches topic A"
            parallel_safe = True

            async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
                return WorkerOutput(
                    artifact=Artifact(type="research", content="A", creator=self.name)
                )

        class ResearcherB(Worker):
            name = "ResearcherB"
            description = "Researches topic B"
            parallel_safe = True

            async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
                return WorkerOutput(
                    artifact=Artifact(type="research", content="B", creator=self.name)
                )

        llm = MockLLM([
            '{"action": "call_independent", "calls": [{"worker": "ResearcherA", "instructions": "A"}, {"worker": "ResearcherB", "instructions": "B"}], "reasoning": "Run both"}'
        ])

        blueprint = Blueprint(
            name="Parallel Research",
            allow_skip_to_done=False,
            steps=[
                Step(
                    name="research",
                    allowed_workers=["ResearcherA", "ResearcherB"],
                    instructions="Run the researchers in one batch.",
                    max_iterations=5
                )
            ]
        )

        orchestrator = Orchestrator(
            llm=llm,
            workers=[ResearcherA(), ResearcherB()],
            enable_parallel=True
        )

        result = await orchestrator.run(
            goal="Research both topics",
            max_steps=3,
            blueprint=blueprint
        )

        assert result.status == Status.DONE
        assert result.step_count == 1
        assert result.metadata["_blueprint_step_index"] == 1
        assert len(result.artifacts) == 2


class TestWorkerInput:
    """Tests for worker input schemas."""
    
    def test_worker_input_creation(self):
        """Test creating a WorkerInput."""
        inputs = WorkerInput(instructions="Do something")
        assert inputs.instructions == "Do something"
    
    def test_custom_worker_input(self):
        """Test custom worker input schema."""
        
        class CustomInput(WorkerInput):
            language: str = "python"
            include_tests: bool = False
        
        inputs = CustomInput(instructions="Generate code", language="rust", include_tests=True)
        assert inputs.language == "rust"
        assert inputs.include_tests is True


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
