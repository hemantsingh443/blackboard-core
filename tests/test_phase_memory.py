"""Tests for tool calling, memory, and auto-summarization."""

import pytest
import tempfile
from pathlib import Path
from typing import Optional

from blackboard import (
    Worker, WorkerOutput, WorkerInput,
    Artifact, Feedback, Blackboard, Status,
    Orchestrator, LLMResponse, LLMUsage
)
from blackboard.tools import (
    ToolDefinition, ToolParameter, ToolCall,
    worker_to_tool_definition, workers_to_tool_definitions
)
from blackboard.memory import (
    Memory, MemoryEntry, SimpleVectorMemory, MemoryWorker, MemoryInput
)
from blackboard.middleware import AutoSummarizationMiddleware, BudgetMiddleware, StepContext


class TestToolDefinition:
    """Tests for tool definition."""
    
    def test_tool_definition_creation(self):
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="Writer",
            description="Writes text",
            parameters=[
                ToolParameter(name="instructions", type="string", description="Task")
            ]
        )
        
        assert tool.name == "Writer"
        assert len(tool.parameters) == 1
    
    def test_to_openai_format(self):
        """Test OpenAI format conversion."""
        tool = ToolDefinition(
            name="Writer",
            description="Writes text",
            parameters=[
                ToolParameter(name="topic", type="string", description="Topic to write about"),
                ToolParameter(name="length", type="number", description="Word count", required=False)
            ]
        )
        
        openai_format = tool.to_openai_format()
        
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "Writer"
        assert "topic" in openai_format["function"]["parameters"]["properties"]
        assert "topic" in openai_format["function"]["parameters"]["required"]
        assert "length" not in openai_format["function"]["parameters"]["required"]
    
    def test_to_anthropic_format(self):
        """Test Anthropic format conversion."""
        tool = ToolDefinition(
            name="Writer",
            description="Writes text",
            parameters=[
                ToolParameter(name="topic", type="string", description="Topic")
            ]
        )
        
        anthropic_format = tool.to_anthropic_format()
        
        assert anthropic_format["name"] == "Writer"
        assert "input_schema" in anthropic_format
        assert "topic" in anthropic_format["input_schema"]["properties"]
    
    def test_worker_to_tool_definition(self):
        """Test converting a worker to tool definition."""
        
        class TestWorker(Worker):
            name = "Tester"
            description = "Tests things"
            
            async def run(self, state, inputs=None):
                return WorkerOutput()
        
        tool = worker_to_tool_definition(TestWorker())
        
        assert tool.name == "Tester"
        assert tool.description == "Tests things"
        assert len(tool.parameters) == 1  # Default instructions param
    
    def test_worker_with_schema_to_tool(self):
        """Test converting a worker with input schema."""
        
        class CustomInput(WorkerInput):
            language: str = "python"
            include_tests: bool = False
        
        class CodeWorker(Worker):
            name = "Coder"
            description = "Generates code"
            input_schema = CustomInput
            
            async def run(self, state, inputs=None):
                return WorkerOutput()
        
        tool = worker_to_tool_definition(CodeWorker())
        
        assert tool.name == "Coder"
        # Should have parameters from schema
        param_names = [p.name for p in tool.parameters]
        assert "language" in param_names or "instructions" in param_names


class TestSimpleVectorMemory:
    """Tests for simple vector memory."""
    
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        """Test adding and searching memories."""
        memory = SimpleVectorMemory()
        
        await memory.add("User prefers Python 3.11")
        await memory.add("Database is PostgreSQL 15")
        await memory.add("Deployment target is AWS")
        
        results = await memory.search("What Python version?", limit=2)
        
        assert len(results) > 0
        assert "Python" in results[0].entry.content
    
    @pytest.mark.asyncio
    async def test_search_relevance(self):
        """Test that search returns relevant results."""
        memory = SimpleVectorMemory()
        
        await memory.add("The user likes cats")
        await memory.add("Python is the preferred language")
        await memory.add("Database queries should be optimized")
        
        results = await memory.search("programming language", limit=1)
        
        assert len(results) == 1
        assert "Python" in results[0].entry.content
    
    @pytest.mark.asyncio
    async def test_get_by_id(self):
        """Test getting memory by ID."""
        memory = SimpleVectorMemory()
        
        entry = await memory.add("Test content")
        
        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Test content"
    
    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting a memory."""
        memory = SimpleVectorMemory()
        
        entry = await memory.add("To be deleted")
        assert await memory.delete(entry.id) is True
        assert await memory.get(entry.id) is None
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all memories."""
        memory = SimpleVectorMemory()
        
        await memory.add("One")
        await memory.add("Two")
        
        count = await memory.clear()
        
        assert count == 2
        assert len(await memory.get_all()) == 0
    
    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        """Test persisting memories to disk."""
        path = tmp_path / "memory.json"
        
        # Create and populate
        memory1 = SimpleVectorMemory(persist_path=str(path))
        await memory1.add("Persistent memory")
        
        # Load in new instance
        memory2 = SimpleVectorMemory(persist_path=str(path))
        
        all_memories = await memory2.get_all()
        assert len(all_memories) == 1
        assert all_memories[0].content == "Persistent memory"


class TestMemoryWorker:
    """Tests for the memory worker."""
    
    @pytest.mark.asyncio
    async def test_memory_search(self):
        """Test memory search operation."""
        memory = SimpleVectorMemory()
        await memory.add("User prefers dark mode")
        
        worker = MemoryWorker(memory)
        state = Blackboard(goal="Test")
        
        inputs = MemoryInput(operation="search", query="dark mode")
        output = await worker.run(state, inputs)
        
        assert output.has_artifact()
        assert "dark mode" in output.artifact.content.lower()
    
    @pytest.mark.asyncio
    async def test_memory_add(self):
        """Test memory add operation."""
        memory = SimpleVectorMemory()
        worker = MemoryWorker(memory)
        state = Blackboard(goal="Test")
        
        inputs = MemoryInput(operation="add", content="New important fact")
        output = await worker.run(state, inputs)
        
        assert output.has_artifact()
        assert len(await memory.get_all()) == 1
    
    @pytest.mark.asyncio
    async def test_memory_explicit_operation(self):
        """Test that operation must be explicitly specified."""
        memory = SimpleVectorMemory()
        await memory.add("Some existing data")
        
        worker = MemoryWorker(memory)
        state = Blackboard(goal="Test")
        
        # Explicit operation='add' required (no heuristic parsing)
        inputs = MemoryInput(operation="add", content="User likes blue")
        output = await worker.run(state, inputs)
        
        assert len(await memory.get_all()) == 2  # Original + new


class TestAutoSummarization:
    """Tests for auto-summarization middleware."""
    
    def test_should_trigger_on_threshold(self):
        """Test that summarization triggers on threshold."""
        import asyncio
        
        class MockLLM:
            call_count = 0
            def generate(self, prompt):
                MockLLM.call_count += 1
                return "This is a summary of the session."
        
        MockLLM.call_count = 0
        
        summarizer = AutoSummarizationMiddleware(
            llm=MockLLM(),
            artifact_threshold=3,
            step_threshold=100
        )
        
        state = Blackboard(goal="Test")
        state.step_count = 10  # Need step count >= 5 to trigger
        
        # Add many artifacts to trigger threshold
        for i in range(5):
            state.add_artifact(Artifact(type="text", content=f"Art {i}", creator="W"))
        
        ctx = StepContext(step_number=10, state=state)
        
        # Now async
        asyncio.run(summarizer.after_step(ctx))
        
        assert MockLLM.call_count == 1
        assert state.context_summary != ""
    
    def test_compacts_after_summarization(self):
        """Test that artifacts are compacted after summarization."""
        import asyncio
        
        class MockLLM:
            def generate(self, prompt):
                return "Summary of old context"
        
        summarizer = AutoSummarizationMiddleware(
            llm=MockLLM(),
            artifact_threshold=3,
            keep_recent_artifacts=2
        )
        
        state = Blackboard(goal="Test")
        state.step_count = 10  # Need step count >= 5 to trigger
        
        for i in range(10):
            state.add_artifact(Artifact(type="text", content=f"Art {i}", creator="W"))
        
        ctx = StepContext(step_number=10, state=state)
        
        # Now async
        asyncio.run(summarizer.after_step(ctx))
        
        # Should have compacted to keep_recent_artifacts
        assert len(state.artifacts) == 2
        assert state.artifacts[-1].content == "Art 9"  # Most recent


class TestToolCall:
    """Tests for tool call handling."""
    
    def test_tool_call_creation(self):
        """Test creating a tool call."""
        call = ToolCall(
            id="call_123",
            name="Writer",
            arguments={"instructions": "Write a poem"}
        )
        
        assert call.id == "call_123"
        assert call.name == "Writer"
        assert call.arguments["instructions"] == "Write a poem"
    
    def test_to_worker_inputs(self):
        """Test converting tool call to worker inputs."""
        call = ToolCall(
            id="call_123",
            name="Writer",
            arguments={"topic": "Python", "length": 500}
        )
        
        inputs = call.to_worker_inputs()
        
        assert inputs["topic"] == "Python"
        assert inputs["length"] == 500


class UsageWorker(Worker):
    """A simple worker used for usage and budget contract tests."""

    name = "UsageWorker"
    description = "Produces a small text artifact"

    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        return WorkerOutput(
            artifact=Artifact(type="text", content="usage artifact", creator=self.name)
        )


class TestUsageAndBudgetContracts:
    """Tests for usage tracking and budget enforcement."""

    @pytest.mark.asyncio
    async def test_json_supervisor_usage_recorded(self):
        """JSON supervisor calls should populate the canonical last_usage payload."""

        class JsonUsageLLM:
            def generate(self, prompt):
                return LLMResponse(
                    content='{"action": "done", "reasoning": "Complete"}',
                    usage=LLMUsage(input_tokens=11, output_tokens=5, model="json-model")
                )

        orchestrator = Orchestrator(llm=JsonUsageLLM(), workers=[UsageWorker()])
        result = await orchestrator.run(goal="Record JSON usage", max_steps=1)

        assert result.metadata["last_usage"] == {
            "input_tokens": 11,
            "output_tokens": 5,
            "model": "json-model",
            "context": "supervisor",
        }
        assert "last_model" not in result.metadata

    @pytest.mark.asyncio
    async def test_tool_calling_supervisor_usage_recorded(self):
        """Tool-calling supervisor calls should also populate last_usage."""

        class ToolUsageLLM:
            def __init__(self):
                self.last_usage = None

            def generate(self, prompt):
                raise AssertionError("JSON fallback should not be used in this test")

            def generate_with_tools(self, prompt, tools):
                self.last_usage = LLMUsage(
                    input_tokens=21,
                    output_tokens=7,
                    model="tool-model"
                )
                return [ToolCall(id="done", name="mark_done", arguments={"reason": "Complete"})]

        orchestrator = Orchestrator(llm=ToolUsageLLM(), workers=[UsageWorker()])
        result = await orchestrator.run(goal="Record tool usage", max_steps=1)

        assert result.metadata["last_usage"] == {
            "input_tokens": 21,
            "output_tokens": 7,
            "model": "tool-model",
            "context": "supervisor",
        }

    @pytest.mark.asyncio
    async def test_summarization_usage_is_recorded(self):
        """Auto-summarization should contribute usage records to the current step."""

        class SummarizationLLM:
            def __init__(self):
                self.call_count = 0

            def generate(self, prompt):
                self.call_count += 1
                if "Summarize the following session history" in prompt:
                    return LLMResponse(
                        content="Condensed session summary",
                        usage=LLMUsage(
                            input_tokens=30,
                            output_tokens=9,
                            model="summary-model"
                        )
                    )
                return LLMResponse(
                    content='{"action": "done", "reasoning": "Finished after summary"}',
                    usage=LLMUsage(
                        input_tokens=14,
                        output_tokens=6,
                        model="supervisor-model"
                    )
                )

        state = Blackboard(goal="Trigger summarization")
        state.add_artifact(Artifact(type="text", content="First artifact", creator="A"))
        state.add_artifact(Artifact(type="text", content="Second artifact", creator="B"))

        orchestrator = Orchestrator(
            llm=SummarizationLLM(),
            workers=[UsageWorker()],
            auto_summarize=True,
            summarize_thresholds={"artifacts": 1, "feedback": 99, "steps": 99}
        )
        result = await orchestrator.run(state=state, max_steps=1)

        usage_records = result.metadata["_step_usage_records"]
        assert any(record["context"] == "summarizer" for record in usage_records)
        assert any(record["context"] == "supervisor" for record in usage_records)

    @pytest.mark.asyncio
    async def test_budget_middleware_halts_from_canonical_usage_contract(self):
        """BudgetMiddleware should halt based on recorded last_usage data."""

        class OverBudgetLLM:
            def __init__(self):
                self.call_count = 0

            def generate(self, prompt):
                self.call_count += 1
                if self.call_count > 1:
                    raise AssertionError("BudgetMiddleware should stop execution before a second LLM call")
                return LLMResponse(
                    content='{"action": "call", "worker": "UsageWorker", "instructions": "Do work"}',
                    usage=LLMUsage(
                        input_tokens=40,
                        output_tokens=20,
                        model="json-model"
                    )
                )

        budget = BudgetMiddleware(max_tokens=50)
        orchestrator = Orchestrator(
            llm=OverBudgetLLM(),
            workers=[UsageWorker()],
            middleware=[budget]
        )

        result = await orchestrator.run(goal="Stop when over budget", max_steps=3)

        assert result.status == Status.FAILED
        assert result.metadata["budget"]["total_tokens"] == 60
