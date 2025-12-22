"""Tests for middleware, usage tracking, and JSON parsing."""

import pytest
from typing import Optional

from blackboard import (
    Orchestrator, Worker, WorkerOutput, WorkerInput,
    Artifact, Feedback, Blackboard, Status
)
from blackboard.middleware import (
    Middleware, MiddlewareStack, StepContext, WorkerContext,
    BudgetMiddleware
)
from blackboard.usage import LLMResponse, LLMUsage, UsageTracker


class MockLLM:
    """A mock LLM that returns predefined responses."""
    
    def __init__(self, responses):
        self.responses = responses
        self.call_index = 0
    
    def generate(self, prompt: str) -> str:
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return '{"action": "fail", "reasoning": "No more mock responses"}'


class SimpleWorker(Worker):
    name = "Simple"
    description = "Simple worker"
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        return WorkerOutput(
            artifact=Artifact(type="text", content="Test", creator=self.name)
        )


class TestMiddleware:
    """Tests for middleware system."""
    
    def test_middleware_stack_add_remove(self):
        """Test adding and removing middleware."""
        stack = MiddlewareStack()
        
        class TestMiddleware(Middleware):
            name = "TestMW"
        
        mw = TestMiddleware()
        stack.add(mw)
        
        assert len(stack) == 1
        assert stack.remove("TestMW") is True
        assert len(stack) == 0
    
    def test_step_context(self):
        """Test step context creation."""
        state = Blackboard(goal="Test")
        ctx = StepContext(step_number=1, state=state)
        
        assert ctx.step_number == 1
        assert ctx.skip_step is False
        assert ctx.modified_decision is None
    
    @pytest.mark.asyncio
    async def test_middleware_before_step(self):
        """Test before_step hook."""
        stack = MiddlewareStack()
        calls = []
        
        class TrackingMiddleware(Middleware):
            name = "Tracker"
            
            async def before_step(self, ctx: StepContext):
                calls.append("before")
        
        stack.add(TrackingMiddleware())
        
        state = Blackboard(goal="Test")
        ctx = StepContext(step_number=1, state=state)
        await stack.before_step(ctx)
        
        assert calls == ["before"]
    
    @pytest.mark.asyncio
    async def test_middleware_skip_step(self):
        """Test middleware can skip step."""
        stack = MiddlewareStack()
        
        class SkipMiddleware(Middleware):
            name = "Skipper"
            
            async def before_step(self, ctx: StepContext):
                ctx.skip_step = True
        
        stack.add(SkipMiddleware())
        
        state = Blackboard(goal="Test")
        ctx = StepContext(step_number=1, state=state)
        await stack.before_step(ctx)
        
        assert ctx.skip_step is True
    
    @pytest.mark.asyncio
    async def test_middleware_integration(self):
        """Test middleware integrates with orchestrator."""
        calls = []
        
        class RecordingMiddleware(Middleware):
            name = "Recorder"
            
            async def before_step(self, ctx: StepContext):
                calls.append(f"before_step_{ctx.step_number}")
            
            async def after_step(self, ctx: StepContext):
                calls.append(f"after_step_{ctx.step_number}")
        
        llm = MockLLM(['{"action": "done", "reasoning": "Quick"}'])
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWorker()],
            middleware=[RecordingMiddleware()]
        )
        
        await orch.run(goal="Test", max_steps=5)
        
        assert "before_step_1" in calls
        assert "after_step_1" in calls


class TestUsageTracking:
    """Tests for usage tracking."""
    
    def test_llm_usage(self):
        """Test LLMUsage dataclass."""
        usage = LLMUsage(input_tokens=100, output_tokens=50)
        
        assert usage.total_tokens == 150
    
    def test_llm_response(self):
        """Test LLMResponse dataclass."""
        usage = LLMUsage(input_tokens=100, output_tokens=50)
        response = LLMResponse(content="Hello", usage=usage)
        
        assert response.content == "Hello"
        assert str(response) == "Hello"
        assert response.usage.total_tokens == 150
    
    def test_usage_tracker_record(self):
        """Test recording usage."""
        tracker = UsageTracker(cost_per_1k_input=0.01, cost_per_1k_output=0.03)
        
        usage = LLMUsage(input_tokens=1000, output_tokens=500)
        cost = tracker.record(usage, context="test")
        
        # 1.0 * 0.01 + 0.5 * 0.03 = 0.01 + 0.015 = 0.025
        assert abs(cost - 0.025) < 0.001
        assert tracker.call_count == 1
        assert tracker.total_tokens == 1500
    
    def test_usage_tracker_summary(self):
        """Test usage summary."""
        tracker = UsageTracker()
        
        tracker.record(LLMUsage(input_tokens=100, output_tokens=50))
        tracker.record(LLMUsage(input_tokens=200, output_tokens=100))
        
        summary = tracker.get_summary()
        
        assert summary["call_count"] == 2
        assert summary["total_tokens"] == 450
        assert summary["input_tokens"] == 300
        assert summary["output_tokens"] == 150
    
    def test_usage_context_grouping(self):
        """Test grouping usage by context."""
        tracker = UsageTracker()
        
        tracker.record(LLMUsage(input_tokens=100, output_tokens=50), context="supervisor")
        tracker.record(LLMUsage(input_tokens=100, output_tokens=50), context="supervisor")
        tracker.record(LLMUsage(input_tokens=200, output_tokens=100), context="worker:Writer")
        
        by_ctx = tracker.get_context_summary()
        
        assert by_ctx["supervisor"]["call_count"] == 2
        assert by_ctx["worker:Writer"]["call_count"] == 1
    
    @pytest.mark.asyncio
    async def test_usage_tracking_integration(self):
        """Test usage tracking integrates with orchestrator."""
        
        class MockLLMWithUsage:
            def generate(self, prompt: str) -> LLMResponse:
                return LLMResponse(
                    content='{"action": "done", "reasoning": "Quick"}',
                    usage=LLMUsage(input_tokens=100, output_tokens=20)
                )
        
        tracker = UsageTracker()
        orch = Orchestrator(
            llm=MockLLMWithUsage(),
            workers=[SimpleWorker()],
            usage_tracker=tracker
        )
        
        await orch.run(goal="Test", max_steps=5)
        
        assert tracker.call_count == 1
        assert tracker.total_tokens == 120


class TestJSONParsing:
    """Tests for improved JSON parsing via reasoning strategies."""
    
    def test_parse_json_in_code_block(self):
        """Test parsing JSON from markdown code block."""
        from blackboard.reasoning import OneShotStrategy
        
        strategy = OneShotStrategy()
        decision = strategy.parse_response(
            '```json\n{"action": "done", "reasoning": "Test"}\n```'
        )
        
        assert decision.action == "done"
    
    def test_parse_json_with_chatter(self):
        """Test parsing JSON with surrounding text."""
        from blackboard.reasoning import OneShotStrategy
        
        strategy = OneShotStrategy()
        response = """
        Based on the current state, I think we should proceed.
        
        {"action": "call", "worker": "Simple", "instructions": "Do it", "reasoning": "Because"}
        
        Let me know if you have questions!
        """
        
        decision = strategy.parse_response(response)
        
        assert decision.action == "call"
        assert decision.calls[0]["worker_name"] == "Simple"
    
    def test_parse_json_handles_invalid(self):
        """Test JSON parsing fails gracefully for invalid JSON."""
        from blackboard.reasoning import OneShotStrategy
        
        strategy = OneShotStrategy()
        decision = strategy.parse_response("not valid json at all")
        
        assert decision.action == "fail"


class TestContextSummarization:
    """Tests for context summarization."""
    
    def test_should_summarize_false(self):
        """Test should_summarize returns False for small state."""
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="text", content="A", creator="W"))
        
        assert state.should_summarize() is False
    
    def test_should_summarize_true_artifacts(self):
        """Test should_summarize returns True when artifacts exceed threshold."""
        state = Blackboard(goal="Test")
        
        for i in range(15):
            state.add_artifact(Artifact(type="text", content=f"A{i}", creator="W"))
        
        assert state.should_summarize(artifact_threshold=10) is True
    
    def test_update_summary(self):
        """Test updating context summary."""
        state = Blackboard(goal="Test")
        state.update_summary("This is a summary of earlier work.")
        
        assert state.context_summary == "This is a summary of earlier work."
    
    def test_context_string_includes_summary(self):
        """Test context string includes summary when present."""
        state = Blackboard(goal="Test")
        state.update_summary("Earlier we did X and Y.")
        
        context = state.to_context_string()
        
        assert "Previous Context Summary" in context
        assert "Earlier we did X and Y." in context
    
    def test_compact_history(self):
        """Test compacting history log."""
        state = Blackboard(goal="Test")
        
        # Add many artifacts to generate history
        for i in range(30):
            state.add_artifact(Artifact(type="text", content=f"A{i}", creator="W"))
        
        initial_history = len(state.history)
        removed = state.compact_history(keep_last=10)
        
        assert removed == initial_history - 10
        assert len(state.history) == 10
