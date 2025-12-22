"""
Tests for SDK Production Hardening

Tests for the production hardening improvements:
- BlackboardConfig
- InsecureLocalExecutor warning
- SimpleVectorMemory warning  
- CircuitBreakerMiddleware
- DatasetLoggingMiddleware
- Graceful shutdown
"""

import pytest
import warnings
import os
import json
import tempfile
from typing import List, Optional
from unittest.mock import patch, MagicMock

from blackboard import (
    Orchestrator, Worker, WorkerOutput, WorkerInput,
    Artifact, Feedback, Blackboard, Status, BlackboardConfig
)
from blackboard.middleware import (
    CircuitBreakerMiddleware, DatasetLoggingMiddleware,
    StepContext, WorkerContext
)


# =============================================================================
# Test Helpers
# =============================================================================

class MockLLM:
    """A mock LLM that returns predefined responses."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_index = 0
    
    def generate(self, prompt: str) -> str:
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return '{"action": "done", "reasoning": "No more responses"}'


class SimpleWorker(Worker):
    """A simple worker for testing."""
    name = "SimpleWorker"
    description = "Does simple work"
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        return WorkerOutput(
            artifact=Artifact(type="test", content="done", creator=self.name)
        )


class FailingWorker(Worker):
    """A worker that always fails."""
    name = "FailingWorker"
    description = "Always fails"
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        raise Exception("Worker failed intentionally")


# =============================================================================
# BlackboardConfig Tests
# =============================================================================

class TestBlackboardConfig:
    """Tests for BlackboardConfig."""
    
    def test_default_config(self):
        """Default config has sensible defaults."""
        config = BlackboardConfig()
        assert config.max_steps == 20
        assert config.allow_unsafe_execution is False
        assert config.enable_parallel is True
    
    def test_config_from_kwargs(self):
        """Config can be created with custom values."""
        config = BlackboardConfig(
            max_steps=50,
            allow_unsafe_execution=True,
            simple_prompts=True
        )
        assert config.max_steps == 50
        assert config.allow_unsafe_execution is True
        assert config.simple_prompts is True
    
    def test_config_from_env(self):
        """Config reads from environment variables."""
        with patch.dict(os.environ, {
            "BLACKBOARD_MAX_STEPS": "100",
            "BLACKBOARD_ALLOW_UNSAFE_EXECUTION": "true",
        }):
            config = BlackboardConfig.from_env()
            assert config.max_steps == 100
            assert config.allow_unsafe_execution is True
    
    def test_config_env_override(self):
        """Explicit kwargs override env vars."""
        with patch.dict(os.environ, {"BLACKBOARD_MAX_STEPS": "100"}):
            config = BlackboardConfig.from_env(max_steps=50)
            assert config.max_steps == 50
    
    def test_orchestrator_accepts_config(self):
        """Orchestrator accepts config parameter."""
        llm = MockLLM(['{"action": "done"}'])
        config = BlackboardConfig(verbose=True, enable_parallel=False)
        
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWorker()],
            config=config
        )
        
        assert orch.config == config
        assert orch.verbose is True
        assert orch.enable_parallel is False


# =============================================================================
# Sandbox Security Tests
# =============================================================================

class TestSandboxSecurity:
    """Tests for sandbox security improvements."""
    
    def test_insecure_executor_warns(self, caplog):
        """InsecureLocalExecutor logs warning by default."""
        import logging
        caplog.set_level(logging.WARNING)
        
        from blackboard.sandbox import InsecureLocalExecutor
        executor = InsecureLocalExecutor()
        
        assert "HOST PRIVILEGES" in caplog.text
    
    def test_insecure_executor_acknowledged(self, caplog):
        """InsecureLocalExecutor suppresses warning when acknowledged."""
        import logging
        caplog.set_level(logging.WARNING)
        
        from blackboard.sandbox import InsecureLocalExecutor
        executor = InsecureLocalExecutor(_unsafe_acknowledged=True)
        
        assert "HOST PRIVILEGES" not in caplog.text


# =============================================================================
# Memory Warning Tests  
# =============================================================================

class TestMemoryWarnings:
    """Tests for SimpleVectorMemory O(N) warning."""
    
    def test_simple_memory_warns(self):
        """SimpleVectorMemory warns about O(N) search."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from blackboard.memory import SimpleVectorMemory
            memory = SimpleVectorMemory()
            
            assert len(w) == 1
            assert "O(N)" in str(w[0].message)
            assert "linear search" in str(w[0].message)
    
    def test_simple_memory_suppressed(self):
        """Warning can be suppressed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from blackboard.memory import SimpleVectorMemory
            memory = SimpleVectorMemory(_suppress_warning=True)
            
            assert len(w) == 0


# =============================================================================
# CircuitBreaker Tests
# =============================================================================

class TestCircuitBreakerMiddleware:
    """Tests for CircuitBreakerMiddleware."""
    
    @pytest.mark.asyncio
    async def test_circuit_starts_closed(self):
        """Circuit starts in closed state."""
        cb = CircuitBreakerMiddleware(failure_threshold=3)
        state = Blackboard(goal="test")
        worker = SimpleWorker()
        call = MagicMock()
        call.instructions = ""
        
        ctx = WorkerContext(worker=worker, call=call, state=state)
        
        await cb.before_worker(ctx)
        
        assert ctx.skip_worker is False
        assert state.metadata.get("circuit_breaker", {}).get("state") == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Circuit opens after failure threshold."""
        cb = CircuitBreakerMiddleware(failure_threshold=2)
        state = Blackboard(goal="test")
        worker = SimpleWorker()
        call = MagicMock()
        call.instructions = ""
        
        ctx = WorkerContext(worker=worker, call=call, state=state)
        ctx.error = Exception("failure")
        
        # First failure
        await cb.on_error(ctx)
        assert state.metadata["circuit_breaker"]["failures"] == 1
        assert state.metadata["circuit_breaker"]["state"] == "closed"
        
        # Second failure - should open
        await cb.on_error(ctx)
        assert state.metadata["circuit_breaker"]["failures"] == 2
        assert state.metadata["circuit_breaker"]["state"] == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_open_skips_worker(self):
        """Open circuit skips worker execution."""
        cb = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout=60)
        state = Blackboard(goal="test")
        state.metadata["circuit_breaker"] = {
            "state": "open",
            "failures": 5,
            "last_failure": 9999999999.0  # Future timestamp to prevent half-open
        }
        worker = SimpleWorker()
        call = MagicMock()
        call.instructions = ""
        
        ctx = WorkerContext(worker=worker, call=call, state=state)
        
        await cb.before_worker(ctx)
        
        assert ctx.skip_worker is True
        assert ctx.error is not None
    
    @pytest.mark.asyncio
    async def test_circuit_resets_on_success(self):
        """Circuit resets on successful execution."""
        cb = CircuitBreakerMiddleware(failure_threshold=3)
        state = Blackboard(goal="test")
        state.metadata["circuit_breaker"] = {
            "state": "half-open",
            "failures": 2,
            "last_failure": None,
            "last_success": None
        }
        worker = SimpleWorker()
        call = MagicMock()
        call.instructions = ""
        
        ctx = WorkerContext(worker=worker, call=call, state=state)
        ctx.error = None  # Success
        
        await cb.after_worker(ctx)
        
        assert state.metadata["circuit_breaker"]["state"] == "closed"
        assert state.metadata["circuit_breaker"]["failures"] == 0


# =============================================================================
# DatasetLoggingMiddleware Tests
# =============================================================================

class TestDatasetLoggingMiddleware:
    """Tests for DatasetLoggingMiddleware."""
    
    @pytest.mark.asyncio
    async def test_logs_supervisor_decisions(self, tmp_path):
        """Dataset logger writes supervisor decisions to JSONL."""
        filepath = tmp_path / "dataset.jsonl"
        
        middleware = DatasetLoggingMiddleware(filepath=str(filepath))
        state = Blackboard(goal="test goal")
        
        # Create mock decision
        decision = MagicMock()
        decision.action = "call"
        decision.reasoning = "test reasoning"
        decision.calls = []
        
        ctx = StepContext(step_number=1, state=state, decision=decision)
        
        await middleware.after_step(ctx)
        
        # Read logged data
        with open(filepath) as f:
            entry = json.loads(f.readline())
        
        assert entry["type"] == "supervisor_decision"
        assert entry["step"] == 1
        assert "prompt" in entry
        assert "completion" in entry
    
    @pytest.mark.asyncio
    async def test_truncates_long_prompts(self, tmp_path):
        """Dataset logger truncates long prompts."""
        filepath = tmp_path / "dataset.jsonl"
        
        middleware = DatasetLoggingMiddleware(
            filepath=str(filepath),
            max_prompt_length=100
        )
        
        state = Blackboard(goal="x" * 10000)  # Very long goal
        
        decision = MagicMock()
        decision.action = "done"
        decision.reasoning = "done"
        decision.calls = []
        
        ctx = StepContext(step_number=1, state=state, decision=decision)
        
        await middleware.after_step(ctx)
        
        with open(filepath) as f:
            entry = json.loads(f.readline())
        
        assert len(entry["prompt"]) <= 100


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================

class TestGracefulShutdown:
    """Tests for graceful shutdown signal handling."""
    
    def test_shutdown_flag_exists(self):
        """Orchestrator has shutdown flag."""
        llm = MockLLM([])
        orch = Orchestrator(llm=llm, workers=[SimpleWorker()])
        
        assert hasattr(orch, "_shutdown_requested")
        assert orch._shutdown_requested is False
    
    def test_shutdown_handler_exists(self):
        """Orchestrator has shutdown handler method."""
        llm = MockLLM([])
        orch = Orchestrator(llm=llm, workers=[SimpleWorker()])
        
        assert hasattr(orch, "_handle_shutdown_signal")
        
        # Simulate signal
        orch._handle_shutdown_signal(15, None)  # SIGTERM
        assert orch._shutdown_requested is True


# =============================================================================
# Supervisor Schema Tests
# =============================================================================

class TestSupervisorSchema:
    """Tests for supervisor prompt schema improvements."""
    
    def test_simple_prompts_uses_default_strategy(self):
        """simple_prompts=True should still use default strategy (prompts are in strategy now)."""
        llm = MockLLM([])
        config = BlackboardConfig(simple_prompts=True)
        
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWorker()],
            config=config
        )
        
        # Strategy should be the default OneShot strategy
        from blackboard.reasoning import OneShotStrategy
        assert isinstance(orch.strategy, OneShotStrategy)
    
    def test_cot_config_uses_cot_strategy(self):
        """reasoning_strategy='cot' should use ChainOfThoughtStrategy."""
        llm = MockLLM([])
        config = BlackboardConfig(reasoning_strategy="cot")
        
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWorker()],
            config=config
        )
        
        from blackboard.reasoning import ChainOfThoughtStrategy
        assert isinstance(orch.strategy, ChainOfThoughtStrategy)
    
    def test_default_uses_oneshot_strategy(self):
        """Default config should use OneShotStrategy."""
        llm = MockLLM([])
        
        orch = Orchestrator(
            llm=llm,
            workers=[SimpleWorker()]
        )
        
        from blackboard.reasoning import OneShotStrategy
        assert isinstance(orch.strategy, OneShotStrategy)
    
    def test_pydantic_schema_available(self):
        """Pydantic schemas should be importable if pydantic is installed."""
        try:
            from blackboard.schemas import (
                PYDANTIC_AVAILABLE,
                get_supervisor_json_schema,
                get_simple_prompt_schema,
                validate_supervisor_response
            )
            
            if PYDANTIC_AVAILABLE:
                schema = get_supervisor_json_schema()
                assert "oneOf" in schema or "examples" in schema
                
                # Test validation
                valid_call = {"action": "call", "worker": "Test", "instructions": "Do it"}
                error = validate_supervisor_response(valid_call)
                assert error is None
                
                # Test invalid
                invalid = {"action": "unknown"}
                error = validate_supervisor_response(invalid)
                assert error is not None
            
        except ImportError:
            pass  # Pydantic not installed
    
    def test_simple_prompt_schema_text(self):
        """get_simple_prompt_schema should return readable text."""
        from blackboard.schemas import get_simple_prompt_schema
        
        schema_text = get_simple_prompt_schema()
        assert "call" in schema_text
        assert "done" in schema_text
        assert "fail" in schema_text
