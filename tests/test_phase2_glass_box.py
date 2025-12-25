"""
Tests for Phase 2: The "Glass Box" components.

Tests cover:
- Structured Logging (blackboard/logging.py)
- Pricing (blackboard/pricing.py)
- Telemetry (blackboard/telemetry.py)
- Testing Harness (blackboard/testing/)
"""

import pytest
import asyncio
import re


# =============================================================================
# Logging Tests
# =============================================================================

class TestStructuredLogging:
    """Tests for blackboard.logging module."""
    
    def test_context_var_set_and_get(self):
        """Test context variable setters and getters."""
        from blackboard.logging import (
            set_session_id, get_session_id,
            set_trace_id, get_trace_id,
            set_step_index, get_step_index,
            set_worker_name, get_worker_name,
            clear_context
        )
        
        # Set values
        set_session_id("sess_123")
        set_trace_id("trace_456")
        set_step_index(5)
        set_worker_name("TestWorker")
        
        # Verify
        assert get_session_id() == "sess_123"
        assert get_trace_id() == "trace_456"
        assert get_step_index() == 5
        assert get_worker_name() == "TestWorker"
        
        # Clear
        clear_context()
        assert get_session_id() is None
        assert get_trace_id() is None
        assert get_step_index() is None
        assert get_worker_name() is None
    
    def test_bind_context_manager(self):
        """Test bind_context context manager."""
        from blackboard.logging import bind_context, get_session_id, get_step_index, clear_context
        
        clear_context()
        
        with bind_context(session_id="outer", step=1):
            assert get_session_id() == "outer"
            assert get_step_index() == 1
            
            with bind_context(session_id="inner", step=2):
                assert get_session_id() == "inner"
                assert get_step_index() == 2
            
            # Restored after inner context exits
            assert get_session_id() == "outer"
            assert get_step_index() == 1
        
        # Restored after outer context exits
        assert get_session_id() is None
        clear_context()
    
    def test_generate_trace_id(self):
        """Test trace ID generation."""
        from blackboard.logging import generate_trace_id
        
        id1 = generate_trace_id()
        id2 = generate_trace_id()
        
        assert id1 != id2
        assert len(id1) == 36  # UUID format
    
    def test_get_logger(self):
        """Test logger factory."""
        from blackboard.logging import get_logger
        
        log = get_logger("test_module")
        assert log is not None
        
        # Should not raise
        log.info("test_event", key="value")


# =============================================================================
# Pricing Tests
# =============================================================================

class TestPricing:
    """Tests for blackboard.pricing module."""
    
    def test_get_model_cost_known_model(self):
        """Test getting cost for a known model."""
        from blackboard.pricing import get_model_cost
        
        input_cost, output_cost = get_model_cost("gpt-4o")
        
        assert input_cost == 0.0025
        assert output_cost == 0.01
    
    def test_get_model_cost_prefix_match(self):
        """Test prefix matching for model variants (when LiteLLM doesn't have it)."""
        from blackboard.pricing import get_model_cost
        
        # Use a model in DEFAULT_PRICING but not in LiteLLM
        # gemini-3-pro-preview should match our DEFAULT_PRICING
        input_cost, output_cost = get_model_cost("gemini-3-pro-preview-v2")
        
        # Should match gemini-3-pro-preview pricing: (0.002, 0.012)
        assert input_cost == pytest.approx(0.002)
        assert output_cost == pytest.approx(0.012)
    
    def test_get_model_cost_unknown_model(self):
        """Test fallback for unknown model."""
        from blackboard.pricing import get_model_cost
        
        input_cost, output_cost = get_model_cost("unknown-model-xyz")
        
        # Should return default pricing
        assert input_cost == pytest.approx(0.001)
        assert output_cost == pytest.approx(0.002)
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        from blackboard.pricing import estimate_cost
        
        # gpt-4o: $0.0025/$0.01 per 1k
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        
        expected = (1000/1000)*0.0025 + (500/1000)*0.01
        assert cost == pytest.approx(expected)
    
    def test_configure_custom_pricing(self):
        """Test custom pricing override."""
        from blackboard.pricing import configure_pricing, get_model_cost
        
        configure_pricing({"my-custom-model": (0.05, 0.10)})
        
        input_cost, output_cost = get_model_cost("my-custom-model")
        
        assert input_cost == 0.05
        assert output_cost == 0.10
    
    def test_budget_exceeded_error(self):
        """Test BudgetExceededError exception."""
        from blackboard.pricing import BudgetExceededError
        
        error = BudgetExceededError(
            accumulated_cost=1.50,
            budget=1.00,
            model="gpt-4o"
        )
        
        assert error.accumulated_cost == 1.50
        assert error.budget == 1.00
        assert error.model == "gpt-4o"
        assert "1.5" in str(error)
        assert "1.0" in str(error)


# =============================================================================
# Telemetry Tests
# =============================================================================

class TestTelemetry:
    """Tests for blackboard.telemetry module."""
    
    def test_llm_span_creation(self):
        """Test LLMSpan dataclass."""
        from blackboard.telemetry import LLMSpan
        
        span = LLMSpan(
            model="gpt-4o",
            system="openai",
            input_tokens=100,
            output_tokens=50,
            duration_ms=150.5,
            cost=0.0035,
            prompt="Hello",
            completion="Hi there!",
            session_id="sess_123"
        )
        
        assert span.model == "gpt-4o"
        assert span.system == "openai"
        assert span.input_tokens == 100
        assert span.output_tokens == 50
        assert span.prompt == "Hello"
    
    def test_trace_exporter_protocol(self):
        """Test TraceExporter is a Protocol."""
        from blackboard.telemetry import TraceExporter, LLMSpan
        
        class MyExporter:
            async def export_llm_span(self, span: LLMSpan) -> None:
                pass
        
        exporter = MyExporter()
        assert isinstance(exporter, TraceExporter)
    
    def test_redaction_callback(self):
        """Test PII redaction callback."""
        import re
        from blackboard.telemetry import configure_redaction, redact_text
        
        def redact_ssn(text: str) -> str:
            return re.sub(r'\d{3}-\d{2}-\d{4}', '[SSN REDACTED]', text)
        
        configure_redaction(redact_ssn)
        
        result = redact_text("My SSN is 123-45-6789")
        assert result == "My SSN is [SSN REDACTED]"
        
        # Reset callback
        configure_redaction(lambda x: x)
    
    def test_redact_text_none(self):
        """Test redaction with None input."""
        from blackboard.telemetry import redact_text
        
        result = redact_text(None)
        assert result is None
    
    def test_span_attributes_genai(self):
        """Test GenAI semantic convention attributes exist."""
        from blackboard.telemetry import SpanAttributes
        
        assert SpanAttributes.GEN_AI_SYSTEM == "gen_ai.system"
        assert SpanAttributes.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
        assert SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"


# =============================================================================
# Testing Harness Tests
# =============================================================================

class TestMockLLMClient:
    """Tests for MockLLMClient."""
    
    @pytest.mark.asyncio
    async def test_sequential_mode(self):
        """Test sequential response mode."""
        from blackboard.testing import MockLLMClient
        
        mock = MockLLMClient(sequence=[
            "Response 1",
            "Response 2",
            "Response 3"
        ])
        
        r1 = await mock.generate("Any prompt")
        r2 = await mock.generate("Different prompt")
        r3 = await mock.generate("Third prompt")
        
        assert r1.content == "Response 1"
        assert r2.content == "Response 2"
        assert r3.content == "Response 3"
    
    @pytest.mark.asyncio
    async def test_pattern_mode(self):
        """Test pattern matching mode."""
        from blackboard.testing import MockLLMClient
        
        mock = MockLLMClient(responses=[
            (r".*search.*", "Search results found"),
            (r".*plan.*", "Here is the plan"),
        ])
        
        r1 = await mock.generate("Please search for cats")
        r2 = await mock.generate("Create a plan")
        
        assert r1.content == "Search results found"
        assert r2.content == "Here is the plan"
    
    @pytest.mark.asyncio
    async def test_call_tracking(self):
        """Test call history tracking."""
        from blackboard.testing import MockLLMClient
        
        mock = MockLLMClient()
        
        await mock.generate("First call")
        await mock.generate("Second call")
        
        assert mock.get_call_count() == 2
        history = mock.get_call_history()
        assert history[0]["prompt"] == "First call"
        assert history[1]["prompt"] == "Second call"
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test reset functionality."""
        from blackboard.testing import MockLLMClient
        
        mock = MockLLMClient(sequence=["A", "B"])
        
        await mock.generate("x")
        assert mock.get_call_count() == 1
        
        mock.reset()
        
        assert mock.get_call_count() == 0
        r = await mock.generate("y")
        assert r.content == "A"  # Sequence reset


class TestOrchestratorTestFixture:
    """Tests for OrchestratorTestFixture."""
    
    def test_fixture_creation(self):
        """Test fixture can be created."""
        from blackboard.testing import OrchestratorTestFixture
        
        fixture = OrchestratorTestFixture()
        
        assert fixture.orchestrator is not None
        assert fixture.llm is not None
        assert fixture.persistence is not None
    
    def test_deterministic_id_generation(self):
        """Test deterministic ID generation."""
        from blackboard.testing import OrchestratorTestFixture
        
        fixture = OrchestratorTestFixture(use_deterministic_ids=True)
        
        id1 = fixture.generate_id("test")
        id2 = fixture.generate_id("test")
        
        assert id1 == "test_0001"
        assert id2 == "test_0002"
    
    def test_fixture_reset(self):
        """Test fixture reset."""
        from blackboard.testing import OrchestratorTestFixture
        
        fixture = OrchestratorTestFixture()
        
        fixture.generate_id("x")
        fixture.generate_id("x")
        fixture.reset()
        
        id1 = fixture.generate_id("x")
        assert id1 == "x_0001"


# =============================================================================
# Middleware Tests
# =============================================================================

class TestBudgetMiddleware:
    """Tests for enhanced BudgetMiddleware."""
    
    def test_budget_middleware_init(self):
        """Test BudgetMiddleware initialization."""
        from blackboard.middleware import BudgetMiddleware
        
        middleware = BudgetMiddleware(
            max_cost_usd=5.00,
            max_tokens=10000
        )
        
        assert middleware.max_cost_usd == 5.00
        assert middleware.max_tokens == 10000
        assert middleware.total_cost == 0.0
        assert middleware.total_tokens == 0
    
    def test_budget_middleware_custom_pricing(self):
        """Test custom pricing in BudgetMiddleware."""
        from blackboard.middleware import BudgetMiddleware
        
        middleware = BudgetMiddleware(
            max_cost_usd=1.00,
            custom_pricing={"my-model": (0.01, 0.02)}
        )
        
        input_cost, output_cost = middleware._get_model_cost("my-model")
        
        assert input_cost == 0.01
        assert output_cost == 0.02
