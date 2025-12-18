"""Tests for Phase 3 Observability: Telemetry and Replay."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from blackboard import Blackboard
from blackboard.events import Event, EventBus, EventType
from blackboard.middleware import StepContext, WorkerContext


class TestTelemetryImport:
    """Test telemetry module imports."""
    
    def test_import_telemetry(self):
        """Test that telemetry module can be imported."""
        from blackboard.telemetry import (
            OpenTelemetryMiddleware,
            SpanAttributes,
            TelemetryMetrics,
            MetricsCollector
        )
        
        assert OpenTelemetryMiddleware is not None
        assert SpanAttributes is not None
        assert TelemetryMetrics is not None
        assert MetricsCollector is not None


class TestTelemetryMetrics:
    """Tests for TelemetryMetrics."""
    
    def test_metrics_creation(self):
        """Test creating TelemetryMetrics."""
        from blackboard.telemetry import TelemetryMetrics
        
        metrics = TelemetryMetrics()
        
        assert metrics.total_steps == 0
        assert metrics.total_workers_called == 0
        assert metrics.total_errors == 0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dict."""
        from blackboard.telemetry import TelemetryMetrics
        
        metrics = TelemetryMetrics(
            total_steps=5,
            total_workers_called=10,
            total_input_tokens=1000,
            total_output_tokens=500
        )
        
        d = metrics.to_dict()
        
        assert d["total_steps"] == 5
        assert d["total_workers_called"] == 10
        assert d["total_input_tokens"] == 1000


class TestOpenTelemetryMiddleware:
    """Tests for OpenTelemetryMiddleware."""
    
    def test_middleware_creation(self):
        """Test creating middleware without OTEL installed."""
        from blackboard.telemetry import OpenTelemetryMiddleware
        
        middleware = OpenTelemetryMiddleware(service_name="test-service")
        
        assert middleware.name == "OpenTelemetryMiddleware"
        assert middleware.service_name == "test-service"
    
    def test_start_end_session(self):
        """Test session lifecycle without OTEL."""
        from blackboard.telemetry import OpenTelemetryMiddleware
        
        middleware = OpenTelemetryMiddleware()
        
        # Should not raise even without OTEL
        middleware.start_session(goal="Test goal", session_id="123")
        middleware.end_session(status="done")
    
    def test_record_llm_call(self):
        """Test recording LLM metrics."""
        from blackboard.telemetry import OpenTelemetryMiddleware
        
        middleware = OpenTelemetryMiddleware()
        
        middleware.record_llm_call(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.01
        )
        
        assert middleware.metrics.total_input_tokens == 100
        assert middleware.metrics.total_output_tokens == 50
        assert middleware.metrics.total_cost == 0.01
    
    def test_get_metrics(self):
        """Test getting session metrics."""
        from blackboard.telemetry import OpenTelemetryMiddleware
        
        middleware = OpenTelemetryMiddleware()
        middleware.metrics.total_steps = 5
        
        metrics = middleware.get_metrics()
        
        assert metrics.total_steps == 5
    
    def test_reset(self):
        """Test resetting middleware state."""
        from blackboard.telemetry import OpenTelemetryMiddleware
        
        middleware = OpenTelemetryMiddleware()
        middleware.metrics.total_steps = 10
        
        middleware.reset()
        
        assert middleware.metrics.total_steps == 0


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_collector_creation(self):
        """Test creating MetricsCollector."""
        from blackboard.telemetry import MetricsCollector
        
        collector = MetricsCollector()
        
        assert len(collector.events) == 0
    
    def test_on_event(self):
        """Test recording events."""
        from blackboard.telemetry import MetricsCollector
        
        collector = MetricsCollector()
        
        event = Event(type=EventType.STEP_COMPLETED, data={"step": 1})
        collector.on_event(event)
        
        assert len(collector.events) == 1
        assert collector.events[0]["type"] == "step_completed"
    
    def test_get_summary(self):
        """Test getting metrics summary."""
        from blackboard.telemetry import MetricsCollector
        
        collector = MetricsCollector()
        
        collector.on_event(Event(type=EventType.ORCHESTRATOR_STARTED))
        collector.on_event(Event(type=EventType.STEP_COMPLETED))
        collector.on_event(Event(type=EventType.WORKER_COMPLETED))
        collector.on_event(Event(type=EventType.ORCHESTRATOR_COMPLETED))
        
        summary = collector.get_summary()
        
        assert summary["total_events"] == 4
        assert summary["total_steps"] == 1


class TestReplayImport:
    """Test replay module imports."""
    
    def test_import_replay(self):
        """Test that replay module can be imported."""
        from blackboard.replay import (
            RecordedSession,
            SessionRecorder,
            RecordingLLMClient,
            ReplayLLMClient,
            ReplayOrchestrator,
            compare_sessions
        )
        
        assert RecordedSession is not None
        assert SessionRecorder is not None
        assert ReplayLLMClient is not None


class TestRecordedLLMCall:
    """Tests for RecordedLLMCall."""
    
    def test_to_dict(self):
        """Test serialization."""
        from blackboard.replay import RecordedLLMCall
        
        call = RecordedLLMCall(
            prompt="Hello",
            response="Hi there",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            timestamp="2024-01-01T00:00:00"
        )
        
        d = call.to_dict()
        
        assert d["prompt"] == "Hello"
        assert d["response"] == "Hi there"
    
    def test_from_dict(self):
        """Test deserialization."""
        from blackboard.replay import RecordedLLMCall
        
        data = {
            "prompt": "Test",
            "response": "Response",
            "model": "test-model",
            "input_tokens": 100,
            "output_tokens": 50,
            "timestamp": "2024-01-01"
        }
        
        call = RecordedLLMCall.from_dict(data)
        
        assert call.prompt == "Test"
        assert call.model == "test-model"


class TestRecordedSession:
    """Tests for RecordedSession."""
    
    def test_to_dict(self):
        """Test session serialization."""
        from blackboard.replay import RecordedSession, RecordedLLMCall
        
        session = RecordedSession(
            goal="Test goal",
            events=[{"type": "step_completed"}],
            llm_calls=[
                RecordedLLMCall("p", "r", "m", 0, 0, "t")
            ],
            final_status="done",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:01:00"
        )
        
        d = session.to_dict()
        
        assert d["goal"] == "Test goal"
        assert len(d["llm_calls"]) == 1
    
    def test_from_dict(self):
        """Test session deserialization."""
        from blackboard.replay import RecordedSession
        
        data = {
            "goal": "My goal",
            "events": [],
            "llm_calls": [],
            "final_status": "done",
            "start_time": "",
            "end_time": ""
        }
        
        session = RecordedSession.from_dict(data)
        
        assert session.goal == "My goal"
        assert session.final_status == "done"


class TestSessionRecorder:
    """Tests for SessionRecorder."""
    
    def test_recorder_creation(self):
        """Test creating recorder."""
        from blackboard.replay import SessionRecorder
        
        recorder = SessionRecorder()
        
        assert recorder.event_bus is not None
    
    def test_record_event(self):
        """Test recording events."""
        from blackboard.replay import SessionRecorder
        
        recorder = SessionRecorder()
        
        # Simulate event
        event = Event(type=EventType.STEP_COMPLETED, data={"step": 1})
        recorder.event_bus.publish(event)
        
        session = recorder.get_session()
        
        assert len(session.events) == 1
    
    def test_record_llm_call(self):
        """Test recording LLM calls."""
        from blackboard.replay import SessionRecorder
        
        recorder = SessionRecorder()
        
        recorder.record_llm_call(
            prompt="Test prompt",
            response="Test response",
            model="gpt-4o"
        )
        
        session = recorder.get_session()
        
        assert len(session.llm_calls) == 1
        assert session.llm_calls[0].prompt == "Test prompt"
    
    def test_reset(self):
        """Test resetting recorder."""
        from blackboard.replay import SessionRecorder
        
        recorder = SessionRecorder()
        recorder.record_llm_call("p", "r")
        
        recorder.reset()
        
        assert len(recorder.get_session().llm_calls) == 0


class TestReplayLLMClient:
    """Tests for ReplayLLMClient."""
    
    def test_replay_responses(self):
        """Test replaying recorded responses."""
        from blackboard.replay import ReplayLLMClient, RecordedLLMCall
        
        calls = [
            RecordedLLMCall("p1", "r1", "m", 10, 5, "t"),
            RecordedLLMCall("p2", "r2", "m", 20, 10, "t"),
        ]
        
        client = ReplayLLMClient(calls)
        
        resp1 = client.generate("p1")
        resp2 = client.generate("p2")
        
        assert resp1.content == "r1"
        assert resp2.content == "r2"
        assert client.calls_remaining == 0
    
    def test_replay_exhausted_strict(self):
        """Test error when replay exhausted in strict mode."""
        from blackboard.replay import ReplayLLMClient
        
        client = ReplayLLMClient([], strict=True)
        
        with pytest.raises(RuntimeError, match="Replay exhausted"):
            client.generate("test")
    
    def test_replay_exhausted_non_strict(self):
        """Test empty response when replay exhausted in non-strict mode."""
        from blackboard.replay import ReplayLLMClient
        
        client = ReplayLLMClient([], strict=False)
        
        response = client.generate("test")
        
        assert response.content == ""
    
    def test_reset(self):
        """Test resetting replay client."""
        from blackboard.replay import ReplayLLMClient, RecordedLLMCall
        
        calls = [RecordedLLMCall("p", "r", "m", 0, 0, "t")]
        client = ReplayLLMClient(calls)
        
        client.generate("p")
        assert client.calls_remaining == 0
        
        client.reset()
        assert client.calls_remaining == 1
    
    @pytest.mark.asyncio
    async def test_async_replay(self):
        """Test async replay."""
        from blackboard.replay import ReplayLLMClient, RecordedLLMCall
        
        calls = [RecordedLLMCall("p", "response", "m", 0, 0, "t")]
        client = ReplayLLMClient(calls)
        
        response = await client.agenerate("p")
        
        assert response.content == "response"


class TestSessionDiff:
    """Tests for session comparison."""
    
    def test_compare_sessions(self):
        """Test comparing sessions."""
        from blackboard.replay import RecordedSession, SessionDiff, compare_sessions
        from blackboard import Blackboard, Status
        
        original = RecordedSession(
            goal="Test",
            events=[
                {"type": "step_completed"},
                {"type": "artifact_created"}
            ],
            llm_calls=[],
            final_status="done",
            start_time="",
            end_time=""
        )
        
        replayed = Blackboard(goal="Test")
        replayed.status = Status.DONE
        replayed._step_count = 1
        # Note: artifacts would be added via add_artifact in real usage
        
        diff = compare_sessions(original, replayed)
        
        assert diff.status_match is True
