"""Tests for event bus and retry mechanism."""

import pytest
import asyncio

from blackboard.events import EventBus, Event, EventType
from blackboard.retry import RetryPolicy, retry_with_backoff, is_transient_error


class TestEventBus:
    """Tests for the event bus."""
    
    def test_subscribe_and_publish(self):
        """Test basic subscribe and publish."""
        bus = EventBus()
        received = []
        
        def handler(event: Event):
            received.append(event)
        
        bus.subscribe(EventType.ARTIFACT_CREATED, handler)
        bus.publish(Event(EventType.ARTIFACT_CREATED, {"id": "123"}))
        
        assert len(received) == 1
        assert received[0].data["id"] == "123"
    
    def test_subscribe_all(self):
        """Test subscribing to all events."""
        bus = EventBus()
        received = []
        
        bus.subscribe_all(lambda e: received.append(e))
        
        bus.publish(Event(EventType.ARTIFACT_CREATED, {"a": 1}))
        bus.publish(Event(EventType.FEEDBACK_ADDED, {"b": 2}))
        
        assert len(received) == 2
    
    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        received = []
        
        def handler(event):
            received.append(event)
        
        bus.subscribe(EventType.STEP_STARTED, handler)
        bus.publish(Event(EventType.STEP_STARTED, {}))
        
        bus.unsubscribe(EventType.STEP_STARTED, handler)
        bus.publish(Event(EventType.STEP_STARTED, {}))
        
        assert len(received) == 1
    
    @pytest.mark.asyncio
    async def test_publish_async(self):
        """Test async event publishing."""
        bus = EventBus()
        received = []
        
        async def async_handler(event: Event):
            await asyncio.sleep(0.01)
            received.append(event)
        
        bus.subscribe_async(EventType.WORKER_COMPLETED, async_handler)
        await bus.publish_async(Event(EventType.WORKER_COMPLETED, {"worker": "Test"}))
        
        assert len(received) == 1
    
    def test_event_bus_isolation(self):
        """Test that separate EventBus instances are isolated."""
        bus1 = EventBus()
        bus2 = EventBus()
        
        received1 = []
        received2 = []
        
        bus1.subscribe(EventType.ARTIFACT_CREATED, lambda e: received1.append(e))
        bus2.subscribe(EventType.ARTIFACT_CREATED, lambda e: received2.append(e))
        
        bus1.publish(Event(EventType.ARTIFACT_CREATED, {"source": "bus1"}))
        
        assert len(received1) == 1
        assert len(received2) == 0  # bus2 should not receive bus1's events
    
    def test_event_to_dict(self):
        """Test event serialization."""
        event = Event(EventType.ARTIFACT_CREATED, {"id": "abc"})
        data = event.to_dict()
        
        assert data["type"] == "artifact_created"
        assert data["data"]["id"] == "abc"
        assert "timestamp" in data


class TestRetryPolicy:
    """Tests for retry policy."""
    
    def test_default_policy(self):
        """Test default retry policy values."""
        policy = RetryPolicy()
        
        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.backoff_factor == 2.0
    
    def test_should_retry(self):
        """Test retry decision logic."""
        policy = RetryPolicy(max_retries=3)
        
        # Should retry timeout
        assert policy.should_retry(TimeoutError(), 0) is True
        assert policy.should_retry(TimeoutError(), 2) is True
        
        # Should not retry after max attempts
        assert policy.should_retry(TimeoutError(), 3) is False
        
        # Should not retry non-retryable exceptions
        assert policy.should_retry(ValueError(), 0) is False
    
    def test_get_delay(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(initial_delay=1.0, backoff_factor=2.0, max_delay=10.0)
        
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(5) == 10.0  # Capped at max_delay
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test successful execution without retry."""
        call_count = 0
        
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await retry_with_backoff(succeed, RetryPolicy(max_retries=3))
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_retry_then_success(self):
        """Test retry behavior with eventual success."""
        call_count = 0
        
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Transient failure")
            return "success"
        
        policy = RetryPolicy(max_retries=3, initial_delay=0.01)
        result = await retry_with_backoff(fail_then_succeed, policy)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_exhausted(self):
        """Test behavior when retries are exhausted."""
        
        async def always_fail():
            raise TimeoutError("Always fails")
        
        policy = RetryPolicy(max_retries=2, initial_delay=0.01)
        
        with pytest.raises(TimeoutError):
            await retry_with_backoff(always_fail, policy)
    
    def test_is_transient_error(self):
        """Test transient error detection."""
        assert is_transient_error(TimeoutError()) is True
        assert is_transient_error(ConnectionError()) is True
        assert is_transient_error(ValueError()) is False
        
        # Test error message detection
        assert is_transient_error(Exception("connection refused")) is True
        assert is_transient_error(Exception("rate limit exceeded")) is True
