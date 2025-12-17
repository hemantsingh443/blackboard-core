"""
Event Bus for Blackboard System

Pub/Sub pattern for observability and real-time event handling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Set
import asyncio
import logging

logger = logging.getLogger("blackboard.events")


class EventType(str, Enum):
    """Types of events that can be published."""
    # Orchestrator events
    ORCHESTRATOR_STARTED = "orchestrator_started"
    ORCHESTRATOR_COMPLETED = "orchestrator_completed"
    ORCHESTRATOR_PAUSED = "orchestrator_paused"
    ORCHESTRATOR_RESUMED = "orchestrator_resumed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    
    # Worker events
    WORKER_CALLED = "worker_called"
    WORKER_COMPLETED = "worker_completed"
    WORKER_ERROR = "worker_error"
    WORKER_RETRY = "worker_retry"
    
    # State events
    ARTIFACT_CREATED = "artifact_created"
    FEEDBACK_ADDED = "feedback_added"
    STATUS_CHANGED = "status_changed"
    
    # Persistence events
    STATE_SAVED = "state_saved"
    STATE_LOADED = "state_loaded"
    
    # Streaming events
    STREAM_START = "stream_start"
    STREAM_TOKEN = "stream_token"
    STREAM_END = "stream_end"


@dataclass
class Event:
    """An event that can be published to subscribers."""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


# Type alias for event callbacks
EventCallback = Callable[[Event], None]
AsyncEventCallback = Callable[[Event], Any]  # Can be async


class EventBus:
    """
    Pub/Sub event bus for the blackboard system.
    
    Allows components to subscribe to specific event types and receive
    notifications when those events occur.
    
    Example:
        bus = EventBus()
        
        def on_artifact(event: Event):
            print(f"New artifact: {event.data}")
        
        bus.subscribe(EventType.ARTIFACT_CREATED, on_artifact)
        bus.publish(Event(EventType.ARTIFACT_CREATED, {"id": "123"}))
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, Set[EventCallback]] = {}
        self._async_subscribers: Dict[EventType, Set[AsyncEventCallback]] = {}
        self._all_subscribers: Set[EventCallback] = set()
        self._async_all_subscribers: Set[AsyncEventCallback] = set()
    
    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)
        logger.debug(f"Subscribed to {event_type.value}")
    
    def subscribe_async(self, event_type: EventType, callback: AsyncEventCallback) -> None:
        """Subscribe with an async callback."""
        if event_type not in self._async_subscribers:
            self._async_subscribers[event_type] = set()
        self._async_subscribers[event_type].add(callback)
    
    def subscribe_all(self, callback: EventCallback) -> None:
        """Subscribe to all event types."""
        self._all_subscribers.add(callback)
    
    def subscribe_all_async(self, callback: AsyncEventCallback) -> None:
        """Subscribe to all event types with async callback."""
        self._async_all_subscribers.add(callback)
    
    def unsubscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Unsubscribe from a specific event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
    
    def unsubscribe_async(self, event_type: EventType, callback: AsyncEventCallback) -> None:
        """Unsubscribe async callback."""
        if event_type in self._async_subscribers:
            self._async_subscribers[event_type].discard(callback)
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers (sync).
        
        Args:
            event: The event to publish
        """
        logger.debug(f"Publishing {event.type.value}: {event.data}")
        
        # Call type-specific subscribers
        for callback in self._subscribers.get(event.type, set()):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
        
        # Call all-event subscribers
        for callback in self._all_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    async def publish_async(self, event: Event) -> None:
        """
        Publish an event to all subscribers (async).
        
        Calls both sync and async subscribers.
        """
        logger.debug(f"Publishing async {event.type.value}: {event.data}")
        
        # Call sync subscribers first
        self.publish(event)
        
        # Call async type-specific subscribers
        async_tasks = []
        for callback in self._async_subscribers.get(event.type, set()):
            result = callback(event)
            if asyncio.iscoroutine(result):
                async_tasks.append(result)
        
        # Call async all-event subscribers
        for callback in self._async_all_subscribers:
            result = callback(event)
            if asyncio.iscoroutine(result):
                async_tasks.append(result)
        
        # Wait for all async callbacks
        if async_tasks:
            await asyncio.gather(*async_tasks, return_exceptions=True)
    
    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
        self._async_subscribers.clear()
        self._all_subscribers.clear()
        self._async_all_subscribers.clear()
