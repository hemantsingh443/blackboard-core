"""
Blackboard-Core SDK

A Python SDK implementing the Blackboard Pattern for centralized state multi-agent systems.

Quick Start:
    from blackboard import Orchestrator, Worker, WorkerOutput, Artifact, Blackboard
    
    class MyWriter(Worker):
        name = "Writer"
        description = "Generates text content"
        
        async def run(self, state: Blackboard) -> WorkerOutput:
            return WorkerOutput(
                artifact=Artifact(type="text", content="Hello!", creator=self.name)
            )
    
    orchestrator = Orchestrator(llm=my_llm_client, workers=[MyWriter()])
    result = await orchestrator.run(goal="Write a greeting")
    
    # Save and resume
    result.save_to_json("session.json")
    resumed = Blackboard.load_from_json("session.json")
"""

from .state import Artifact, Feedback, Blackboard, Status
from .protocols import Worker, WorkerOutput, WorkerRegistry, WorkerInput
from .core import Orchestrator, LLMClient, SupervisorDecision, WorkerCall, run_blackboard, run_blackboard_sync
from .events import EventBus, Event, EventType, get_event_bus, reset_event_bus
from .retry import RetryPolicy, retry_with_backoff, DEFAULT_RETRY_POLICY, NO_RETRY, is_transient_error

__version__ = "0.3.0"

__all__ = [
    # State models
    "Artifact",
    "Feedback", 
    "Blackboard",
    "Status",
    # Worker protocol
    "Worker",
    "WorkerOutput",
    "WorkerRegistry",
    "WorkerInput",
    # Orchestrator
    "Orchestrator",
    "LLMClient",
    "SupervisorDecision",
    "WorkerCall",
    "run_blackboard",
    "run_blackboard_sync",
    # Events
    "EventBus",
    "Event",
    "EventType",
    "get_event_bus",
    "reset_event_bus",
    # Retry
    "RetryPolicy",
    "retry_with_backoff",
    "DEFAULT_RETRY_POLICY",
    "NO_RETRY",
    "is_transient_error",
]
