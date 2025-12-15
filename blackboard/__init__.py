"""
Blackboard-Core SDK

A Python SDK implementing the Blackboard Pattern for centralized state multi-agent systems.

Quick Start:
    from blackboard import Orchestrator, Worker, WorkerOutput, Artifact, Blackboard
    
    class MyWriter(Worker):
        name = "Writer"
        description = "Generates text content"
        
        def run(self, state: Blackboard) -> WorkerOutput:
            return WorkerOutput(
                artifact=Artifact(type="text", content="Hello!", creator=self.name)
            )
    
    orchestrator = Orchestrator(llm=my_llm_client, workers=[MyWriter()])
    result = orchestrator.run(goal="Write a greeting")
"""

from .state import Artifact, Feedback, Blackboard, Status
from .protocols import Worker, WorkerOutput, WorkerRegistry
from .core import Orchestrator, LLMClient, SupervisorDecision, run_blackboard

__version__ = "0.1.0"

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
    # Orchestrator
    "Orchestrator",
    "LLMClient",
    "SupervisorDecision",
    "run_blackboard",
]
