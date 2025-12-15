"""
Worker Protocols

The "Contract" that any agent must follow to participate in the blackboard system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Artifact, Feedback, Status, Blackboard


@dataclass
class WorkerOutput:
    """
    The structured output from a worker's run() method.
    
    A worker can return:
    - An artifact: A new piece of content (code, text, image, etc.)
    - Feedback: A critique of an existing artifact
    - Status update: A request to change the blackboard status
    - Any combination of the above
    
    Examples:
        # Generator returning an artifact
        return WorkerOutput(artifact=Artifact(type="text", content="Hello", creator="Writer"))
        
        # Critic returning feedback
        return WorkerOutput(feedback=Feedback(source="Critic", critique="Looks good", passed=True))
        
        # Worker signaling completion
        return WorkerOutput(status_update=Status.DONE)
    """
    artifact: Optional["Artifact"] = None
    feedback: Optional["Feedback"] = None
    status_update: Optional["Status"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_artifact(self) -> bool:
        """Check if this output contains an artifact."""
        return self.artifact is not None

    def has_feedback(self) -> bool:
        """Check if this output contains feedback."""
        return self.feedback is not None

    def has_status_update(self) -> bool:
        """Check if this output requests a status update."""
        return self.status_update is not None


class Worker(ABC):
    """
    Abstract base class for all workers in the blackboard system.
    
    Workers are the "spokes" in the hub-and-spoke model:
    - They read from the blackboard to understand context
    - They perform specialized work (generate, validate, transform)
    - They write results back to the blackboard
    
    Workers should NOT communicate directly with each other.
    All communication happens through the shared blackboard state.
    
    Example:
        class TextWriter(Worker):
            name = "Writer"
            description = "Generates text content based on the goal"
            
            def run(self, state: Blackboard) -> WorkerOutput:
                # Generate content based on state.goal
                content = f"Generated content for: {state.goal}"
                return WorkerOutput(
                    artifact=Artifact(type="text", content=content, creator=self.name)
                )
    """
    
    # Class attributes that subclasses should override
    name: str = "UnnamedWorker"
    description: str = "A worker in the blackboard system"
    
    @abstractmethod
    def run(self, state: "Blackboard") -> WorkerOutput:
        """
        Execute the worker's logic and return the result.
        
        Args:
            state: The current blackboard state (read-only view recommended)
            
        Returns:
            WorkerOutput containing artifact, feedback, or status update
            
        Note:
            Workers should treat the state as read-only and return their
            changes via WorkerOutput. The Orchestrator handles actually
            updating the blackboard.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class WorkerRegistry:
    """
    A registry for managing workers.
    
    Provides lookup by name and validation of worker uniqueness.
    """
    
    def __init__(self):
        self._workers: Dict[str, Worker] = {}
    
    def register(self, worker: Worker) -> None:
        """Register a worker. Raises if name already exists."""
        if worker.name in self._workers:
            raise ValueError(f"Worker with name '{worker.name}' already registered")
        self._workers[worker.name] = worker
    
    def get(self, name: str) -> Optional[Worker]:
        """Get a worker by name."""
        return self._workers.get(name)
    
    def list_workers(self) -> Dict[str, str]:
        """Return a dict of worker names to descriptions."""
        return {name: w.description for name, w in self._workers.items()}
    
    def __contains__(self, name: str) -> bool:
        return name in self._workers
    
    def __iter__(self):
        return iter(self._workers.values())
    
    def __len__(self) -> int:
        return len(self._workers)
