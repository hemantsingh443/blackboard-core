"""
Worker Protocols

The "Contract" that any agent must follow to participate in the blackboard system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .state import Artifact, Feedback, Status, Blackboard


class WorkerInput(BaseModel):
    """
    Base class for worker input schemas.
    
    Workers can define their own input schema by subclassing this.
    The LLM can then provide structured inputs instead of free-form instructions.
    
    Example:
        class CodeGeneratorInput(WorkerInput):
            language: str = "python"
            description: str
            include_tests: bool = False
    """
    instructions: str = ""  # Default fallback field
    
    model_config = {"extra": "allow"}


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
    
    Attributes:
        name: Unique identifier for the worker
        description: Description for the supervisor LLM
        input_schema: Optional Pydantic model for structured inputs
        parallel_safe: Whether this worker can run in parallel with others
    
    Example:
        class TextWriter(Worker):
            name = "Writer"
            description = "Generates text content based on the goal"
            input_schema = WriterInput  # Optional structured input
            
            async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
                lang = inputs.language if inputs else "python"
                content = f"Generated content for: {state.goal}"
                return WorkerOutput(
                    artifact=Artifact(type="text", content=content, creator=self.name)
                )
    """
    
    # Class attributes that subclasses should override
    name: str = "UnnamedWorker"
    description: str = "A worker in the blackboard system"
    input_schema: Optional[Type[WorkerInput]] = None  # Optional structured input
    parallel_safe: bool = True  # Can run in parallel with other workers
    
    @abstractmethod
    async def run(
        self,
        state: "Blackboard",
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        """
        Execute the worker's logic and return the result.
        
        Args:
            state: The current blackboard state (read-only view recommended)
            inputs: Optional structured inputs from the supervisor
            
        Returns:
            WorkerOutput containing artifact, feedback, or status update
            
        Note:
            Workers should treat the state as read-only and return their
            changes via WorkerOutput. The Orchestrator handles actually
            updating the blackboard.
        """
        pass

    def get_schema_json(self) -> Optional[Dict[str, Any]]:
        """Get JSON schema for this worker's inputs."""
        if self.input_schema is None:
            return None
        return self.input_schema.model_json_schema()

    def parse_inputs(self, data: Dict[str, Any]) -> WorkerInput:
        """Parse input data into the worker's input schema."""
        if self.input_schema is None:
            return WorkerInput(**data)
        return self.input_schema.model_validate(data)

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
    
    def get_many(self, names: List[str]) -> List[Worker]:
        """Get multiple workers by name. Returns only found workers."""
        return [w for name in names if (w := self._workers.get(name)) is not None]
    
    def list_workers(self) -> Dict[str, str]:
        """Return a dict of worker names to descriptions."""
        return {name: w.description for name, w in self._workers.items()}
    
    def list_workers_with_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Return worker info including input schemas."""
        result = {}
        for name, worker in self._workers.items():
            result[name] = {
                "description": worker.description,
                "parallel_safe": worker.parallel_safe,
                "input_schema": worker.get_schema_json()
            }
        return result
    
    def get_parallel_safe(self) -> List[Worker]:
        """Get all workers that can run in parallel."""
        return [w for w in self._workers.values() if w.parallel_safe]
    
    def __contains__(self, name: str) -> bool:
        return name in self._workers
    
    def __iter__(self):
        return iter(self._workers.values())
    
    def __len__(self) -> int:
        return len(self._workers)