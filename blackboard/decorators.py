"""
Functional Worker Decorators

Provides a decorator-based API for creating workers with automatic
input schema generation from type hints.

Example:
    @worker
    def add(a: int, b: int) -> int:
        '''Adds two numbers.'''
        return a + b

    @worker(artifact_type="code")
    async def write_code(topic: str, state: Blackboard) -> str:
        '''Writes code about a topic.'''
        return f"def hello_{topic}(): pass"
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from pydantic import BaseModel, create_model

from .protocols import Worker, WorkerOutput, WorkerInput
from .state import Artifact, Blackboard, Feedback


class FunctionalWorker(Worker):
    """
    Worker that wraps a standard Python function.
    
    Handles:
    - Dynamic input parsing from Pydantic models
    - Automatic state injection for parameters named 'state'
    - Output wrapping (raw values -> Artifact)
    """
    
    def __init__(
        self,
        fn: Callable,
        worker_name: str,
        worker_description: str,
        worker_artifact_type: str = "text",
        worker_parallel_safe: bool = False,
        worker_input_schema: Optional[Type[BaseModel]] = None
    ):
        self._fn = fn
        self._name = worker_name
        self._description = worker_description
        self._artifact_type = worker_artifact_type
        self._parallel_safe = worker_parallel_safe
        self._input_schema = worker_input_schema
        self._is_async = asyncio.iscoroutinefunction(fn)
        
        # Analyze signature for state injection
        sig = inspect.signature(fn)
        self._params = list(sig.parameters.keys())
        self._has_state_param = "state" in self._params
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parallel_safe(self) -> bool:
        return self._parallel_safe
    
    @property
    def input_schema(self) -> Optional[Type[BaseModel]]:
        return self._input_schema
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        """Execute the wrapped function."""
        
        # Build kwargs from inputs
        kwargs: Dict[str, Any] = {}
        
        if inputs:
            # Get all fields from the input model
            input_data = inputs.model_dump(exclude_unset=False)
            # Filter to only include params the function expects
            for key, value in input_data.items():
                if key in self._params and key != "state":
                    kwargs[key] = value
        
        # Inject state if function expects it
        if self._has_state_param:
            kwargs["state"] = state
        
        # Call function
        try:
            if self._is_async:
                result = await self._fn(**kwargs)
            else:
                result = self._fn(**kwargs)
        except TypeError as e:
            # Handle missing required arguments gracefully
            raise RuntimeError(
                f"Worker '{self._name}' call failed. "
                f"Expected params: {self._params}, got: {list(kwargs.keys())}. "
                f"Error: {e}"
            ) from e
        
        # Wrap result
        if isinstance(result, WorkerOutput):
            return result
        elif isinstance(result, Artifact):
            return WorkerOutput(artifact=result)
        elif isinstance(result, Feedback):
            return WorkerOutput(feedback=result)
        else:
            content = str(result) if result is not None else "Done"
            return WorkerOutput(
                artifact=Artifact(
                    type=self._artifact_type,
                    content=content,
                    creator=self._name
                )
            )
    
    def __repr__(self) -> str:
        return f"FunctionalWorker({self._name})"


def worker(
    _fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    artifact_type: str = "text",
    parallel_safe: bool = False,
    input_schema: Optional[Type[WorkerInput]] = None
) -> Union[Worker, Callable[[Callable], Worker]]:
    """
    Decorator to create a Worker from a function.
    
    Automatically generates input schema from type hints if not provided.
    Description is extracted from the docstring if not provided.
    
    Args:
        name: Worker name (defaults to function name, capitalized)
        description: Worker description (defaults to first line of docstring)
        artifact_type: Type of artifact produced (e.g., "text", "code")
        parallel_safe: Whether this worker can run concurrently
        input_schema: Manual schema override (auto-generated if None)
        
    Usage:
        # Minimal - everything auto-inferred
        @worker
        def greet(name: str) -> str:
            '''Greets a person.'''
            return f"Hello, {name}!"
        
        # With options
        @worker(artifact_type="code", parallel_safe=True)
        def generate_code(language: str = "python") -> str:
            '''Generates boilerplate code.'''
            return "# code here"
        
        # With state access
        @worker
        def summarize(state: Blackboard) -> str:
            '''Summarizes the current artifacts.'''
            return f"Found {len(state.artifacts)} artifacts"
    """
    
    def make_worker(fn: Callable) -> FunctionalWorker:
        # 1. Determine name
        worker_name = name or fn.__name__.replace("_", " ").title().replace(" ", "")
        
        # 2. Determine description from docstring
        worker_description = description
        if not worker_description:
            doc = inspect.getdoc(fn)
            if doc:
                worker_description = doc.strip().split('\n')[0]
            else:
                worker_description = f"Worker: {worker_name}"
        
        # 3. Build input schema from signature
        schema = input_schema
        if schema is None:
            schema = _build_input_schema(fn, worker_name)
        
        return FunctionalWorker(
            fn=fn,
            worker_name=worker_name,
            worker_description=worker_description,
            worker_artifact_type=artifact_type,
            worker_parallel_safe=parallel_safe,
            worker_input_schema=schema
        )
    
    # Support @worker without parentheses
    if _fn is not None:
        return make_worker(_fn)
    
    return make_worker


def _build_input_schema(fn: Callable, worker_name: str) -> Type[WorkerInput]:
    """
    Build a Pydantic model from function signature.
    
    Excludes 'state' parameter (injected at runtime).
    """
    sig = inspect.signature(fn)
    
    # Try to get type hints, fall back to empty dict
    try:
        type_hints = get_type_hints(fn)
    except Exception:
        type_hints = {}
    
    fields: Dict[str, Any] = {}
    
    for param_name, param in sig.parameters.items():
        # Skip state parameter
        if param_name == "state":
            continue
        
        # Get type annotation
        annotation = type_hints.get(param_name, Any)
        
        # Handle Blackboard type (also skip)
        if annotation is Blackboard:
            continue
        
        # Get default value
        if param.default is inspect.Parameter.empty:
            default = ...  # Required field
        else:
            default = param.default
        
        fields[param_name] = (annotation, default)
    
    if not fields:
        # No additional fields needed
        return WorkerInput
    
    # Create dynamic model
    return create_model(
        f"{worker_name}Input",
        __base__=WorkerInput,
        **fields
    )


class CriticWorker(Worker):
    """
    Worker specialized for providing feedback on artifacts.
    
    The wrapped function should return:
    - bool: Simple pass/fail
    - tuple[bool, str]: Pass/fail with explanation
    """
    
    def __init__(
        self,
        fn: Callable,
        worker_name: str,
        worker_description: str,
        worker_parallel_safe: bool = False
    ):
        self._fn = fn
        self._name = worker_name
        self._description = worker_description
        self._parallel_safe = worker_parallel_safe
        self._is_async = asyncio.iscoroutinefunction(fn)
        
        sig = inspect.signature(fn)
        self._has_state_param = "state" in sig.parameters
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parallel_safe(self) -> bool:
        return self._parallel_safe
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        """Execute the critic function."""
        
        # Call function
        if self._has_state_param:
            result = self._fn(state)
        else:
            result = self._fn()
        
        if asyncio.iscoroutine(result):
            result = await result
        
        # Parse result
        if isinstance(result, tuple) and len(result) == 2:
            passed, critique = result
        elif isinstance(result, bool):
            passed = result
            critique = "Approved" if passed else "Needs revision"
        else:
            raise ValueError(
                f"Critic '{self._name}' must return bool or (bool, str), "
                f"got {type(result).__name__}"
            )
        
        last_artifact = state.get_last_artifact()
        
        return WorkerOutput(
            feedback=Feedback(
                source=self._name,
                passed=passed,
                critique=critique,
                artifact_id=last_artifact.id if last_artifact else None
            )
        )
    
    def __repr__(self) -> str:
        return f"CriticWorker({self._name})"


def critic(
    name: str,
    description: str,
    parallel_safe: bool = False
) -> Callable[[Callable], Worker]:
    """
    Decorator for creating critic/reviewer workers.
    
    The decorated function should return:
    - bool: Simple pass/fail (auto-generates critique message)
    - tuple[bool, str]: Pass/fail with custom critique
    
    Example:
        @critic(name="CodeReviewer", description="Reviews code quality")
        def review_code(state: Blackboard) -> tuple[bool, str]:
            last = state.get_last_artifact()
            if "def " not in last.content:
                return False, "No function definitions found"
            return True, "Code looks good!"
    """
    
    def decorator(fn: Callable) -> CriticWorker:
        return CriticWorker(
            fn=fn,
            worker_name=name,
            worker_description=description,
            worker_parallel_safe=parallel_safe
        )
    
    return decorator
