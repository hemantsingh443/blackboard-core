"""
Functional Worker Decorators

Provides a simple decorator-based API for creating workers.
Reduces boilerplate from ~15 lines to ~3 lines.

Example:
    @worker(name="Adder", description="Adds two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    @worker(name="CodeWriter", description="Writes code", artifact_type="code")
    async def write_code(state: Blackboard, inputs: WorkerInput) -> str:
        return "def hello(): pass"
"""

import asyncio
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, get_type_hints, Union

from .protocols import Worker, WorkerOutput, WorkerInput
from .state import Artifact, Blackboard


class FunctionalWorker(Worker):
    """Worker that wraps a function."""
    
    def __init__(
        self,
        fn: Callable,
        worker_name: str,
        worker_description: str,
        worker_artifact_type: str = "text",
        worker_parallel_safe: bool = False,
        worker_input_schema: Optional[Type[WorkerInput]] = None
    ):
        self._fn = fn
        self._name = worker_name
        self._description = worker_description
        self._artifact_type = worker_artifact_type
        self._parallel_safe = worker_parallel_safe
        self._input_schema = worker_input_schema
        
        # Analyze function signature
        sig = inspect.signature(fn)
        self._params = list(sig.parameters.values())
        self._is_async = asyncio.iscoroutinefunction(fn)
        
        # Determine function type
        self._has_state_param = len(self._params) > 0 and self._params[0].name == "state"
        self._has_inputs_param = len(self._params) > 1 and self._params[1].name == "inputs"
    
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
    def input_schema(self) -> Optional[Type[WorkerInput]]:
        return self._input_schema
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        """Execute the wrapped function and return WorkerOutput."""
        
        # Build arguments based on function signature
        if self._has_state_param and self._has_inputs_param:
            # Full signature: (state, inputs)
            result = self._fn(state, inputs)
        elif self._has_state_param:
            # State only: (state)
            result = self._fn(state)
        elif inputs and hasattr(inputs, '__dict__'):
            # Custom args from inputs
            kwargs = {}
            for param in self._params:
                if hasattr(inputs, param.name):
                    kwargs[param.name] = getattr(inputs, param.name)
                elif param.default != inspect.Parameter.empty:
                    kwargs[param.name] = param.default
            result = self._fn(**kwargs)
        else:
            # No args - use defaults
            kwargs = {}
            for param in self._params:
                if param.default != inspect.Parameter.empty:
                    kwargs[param.name] = param.default
            if kwargs:
                result = self._fn(**kwargs)
            else:
                result = self._fn()
        
        # Await if async
        if asyncio.iscoroutine(result):
            result = await result
        
        # Handle different return types
        if isinstance(result, WorkerOutput):
            return result
        elif isinstance(result, Artifact):
            return WorkerOutput(artifact=result)
        else:
            # Wrap raw result in Artifact
            return WorkerOutput(
                artifact=Artifact(
                    type=self._artifact_type,
                    content=str(result) if not isinstance(result, str) else result,
                    creator=self._name
                )
            )
    
    def __repr__(self) -> str:
        return f"FunctionalWorker({self._name})"


def worker(
    name: str,
    description: str,
    artifact_type: str = "text",
    parallel_safe: bool = False,
    input_schema: Optional[Type[WorkerInput]] = None
) -> Callable[[Callable], Worker]:
    """
    Decorator to create a Worker from a function.
    
    The decorated function can have one of these signatures:
    1. Simple: () -> result
    2. With state: (state: Blackboard) -> result
    3. With inputs: (state: Blackboard, inputs: WorkerInput) -> result
    4. Custom args: (arg1: type, arg2: type, ...) -> result
    
    Args:
        name: Worker name (must be unique)
        description: What this worker does
        artifact_type: Type of artifact produced (e.g., "text", "code", "data")
        parallel_safe: Whether this worker can run in parallel
        input_schema: Optional custom input schema class
        
    Returns:
        A Worker instance wrapping the function
        
    Example:
        @worker(name="Greeter", description="Says hello")
        def greet(name: str = "World") -> str:
            return f"Hello, {name}!"
        
        # With state access
        @worker(name="Summarizer", description="Summarizes artifacts")
        async def summarize(state: Blackboard) -> str:
            last = state.get_last_artifact()
            return f"Summary of {last.type}: ..."
    """
    def decorator(fn: Callable) -> Worker:
        # Build input schema from function parameters if not provided
        schema = input_schema
        if schema is None:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            has_state_param = len(params) > 0 and params[0].name == "state"
            
            if not has_state_param and params:
                # Create dynamic schema from function params
                schema_fields = {}
                for param in params:
                    if param.annotation != inspect.Parameter.empty:
                        schema_fields[param.name] = param.annotation
                
                if schema_fields:
                    # Create a dynamic WorkerInput subclass
                    schema = type(
                        f"{name}Input",
                        (WorkerInput,),
                        {"__annotations__": schema_fields}
                    )
        
        return FunctionalWorker(
            fn=fn,
            worker_name=name,
            worker_description=description,
            worker_artifact_type=artifact_type,
            worker_parallel_safe=parallel_safe,
            worker_input_schema=schema
        )
    
    return decorator


class CriticWorker(Worker):
    """Worker that gives feedback on artifacts."""
    
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
        
        # Analyze function signature
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        self._has_state_param = len(params) > 0 and params[0].name == "state"
    
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
        """Execute the critic function and return feedback."""
        from .state import Feedback
        
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
                f"Critic function must return (bool, str) or bool, got {type(result)}"
            )
        
        # Get artifact to link feedback to
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
    
    The decorated function should return a tuple of (passed: bool, critique: str)
    or just a bool (with auto-generated critique).
    
    Example:
        @critic(name="CodeReviewer", description="Reviews code quality")
        def review_code(state: Blackboard) -> tuple[bool, str]:
            last = state.get_last_artifact()
            if "def " not in last.content:
                return False, "No function definitions found"
            return True, "Code looks good!"
    """
    def decorator(fn: Callable) -> Worker:
        return CriticWorker(
            fn=fn,
            worker_name=name,
            worker_description=description,
            worker_parallel_safe=parallel_safe
        )
    
    return decorator
