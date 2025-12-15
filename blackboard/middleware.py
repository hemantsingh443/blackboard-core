"""
Middleware System for Blackboard Orchestrator

Provides hooks for intercepting and modifying orchestration flow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Blackboard
    from .core import SupervisorDecision, WorkerCall
    from .protocols import Worker, WorkerOutput


@dataclass
class StepContext:
    """Context passed to middleware during step execution."""
    step_number: int
    state: "Blackboard"
    decision: Optional["SupervisorDecision"] = None
    
    # For modification by middleware
    skip_step: bool = False
    modified_decision: Optional["SupervisorDecision"] = None


@dataclass
class WorkerContext:
    """Context passed to middleware during worker execution."""
    worker: "Worker"
    call: "WorkerCall"
    state: "Blackboard"
    
    # For modification by middleware
    skip_worker: bool = False
    modified_output: Optional["WorkerOutput"] = None
    error: Optional[Exception] = None


class Middleware(ABC):
    """
    Base class for orchestrator middleware.
    
    Middleware can intercept and modify the orchestration flow at various points:
    - Before/after each step
    - Before/after each worker execution
    
    Example:
        class BudgetMiddleware(Middleware):
            def __init__(self, max_cost: float):
                self.max_cost = max_cost
                self.total_cost = 0.0
            
            def after_step(self, ctx: StepContext) -> None:
                usage = ctx.state.metadata.get("usage", {})
                self.total_cost += usage.get("step_cost", 0)
                if self.total_cost > self.max_cost:
                    ctx.state.update_status(Status.FAILED)
                    ctx.skip_step = True
    """
    
    name: str = "UnnamedMiddleware"
    
    def before_step(self, ctx: StepContext) -> None:
        """Called before each orchestration step."""
        pass
    
    def after_step(self, ctx: StepContext) -> None:
        """Called after each orchestration step."""
        pass
    
    def before_worker(self, ctx: WorkerContext) -> None:
        """Called before a worker is executed."""
        pass
    
    def after_worker(self, ctx: WorkerContext) -> None:
        """Called after a worker is executed."""
        pass
    
    def on_error(self, ctx: WorkerContext) -> None:
        """Called when a worker raises an exception."""
        pass


class MiddlewareStack:
    """
    Manages a stack of middleware.
    
    Middleware is executed in order for "before" hooks,
    and in reverse order for "after" hooks (like a stack).
    """
    
    def __init__(self):
        self._middleware: List[Middleware] = []
    
    def add(self, middleware: Middleware) -> None:
        """Add middleware to the stack."""
        self._middleware.append(middleware)
    
    def remove(self, name: str) -> bool:
        """Remove middleware by name. Returns True if found."""
        for i, m in enumerate(self._middleware):
            if m.name == name:
                self._middleware.pop(i)
                return True
        return False
    
    def before_step(self, ctx: StepContext) -> None:
        """Run all before_step hooks."""
        for middleware in self._middleware:
            middleware.before_step(ctx)
            if ctx.skip_step:
                break
    
    def after_step(self, ctx: StepContext) -> None:
        """Run all after_step hooks (reverse order)."""
        for middleware in reversed(self._middleware):
            middleware.after_step(ctx)
    
    def before_worker(self, ctx: WorkerContext) -> None:
        """Run all before_worker hooks."""
        for middleware in self._middleware:
            middleware.before_worker(ctx)
            if ctx.skip_worker:
                break
    
    def after_worker(self, ctx: WorkerContext) -> None:
        """Run all after_worker hooks (reverse order)."""
        for middleware in reversed(self._middleware):
            middleware.after_worker(ctx)
    
    def on_error(self, ctx: WorkerContext) -> None:
        """Run all on_error hooks."""
        for middleware in self._middleware:
            middleware.on_error(ctx)
    
    def __len__(self) -> int:
        return len(self._middleware)
    
    def __iter__(self):
        return iter(self._middleware)


# =============================================================================
# Built-in Middleware Examples
# =============================================================================

class BudgetMiddleware(Middleware):
    """
    Tracks token usage and stops execution when budget is exceeded.
    
    Example:
        budget = BudgetMiddleware(max_tokens=10000, cost_per_1k=0.01)
        orchestrator = Orchestrator(..., middleware=[budget])
    """
    
    name = "BudgetMiddleware"
    
    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_cost: Optional[float] = None,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0
    ):
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def after_step(self, ctx: StepContext) -> None:
        """Check budget after each step."""
        usage = ctx.state.metadata.get("last_usage", {})
        
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += (
            (input_tokens / 1000) * self.cost_per_1k_input +
            (output_tokens / 1000) * self.cost_per_1k_output
        )
        
        # Update metadata
        ctx.state.metadata["budget"] = {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "remaining_tokens": (self.max_tokens - self.total_tokens) if self.max_tokens else None,
            "remaining_budget": (self.max_cost - self.total_cost) if self.max_cost else None
        }
        
        # Check limits
        if self.max_tokens and self.total_tokens > self.max_tokens:
            from .state import Status
            ctx.state.update_status(Status.FAILED)
            ctx.state.metadata["failure_reason"] = "Token budget exceeded"
            ctx.skip_step = True
        
        if self.max_cost and self.total_cost > self.max_cost:
            from .state import Status
            ctx.state.update_status(Status.FAILED)
            ctx.state.metadata["failure_reason"] = "Cost budget exceeded"
            ctx.skip_step = True


class LoggingMiddleware(Middleware):
    """
    Logs all orchestration events for debugging.
    """
    
    name = "LoggingMiddleware"
    
    def __init__(self, logger=None):
        import logging
        self.logger = logger or logging.getLogger("blackboard.middleware")
    
    def before_step(self, ctx: StepContext) -> None:
        self.logger.debug(f"[Step {ctx.step_number}] Starting")
    
    def after_step(self, ctx: StepContext) -> None:
        self.logger.debug(f"[Step {ctx.step_number}] Completed")
    
    def before_worker(self, ctx: WorkerContext) -> None:
        self.logger.info(f"[Worker] Calling {ctx.worker.name}")
    
    def after_worker(self, ctx: WorkerContext) -> None:
        self.logger.info(f"[Worker] {ctx.worker.name} completed")
    
    def on_error(self, ctx: WorkerContext) -> None:
        self.logger.error(f"[Worker] {ctx.worker.name} failed: {ctx.error}")


class HumanApprovalMiddleware(Middleware):
    """
    Requires human approval before executing certain workers.
    
    Example:
        approval = HumanApprovalMiddleware(
            require_approval_for=["Deployer", "DataDeleter"],
            approval_callback=lambda w, i: input(f"Approve {w}? (y/n): ") == "y"
        )
    """
    
    name = "HumanApprovalMiddleware"
    
    def __init__(
        self,
        require_approval_for: List[str] = None,
        approval_callback: Callable[[str, str], bool] = None
    ):
        self.require_approval_for = require_approval_for or []
        self.approval_callback = approval_callback or self._default_approval
    
    def _default_approval(self, worker_name: str, instructions: str) -> bool:
        """Default approval via console input."""
        print(f"\n[APPROVAL REQUIRED]")
        print(f"Worker: {worker_name}")
        print(f"Instructions: {instructions}")
        response = input("Approve? (y/n): ").strip().lower()
        return response == 'y'
    
    def before_worker(self, ctx: WorkerContext) -> None:
        if ctx.worker.name in self.require_approval_for:
            approved = self.approval_callback(ctx.worker.name, ctx.call.instructions)
            if not approved:
                ctx.skip_worker = True
                ctx.state.metadata["skipped_workers"] = ctx.state.metadata.get("skipped_workers", [])
                ctx.state.metadata["skipped_workers"].append(ctx.worker.name)
