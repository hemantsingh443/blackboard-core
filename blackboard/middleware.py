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
            from .state import Status, Feedback
            ctx.state.update_status(Status.FAILED)
            ctx.state.metadata["failure_reason"] = "Token budget exceeded"
            ctx.state.add_feedback(Feedback(
                source="System:BudgetMiddleware",
                critique="Action blocked: Token budget exceeded. No further actions will be processed.",
                passed=False
            ))
            ctx.skip_step = True
        
        if self.max_cost and self.total_cost > self.max_cost:
            from .state import Status, Feedback
            ctx.state.update_status(Status.FAILED)
            ctx.state.metadata["failure_reason"] = "Cost budget exceeded"
            ctx.state.add_feedback(Feedback(
                source="System:BudgetMiddleware",
                critique="Action blocked: Cost budget exceeded. No further actions will be processed.",
                passed=False
            ))
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


class ApprovalRequired(Exception):
    """
    Raised when human approval is required before proceeding.
    
    Catch this exception to implement custom approval flows:
    - Web: Save state, return 202 Accepted, await webhook callback
    - CLI: Prompt user and call orchestrator.resume()
    - Async: Check database flag, retry later
    
    Example:
        try:
            result = await orchestrator.run(goal="Deploy to prod")
        except ApprovalRequired as e:
            # Save state for later resume
            orchestrator.state.save_to_json("pending_approval.json")
            notify_admin(e.worker_name, e.instructions)
    """
    def __init__(self, worker_name: str, instructions: str):
        self.worker_name = worker_name
        self.instructions = instructions
        super().__init__(f"Approval required for worker '{worker_name}'")


class HumanApprovalMiddleware(Middleware):
    """
    Requires human approval before executing certain workers.
    
    .. warning::
        Default callback raises ApprovalRequired exception.
        For server deployments, provide an async-compatible callback
        or catch the exception to implement pause-and-resume.
    
    Example (async callback):
        async def check_approval_db(worker, instructions):
            return await db.check_approval_flag(worker)
        
        approval = HumanApprovalMiddleware(
            require_approval_for=["Deployer"],
            approval_callback=check_approval_db
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
        """
        Default: Raise exception to pause execution.
        
        Override with a non-blocking callback for server deployments.
        """
        raise ApprovalRequired(worker_name, instructions)
    
    def before_worker(self, ctx: WorkerContext) -> None:
        if ctx.worker.name in self.require_approval_for:
            approved = self.approval_callback(ctx.worker.name, ctx.call.instructions)
            if not approved:
                ctx.skip_worker = True
                # Set status to PAUSED so LLM knows what happened
                from .state import Status
                ctx.state.update_status(Status.PAUSED)
                ctx.state.metadata["paused_for"] = {
                    "worker": ctx.worker.name,
                    "reason": "Awaiting human approval"
                }


class AutoSummarizationMiddleware(Middleware):
    """
    Automatically summarizes context when it grows too large.
    
    Uses the LLM to compress history and artifacts into a summary,
    then clears the raw data to free up context window space.
    
    Example:
        summarizer = AutoSummarizationMiddleware(
            llm=my_llm,
            artifact_threshold=10,
            step_threshold=20
        )
        orchestrator = Orchestrator(..., middleware=[summarizer])
    """
    
    name = "AutoSummarizationMiddleware"
    
    SUMMARIZE_PROMPT = '''Summarize the following session history into a concise summary.
Focus on:
1. Key decisions made
2. Important artifacts created
3. Feedback received
4. Current progress toward the goal

Goal: {goal}

History:
{history}

Provide a 2-3 paragraph summary that captures the essential context:'''
    
    def __init__(
        self,
        llm,
        artifact_threshold: int = 10,
        step_threshold: int = 20,
        feedback_threshold: int = 15,
        keep_recent_artifacts: int = 3,
        keep_recent_feedback: int = 3
    ):
        """
        Initialize the summarization middleware.
        
        Args:
            llm: LLM client to use for summarization
            artifact_threshold: Summarize when artifacts exceed this
            step_threshold: Summarize when steps exceed this
            feedback_threshold: Summarize when feedback exceeds this
            keep_recent_artifacts: Keep this many recent artifacts
            keep_recent_feedback: Keep this many recent feedback entries
        """
        self.llm = llm
        self.artifact_threshold = artifact_threshold
        self.step_threshold = step_threshold
        self.feedback_threshold = feedback_threshold
        self.keep_recent_artifacts = keep_recent_artifacts
        self.keep_recent_feedback = keep_recent_feedback
        self._last_summarized_step = 0
    
    def after_step(self, ctx: StepContext) -> None:
        """Check if summarization is needed after each step."""
        state = ctx.state
        
        # Check thresholds
        should_summarize = (
            len(state.artifacts) > self.artifact_threshold or
            state.step_count > self.step_threshold or
            len(state.feedback) > self.feedback_threshold
        )
        
        # Don't summarize too frequently
        if should_summarize and (state.step_count - self._last_summarized_step) >= 5:
            self._summarize(state)
            self._last_summarized_step = state.step_count
    
    def _summarize(self, state) -> None:
        """Perform the summarization."""
        import asyncio
        
        # Build history text
        history_parts = []
        
        if state.context_summary:
            history_parts.append(f"Previous Summary:\n{state.context_summary}")
        
        history_parts.append(f"\nSteps completed: {state.step_count}")
        
        # Include older artifacts (the ones we'll compress)
        old_artifacts = state.artifacts[:-self.keep_recent_artifacts] if len(state.artifacts) > self.keep_recent_artifacts else []
        if old_artifacts:
            history_parts.append("\nArtifacts to summarize:")
            for a in old_artifacts:
                preview = str(a.content)[:200]
                history_parts.append(f"- {a.type} by {a.creator}: {preview}")
        
        # Include older feedback
        old_feedback = state.feedback[:-self.keep_recent_feedback] if len(state.feedback) > self.keep_recent_feedback else []
        if old_feedback:
            history_parts.append("\nFeedback to summarize:")
            for f in old_feedback:
                status = "PASSED" if f.passed else "FAILED"
                history_parts.append(f"- [{status}] {f.source}: {f.critique[:100]}")
        
        history_text = "\n".join(history_parts)
        
        # Generate summary
        prompt = self.SUMMARIZE_PROMPT.format(
            goal=state.goal,
            history=history_text
        )
        
        try:
            result = self.llm.generate(prompt)
            if asyncio.iscoroutine(result):
                # Can't await in sync context, skip summarization
                return
            
            # Handle LLMResponse or string
            if hasattr(result, 'content'):
                summary = result.content
            else:
                summary = result
            
            # Update state
            state.update_summary(summary)
            
            # Compact: keep only recent artifacts and feedback
            if len(state.artifacts) > self.keep_recent_artifacts:
                state.artifacts = state.artifacts[-self.keep_recent_artifacts:]
            
            if len(state.feedback) > self.keep_recent_feedback:
                state.feedback = state.feedback[-self.keep_recent_feedback:]
            
            # Compact history
            state.compact_history(keep_last=10)
            
        except Exception as e:
            import logging
            logging.getLogger("blackboard.middleware").warning(f"Summarization failed: {e}")

