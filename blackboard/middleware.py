"""
Middleware System for Blackboard Orchestrator

Provides hooks for intercepting and modifying orchestration flow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Blackboard, Status, Feedback
    from .core import SupervisorDecision, WorkerCall
    from .protocols import Worker, WorkerOutput

# Runtime imports (avoid circular - imported at type check time only)
from .state import Status, Feedback


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
    
    All methods are async to support async operations (LLM calls, DB access, webhooks).
    
    Example:
        class BudgetMiddleware(Middleware):
            def __init__(self, max_cost: float):
                self.max_cost = max_cost
                self.total_cost = 0.0
            
            async def after_step(self, ctx: StepContext) -> None:
                usage = ctx.state.metadata.get("usage", {})
                self.total_cost += usage.get("step_cost", 0)
                if self.total_cost > self.max_cost:
                    ctx.state.update_status(Status.FAILED)
                    ctx.skip_step = True
    """
    
    name: str = "UnnamedMiddleware"
    
    async def before_step(self, ctx: StepContext) -> None:
        """Called before each orchestration step."""
        pass
    
    async def after_step(self, ctx: StepContext) -> None:
        """Called after each orchestration step."""
        pass
    
    async def before_worker(self, ctx: WorkerContext) -> None:
        """Called before a worker is executed."""
        pass
    
    async def after_worker(self, ctx: WorkerContext) -> None:
        """Called after a worker is executed."""
        pass
    
    async def on_error(self, ctx: WorkerContext) -> None:
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
    
    async def before_step(self, ctx: StepContext) -> None:
        """Run all before_step hooks."""
        for middleware in self._middleware:
            await middleware.before_step(ctx)
            if ctx.skip_step:
                break
    
    async def after_step(self, ctx: StepContext) -> None:
        """Run all after_step hooks (reverse order)."""
        for middleware in reversed(self._middleware):
            await middleware.after_step(ctx)
    
    async def before_worker(self, ctx: WorkerContext) -> None:
        """Run all before_worker hooks."""
        for middleware in self._middleware:
            await middleware.before_worker(ctx)
            if ctx.skip_worker:
                break
    
    async def after_worker(self, ctx: WorkerContext) -> None:
        """Run all after_worker hooks (reverse order)."""
        for middleware in reversed(self._middleware):
            await middleware.after_worker(ctx)
    
    async def on_error(self, ctx: WorkerContext) -> None:
        """Run all on_error hooks."""
        for middleware in self._middleware:
            await middleware.on_error(ctx)
    
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
    
    Implements REACTIVE budget enforcement:
    - Checks accumulated cost BEFORE each step
    - If over budget, halts immediately (no further execution)
    - Integrates with blackboard.pricing for LiteLLM-based cost lookup
    
    Example:
        from blackboard.middleware import BudgetMiddleware
        
        budget = BudgetMiddleware(max_cost_usd=1.00)
        orchestrator = Orchestrator(..., middleware=[budget])
    """
    
    name = "BudgetMiddleware"
    
    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        custom_pricing: Optional[Dict[str, tuple]] = None
    ):
        """
        Initialize budget middleware.
        
        Args:
            max_tokens: Optional hard limit on total tokens
            max_cost_usd: Optional hard limit on total cost in USD
            custom_pricing: Optional dict of model -> (input_per_1k, output_per_1k)
        """
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.custom_pricing = custom_pricing or {}
        
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def _get_model_cost(self, model: str) -> tuple:
        """Get cost per 1k tokens for a model."""
        # Check custom overrides first
        if model in self.custom_pricing:
            return self.custom_pricing[model]
        
        # Use pricing module
        try:
            from .pricing import get_model_cost
            return get_model_cost(model)
        except ImportError:
            return (0.001, 0.002)  # Safe fallback
    
    async def before_step(self, ctx: StepContext) -> None:
        """
        REACTIVE CHECK: Halt immediately if already over budget.
        
        This runs BEFORE each step to prevent further execution
        once budget is exceeded.
        """
        # Check if we're already over budget
        if self.max_cost_usd and self.total_cost >= self.max_cost_usd:
            from .pricing import BudgetExceededError  # Only import error class lazily
            
            ctx.state.update_status(Status.FAILED)
            ctx.state.metadata["failure_reason"] = "Budget exceeded"
            ctx.state.metadata["budget"] = {
                "total_cost": self.total_cost,
                "max_cost": self.max_cost_usd,
            }
            ctx.state.add_feedback(Feedback(
                source="System:BudgetMiddleware",
                critique=f"HALTED: Budget exceeded (${self.total_cost:.4f} / ${self.max_cost_usd:.4f}). No further steps allowed.",
                passed=False
            ))
            ctx.skip_step = True
            return
        
        if self.max_tokens and self.total_tokens >= self.max_tokens:
            
            ctx.state.update_status(Status.FAILED)
            ctx.state.metadata["failure_reason"] = "Token budget exceeded"
            ctx.state.add_feedback(Feedback(
                source="System:BudgetMiddleware",
                critique=f"HALTED: Token budget exceeded ({self.total_tokens} / {self.max_tokens}). No further steps allowed.",
                passed=False
            ))
            ctx.skip_step = True
    
    async def after_step(self, ctx: StepContext) -> None:
        """Update accumulated costs after each step completes."""
        usage = ctx.state.metadata.get("last_usage", {})
        model = ctx.state.metadata.get("last_model", "gpt-4o-mini")
        
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        # Use model-aware pricing
        input_per_1k, output_per_1k = self._get_model_cost(model)
        step_cost = (
            (input_tokens / 1000) * input_per_1k +
            (output_tokens / 1000) * output_per_1k
        )
        
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += step_cost
        
        # Update metadata for visibility
        ctx.state.metadata["budget"] = {
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "remaining_tokens": (self.max_tokens - self.total_tokens) if self.max_tokens else None,
            "remaining_budget": round(self.max_cost_usd - self.total_cost, 6) if self.max_cost_usd else None
        }


class LoggingMiddleware(Middleware):
    """
    Logs all orchestration events for debugging.
    """
    
    name = "LoggingMiddleware"
    
    def __init__(self, logger=None):
        import logging
        self.logger = logger or logging.getLogger("blackboard.middleware")
    
    async def before_step(self, ctx: StepContext) -> None:
        self.logger.debug(f"[Step {ctx.step_number}] Starting")
    
    async def after_step(self, ctx: StepContext) -> None:
        self.logger.debug(f"[Step {ctx.step_number}] Completed")
    
    async def before_worker(self, ctx: WorkerContext) -> None:
        self.logger.info(f"[Worker] Calling {ctx.worker.name}")
    
    async def after_worker(self, ctx: WorkerContext) -> None:
        self.logger.info(f"[Worker] {ctx.worker.name} completed")
    
    async def on_error(self, ctx: WorkerContext) -> None:
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
    
    async def before_worker(self, ctx: WorkerContext) -> None:
        if ctx.worker.name in self.require_approval_for:
            # Support both sync and async callbacks
            import asyncio
            result = self.approval_callback(ctx.worker.name, ctx.call.instructions)
            if asyncio.iscoroutine(result):
                approved = await result
            else:
                approved = result
            
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
    
    async def after_step(self, ctx: StepContext) -> None:
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
            await self._summarize(state)
            self._last_summarized_step = state.step_count
    
    async def _summarize(self, state) -> None:
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
            # Properly await if async LLM
            if asyncio.iscoroutine(result):
                result = await result
            
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


class ConsoleLoggingMiddleware(Middleware):
    """
    Prints colored events to stdout for easy observability.
    
    No logging configuration required - just add to middleware stack.
    
    Args:
        show_state: If True, print full state (verbose). Default: False
        use_colors: If True, use ANSI colors. Default: True
        
    Example:
        orchestrator = Orchestrator(
            llm=my_llm,
            workers=[...],
            middleware=[ConsoleLoggingMiddleware()]
        )
    """
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "red": "\033[31m",
    }
    
    def __init__(self, show_state: bool = False, use_colors: bool = True):
        self.show_state = show_state
        self.use_colors = use_colors
    
    def _c(self, color: str, text: str) -> str:
        """Apply color if enabled."""
        if self.use_colors:
            return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
        return text
    
    async def before_step(self, ctx: "StepContext") -> None:
        step = ctx.step_number
        status = ctx.state.status.value if ctx.state.status else "unknown"
        print(f"\n{self._c('bold', '━' * 50)}")
        print(f"{self._c('cyan', '▶ STEP')} {self._c('bold', str(step))} | Status: {self._c('yellow', status)}")
        print(f"{self._c('dim', 'Goal:')} {ctx.state.goal[:60]}..." if len(ctx.state.goal) > 60 else f"{self._c('dim', 'Goal:')} {ctx.state.goal}")
    
    async def after_step(self, ctx: "StepContext") -> None:
        if ctx.decision:
            action = ctx.decision.action
            if action == "call":
                workers = ", ".join(ctx.decision.workers or [])
                print(f"{self._c('green', '✓ Decision:')} call [{self._c('magenta', workers)}]")
            elif action == "done":
                print(f"{self._c('green', '✓ Decision:')} {self._c('bold', 'DONE')}")
            elif action == "fail":
                print(f"{self._c('red', '✗ Decision:')} FAIL - {ctx.decision.reason}")
            else:
                print(f"{self._c('yellow', '? Decision:')} {action}")
        
        if self.show_state:
            print(f"{self._c('dim', 'Artifacts:')} {len(ctx.state.artifacts)} | {self._c('dim', 'Feedback:')} {len(ctx.state.feedback)}")
    
    async def before_worker(self, ctx: "WorkerContext") -> None:
        worker_name = ctx.worker.name
        instructions = ctx.call.instructions[:50] + "..." if len(ctx.call.instructions) > 50 else ctx.call.instructions
        print(f"  {self._c('blue', '→')} {self._c('bold', worker_name)}: {self._c('dim', instructions)}")
    
    async def after_worker(self, ctx: "WorkerContext") -> None:
        worker_name = ctx.worker.name
        if ctx.error:
            print(f"  {self._c('red', '✗')} {worker_name} error: {ctx.error}")
        elif ctx.modified_output and ctx.modified_output.has_artifact():
            print(f"  {self._c('green', '✓')} {worker_name} created artifact: {ctx.modified_output.artifact.type}")
        else:
            print(f"  {self._c('green', '✓')} {worker_name} completed")


# =============================================================================
# Production-Grade Middleware
# =============================================================================

class CircuitBreakerMiddleware(Middleware):
    """
    Implements circuit breaker pattern for worker execution.
    
    Prevents cascading failures by tracking errors and temporarily
    blocking worker execution when failures exceed threshold.
    
    State is persisted in Blackboard.metadata["circuit_breaker"] to survive
    across HTTP requests and distributed runs.
    
    States:
    - closed: Normal operation, errors increment failure count
    - open: Failing fast, worker execution skipped
    - half-open: Testing recovery, one execution allowed
    
    Args:
        failure_threshold: Failures before opening circuit (default: 5)
        recovery_timeout: Seconds before trying half-open (default: 30)
        worker_patterns: Optional list of worker name patterns to apply to
        
    Example:
        middleware = [CircuitBreakerMiddleware(failure_threshold=3)]
        orchestrator = Orchestrator(llm=llm, workers=workers, middleware=middleware)
    """
    
    name = "CircuitBreaker"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        worker_patterns: Optional[List[str]] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.worker_patterns = worker_patterns
    
    def _get_state(self, ctx: WorkerContext) -> dict:
        """Get or initialize circuit breaker state from Blackboard metadata."""
        return ctx.state.metadata.setdefault("circuit_breaker", {
            "failures": 0,
            "state": "closed",
            "last_failure": None,
            "last_success": None
        })
    
    def _should_apply(self, worker_name: str) -> bool:
        """Check if circuit breaker applies to this worker."""
        if self.worker_patterns is None:
            return True
        return any(pattern in worker_name for pattern in self.worker_patterns)
    
    async def before_worker(self, ctx: WorkerContext) -> None:
        """Check circuit state before worker execution."""
        import time
        
        if not self._should_apply(ctx.worker.name):
            return
        
        cb = self._get_state(ctx)
        
        if cb["state"] == "open":
            # Check if recovery timeout has passed
            if cb["last_failure"] and time.time() - cb["last_failure"] > self.recovery_timeout:
                cb["state"] = "half-open"
            else:
                # Skip worker execution
                ctx.skip_worker = True
                ctx.error = Exception(f"Circuit breaker OPEN for worker {ctx.worker.name}")
                return
    
    async def after_worker(self, ctx: WorkerContext) -> None:
        """Reset circuit on success."""
        import time
        
        if not self._should_apply(ctx.worker.name):
            return
        
        if ctx.error is None:
            cb = self._get_state(ctx)
            cb["failures"] = 0
            cb["state"] = "closed"
            cb["last_success"] = time.time()
    
    async def on_error(self, ctx: WorkerContext) -> bool:
        """Track failures and open circuit if threshold exceeded."""
        import time
        
        if not self._should_apply(ctx.worker.name):
            return False
        
        cb = self._get_state(ctx)
        cb["failures"] += 1
        cb["last_failure"] = time.time()
        
        if cb["failures"] >= self.failure_threshold:
            cb["state"] = "open"
        
        return False  # Don't suppress error, just track


class DatasetLoggingMiddleware(Middleware):
    """
    Logs context→decision pairs for LLM fine-tuning datasets.
    
    Captures structured data in JSONL format suitable for training:
    - prompt: The context string shown to the supervisor LLM
    - completion: The supervisor's decision (JSON)
    - score: 1.0 for success, 0.0 for failure
    
    Args:
        filepath: Path to JSONL file (default: "finetune_dataset.jsonl")
        max_prompt_length: Truncate prompts to this length (default: 4000)
        include_worker_pairs: Also log worker inputs/outputs (default: False)
        
    Example:
        middleware = [DatasetLoggingMiddleware("training_data.jsonl")]
        orchestrator = Orchestrator(..., middleware=middleware)
        
        # After running, you'll have JSONL entries like:
        # {"prompt": "...", "completion": "...", "score": 1.0}
    """
    
    name = "DatasetLogger"
    
    def __init__(
        self,
        filepath: str = "finetune_dataset.jsonl",
        max_prompt_length: int = 4000,
        include_worker_pairs: bool = False
    ):
        self.filepath = filepath
        self.max_prompt_length = max_prompt_length
        self.include_worker_pairs = include_worker_pairs
    
    async def after_step(self, ctx: StepContext) -> None:
        """Log supervisor decisions for fine-tuning."""
        import json
        
        if ctx.decision is None:
            return
        
        # Determine score based on status
        from .state import Status
        score = 0.0 if ctx.state.status == Status.FAILED else 1.0
        
        # Build decision dict
        decision_dict = {
            "action": ctx.decision.action,
            "reasoning": ctx.decision.reasoning,
            "calls": [
                {"worker": c.worker_name, "instructions": c.instructions}
                for c in ctx.decision.calls
            ]
        }
        
        entry = {
            "type": "supervisor_decision",
            "prompt": ctx.state.to_context_string()[:self.max_prompt_length],
            "completion": json.dumps(decision_dict),
            "score": score,
            "step": ctx.step_number
        }
        
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Don't fail orchestration on logging errors
    
    async def after_worker(self, ctx: WorkerContext) -> None:
        """Optionally log worker input/output pairs."""
        if not self.include_worker_pairs:
            return
        
        import json
        
        entry = {
            "type": "worker_execution",
            "worker": ctx.worker.name,
            "instructions": ctx.call.instructions[:1000] if ctx.call else "",
            "has_artifact": ctx.modified_output.has_artifact() if ctx.modified_output else False,
            "has_feedback": ctx.modified_output.has_feedback() if ctx.modified_output else False,
            "error": str(ctx.error) if ctx.error else None
        }
        
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

