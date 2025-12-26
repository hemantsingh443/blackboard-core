"""
Orchestrator (Supervisor)

The "Prefrontal Cortex" of the blackboard system.
An LLM-driven supervisor that manages worker execution based on state.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable, Awaitable, Union

from .state import Blackboard, Status, Artifact, Feedback
from .protocols import Worker, WorkerOutput, WorkerRegistry, WorkerInput
from .events import EventBus, Event, EventType
from .retry import RetryPolicy, retry_with_backoff, DEFAULT_RETRY_POLICY, is_transient_error
from .usage import LLMResponse, LLMUsage, UsageTracker
from .middleware import Middleware, MiddlewareStack, StepContext, WorkerContext
from .tools import (
    ToolDefinition, ToolCall, ToolCallingLLMClient, 
    worker_to_tool_definition, workers_to_tool_definitions,
    DONE_TOOL, FAIL_TOOL
)
from .reasoning import (
    ReasoningStrategy,
    OneShotStrategy,
    ChainOfThoughtStrategy,
    get_strategy,
    DEFAULT_STRATEGY
)


# Configure module logger
logger = logging.getLogger("blackboard")


# =============================================================================
# Event Loop Safety Helper
# =============================================================================

def _run_sync(coro):
    """
    Safely run a coroutine in a sync context.
    
    Handles the case where an event loop is already running (e.g., FastAPI,
    Jupyter, asyncio REPL). Falls back to running in a thread pool.
    
    Args:
        coro: Coroutine to execute
        
    Returns:
        The result of the coroutine
    """
    try:
        asyncio.get_running_loop()
        # Loop is running - can't use asyncio.run()
        # Execute in a new thread with its own event loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No loop running - safe to use asyncio.run()
        return asyncio.run(coro)



# Type for LLM responses - can be string or LLMResponse
LLMResult = Union[str, LLMResponse]


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM providers.
    
    Supports returning either:
    - str: Simple text response (backward compatible)
    - LLMResponse: Structured response with usage stats
    
    Examples:
        # Simple string response
        class SimpleLLM:
            def generate(self, prompt: str) -> str:
                return "response text"
        
        # With usage tracking
        class OpenAIClient:
            def generate(self, prompt: str) -> LLMResponse:
                response = openai.chat.completions.create(...)
                return LLMResponse(
                    content=response.choices[0].message.content,
                    usage=LLMUsage(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens
                    )
                )
    """
    
    def generate(self, prompt: str) -> Union[LLMResult, Awaitable[LLMResult]]:
        """Generate a response for the given prompt."""
        ...


@dataclass
class WorkerCall:
    """A single worker call specification."""
    worker_name: str
    instructions: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)  # Structured inputs


@dataclass
class SupervisorDecision:
    """
    The parsed decision from the supervisor LLM.
    
    Supports both single and independent parallel worker calls.
    
    Attributes:
        action: The action to take ("call", "call_independent", "done", "fail")
        calls: List of worker calls (supports parallel execution)
        reasoning: The supervisor's reasoning for this decision
        
    Note:
        For "call_independent", workers read state at call time. Their outputs
        are applied sequentially after all complete - later workers will NOT
        see earlier workers' results within the same call_independent batch.
    """
    action: str  # "call", "call_independent", "done", "fail"
    calls: List[WorkerCall] = field(default_factory=list)
    reasoning: str = ""


class Orchestrator:
    """
    The LLM-driven Supervisor that orchestrates worker execution.
    
    The orchestrator follows the Observe-Reason-Act loop:
    1. Observe: Read the current blackboard state
    2. Reason: Ask the LLM which worker to call next
    3. Act: Execute the worker(s) and update the blackboard
    4. Check: If done or failed, stop; otherwise repeat
    
    Features:
    - Async-first with sync wrapper
    - Parallel worker execution with asyncio.gather
    - Retry mechanism with exponential backoff
    - Event bus integration for observability
    - Resume from saved state
    - Worker input schemas
    
    Example:
        llm = MyLLMClient()
        workers = [TextWriter(), TextReviewer()]
        orchestrator = Orchestrator(llm=llm, workers=workers)
        
        # Async usage
        result = await orchestrator.run(goal="Write a haiku")
        
        # Resume from saved state
        state = Blackboard.load_from_json("session.json")
        result = await orchestrator.run(state=state)
    """

    def __init__(
        self,
        llm: LLMClient,
        workers: List[Worker],
        config: Optional["BlackboardConfig"] = None,
        verbose: bool = False,
        on_step: Optional[Callable[[int, Blackboard, SupervisorDecision], None]] = None,
        event_bus: Optional[EventBus] = None,
        retry_policy: Optional[RetryPolicy] = None,
        auto_save_path: Optional[str] = None,
        enable_parallel: bool = True,
        middleware: Optional[List[Middleware]] = None,
        usage_tracker: Optional[UsageTracker] = None,
        use_tool_calling: bool = True,
        allow_json_fallback: bool = True,
        strict_tools: bool = False,
        auto_summarize: bool = False,
        summarize_thresholds: Optional[Dict[str, int]] = None,
        supervisor_prompt: Optional[str] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm: An LLM client with a generate() method (sync or async)
            workers: List of workers to manage
            config: BlackboardConfig for centralized configuration (overrides loose kwargs)
            verbose: If True, enable INFO level logging
            on_step: Optional callback called after each step
            event_bus: Event bus for observability (creates new if not provided)
            retry_policy: Retry policy for worker execution (default: 3 retries)
            auto_save_path: If provided, auto-save state after each step
            enable_parallel: If True, allow parallel worker execution
            middleware: List of middleware to add to the stack
            usage_tracker: Tracker for LLM token usage and costs
            use_tool_calling: If True, use native tool calling when LLM supports it
            allow_json_fallback: If False, raise error when tool calling fails instead of silently falling back
            strict_tools: If True, validate tool definitions at startup and crash on errors
            auto_summarize: If True, automatically summarize context when thresholds exceeded
            summarize_thresholds: Custom thresholds for summarization
            supervisor_prompt: Custom system prompt for the supervisor LLM
        """
        # Validate inputs
        if not workers:
            raise ValueError("At least one worker must be provided")
        
        # Import config here to avoid circular import
        from .config import BlackboardConfig
        
        # Use config if provided, otherwise use individual kwargs
        if config is not None:
            self.config = config
        else:
            # Build config from kwargs for backward compatibility
            self.config = BlackboardConfig(
                max_steps=20,  # Default, overridden in run()
                allow_unsafe_execution=False,
                simple_prompts=False,
                use_tool_calling=use_tool_calling,
                allow_json_fallback=allow_json_fallback,
                strict_tools=strict_tools,
                enable_parallel=enable_parallel,
                auto_summarize=auto_summarize,
                auto_save_path=auto_save_path,
                verbose=verbose,
                summarize_thresholds=summarize_thresholds or {
                    "artifacts": 10,
                    "feedback": 20,
                    "steps": 50
                }
            )
        
        self.llm = llm
        self.registry = WorkerRegistry()
        for worker in workers:
            self.registry.register(worker)
        self.verbose = self.config.verbose if config else verbose
        self.on_step = on_step
        self.event_bus = event_bus or EventBus()
        self.retry_policy = retry_policy or DEFAULT_RETRY_POLICY
        self.auto_save_path = self.config.auto_save_path if config else auto_save_path
        self.enable_parallel = self.config.enable_parallel if config else enable_parallel
        self.usage_tracker = usage_tracker
        self.use_tool_calling = self.config.use_tool_calling if config else use_tool_calling
        self.allow_json_fallback = self.config.allow_json_fallback if config else allow_json_fallback
        self.strict_tools = self.config.strict_tools if config else strict_tools
        self.auto_summarize = self.config.auto_summarize if config else auto_summarize
        self.summarize_thresholds = self.config.summarize_thresholds if config else (summarize_thresholds or {
            "artifacts": 10,
            "feedback": 20,
            "steps": 50
        })
        
        # Select reasoning strategy
        # Strategy handles prompt construction and response parsing
        if hasattr(self.config, 'reasoning_strategy') and self.config.reasoning_strategy == "cot":
            self.strategy: ReasoningStrategy = ChainOfThoughtStrategy()
        else:
            self.strategy = DEFAULT_STRATEGY
        
        # Keep supervisor_prompt for backward compat (custom prompts)
        self._custom_supervisor_prompt = supervisor_prompt
        
        self.persistence = None  # Set via set_persistence()
        self._heartbeat_task = None  # Background heartbeat for zombie detection
        self._heartbeat_interval = 30  # seconds between heartbeat pulses
        
        # Graceful shutdown support
        self._shutdown_requested = False
        self._setup_signal_handlers()
        
        # Check if LLM supports tool calling
        self._supports_tool_calling = isinstance(llm, ToolCallingLLMClient)
        
        # Cache tool definitions if tool calling is available
        self._tool_definitions: List[ToolDefinition] = []
        if self._supports_tool_calling and self.use_tool_calling:
            try:
                self._tool_definitions = self._build_tool_definitions(workers)
                self._tool_definitions.extend([DONE_TOOL, FAIL_TOOL])
                
                # Validate tool definitions if strict mode
                if self.strict_tools:
                    self._validate_tool_definitions()
            except Exception as e:
                if self.strict_tools:
                    raise ValueError(f"Tool definition error (strict_tools=True): {e}") from e
                else:
                    logger.warning(f"Tool definition error, tool calling disabled: {e}")
                    self._supports_tool_calling = False
        
        # Initialize middleware stack
        self.middleware = MiddlewareStack()
        if middleware:
            for mw in middleware:
                self.middleware.add(mw)
        
        # Configure logging based on verbose flag
        if self.verbose and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        import signal
        import sys
        
        self._shutdown_count = 0  # Track number of shutdown signals
        
        # Only set up signal handlers on main thread and non-Windows for SIGTERM
        try:
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        except ValueError:
            # Signal only works in main thread
            pass
    
    def _handle_shutdown_signal(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        import sys
        
        self._shutdown_count = getattr(self, '_shutdown_count', 0) + 1
        self._shutdown_requested = True
        
        if self._shutdown_count == 1:
            logger.warning(f"Shutdown signal received. Press Ctrl+C again to force quit.")
        else:
            logger.warning(f"Force quit requested. Exiting immediately.")
            sys.exit(1)
    
    def set_persistence(self, persistence: Any) -> None:
        """
        Set persistence layer for step-by-step state saving.
        
        When set, the orchestrator will save state to the persistence layer
        after each step (in addition to any JSON auto_save_path).
        
        The state.metadata["session_id"] is used as the session identifier.
        If not set, a default ID will be used.
        
        Args:
            persistence: A persistence layer (e.g., SQLitePersistence)
            
        Example:
            from blackboard.persistence import SQLitePersistence
            
            persistence = SQLitePersistence("./sessions.db")
            await persistence.initialize()
            
            orchestrator.set_persistence(persistence)
            result = await orchestrator.run(goal="...", state=state)
            # State is saved after each step
        """
        self.persistence = persistence
    
    async def _start_heartbeat(self, session_id: str) -> None:
        """
        Start background heartbeat pulse for zombie detection.
        
        Updates the heartbeat_at timestamp in persistence every N seconds
        while the orchestrator is running. This allows external systems to
        detect "zombie" sessions that appear running but have crashed.
        """
        if not self.persistence or not hasattr(self.persistence, 'update_heartbeat'):
            return  # Persistence doesn't support heartbeats
        
        async def pulse():
            while True:
                try:
                    await asyncio.sleep(self._heartbeat_interval)
                    await self.persistence.update_heartbeat(session_id)
                    logger.debug(f"Heartbeat sent for session {session_id}")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")
        
        self._heartbeat_task = asyncio.create_task(pulse())
    
    async def _stop_heartbeat(self) -> None:
        """Stop the heartbeat background task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
    
    @classmethod
    async def recover_session(
        cls,
        persistence,
        session_id: str,
        llm,
        workers: list
    ) -> "Blackboard":
        """
        Recover a zombie session from persistence.
        
        Loads the session state, marks it with recovery_mode=True, and
        adds feedback indicating the previous step may have partially executed.
        
        Args:
            persistence: Persistence layer with the crashed session
            session_id: ID of the zombie session to recover
            llm: LLM client for the new orchestrator
            workers: Workers for the new orchestrator
            
        Returns:
            The recovered Blackboard state (in PAUSED status)
            
        Example:
            # Detect zombies
            zombies = await persistence.find_zombie_sessions()
            
            for session_id in zombies:
                state = await Orchestrator.recover_session(
                    persistence, session_id, llm, workers
                )
                print(f"Recovered {session_id} at step {state.step_count}")
        
        Warning:
            The previous step may have executed side effects (API calls, emails).
            Check state.metadata['recovery_mode'] before retrying workers.
        """
        from .state import Blackboard, Feedback, Status
        
        # Load the crashed session
        state = await persistence.load(session_id)
        
        # Mark as recovered
        state.update_status(Status.PAUSED)
        state.metadata["recovery_mode"] = True
        state.metadata["recovered_from_status"] = state.status.value
        
        # Add recovery feedback
        state.add_feedback(Feedback(
            source="System",
            critique="Session recovered from crash. Previous step may have partially executed. Check recovery_mode flag before retrying any workers.",
            passed=False
        ))
        
        # Save the recovered state
        await persistence.save(state, session_id)
        
        logger.warning(f"Session {session_id} recovered from crash at step {state.step_count}")
        
        return state

    async def fork_session(
        self,
        session_id: str,
        step_index: int,
        fork_suffix: Optional[str] = None
    ) -> str:
        """
        Fork a session at a specific step index for time-travel debugging.
        
        Creates a new session by loading the checkpoint at the given step,
        generating a new session ID, and saving it as a fresh session.
        This enables "what if" experiments without polluting the original session.
        
        Args:
            session_id: ID of the session to fork
            step_index: Step number to fork from (must have a checkpoint)
            fork_suffix: Optional suffix for fork ID (defaults to timestamp)
            
        Returns:
            The new forked session ID
            
        Raises:
            ValueError: If persistence layer doesn't support checkpoints
            PersistenceError: If no checkpoint exists at that step
            
        Example:
            # Original session failed at step 10
            # Fork from step 9 to try a different approach
            fork_id = await orchestrator.fork_session("session-001", 9)
            
            # Load forked state and continue
            forked_state = await persistence.load(fork_id)
            result = await orchestrator.run(state=forked_state)
        """
        if not self.persistence:
            raise ValueError("Persistence layer required for session forking")
        
        if not hasattr(self.persistence, 'load_state_at_step'):
            raise ValueError("Persistence layer doesn't support checkpoints (CheckpointCapable required)")
        
        # Load checkpoint at the specified step
        from datetime import datetime
        forked_state = await self.persistence.load_state_at_step(session_id, step_index)
        
        # Generate fork session ID
        if fork_suffix is None:
            fork_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        fork_session_id = f"{session_id}_fork_{fork_suffix}"
        
        # Update metadata to track lineage
        forked_state.metadata["session_id"] = fork_session_id
        forked_state.metadata["forked_from"] = session_id
        forked_state.metadata["forked_at_step"] = step_index
        forked_state.metadata["fork_timestamp"] = datetime.now().isoformat()
        
        # Reset version for new session
        forked_state.version = 0
        
        # Save as new session
        await self.persistence.save(forked_state, fork_session_id, parent_session_id=session_id)
        
        logger.info(f"Forked session {session_id} at step {step_index} -> {fork_session_id}")
        
        return fork_session_id

    def _build_tool_definitions(self, workers: List[Worker]) -> List[ToolDefinition]:
        """
        Build tool definitions from workers.
        
        This checks if workers provide their own get_tool_definitions() method
        (like MCPToolWorker) and uses those. This enables dynamic tool expansion
        where each MCP tool becomes a separate LLM tool with proper schema.
        
        For workers without get_tool_definitions(), falls back to auto-generating
        a ToolDefinition from the worker's input_schema.
        """
        definitions = []
        
        for worker in workers:
            # Check if worker provides custom tool definitions (dynamic tools)
            if hasattr(worker, 'get_tool_definitions') and callable(worker.get_tool_definitions):
                # Worker provides its own tool definitions
                worker_defs = worker.get_tool_definitions()
                if worker_defs:
                    definitions.extend(worker_defs)
                    continue
            
            # Fall back to auto-generating from worker
            definitions.append(worker_to_tool_definition(worker))
        
        return definitions

    async def run(
        self,
        goal: Optional[str] = None,
        state: Optional[Blackboard] = None,
        max_steps: int = 20,
        blueprint: Optional["Blueprint"] = None
    ) -> Blackboard:
        """
        Execute the main orchestration loop (async).
        
        Args:
            goal: The objective to accomplish (required if state is None)
            state: Existing state to resume from (optional)
            max_steps: Maximum number of steps before stopping
            blueprint: Optional workflow blueprint to constrain execution
            
        Returns:
            The final blackboard state
            
        Raises:
            ValueError: If neither goal nor state is provided, or max_steps < 1
        """
        # Validate max_steps
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        
        # Initialize or resume state
        if state is not None:
            # Resume from existing state
            logger.info(f"Resuming from step {state.step_count}")
            await self._publish_event(EventType.STATE_LOADED, {"step_count": state.step_count})
        elif goal is not None:
            state = Blackboard(goal=goal, status=Status.PLANNING)
        else:
            raise ValueError("Either 'goal' or 'state' must be provided")
        
        # Publish start event
        await self._publish_event(EventType.ORCHESTRATOR_STARTED, {
            "goal": state.goal,
            "max_steps": max_steps
        })
        
        logger.info(f"Goal: {state.goal}")
        logger.info(f"Workers: {list(self.registry.list_workers().keys())}")
        logger.debug("-" * 50)
        
        # Start heartbeat pulse for zombie detection (if persistence supports it)
        session_id = state.metadata.get("session_id", "default")
        await self._start_heartbeat(session_id)
        
        for step in range(max_steps):
            state.increment_step()
            
            await self._publish_event(EventType.STEP_STARTED, {"step": state.step_count})
            logger.debug(f"Step {state.step_count}")
            
            # Auto-summarize if enabled and thresholds exceeded
            if self.auto_summarize and state.should_summarize(
                artifact_threshold=self.summarize_thresholds.get("artifacts", 10),
                feedback_threshold=self.summarize_thresholds.get("feedback", 20),
                step_threshold=self.summarize_thresholds.get("steps", 50)
            ):
                await self._auto_summarize(state)
            
            # Create step context for middleware
            step_ctx = StepContext(step_number=state.step_count, state=state)
            
            # Before step middleware hook
            await self.middleware.before_step(step_ctx)
            if step_ctx.skip_step:
                logger.debug("Step skipped by middleware")
                continue
            
            # 1. OBSERVE: Build context
            context = state.to_context_string()
            
            # 2. REASON: Ask LLM for next action
            decision = await self._get_supervisor_decision(context, state, blueprint)
            step_ctx.decision = decision
            
            logger.debug(f"Decision: {decision.action} -> {[c.worker_name for c in decision.calls]}")
            logger.debug(f"Reasoning: {decision.reasoning}")
            
            # Call step callback if provided
            if self.on_step:
                self.on_step(step, state, decision)
            
            # 3. CHECK: Handle terminal actions
            if decision.action == "done":
                state.update_status(Status.DONE)
                logger.info("Goal achieved!")
                await self.middleware.after_step(step_ctx)
                await self._publish_event(EventType.STEP_COMPLETED, {
                    "step": state.step_count,
                    "action": "done"
                })
                break
            
            if decision.action == "fail":
                state.update_status(Status.FAILED)
                logger.warning("Goal failed")
                await self.middleware.after_step(step_ctx)
                await self._publish_event(EventType.STEP_COMPLETED, {
                    "step": state.step_count,
                    "action": "fail"
                })
                break
            
            # 4. ACT: Execute worker(s)
            if decision.action == "call" and decision.calls:
                # Single worker call
                await self._execute_worker(state, decision.calls[0])
                # Advance blueprint step if needed
                if blueprint:
                    blueprint.increment_step_iteration(state)
                    blueprint.advance_step(state)
            elif decision.action == "call_independent" and decision.calls:
                # Independent parallel worker calls (stale read warning: each sees state at T0)
                await self._execute_workers_parallel(state, decision.calls)
            
            # After step middleware hook
            await self.middleware.after_step(step_ctx)
            
            # Check if middleware skipped further execution
            if step_ctx.skip_step:
                break
            
            # Check if a worker requested pause (e.g., HumanProxyWorker)
            if state.status == Status.PAUSED:
                logger.info("Execution paused (awaiting input)")
                await self._publish_event(EventType.ORCHESTRATOR_PAUSED, {
                    "step": state.step_count,
                    "pending_input": state.pending_input
                })
                break
            
            await self._publish_event(EventType.STEP_COMPLETED, {
                "step": state.step_count,
                "action": decision.action,
                "workers_called": len(decision.calls)
            })
            
            # Auto-save if configured
            if self.auto_save_path:
                state.save_to_json(self.auto_save_path)
                await self._publish_event(EventType.STATE_SAVED, {"path": self.auto_save_path})
            
            # Save to persistence layer if configured (for step-by-step trace recovery)
            if self.persistence:
                session_id = state.metadata.get("session_id", "default")
                parent_session_id = state.metadata.get("parent_session_id")
                try:
                    await self.persistence.save(state, session_id, parent_session_id=parent_session_id)
                    
                    # Save checkpoint for time-travel debugging if persistence supports it
                    if hasattr(self.persistence, 'save_checkpoint'):
                        await self.persistence.save_checkpoint(session_id, state.step_count, state)
                except Exception as e:
                    logger.warning(f"Persistence save failed: {e}")
            
            # Check for graceful shutdown request
            if self._shutdown_requested:
                logger.info("Graceful shutdown: saving state and exiting...")
                if self.auto_save_path:
                    state.save_to_json(self.auto_save_path)
                state.update_status(Status.PAUSED)
                state.metadata["shutdown_reason"] = "graceful_shutdown"
                await self._publish_event(EventType.ORCHESTRATOR_COMPLETED, {
                    "status": "paused",
                    "reason": "graceful_shutdown",
                    "step_count": state.step_count
                })
                break
        else:
            # Max steps reached without completion
            if state.status not in (Status.DONE, Status.FAILED):
                state.update_status(Status.FAILED)
                logger.warning(f"Max steps ({max_steps}) reached")
        
        # Publish completion event
        await self._publish_event(EventType.ORCHESTRATOR_COMPLETED, {
            "status": state.status.value,
            "step_count": state.step_count,
            "artifacts_count": len(state.artifacts)
        })
        
        # Stop heartbeat when run ends
        await self._stop_heartbeat()
        
        return state

    async def _execute_workers_parallel(
        self,
        state: Blackboard,
        calls: List[WorkerCall]
    ) -> List[Optional[WorkerOutput]]:
        """Execute multiple workers in parallel using asyncio.gather."""
        if not self.enable_parallel:
            # Fall back to sequential execution
            results = []
            for call in calls:
                await self._execute_worker(state, call)
                results.append(None)  # Results already applied to state
            return results
        
        # Filter to only parallel-safe workers
        safe_calls = []
        for call in calls:
            worker = self.registry.get(call.worker_name)
            if worker and worker.parallel_safe:
                safe_calls.append(call)
            elif worker:
                logger.warning(f"Worker '{call.worker_name}' is not parallel-safe, executing sequentially")
                await self._execute_worker(state, call)
        
        if not safe_calls:
            return []
        
        logger.debug(f"Executing {len(safe_calls)} workers in parallel")
        
        # Create tasks for parallel execution
        tasks = [
            self._execute_worker_get_output(state, call)
            for call in safe_calls
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Apply results to state (sequentially to maintain consistency)
        for call, result in zip(safe_calls, results):
            if isinstance(result, Exception):
                # Let failures fail - no heuristic retry
                logger.error(f"Parallel worker '{call.worker_name}' failed: {result}")
                state.add_feedback(Feedback(
                    source="Orchestrator",
                    critique=f"Worker '{call.worker_name}' failed: {str(result)}",
                    passed=False
                ))
            elif result is not None:
                worker = self.registry.get(call.worker_name)
                if worker:
                    self._apply_worker_output(state, result, worker.name)
        
        return results

    async def _execute_worker_get_output(
        self,
        state: Blackboard,
        call: WorkerCall
    ) -> Optional[WorkerOutput]:
        """Execute a worker and return output (for parallel execution)."""
        worker = self.registry.get(call.worker_name)
        
        if worker is None:
            logger.warning(f"Worker '{call.worker_name}' not found")
            return None
        
        await self._publish_event(EventType.WORKER_CALLED, {
            "worker": worker.name,
            "instructions": call.instructions,
            "parallel": True
        })
        
        # Parse inputs if worker has a schema
        inputs = None
        if call.inputs:
            inputs = worker.parse_inputs(call.inputs)
        elif call.instructions:
            inputs = WorkerInput(instructions=call.instructions)
        
        # Define the worker execution function
        async def execute():
            return await worker.run(state, inputs)
        
        try:
            output = await retry_with_backoff(execute, policy=self.retry_policy)
            
            # Build event data with optional trace_id
            completed_event_data = {
                "worker": worker.name,
                "has_artifact": output.has_artifact(),
                "has_feedback": output.has_feedback(),
                "parallel": True
            }
            
            # Add trace_id if this was a sub-agent (fractal agent support)
            if output.has_trace():
                completed_event_data["trace_id"] = output.trace_id
                completed_event_data["is_sub_agent"] = True
            
            await self._publish_event(EventType.WORKER_COMPLETED, completed_event_data)
            
            return output
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
            await self._publish_event(EventType.WORKER_ERROR, {
                "worker": worker.name,
                "error": str(e),
                "parallel": True
            })
            raise

    async def _execute_worker(self, state: Blackboard, call: WorkerCall) -> None:
        """Execute a single worker with retry logic and middleware support."""
        from .middleware import WorkerContext
        
        worker = self.registry.get(call.worker_name)
        
        if worker is None:
            logger.warning(f"Worker '{call.worker_name}' not found")
            return
        
        # Create worker context for middleware
        worker_ctx = WorkerContext(worker=worker, call=call, state=state)
        
        # Run before_worker middleware hooks
        await self.middleware.before_worker(worker_ctx)
        
        # Check if middleware requested skip
        if worker_ctx.skip_worker:
            logger.debug(f"Worker '{call.worker_name}' skipped by middleware")
            if worker_ctx.error:
                # Middleware set an error (e.g., circuit breaker open)
                await self.middleware.on_error(worker_ctx)
                state.add_feedback(Feedback(
                    source="Middleware",
                    critique=f"Worker '{call.worker_name}' skipped: {str(worker_ctx.error)}",
                    passed=False
                ))
            return
        
        # Update status based on worker type (heuristic)
        if "critic" in worker.name.lower() or "review" in worker.name.lower():
            state.update_status(Status.CRITIQUING)
        elif "refine" in worker.name.lower() or "fix" in worker.name.lower():
            state.update_status(Status.REFINING)
        else:
            state.update_status(Status.GENERATING)
        

        await self._publish_event(EventType.WORKER_CALLED, {
            "worker": worker.name,
            "instructions": call.instructions
        })
        
        # Parse inputs if worker has a schema
        inputs = None
        if call.inputs:
            inputs = worker.parse_inputs(call.inputs)
        elif call.instructions:
            inputs = WorkerInput(instructions=call.instructions)
        
        # Define the worker execution function
        async def execute():
            return await worker.run(state, inputs)
        
        # Retry callback for observability
        def on_retry(attempt: int, exception: Exception, delay: float):
            logger.warning(f"Retrying {worker.name} (attempt {attempt + 2})")
            self.event_bus.publish(Event(EventType.WORKER_RETRY, {
                "worker": worker.name,
                "attempt": attempt + 1,
                "error": str(exception),
                "delay": delay
            }))
        
        try:
            output = await retry_with_backoff(
                execute,
                policy=self.retry_policy,
                on_retry=on_retry
            )
            
            # Store output in context for middleware
            worker_ctx.modified_output = output
            
            # Run after_worker middleware hooks
            await self.middleware.after_worker(worker_ctx)
            
            # Use modified output if middleware changed it
            final_output = worker_ctx.modified_output or output
            
            self._apply_worker_output(state, final_output, worker.name)
            
            # Build event data with optional trace_id
            completed_event_data = {
                "worker": worker.name,
                "has_artifact": final_output.has_artifact(),
                "has_feedback": final_output.has_feedback()
            }
            
            # Add trace_id if this was a sub-agent (fractal agent support)
            if final_output.has_trace():
                completed_event_data["trace_id"] = final_output.trace_id
                completed_event_data["is_sub_agent"] = True
            
            await self._publish_event(EventType.WORKER_COMPLETED, completed_event_data)
            
            if final_output.has_artifact():
                logger.debug(f"Artifact: {final_output.artifact.type}")
                await self._publish_event(EventType.ARTIFACT_CREATED, {
                    "id": final_output.artifact.id,
                    "type": final_output.artifact.type,
                    "creator": worker.name
                })
            if final_output.has_feedback():
                logger.debug(f"Feedback: passed={final_output.feedback.passed}")
                await self._publish_event(EventType.FEEDBACK_ADDED, {
                    "id": final_output.feedback.id,
                    "passed": final_output.feedback.passed,
                    "source": worker.name
                })
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
            worker_ctx.error = e
            
            # Run on_error middleware hooks
            await self.middleware.on_error(worker_ctx)
            
            await self._publish_event(EventType.WORKER_ERROR, {
                "worker": worker.name,
                "error": str(e),
                "transient": is_transient_error(e)
            })
            state.add_feedback(Feedback(
                source="Orchestrator",
                critique=f"Worker '{call.worker_name}' failed: {str(e)}",
                passed=False
            ))

    def run_sync(
        self,
        goal: Optional[str] = None,
        state: Optional[Blackboard] = None,
        max_steps: int = 20
    ) -> Blackboard:
        """Synchronous wrapper for run(). Safe to call inside existing event loops."""
        return _run_sync(self.run(goal=goal, state=state, max_steps=max_steps))

    def set_persistence(self, persistence) -> None:
        """
        Set the persistence layer for save/resume.
        
        Args:
            persistence: PersistenceLayer implementation
            
        Example:
            from blackboard.persistence import RedisPersistence
            orchestrator.set_persistence(RedisPersistence("redis://localhost"))
        """
        self.persistence = persistence

    async def pause(
        self, 
        state: Blackboard, 
        session_id: str,
        reason: str = "Awaiting input"
    ) -> None:
        """
        Pause execution and save state for later resume.
        
        Args:
            state: Current blackboard state
            session_id: Session ID for persistence
            reason: Reason for pause (stored in metadata)
            
        Example:
            await orchestrator.pause(state, "session-123", reason="Needs approval")
        """
        if not self.persistence:
            raise RuntimeError("No persistence layer configured. Use set_persistence() first.")
        
        state.update_status(Status.PAUSED)
        state.metadata["pause_reason"] = reason
        
        await self.persistence.save(state, session_id)
        
        await self._publish_event(EventType.ORCHESTRATOR_PAUSED, {
            "session_id": session_id,
            "reason": reason,
            "step": state.step_count
        })

    async def resume(
        self,
        session_id: str,
        user_input: Optional[Dict[str, Any]] = None,
        max_steps: int = 20
    ) -> Blackboard:
        """
        Resume a paused session.
        
        Args:
            session_id: Session ID to resume
            user_input: Optional user-provided input to inject
            max_steps: Maximum steps to run
            
        Returns:
            Final blackboard state
            
        Example:
            result = await orchestrator.resume(
                "session-123",
                user_input={"approved": True}
            )
        """
        if not self.persistence:
            raise RuntimeError("No persistence layer configured. Use set_persistence() first.")
        
        # Load state
        state = await self.persistence.load(session_id)
        
        await self._publish_event(EventType.ORCHESTRATOR_RESUMED, {
            "session_id": session_id,
            "step": state.step_count
        })
        
        # Inject user input if provided
        if user_input:
            state.pending_input = user_input
            state.metadata["user_input_received"] = True
        
        # Clear pause status
        if state.status == Status.PAUSED:
            state.update_status(Status.GENERATING)
        
        # Continue execution
        return await self.run(state=state, max_steps=max_steps)


    async def _auto_summarize(self, state: Blackboard) -> None:
        """Automatically summarize context when thresholds are exceeded."""
        logger.info("Auto-summarizing context...")
        
        # Build history text for summarization
        history_parts = []
        
        if state.context_summary:
            history_parts.append(f"Previous Summary:\n{state.context_summary}")
        
        history_parts.append(f"\nSteps completed: {state.step_count}")
        
        # Include artifacts beyond keep threshold
        keep_artifacts = 3
        if len(state.artifacts) > keep_artifacts:
            old_artifacts = state.artifacts[:-keep_artifacts]
            history_parts.append("\nArtifacts to summarize:")
            for a in old_artifacts:
                preview = str(a.content)[:200]
                history_parts.append(f"- {a.type} by {a.creator}: {preview}")
        
        # Include feedback beyond keep threshold
        keep_feedback = 5
        if len(state.feedback) > keep_feedback:
            old_feedback = state.feedback[:-keep_feedback]
            history_parts.append("\nFeedback to summarize:")
            for f in old_feedback:
                status = "PASSED" if f.passed else "FAILED"
                history_parts.append(f"- [{status}] {f.source}: {f.critique[:100]}")
        
        history_text = "\n".join(history_parts)
        
        # Generate summary using LLM
        summarize_prompt = f'''Summarize the following session history into a concise summary.
Focus on key decisions, artifacts created, and feedback received.

Goal: {state.goal}

History:
{history_text}

Provide a 1-2 paragraph summary:'''
        
        try:
            result = self.llm.generate(summarize_prompt)
            if asyncio.iscoroutine(result):
                response = await result
            else:
                response = result
            
            # Handle LLMResponse or string
            if isinstance(response, LLMResponse):
                summary = response.content
            else:
                summary = response
            
            # Update state
            state.update_summary(summary)
            
            # Compact: keep only recent items
            if len(state.artifacts) > keep_artifacts:
                state.artifacts = state.artifacts[-keep_artifacts:]
            if len(state.feedback) > keep_feedback:
                state.feedback = state.feedback[-keep_feedback:]
            
            state.compact_history(keep_last=10)
            
            logger.info("Context summarized and compacted")
            await self._publish_event(EventType.STEP_COMPLETED, {
                "action": "auto_summarize",
                "summary_length": len(summary)
            })
            
        except Exception as e:
            logger.warning(f"Auto-summarization failed: {e}")


    async def _get_supervisor_decision(
        self, 
        context: str, 
        state: Blackboard,
        blueprint: Optional["Blueprint"] = None
    ) -> SupervisorDecision:
        """Ask the LLM supervisor what to do next."""
        
        # Use tool calling if available
        if self._supports_tool_calling and self.use_tool_calling and self._tool_definitions:
            return await self._get_decision_via_tools(context, state, blueprint)
        
        # Fallback to JSON-based approach
        return await self._get_decision_via_json(context, state, blueprint)
    
    async def _get_decision_via_tools(
        self, 
        context: str, 
        state: Blackboard,
        blueprint: Optional["Blueprint"] = None
    ) -> SupervisorDecision:
        """Get decision using native tool calling."""
        # Build prompt with optional blueprint context
        blueprint_context = blueprint.get_prompt_context(state) if blueprint else ""
        
        simple_prompt = f"""You are a supervisor coordinating workers to achieve the goal.

## Current State
{context}
{blueprint_context}
Choose the best action: call a worker tool, mark_done if complete, or mark_failed if impossible."""
        
        try:
            result = self.llm.generate_with_tools(simple_prompt, self._tool_definitions)
            if asyncio.iscoroutine(result):
                response = await result
            else:
                response = result
            
            # Handle tool calls
            if isinstance(response, list) and response:
                # LLM returned tool calls
                calls = []
                for tool_call in response:
                    if isinstance(tool_call, ToolCall):
                        if tool_call.name == "mark_done":
                            return SupervisorDecision(
                                action="done",
                                reasoning=tool_call.arguments.get("reason", "Goal achieved")
                            )
                        elif tool_call.name == "mark_failed":
                            return SupervisorDecision(
                                action="fail",
                                reasoning=tool_call.arguments.get("reason", "Cannot complete")
                            )
                        else:
                            # Worker call
                            calls.append(WorkerCall(
                                worker_name=tool_call.name,
                                instructions=tool_call.arguments.get("instructions", ""),
                                inputs=tool_call.arguments
                            ))
                
                if calls:
                    action = "call_independent" if len(calls) > 1 else "call"
                    return SupervisorDecision(action=action, calls=calls, reasoning="Via tool calling")
            
            # String response - use strategy to parse
            if isinstance(response, str):
                decision = self.strategy.parse_response(response)
                # Convert Strategy Decision -> SupervisorDecision
                calls = []
                for call in decision.calls:
                    calls.append(WorkerCall(
                        worker_name=call["worker_name"],
                        instructions=call.get("instructions", ""),
                        inputs=call.get("inputs", {})
                    ))
                return SupervisorDecision(
                    action=decision.action,
                    calls=calls,
                    reasoning=decision.reasoning
                )
            
            return SupervisorDecision(action="fail", reasoning="Unexpected response format")
            
        except Exception as e:
            if self.allow_json_fallback:
                logger.warning(f"Tool calling failed, falling back to JSON: {e}")
                return await self._get_decision_via_json(context, state)
            else:
                raise RuntimeError(
                    f"Tool calling failed and allow_json_fallback=False: {e}"
                ) from e
    
    async def _get_decision_via_json(
        self, 
        context: str, 
        state: Blackboard,
        blueprint: Optional["Blueprint"] = None
    ) -> SupervisorDecision:
        """Get decision using the configured reasoning strategy."""
        # Build worker list for the strategy
        worker_info = self.registry.list_workers_with_schemas()
        
        # Filter workers if blueprint is active
        if blueprint:
            allowed = blueprint.filter_workers(list(worker_info.keys()), state)
            worker_info = {k: v for k, v in worker_info.items() if k in allowed}
        
        # Convert to dict format: {name: description}
        workers_dict = {}
        for name, info in worker_info.items():
            desc = info['description']
            if info.get('input_schema'):
                desc += " (accepts structured input)"
            workers_dict[name] = desc
        
        # Build prompt using strategy
        full_prompt = self.strategy.build_prompt(
            context=context,
            workers=workers_dict
        )
        
        # Append blueprint context if active
        if blueprint:
            full_prompt += blueprint.get_prompt_context(state)
        
        try:
            result = self.llm.generate(full_prompt)
            if asyncio.iscoroutine(result):
                response = await result
            else:
                response = result
            
            # Handle LLMResponse or plain string
            content: str
            if isinstance(response, LLMResponse):
                content = response.content
                # Track usage
                if response.usage and self.usage_tracker:
                    self.usage_tracker.record(response.usage, context="supervisor")
                # Store in state metadata
                if response.usage:
                    state.metadata["last_usage"] = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "model": response.usage.model
                    }
            else:
                content = response
            
            # Parse response using strategy
            decision = self.strategy.parse_response(content)
            
            # Convert Strategy Decision -> SupervisorDecision
            calls = []
            for call in decision.calls:
                calls.append(WorkerCall(
                    worker_name=call["worker_name"],
                    instructions=call.get("instructions", ""),
                    inputs=call.get("inputs", {})
                ))
            
            return SupervisorDecision(
                action=decision.action,
                calls=calls,
                reasoning=decision.reasoning
            )
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return SupervisorDecision(
                action="fail",
                reasoning=f"LLM error: {str(e)}"
            )

    def _apply_worker_output(self, state: Blackboard, output: WorkerOutput, worker_name: str) -> None:
        """Apply the worker's output to the blackboard."""
        if output.has_artifact():
            if not output.artifact.creator:
                output.artifact.creator = worker_name
            state.add_artifact(output.artifact)
        
        if output.has_feedback():
            if not output.feedback.artifact_id and state.artifacts:
                output.feedback.artifact_id = state.artifacts[-1].id
            if not output.feedback.source:
                output.feedback.source = worker_name
            state.add_feedback(output.feedback)
        
        if output.has_status_update():
            state.update_status(output.status_update)

    async def _publish_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Publish an event to the event bus."""
        event = Event(type=event_type, data=data)
        await self.event_bus.publish_async(event)

    def _validate_tool_definitions(self) -> None:
        """
        Validate all tool definitions have valid schemas.
        
        Raises:
            ValueError: If any tool definition is invalid
        """
        # LLM providers require alphanumeric, underscore, hyphen only
        import re
        VALID_TOOL_NAME = re.compile(r'^[a-zA-Z0-9_-]+$')
        
        for tool in self._tool_definitions:
            # Validate name is not empty
            if not tool.name or not tool.name.strip():
                raise ValueError(f"Tool has empty name: {tool}")
            
            # Validate name format (LLM requirement)
            if not VALID_TOOL_NAME.match(tool.name):
                raise ValueError(
                    f"Tool name '{tool.name}' contains invalid characters. "
                    "Use only alphanumeric, underscore, and hyphen."
                )
            
            # Validate each parameter
            for param in tool.parameters:
                if not param.name or not param.name.strip():
                    raise ValueError(f"Tool '{tool.name}' has parameter with empty name")
                if not param.type:
                    raise ValueError(f"Tool '{tool.name}' parameter '{param.name}' has no type")
            
            # Try to generate OpenAI format to catch schema issues
            try:
                tool.to_openai_format()
            except Exception as e:
                raise ValueError(f"Tool '{tool.name}' has invalid schema: {e}")


# Convenience functions
async def run_blackboard(
    goal: str,
    llm: LLMClient,
    workers: List[Worker],
    max_steps: int = 20,
    verbose: bool = False,
    event_bus: Optional[EventBus] = None,
    enable_parallel: bool = True
) -> Blackboard:
    """Convenience function to run the blackboard system (async)."""
    orchestrator = Orchestrator(
        llm=llm,
        workers=workers,
        verbose=verbose,
        event_bus=event_bus,
        enable_parallel=enable_parallel
    )
    return await orchestrator.run(goal=goal, max_steps=max_steps)


def run_blackboard_sync(
    goal: str,
    llm: LLMClient,
    workers: List[Worker],
    max_steps: int = 20,
    verbose: bool = False
) -> Blackboard:
    """Convenience function to run the blackboard system (sync). Safe inside existing event loops."""
    return _run_sync(run_blackboard(goal, llm, workers, max_steps, verbose))


# =============================================================================
# Agent: Fractal Agent (Worker that wraps an Orchestrator)
# =============================================================================

class RecursionDepthExceededError(Exception):
    """Raised when agent nesting exceeds max_recursion_depth."""
    pass


class Agent(Worker):
    """
    A Worker that encapsulates an Orchestrator for fractal agent architectures.
    
    Enables "Agent-as-Worker" pattern where complex tasks can be delegated to
    sub-agents, each with their own workers and orchestration loop.
    
    Features:
    - Recursion depth limiting (prevents infinite agent loops)
    - Config propagation (security flags passed to children)
    - Context compression (summarizes execution before returning)
    - Trace linking (trace_id in output for debugging)
    - Shared persistence (optional, for DB connection reuse)
    
    Args:
        name: Agent name (used in tool definitions)
        description: Description for parent orchestrator
        llm: LLM client for internal orchestration
        workers: Workers available to this agent
        config: BlackboardConfig (inherits from parent if not provided)
        current_depth: Current recursion depth (set by parent)
        persistence: Optional shared persistence layer
        
    Example:
        # Create a research sub-agent
        research_agent = Agent(
            name="ResearchAgent",
            description="Performs web research on a topic",
            llm=llm,
            workers=[WebSearchWorker(), BrowserWorker()],
            config=parent_config.for_child_agent()
        )
        
        # Use as a worker in parent orchestrator
        parent_orchestrator = Orchestrator(
            llm=llm,
            workers=[research_agent, WriterWorker()],
            ...
        )
    """
    
    # Worker protocol attributes
    name: str = "Agent"
    description: str = "A sub-agent that can perform complex multi-step tasks"
    input_schema = None  # Uses default WorkerInput
    parallel_safe: bool = False  # Sub-agents are stateful
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLMClient,
        workers: List[Worker],
        config: Optional["BlackboardConfig"] = None,
        current_depth: int = 0,
        persistence: Optional[Any] = None,  # SQLitePersistence
        event_bus: Optional[EventBus] = None,
        parent_event_bus: Optional[EventBus] = None,
        verbose: bool = False
    ):
        from .config import BlackboardConfig
        
        self.name = name
        self.description = description
        self.llm = llm
        self.workers = workers
        self.config = config or BlackboardConfig()
        self.current_depth = current_depth
        self.persistence = persistence
        self.parent_event_bus = parent_event_bus
        self.verbose = verbose
        
        # Create internal event bus
        self._event_bus = event_bus or EventBus()
        
        # Set up event bubbling if parent bus exists
        if self.parent_event_bus:
            self._setup_event_bubbling()
    
    def _setup_event_bubbling(self) -> None:
        """Subscribe to internal events and republish to parent with namespacing."""
        from copy import copy
        
        async def bubble_event(event: Event) -> None:
            # Clone the event to avoid mutation
            bubbled = copy(event)
            
            # Namespace the source
            original_source = bubbled.data.get("source", "System")
            bubbled.data["source"] = f"{self.name}:{original_source}"
            bubbled.data["depth"] = self.current_depth
            bubbled.data["agent_name"] = self.name
            
            # Publish to parent
            await self.parent_event_bus.publish_async(bubbled)
        
        # Subscribe to all event types
        for event_type in EventType:
            self._event_bus.subscribe(event_type, bubble_event)
    
    async def run(
        self,
        state: "Blackboard",
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        """
        Execute the agent's internal orchestration loop.
        
        Args:
            state: Parent's blackboard state (for context)
            inputs: Instructions containing the goal for this agent
            
        Returns:
            WorkerOutput with summarized result and trace_id
        """
        from .config import BlackboardConfig
        from .persistence import SQLitePersistence
        import uuid
        
        # Extract goal from inputs
        goal = inputs.instructions if inputs else "Complete the delegated task"
        
        # Check recursion depth
        if self.current_depth >= self.config.max_recursion_depth:
            raise RecursionDepthExceededError(
                f"Agent '{self.name}' exceeded max recursion depth of "
                f"{self.config.max_recursion_depth}. Current depth: {self.current_depth}"
            )
        
        # Generate session ID for this agent run
        session_id = f"{self.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
        
        # Create child config with decremented depth
        child_config = self.config.for_child_agent()
        
        # Create initial state with session_id in metadata for persistence tracking
        initial_state = Blackboard(goal=goal, status=Status.PLANNING)
        initial_state.metadata["session_id"] = session_id
        initial_state.metadata["agent_name"] = self.name
        initial_state.metadata["depth"] = self.current_depth
        
        # Link to parent session if available
        parent_session_id = state.metadata.get("session_id")
        if parent_session_id:
            initial_state.metadata["parent_session_id"] = parent_session_id
        
        # Create internal orchestrator
        orchestrator = Orchestrator(
            llm=self.llm,
            workers=self.workers,
            config=child_config,
            event_bus=self._event_bus,
            verbose=self.verbose
        )
        
        # Set persistence for step-by-step trace recovery
        if self.persistence:
            orchestrator.set_persistence(self.persistence)
        
        # Run the internal orchestration loop
        try:
            result_state = await orchestrator.run(
                state=initial_state,  # Pass initialized state with session_id
                max_steps=child_config.max_steps
            )
            
            # Note: State is saved after each step by the orchestrator (if persistence is set)
            # No additional save needed here - step-by-step saves enable crash recovery
            
            # Context Compression: Summarize the execution
            summary = await self._compress_context(result_state, goal)
            
            # Create artifact with summary
            artifact = Artifact(
                type="agent_result",
                content=summary,
                creator=self.name,
                metadata={
                    "session_id": session_id,
                    "steps_taken": result_state.step_count,
                    "final_status": result_state.status.value,
                    "artifacts_created": len(result_state.artifacts)
                }
            )
            
            # Return with trace_id for debugging
            return WorkerOutput(
                artifact=artifact,
                status_update=result_state.status if result_state.status == Status.DONE else None,
                trace_id=session_id,
                metadata={
                    "agent_name": self.name,
                    "depth": self.current_depth,
                    "steps": result_state.step_count
                }
            )
            
        except RecursionDepthExceededError:
            raise
        except Exception as e:
            logger.error(f"Agent '{self.name}' failed: {e}")
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    critique=f"Agent execution failed: {str(e)}",
                    passed=False
                ),
                trace_id=session_id,
                metadata={"error": str(e)}
            )
    
    async def _compress_context(self, state: "Blackboard", original_goal: str) -> str:
        """
        Compress the agent's execution history into a summary.
        
        This prevents context window explosion when returning to parent.
        """
        # If only a few steps, just return the last artifact
        if state.step_count <= 3 and state.artifacts:
            last_artifact = state.get_latest_artifact()
            if last_artifact:
                return f"Result: {last_artifact.content}"
        
        # For longer executions, create a summary
        summary_parts = [
            f"Task: {original_goal}",
            f"Status: {state.status.value}",
            f"Steps taken: {state.step_count}",
        ]
        
        # Add artifacts summary
        if state.artifacts:
            summary_parts.append(f"Artifacts created: {len(state.artifacts)}")
            # Include content of latest artifact (truncated)
            latest = state.get_latest_artifact()
            if latest:
                content_preview = str(latest.content)[:500]
                if len(str(latest.content)) > 500:
                    content_preview += "..."
                summary_parts.append(f"Final result:\n{content_preview}")
        
        # Add feedback summary
        passed_feedback = [f for f in state.feedback if f.passed]
        failed_feedback = [f for f in state.feedback if not f.passed]
        if failed_feedback:
            summary_parts.append(f"Issues encountered: {len(failed_feedback)}")
        
        return "\n".join(summary_parts)
    
    def get_schema_json(self) -> Optional[Dict[str, Any]]:
        """Return JSON schema for agent inputs."""
        return {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": f"The goal or task for {self.name} to accomplish"
                }
            },
            "required": ["instructions"]
        }

