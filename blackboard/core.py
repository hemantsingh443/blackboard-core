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
from .events import EventBus, Event, EventType, get_event_bus
from .retry import RetryPolicy, retry_with_backoff, DEFAULT_RETRY_POLICY, is_transient_error
from .usage import LLMResponse, LLMUsage, UsageTracker
from .middleware import Middleware, MiddlewareStack, StepContext, WorkerContext
from .tools import (
    ToolDefinition, ToolCall, ToolCallingLLMClient, 
    worker_to_tool_definition, workers_to_tool_definitions,
    DONE_TOOL, FAIL_TOOL
)


# Configure module logger
logger = logging.getLogger("blackboard")


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
    
    # Backward compatibility
    @property
    def worker_name(self) -> Optional[str]:
        """Get the first worker name (for backward compat)."""
        return self.calls[0].worker_name if self.calls else None
    
    @property
    def instructions(self) -> str:
        """Get the first worker's instructions (for backward compat)."""
        return self.calls[0].instructions if self.calls else ""


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
    
    SUPERVISOR_SYSTEM_PROMPT = '''You are a Supervisor managing a team of AI workers to accomplish a goal.

## Your Role
- You NEVER do the work yourself
- You ONLY decide which worker(s) to call next based on the current state
- You route tasks and provide specific instructions to workers

## Available Workers
{worker_list}

## Response Format
You MUST respond with valid JSON in one of these formats:

### Single Worker Call
```json
{{
    "reasoning": "Brief explanation of why you're making this decision",
    "action": "call",
    "worker": "WorkerName",
    "instructions": "Specific instructions for the worker"
}}
```

### Independent Worker Calls (parallel, NO dependencies)
IMPORTANT: Only use this when tasks do NOT depend on each other's outputs.
Each worker reads state at call time - workers will NOT see other workers' results from this batch.
```json
{{
    "reasoning": "These tasks are fully independent - no worker needs another's output",
    "action": "call_independent",
    "calls": [
        {{"worker": "Worker1", "instructions": "Task 1"}},
        {{"worker": "Worker2", "instructions": "Task 2"}}
    ]
}}
```

### Terminal Actions
```json
{{"action": "done", "reasoning": "Goal achieved"}}
{{"action": "fail", "reasoning": "Cannot complete"}}
```

## Rules
1. If there's no artifact yet, call a Generator/Writer worker
2. If there's an artifact but no feedback, call a Critic/Reviewer worker
3. If feedback says "passed: false", call the Generator again with the critique
4. If feedback says "passed: true", mark as "done"
5. Use "call_independent" ONLY for truly independent tasks (e.g., researching separate topics)
6. Don't call the same worker twice in a row without new information
'''

    def __init__(
        self,
        llm: LLMClient,
        workers: List[Worker],
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
        auto_summarize: bool = False,
        summarize_thresholds: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm: An LLM client with a generate() method (sync or async)
            workers: List of workers to manage
            verbose: If True, enable INFO level logging
            on_step: Optional callback called after each step
            event_bus: Event bus for observability (uses global if not provided)
            retry_policy: Retry policy for worker execution (default: 3 retries)
            auto_save_path: If provided, auto-save state after each step
            enable_parallel: If True, allow parallel worker execution
            middleware: List of middleware to add to the stack
            usage_tracker: Tracker for LLM token usage and costs
            use_tool_calling: If True, use native tool calling when LLM supports it
            allow_json_fallback: If False, raise error when tool calling fails instead of silently falling back
            auto_summarize: If True, automatically summarize context when thresholds exceeded
            summarize_thresholds: Custom thresholds for summarization
        """
        self.llm = llm
        self.registry = WorkerRegistry()
        for worker in workers:
            self.registry.register(worker)
        self.verbose = verbose
        self.on_step = on_step
        self.event_bus = event_bus or get_event_bus()
        self.retry_policy = retry_policy or DEFAULT_RETRY_POLICY
        self.auto_save_path = auto_save_path
        self.enable_parallel = enable_parallel
        self.usage_tracker = usage_tracker
        self.use_tool_calling = use_tool_calling
        self.allow_json_fallback = allow_json_fallback
        self.auto_summarize = auto_summarize
        self.summarize_thresholds = summarize_thresholds or {
            "artifacts": 10,
            "feedback": 20,
            "steps": 50
        }
        
        # Check if LLM supports tool calling
        self._supports_tool_calling = isinstance(llm, ToolCallingLLMClient)
        
        # Cache tool definitions if tool calling is available
        self._tool_definitions: List[ToolDefinition] = []
        if self._supports_tool_calling and use_tool_calling:
            self._tool_definitions = workers_to_tool_definitions(workers)
            self._tool_definitions.extend([DONE_TOOL, FAIL_TOOL])
        
        # Initialize middleware stack
        self.middleware = MiddlewareStack()
        if middleware:
            for mw in middleware:
                self.middleware.add(mw)
        
        # Configure logging based on verbose flag
        if verbose and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    async def run(
        self,
        goal: Optional[str] = None,
        state: Optional[Blackboard] = None,
        max_steps: int = 20
    ) -> Blackboard:
        """
        Execute the main orchestration loop (async).
        
        Args:
            goal: The objective to accomplish (required if state is None)
            state: Existing state to resume from (optional)
            max_steps: Maximum number of steps before stopping
            
        Returns:
            The final blackboard state
            
        Raises:
            ValueError: If neither goal nor state is provided
        """
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
            self.middleware.before_step(step_ctx)
            if step_ctx.skip_step:
                logger.debug("Step skipped by middleware")
                continue
            
            # 1. OBSERVE: Build context
            context = state.to_context_string()
            
            # 2. REASON: Ask LLM for next action
            decision = await self._get_supervisor_decision(context, state)
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
            elif decision.action == "call_independent" and decision.calls:
                # Independent parallel worker calls (stale read warning: each sees state at T0)
                await self._execute_workers_parallel(state, decision.calls)
            
            # After step middleware hook
            await self.middleware.after_step(step_ctx)
            
            # Check if middleware skipped further execution
            if step_ctx.skip_step:
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
                # Check if this might be a dependency failure (stale read issue)
                if self._is_potential_dependency_error(result):
                    logger.warning(
                        f"Possible stale read issue for '{call.worker_name}', "
                        f"retrying sequentially: {result}"
                    )
                    # Retry this worker sequentially with updated state
                    try:
                        await self._execute_worker(state, call)
                        continue  # Success - don't add error feedback
                    except Exception as retry_error:
                        logger.error(f"Sequential retry also failed: {retry_error}")
                        result = retry_error  # Fall through to error handling
                
                logger.error(f"Parallel worker error: {result}")
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
            
            await self._publish_event(EventType.WORKER_COMPLETED, {
                "worker": worker.name,
                "has_artifact": output.has_artifact(),
                "has_feedback": output.has_feedback(),
                "parallel": True
            })
            
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
        """Execute a single worker with retry logic."""
        worker = self.registry.get(call.worker_name)
        
        if worker is None:
            logger.warning(f"Worker '{call.worker_name}' not found")
            return
        
        # Update status based on worker type (heuristic)
        if "critic" in worker.name.lower() or "review" in worker.name.lower():
            state.update_status(Status.CRITIQUING)
        elif "refine" in worker.name.lower() or "fix" in worker.name.lower():
            state.update_status(Status.REFINING)
        else:
            state.update_status(Status.GENERATING)
        
        # Inject instructions into metadata for backward compat
        state.metadata["current_instructions"] = call.instructions
        
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
            self._apply_worker_output(state, output, worker.name)
            
            await self._publish_event(EventType.WORKER_COMPLETED, {
                "worker": worker.name,
                "has_artifact": output.has_artifact(),
                "has_feedback": output.has_feedback()
            })
            
            if output.has_artifact():
                logger.debug(f"Artifact: {output.artifact.type}")
                await self._publish_event(EventType.ARTIFACT_CREATED, {
                    "id": output.artifact.id,
                    "type": output.artifact.type,
                    "creator": worker.name
                })
            if output.has_feedback():
                logger.debug(f"Feedback: passed={output.feedback.passed}")
                await self._publish_event(EventType.FEEDBACK_ADDED, {
                    "id": output.feedback.id,
                    "passed": output.feedback.passed,
                    "source": worker.name
                })
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
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
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(goal=goal, state=state, max_steps=max_steps))

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


    async def _get_supervisor_decision(self, context: str, state: Blackboard) -> SupervisorDecision:
        """Ask the LLM supervisor what to do next."""
        
        # Use tool calling if available
        if self._supports_tool_calling and self.use_tool_calling and self._tool_definitions:
            return await self._get_decision_via_tools(context, state)
        
        # Fallback to JSON-based approach
        return await self._get_decision_via_json(context, state)
    
    async def _get_decision_via_tools(self, context: str, state: Blackboard) -> SupervisorDecision:
        """Get decision using native tool calling."""
        simple_prompt = f"""You are a supervisor coordinating workers to achieve the goal.

## Current State
{context}

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
            
            # String response - fall back to JSON parsing
            if isinstance(response, str):
                return self._parse_llm_response(response)
            
            return SupervisorDecision(action="fail", reasoning="Unexpected response format")
            
        except Exception as e:
            if self.allow_json_fallback:
                logger.warning(f"Tool calling failed, falling back to JSON: {e}")
                return await self._get_decision_via_json(context, state)
            else:
                raise RuntimeError(
                    f"Tool calling failed and allow_json_fallback=False: {e}"
                ) from e
    
    async def _get_decision_via_json(self, context: str, state: Blackboard) -> SupervisorDecision:
        """Get decision using JSON-based prompting."""
        # Build the worker list with schemas
        worker_info = self.registry.list_workers_with_schemas()
        worker_lines = []
        for name, info in worker_info.items():
            line = f"- **{name}**: {info['description']}"
            if info.get('input_schema'):
                line += f" (accepts structured input)"
            worker_lines.append(line)
        worker_list = "\n".join(worker_lines)
        
        system_prompt = self.SUPERVISOR_SYSTEM_PROMPT.format(worker_list=worker_list)
        
        full_prompt = f"{system_prompt}\n\n## Current State\n{context}\n\n## Your Decision (JSON only):"
        
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
            
            return self._parse_llm_response(content)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return SupervisorDecision(
                action="fail",
                reasoning=f"LLM error: {str(e)}"
            )

    def _parse_llm_response(self, response: str) -> SupervisorDecision:
        """Parse the JSON response from the LLM."""
        
        # First, try to extract JSON from markdown code blocks
        def extract_from_code_block(text: str) -> Optional[str]:
            """Extract JSON from ```json ... ``` blocks."""
            patterns = [
                r'```json\s*\n?(.*?)\n?```',  # ```json ... ```
                r'```\s*\n?(.*?)\n?```',       # ``` ... ```
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return None
        
        # Try to find JSON object with matching braces
        def find_json_object(text: str) -> Optional[str]:
            start = text.find('{')
            if start == -1:
                return None
            
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            return None
        
        # Try basic JSON repair for common issues
        def repair_json(text: str) -> str:
            """Attempt to fix common JSON issues."""
            # Remove trailing commas before } or ]
            text = re.sub(r',\s*([\}\]])', r'\1', text)
            # Fix single quotes to double quotes
            text = text.replace("'", '"')
            # Fix unquoted keys
            text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', text)
            return text
        
        # Try extraction methods in order
        json_str = extract_from_code_block(response)
        if not json_str:
            json_str = find_json_object(response)
        
        if json_str:
            # Try parsing, then repair and retry
            for attempt in range(2):
                try:
                    data = json.loads(json_str)
                    action = data.get("action", "fail")
                    reasoning = data.get("reasoning", "")
                    
                    # Parse calls
                    calls = []
                    if action in ("call_parallel", "call_independent") and "calls" in data:
                        # Support both for backward compatibility
                        action = "call_independent"  # Normalize to new name
                        for call_data in data["calls"]:
                            calls.append(WorkerCall(
                                worker_name=call_data.get("worker", ""),
                                instructions=call_data.get("instructions", ""),
                                inputs=call_data.get("inputs", {})
                            ))
                    elif action == "call" and "worker" in data:
                        calls.append(WorkerCall(
                            worker_name=data.get("worker", ""),
                            instructions=data.get("instructions", ""),
                            inputs=data.get("inputs", {})
                        ))
                    
                    return SupervisorDecision(
                        action=action,
                        calls=calls,
                        reasoning=reasoning
                    )
                except json.JSONDecodeError:
                    if attempt == 0:
                        # Try repair on first failure
                        json_str = repair_json(json_str)
                    else:
                        pass  # Give up after repair attempt
                pass
        
        # Fallback parsing for non-JSON responses
        response_lower = response.lower()
        if "done" in response_lower:
            return SupervisorDecision(action="done", reasoning="Parsed 'done' from response")
        elif "fail" in response_lower:
            return SupervisorDecision(action="fail", reasoning="Parsed 'fail' from response")
        
        return SupervisorDecision(
            action="fail",
            reasoning=f"Could not parse LLM response: {response[:100]}"
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

    def _is_potential_dependency_error(self, error: Exception) -> bool:
        """
        Check if an error likely indicates a stale read / dependency issue.
        
        These errors often occur when a parallel worker tries to access
        state that another worker was supposed to create but hasn't yet
        (because they ran in parallel with stale state).
        
        Returns:
            True if the error pattern suggests a dependency issue
        """
        # Common dependency-related exceptions
        dependency_exceptions = (
            FileNotFoundError,
            KeyError,
            AttributeError,
            IndexError,
        )
        
        if isinstance(error, dependency_exceptions):
            return True
        
        # Check error message patterns
        error_str = str(error).lower()
        dependency_patterns = [
            "not found",
            "does not exist",
            "missing",
            "no such file",
            "undefined",
            "null",
            "none",
        ]
        
        return any(pattern in error_str for pattern in dependency_patterns)


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
    """Convenience function to run the blackboard system (sync)."""
    return asyncio.run(run_blackboard(goal, llm, workers, max_steps, verbose))
