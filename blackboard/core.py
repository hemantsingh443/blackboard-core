"""
Orchestrator (Supervisor)

The "Prefrontal Cortex" of the blackboard system.
An LLM-driven supervisor that manages worker execution based on state.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable, Awaitable, Union

from .state import Blackboard, Status, Artifact, Feedback
from .protocols import Worker, WorkerOutput, WorkerRegistry


# Configure module logger
logger = logging.getLogger("blackboard")


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM providers.
    
    Any class with a `generate(prompt: str) -> str` method works.
    Supports both sync and async implementations.
    
    Examples:
        # OpenAI-style client (sync)
        class OpenAIClient:
            def generate(self, prompt: str) -> str:
                response = openai.chat.completions.create(...)
                return response.choices[0].message.content
        
        # Anthropic-style client (async)
        class AnthropicClient:
            async def generate(self, prompt: str) -> str:
                response = await anthropic.messages.create(...)
                return response.content[0].text
    """
    
    def generate(self, prompt: str) -> Union[str, Awaitable[str]]:
        """Generate a response for the given prompt."""
        ...


@dataclass
class SupervisorDecision:
    """
    The parsed decision from the supervisor LLM.
    
    Attributes:
        action: The action to take ("call", "done", "fail")
        worker_name: Name of the worker to call (if action is "call")
        instructions: Specific instructions for the worker
        reasoning: The supervisor's reasoning for this decision
    """
    action: str  # "call", "done", "fail"
    worker_name: Optional[str] = None
    instructions: str = ""
    reasoning: str = ""


class Orchestrator:
    """
    The LLM-driven Supervisor that orchestrates worker execution.
    
    The orchestrator follows the Observe-Reason-Act loop:
    1. Observe: Read the current blackboard state
    2. Reason: Ask the LLM which worker to call next
    3. Act: Execute the worker and update the blackboard
    4. Check: If done or failed, stop; otherwise repeat
    
    Example:
        llm = MyLLMClient()
        workers = [TextWriter(), TextReviewer()]
        orchestrator = Orchestrator(llm=llm, workers=workers)
        
        # Async usage
        result = await orchestrator.run(
            goal="Write a haiku about coding",
            max_steps=10
        )
        
        # Sync convenience
        result = orchestrator.run_sync(goal="...", max_steps=10)
    """
    
    SUPERVISOR_SYSTEM_PROMPT = '''You are a Supervisor managing a team of AI workers to accomplish a goal.

## Your Role
- You NEVER do the work yourself
- You ONLY decide which worker to call next based on the current state
- You route tasks and provide specific instructions to workers

## Available Workers
{worker_list}

## Response Format
You MUST respond with valid JSON in this exact format:
```json
{{
    "reasoning": "Brief explanation of why you're making this decision",
    "action": "call" | "done" | "fail",
    "worker": "WorkerName",
    "instructions": "Specific instructions for the worker"
}}
```

Actions:
- "call": Call a worker with instructions
- "done": The goal has been achieved (an artifact passed review)
- "fail": The goal cannot be achieved (after multiple failures)

## Rules
1. If there's no artifact yet, call a Generator/Writer worker
2. If there's an artifact but no feedback, call a Critic/Reviewer worker
3. If feedback says "passed: false", call the Generator again with the critique
4. If feedback says "passed: true", mark as "done"
5. Don't call the same worker twice in a row without new information
'''

    def __init__(
        self,
        llm: LLMClient,
        workers: List[Worker],
        verbose: bool = False,
        on_step: Optional[Callable[[int, Blackboard, SupervisorDecision], None]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm: An LLM client with a generate() method (sync or async)
            workers: List of workers to manage
            verbose: If True, enable INFO level logging
            on_step: Optional callback called after each step
        """
        self.llm = llm
        self.registry = WorkerRegistry()
        for worker in workers:
            self.registry.register(worker)
        self.verbose = verbose
        self.on_step = on_step
        
        # Configure logging based on verbose flag
        if verbose and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    async def run(self, goal: str, max_steps: int = 20) -> Blackboard:
        """
        Execute the main orchestration loop (async).
        
        Args:
            goal: The objective to accomplish
            max_steps: Maximum number of steps before stopping
            
        Returns:
            The final blackboard state
        """
        # Initialize the blackboard
        state = Blackboard(goal=goal, status=Status.PLANNING)
        
        logger.info(f"Goal: {goal}")
        logger.info(f"Workers: {list(self.registry.list_workers().keys())}")
        logger.debug("-" * 50)
        
        for step in range(max_steps):
            state.increment_step()
            
            logger.debug(f"Step {state.step_count}")
            
            # 1. OBSERVE: Build context
            context = state.to_context_string()
            
            # 2. REASON: Ask LLM for next action
            decision = await self._get_supervisor_decision(context)
            
            logger.debug(f"Decision: {decision.action} -> {decision.worker_name}")
            logger.debug(f"Reasoning: {decision.reasoning}")
            
            # Call step callback if provided
            if self.on_step:
                self.on_step(step, state, decision)
            
            # 3. CHECK: Handle terminal actions
            if decision.action == "done":
                state.update_status(Status.DONE)
                logger.info("Goal achieved!")
                break
            
            if decision.action == "fail":
                state.update_status(Status.FAILED)
                logger.warning("Goal failed")
                break
            
            # 4. ACT: Execute the worker
            if decision.action == "call" and decision.worker_name:
                worker = self.registry.get(decision.worker_name)
                
                if worker is None:
                    logger.warning(f"Worker '{decision.worker_name}' not found")
                    continue
                
                # Update status based on worker type (heuristic)
                if "critic" in worker.name.lower() or "review" in worker.name.lower():
                    state.update_status(Status.CRITIQUING)
                elif "refine" in worker.name.lower() or "fix" in worker.name.lower():
                    state.update_status(Status.REFINING)
                else:
                    state.update_status(Status.GENERATING)
                
                # Inject instructions into metadata for the worker
                state.metadata["current_instructions"] = decision.instructions
                
                try:
                    output = await worker.run(state)
                    self._apply_worker_output(state, output, worker.name)
                    
                    if output.has_artifact():
                        logger.debug(f"Artifact: {output.artifact.type}")
                    if output.has_feedback():
                        logger.debug(f"Feedback: passed={output.feedback.passed}")
                        
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    state.add_feedback(Feedback(
                        source="Orchestrator",
                        critique=f"Worker '{decision.worker_name}' failed: {str(e)}",
                        passed=False
                    ))
        else:
            # Max steps reached without completion
            if state.status not in (Status.DONE, Status.FAILED):
                state.update_status(Status.FAILED)
                logger.warning(f"Max steps ({max_steps}) reached")
        
        return state

    def run_sync(self, goal: str, max_steps: int = 20) -> Blackboard:
        """
        Synchronous wrapper for run().
        
        Convenience method for non-async contexts.
        
        Args:
            goal: The objective to accomplish
            max_steps: Maximum number of steps before stopping
            
        Returns:
            The final blackboard state
        """
        return asyncio.run(self.run(goal=goal, max_steps=max_steps))

    async def _get_supervisor_decision(self, context: str) -> SupervisorDecision:
        """Ask the LLM supervisor what to do next."""
        # Build the worker list for the system prompt
        worker_list = "\n".join([
            f"- **{name}**: {desc}"
            for name, desc in self.registry.list_workers().items()
        ])
        
        system_prompt = self.SUPERVISOR_SYSTEM_PROMPT.format(worker_list=worker_list)
        
        full_prompt = f"{system_prompt}\n\n## Current State\n{context}\n\n## Your Decision (JSON only):"
        
        try:
            # Support both sync and async LLM clients
            result = self.llm.generate(full_prompt)
            if asyncio.iscoroutine(result):
                response = await result
            else:
                response = result
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            # Fallback: try to continue sensibly
            return SupervisorDecision(
                action="fail",
                reasoning=f"LLM error: {str(e)}"
            )

    def _parse_llm_response(self, response: str) -> SupervisorDecision:
        """Parse the JSON response from the LLM."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                return SupervisorDecision(
                    action=data.get("action", "fail"),
                    worker_name=data.get("worker"),
                    instructions=data.get("instructions", ""),
                    reasoning=data.get("reasoning", "")
                )
            except json.JSONDecodeError:
                pass
        
        # Fallback parsing for non-JSON responses
        response_lower = response.lower()
        if "done" in response_lower:
            return SupervisorDecision(action="done", reasoning="Parsed 'done' from response")
        elif "fail" in response_lower:
            return SupervisorDecision(action="fail", reasoning="Parsed 'fail' from response")
        
        # Default to fail if we can't parse
        return SupervisorDecision(
            action="fail",
            reasoning=f"Could not parse LLM response: {response[:100]}"
        )

    def _apply_worker_output(self, state: Blackboard, output: WorkerOutput, worker_name: str) -> None:
        """Apply the worker's output to the blackboard."""
        if output.has_artifact():
            # Ensure creator is set
            if not output.artifact.creator:
                output.artifact.creator = worker_name
            state.add_artifact(output.artifact)
        
        if output.has_feedback():
            # Link to last artifact if not specified
            if not output.feedback.artifact_id and state.artifacts:
                output.feedback.artifact_id = state.artifacts[-1].id
            if not output.feedback.source:
                output.feedback.source = worker_name
            state.add_feedback(output.feedback)
        
        if output.has_status_update():
            state.update_status(output.status_update)


# Convenience function for simple use cases
async def run_blackboard(
    goal: str,
    llm: LLMClient,
    workers: List[Worker],
    max_steps: int = 20,
    verbose: bool = False
) -> Blackboard:
    """
    Convenience function to run the blackboard system (async).
    
    Args:
        goal: The objective to accomplish
        llm: An LLM client
        workers: List of workers
        max_steps: Maximum iterations
        verbose: Enable debug logging
        
    Returns:
        Final blackboard state
    """
    orchestrator = Orchestrator(llm=llm, workers=workers, verbose=verbose)
    return await orchestrator.run(goal=goal, max_steps=max_steps)


def run_blackboard_sync(
    goal: str,
    llm: LLMClient,
    workers: List[Worker],
    max_steps: int = 20,
    verbose: bool = False
) -> Blackboard:
    """
    Convenience function to run the blackboard system (sync).
    
    Args:
        goal: The objective to accomplish
        llm: An LLM client
        workers: List of workers
        max_steps: Maximum iterations
        verbose: Enable debug logging
        
    Returns:
        Final blackboard state
    """
    return asyncio.run(run_blackboard(goal, llm, workers, max_steps, verbose))
