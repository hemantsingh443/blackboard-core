"""
Human Proxy Worker

Pauses execution to ask the human a question and returns their answer.
"""

import logging
from typing import Any, Callable, Dict, Optional, Awaitable

from pydantic import Field

from blackboard.protocols import Worker, WorkerInput, WorkerOutput
from blackboard.state import Blackboard, Artifact, Status

logger = logging.getLogger("blackboard.stdlib.human")


class HumanProxyInput(WorkerInput):
    """Input schema for HumanProxyWorker."""
    question: str = Field(..., description="The question to ask the human")
    context: Optional[str] = Field(default=None, description="Additional context for the question")
    options: Optional[list] = Field(default=None, description="Optional list of choices")
    required: bool = Field(default=True, description="Whether an answer is required to continue")


# Type for input callbacks
InputCallback = Callable[[str, Optional[str], Optional[list]], Awaitable[str]]


class HumanProxyWorker(Worker):
    """
    Human-in-the-loop worker that pauses execution for user input.
    
    This worker enables human oversight by pausing the orchestration
    and requesting input. It works in two modes:
    
    1. **API Mode** (default): Sets state.status to PAUSED and stores
       the question in state.pending_input. The orchestrator loop breaks,
       allowing external systems (like the HTTP API) to collect input
       and resume execution.
       
    2. **Callback Mode**: Uses a provided async callback function to
       get input immediately (useful for CLI tools or custom integrations).
    
    Args:
        name: Worker name (default: "HumanProxy")
        description: Worker description
        input_callback: Optional async callback for getting input directly
        
    Resume Flow (API Mode):
        1. Worker sets status=PAUSED, pending_input={question: "..."}
        2. Orchestrator.run() breaks and returns the paused state
        3. External system shows question to user, gets answer
        4. External system sets state.pending_input["answer"] = user_answer
        5. External system sets state.status back to GENERATING
        6. Orchestrator.run(state=state) resumes from saved state
        7. HumanProxyWorker sees pending_input["answer"] and returns it
        
    Example:
        # API Mode (pause/resume)
        human = HumanProxyWorker()
        
        # Callback Mode (interactive)
        async def get_input(q, ctx, opts):
            return input(f"{q}: ")
        human = HumanProxyWorker(input_callback=get_input)
    """
    
    name = "HumanProxy"
    description = "Asks the human a question and returns their answer"
    input_schema = HumanProxyInput
    parallel_safe = False  # Modifies state
    
    def __init__(
        self,
        name: str = "HumanProxy",
        description: str = "Asks the human a question and returns their answer",
        input_callback: Optional[InputCallback] = None
    ):
        self.name = name
        self.description = description
        self.input_callback = input_callback
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[HumanProxyInput] = None
    ) -> WorkerOutput:
        """Ask the human a question and return their answer."""
        if not inputs or not inputs.question:
            return WorkerOutput(
                metadata={"error": "question is required"}
            )
        
        # Check if we're resuming with a pending answer
        if state.pending_input and state.pending_input.get("answer") is not None:
            answer = state.pending_input["answer"]
            original_question = state.pending_input.get("question", inputs.question)
            
            logger.info(f"[{self.name}] Resuming with answer: {answer[:50]}...")
            
            # Clear pending input after consuming
            state.pending_input = None
            
            return WorkerOutput(
                artifact=Artifact(
                    type="human_input",
                    content=answer,
                    creator=self.name,
                    metadata={
                        "question": original_question,
                        "resumed": True
                    }
                )
            )
        
        # If we have a callback, use it directly
        if self.input_callback is not None:
            logger.info(f"[{self.name}] Asking via callback: {inputs.question}")
            
            try:
                answer = await self.input_callback(
                    inputs.question,
                    inputs.context,
                    inputs.options
                )
                
                return WorkerOutput(
                    artifact=Artifact(
                        type="human_input",
                        content=answer,
                        creator=self.name,
                        metadata={
                            "question": inputs.question,
                            "context": inputs.context,
                            "options": inputs.options,
                            "via_callback": True
                        }
                    )
                )
            except Exception as e:
                logger.error(f"[{self.name}] Callback error: {e}")
                if inputs.required:
                    return WorkerOutput(
                        metadata={"error": f"Failed to get input: {e}"}
                    )
                else:
                    return WorkerOutput(
                        artifact=Artifact(
                            type="human_input",
                            content="",
                            creator=self.name,
                            metadata={
                                "question": inputs.question,
                                "skipped": True,
                                "error": str(e)
                            }
                        )
                    )
        
        # API Mode: Pause execution and request input
        logger.info(f"[{self.name}] Pausing for human input: {inputs.question}")
        
        # Store the question in pending_input
        state.pending_input = {
            "question": inputs.question,
            "context": inputs.context,
            "options": inputs.options,
            "worker": self.name,
            "required": inputs.required,
            "answer": None  # Will be filled by external system
        }
        
        # Update status to PAUSED - this will cause the orchestrator to break
        state.update_status(Status.PAUSED)
        
        # Return metadata indicating we're waiting for input
        # (this output is mostly for logging - the real action is the status change)
        return WorkerOutput(
            metadata={
                "status": "waiting_for_input",
                "question": inputs.question,
                "context": inputs.context,
                "options": inputs.options,
                "message": "Execution paused. Provide answer via pending_input['answer'] and resume."
            }
        )


class CLIHumanProxyWorker(HumanProxyWorker):
    """
    Human proxy worker that uses CLI input.
    
    Convenience subclass that prompts via stdin/stdout.
    Useful for CLI-based agents.
    
    Example:
        human = CLIHumanProxyWorker()
        orchestrator = Orchestrator(llm=my_llm, workers=[writer, human])
        result = await orchestrator.run("Write a poem about...")
    """
    
    name = "CLIHumanProxy"
    description = "Asks the human a question via CLI and returns their answer"
    
    def __init__(
        self,
        name: str = "CLIHumanProxy",
        description: str = "Asks the human a question via CLI and returns their answer"
    ):
        super().__init__(
            name=name,
            description=description,
            input_callback=self._cli_input
        )
    
    async def _cli_input(
        self,
        question: str,
        context: Optional[str],
        options: Optional[list]
    ) -> str:
        """Get input from CLI."""
        import asyncio
        
        print("\n" + "=" * 60)
        print("🧑 HUMAN INPUT REQUIRED")
        print("=" * 60)
        
        if context:
            print(f"\nContext: {context}")
        
        if options:
            print("\nOptions:")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
        
        print(f"\nQuestion: {question}")
        print()
        
        # Use asyncio to run input in executor (input() is blocking)
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, lambda: input("Your answer: "))
        
        print("=" * 60 + "\n")
        
        return answer.strip()
