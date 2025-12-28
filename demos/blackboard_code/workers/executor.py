"""
Executor Worker - Runs shell commands in the sandbox.

This worker REQUIRES HUMAN APPROVAL before execution.
"""

import subprocess
import asyncio
from pathlib import Path
from typing import Optional
from blackboard import Worker, WorkerOutput, Artifact, Blackboard
from blackboard.protocols import WorkerInput
from demos.blackboard_code.config import SANDBOX_DIR


class ExecutorInput(WorkerInput):
    """Input schema for ExecutorWorker."""
    command: str = ""
    timeout: int = 60


class ExecutorWorker(Worker):
    """
    Executes shell commands in the project sandbox.
    
    ⚠️ This worker requires human approval before execution.
    Commands are run in the sandbox directory.
    """
    
    name = "ExecutorWorker"
    description = "Runs shell commands in the project (requires approval)"
    input_schema = ExecutorInput
    
    # Dangerous commands that should be blocked
    BLOCKED_COMMANDS = [
        "rm -rf /",
        "format",
        "del /f /s",
        ":(){:|:&};:",  # fork bomb
    ]
    
    def __init__(self, sandbox_dir: Path = None):
        self.sandbox_dir = sandbox_dir or SANDBOX_DIR
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        command = inputs.command if inputs and hasattr(inputs, 'command') else ""
        timeout = inputs.timeout if inputs and hasattr(inputs, 'timeout') else 60
        
        if not command:
            return WorkerOutput(feedback_text="No command provided")
        
        # Security check - block dangerous commands
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in command.lower():
                return WorkerOutput(feedback_text=f"Blocked dangerous command: {command}")
        
        # Ensure sandbox exists
        cwd = self.sandbox_dir if self.sandbox_dir else Path.cwd()
        cwd.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run command asynchronously
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return WorkerOutput(
                    feedback_text=f"Command timed out after {timeout}s: {command}"
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            output = stdout_str
            if stderr_str:
                output += f"\n[stderr]\n{stderr_str}"
            
            success = process.returncode == 0
            
            return WorkerOutput(
                artifact=Artifact(
                    type="shell_output",
                    content=output[:5000],  # Limit output size
                    creator=self.name,
                    metadata={
                        "command": command,
                        "exit_code": process.returncode,
                        "success": success,
                        "cwd": str(cwd)
                    }
                ),
                feedback_text=f"{'✓' if success else '✗'} {command} (exit: {process.returncode})"
            )
            
        except Exception as e:
            return WorkerOutput(
                feedback_text=f"Error running command: {e}"
            )
