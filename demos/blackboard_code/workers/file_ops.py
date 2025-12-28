"""
File Worker - Creates and edits files in the sandbox.

This worker REQUIRES HUMAN APPROVAL before execution.
"""

from pathlib import Path
from typing import Optional
from blackboard import Worker, WorkerOutput, Artifact, Blackboard
from blackboard.protocols import WorkerInput
from demos.blackboard_code.config import SANDBOX_DIR


class FileInput(WorkerInput):
    """Input schema for FileWorker."""
    action: str = "create"  # "create" or "edit"
    path: str = ""
    content: str = ""
    artifact_id: str = ""


class FileWorker(Worker):
    """
    Creates or edits files in the project sandbox.
    
    ⚠️ This worker requires human approval before execution.
    It can CREATE new files or EDIT existing files.
    """
    
    name = "FileWorker"
    description = "Creates or edits files in the project (requires approval)"
    input_schema = FileInput
    
    def __init__(self, sandbox_dir: Path = None):
        self.sandbox_dir = sandbox_dir or SANDBOX_DIR
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        action = inputs.action if inputs and hasattr(inputs, 'action') else "create"
        path = inputs.path if inputs and hasattr(inputs, 'path') else ""
        content = inputs.content if inputs and hasattr(inputs, 'content') else ""
        artifact_id = inputs.artifact_id if inputs and hasattr(inputs, 'artifact_id') else ""
        
        if not path:
            return WorkerOutput(feedback_text="No path provided")
        
        # Get content from artifact if specified
        if artifact_id and not content:
            artifact = state.get_artifact(artifact_id)
            if artifact:
                content = artifact.content
        
        # If still no content, try to find most recent code artifact
        if not content:
            for art in reversed(state.artifacts):
                if art.type == "code":
                    content = art.content
                    break
        
        if not content:
            return WorkerOutput(feedback_text="No content provided or found")
        
        if not self.sandbox_dir:
            return WorkerOutput(feedback_text="No sandbox directory configured")
        
        # Ensure sandbox exists
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Build full path
        file_path = self.sandbox_dir / path
        
        # Security check - ensure path is within sandbox
        try:
            file_path.resolve().relative_to(self.sandbox_dir.resolve())
        except ValueError:
            return WorkerOutput(feedback_text=f"Security: Path must be within sandbox")
        
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform action
        if action == "create":
            file_path.write_text(content, encoding="utf-8")
            result_msg = f"Created {path} ({len(content)} bytes)"
        else:  # edit
            if file_path.exists():
                file_path.write_text(content, encoding="utf-8")
                result_msg = f"Updated {path} ({len(content)} bytes)"
            else:
                file_path.write_text(content, encoding="utf-8")
                result_msg = f"Created {path} (file didn't exist)"
        
        return WorkerOutput(
            artifact=Artifact(
                type="file",
                content=content,
                creator=self.name,
                metadata={
                    "path": path,
                    "action": action,
                    "full_path": str(file_path),
                    "size": len(content)
                }
            ),
            feedback_text=result_msg
        )
