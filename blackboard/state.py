"""
Blackboard State Models

The "Single Source of Truth" for the multi-agent system.
All state is stored in typed Pydantic models for strict validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Current phase of the blackboard execution."""
    PLANNING = "planning"
    GENERATING = "generating"
    CRITIQUING = "critiquing"
    REFINING = "refining"
    DONE = "done"
    FAILED = "failed"


class Artifact(BaseModel):
    """
    A versioned output produced by a worker.
    
    Examples:
        - Artifact(type="code", content="def hello(): ...", creator="CodeWriter")
        - Artifact(type="image", content="s3://bucket/image.png", creator="ImageGenerator")
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str = Field(..., description="The artifact type (e.g., 'code', 'text', 'image', 'json')")
    content: Any = Field(..., description="The actual payload")
    creator: str = Field(..., description="Name of the worker that created this artifact")
    version: int = Field(default=1, description="Auto-incrementing version number")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional extra context")

    class Config:
        extra = "allow"


class Feedback(BaseModel):
    """
    A critique or validation result from a worker.
    
    Examples:
        - Feedback(source="Critic", critique="Code has a bug on line 5", passed=False)
        - Feedback(source="Validator", critique="All tests pass", passed=True)
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_id: Optional[str] = Field(default=None, description="Reference to the artifact being critiqued")
    source: str = Field(..., description="Name of the worker that gave this feedback")
    critique: str = Field(..., description="The feedback text")
    passed: bool = Field(..., description="Whether the artifact passed review")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional extra context")

    class Config:
        extra = "allow"


class Blackboard(BaseModel):
    """
    The Shared Global State - the central hub that all agents read from and write to.
    
    This is the "bulletin board" where:
    - The goal is posted (immutable once set)
    - Artifacts are published by workers
    - Feedback is logged by critics
    - Status tracks the current phase
    """
    goal: str = Field(..., description="The immutable objective for this session")
    status: Status = Field(default=Status.PLANNING, description="Current execution phase")
    artifacts: List[Artifact] = Field(default_factory=list, description="Versioned outputs")
    feedback: List[Feedback] = Field(default_factory=list, description="Critique log")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible context")
    step_count: int = Field(default=0, description="Number of steps executed")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Execution history log")

    class Config:
        extra = "allow"

    def get_last_artifact(self, artifact_type: Optional[str] = None) -> Optional[Artifact]:
        """
        Get the most recent artifact, optionally filtered by type.
        
        Args:
            artifact_type: If provided, filter by this type (e.g., "code", "text")
            
        Returns:
            The most recent matching artifact, or None if not found
        """
        if not self.artifacts:
            return None
        
        if artifact_type is None:
            return self.artifacts[-1]
        
        for artifact in reversed(self.artifacts):
            if artifact.type == artifact_type:
                return artifact
        return None

    def get_latest_feedback(self) -> Optional[Feedback]:
        """Get the most recent feedback entry."""
        return self.feedback[-1] if self.feedback else None

    def get_feedback_for_artifact(self, artifact_id: str) -> List[Feedback]:
        """Get all feedback entries for a specific artifact."""
        return [f for f in self.feedback if f.artifact_id == artifact_id]

    def add_artifact(self, artifact: Artifact) -> Artifact:
        """
        Add a new artifact to the blackboard.
        
        Automatically sets the version based on existing artifacts of the same type.
        
        Returns:
            The artifact with updated version number
        """
        # Calculate version based on existing artifacts of same type
        same_type = [a for a in self.artifacts if a.type == artifact.type]
        artifact.version = len(same_type) + 1
        
        self.artifacts.append(artifact)
        self._log_event("artifact_added", {"artifact_id": artifact.id, "type": artifact.type})
        return artifact

    def add_feedback(self, feedback: Feedback) -> Feedback:
        """Add feedback to the blackboard."""
        self.feedback.append(feedback)
        self._log_event("feedback_added", {"feedback_id": feedback.id, "passed": feedback.passed})
        return feedback

    def update_status(self, new_status: Status) -> None:
        """Update the execution status."""
        old_status = self.status
        self.status = new_status
        self._log_event("status_changed", {"from": old_status.value, "to": new_status.value})

    def increment_step(self) -> int:
        """Increment and return the step counter."""
        self.step_count += 1
        return self.step_count

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the execution history."""
        self.history.append({
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "step": self.step_count
        })

    def to_context_string(self) -> str:
        """
        Generate a human-readable context string for the supervisor LLM.
        
        This is what the Supervisor "sees" when deciding what to do next.
        """
        lines = [
            f"## Goal\n{self.goal}",
            f"\n## Status\n{self.status.value.upper()}",
        ]
        
        # Add latest artifact info
        if self.artifacts:
            last = self.artifacts[-1]
            content_preview = str(last.content)[:500]
            if len(str(last.content)) > 500:
                content_preview += "..."
            lines.append(f"\n## Latest Artifact\n- Type: {last.type}\n- Creator: {last.creator}\n- Content:\n{content_preview}")
        
        # Add latest feedback
        if self.feedback:
            last_fb = self.feedback[-1]
            lines.append(f"\n## Latest Feedback\n- From: {last_fb.source}\n- Passed: {last_fb.passed}\n- Critique: {last_fb.critique}")
        
        lines.append(f"\n## Step Count\n{self.step_count}")
        
        return "\n".join(lines)
