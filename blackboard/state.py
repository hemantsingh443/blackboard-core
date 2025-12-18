"""
Blackboard State Models

The "Single Source of Truth" for the multi-agent system.
All state is stored in typed Pydantic models for strict validation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Current phase of the blackboard execution."""
    PLANNING = "planning"
    GENERATING = "generating"
    CRITIQUING = "critiquing"
    REFINING = "refining"
    PAUSED = "paused"  # For pause-and-resume pattern (e.g., awaiting approval)
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

    model_config = {"extra": "allow"}


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

    model_config = {"extra": "allow"}


class StateConflictError(Exception):
    """
    Raised when a state save detects a version conflict.
    
    This prevents "last write wins" data loss when multiple agents/processes
    try to update the state file concurrently.
    """
    pass


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
    version: int = Field(default=1, description="Optimistic locking version number")
    artifacts: List[Artifact] = Field(default_factory=list, description="Versioned outputs")
    feedback: List[Feedback] = Field(default_factory=list, description="Critique log")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible context")
    step_count: int = Field(default=0, description="Number of steps executed")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Execution history log")
    max_history: int = Field(default=1000, description="Max history entries (ring buffer, prevents OOM)")
    context_summary: str = Field(default="", description="Summary of earlier context (for long sessions)")
    pending_input: Optional[Dict[str, Any]] = Field(default=None, description="Pending user input for resume")

    model_config = {"extra": "allow"}

    # =========================================================================
    # Persistence Methods
    # =========================================================================
    
    def save_to_json(self, path: Union[str, Path], skip_version_check: bool = False) -> None:
        """
        Save the blackboard state to a JSON file.
        
        Uses optimistic locking to prevent data loss from concurrent updates.
        Uses atomic write (temp file + rename) to prevent corruption on crash.
        
        Args:
            path: Path to save the JSON file
            skip_version_check: If True, skip version check (use with caution)
            
        Raises:
            StateConflictError: If disk version is newer than memory version
            
        Example:
            state.save_to_json("session_001.json")
        """
        import tempfile
        import os
        
        path = Path(path)
        
        # Optimistic locking: check if disk version is newer
        if not skip_version_check and path.exists():
            try:
                existing = Blackboard.load_from_json(path)
                if existing.version > self.version:
                    raise StateConflictError(
                        f"State conflict: disk version ({existing.version}) is newer than "
                        f"memory version ({self.version}). Another process may have updated the state."
                    )
            except StateConflictError:
                raise  # Re-raise conflict errors
            except Exception:
                pass  # If we can't load, proceed with save
        
        # Increment version
        self.version += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write: write to temp file then rename
        # This prevents corruption if the process crashes mid-write
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='.blackboard_',
            dir=path.parent
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(self.model_dump_json(indent=2))
            # Atomic rename (on POSIX) or replace (on Windows)
            os.replace(temp_path, path)
        except:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except:
                pass
            raise
    
    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> "Blackboard":
        """
        Load blackboard state from a JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Restored Blackboard instance
            
        Example:
            state = Blackboard.load_from_json("session_001.json")
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for custom serialization)."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Blackboard":
        """Create from dictionary."""
        return cls.model_validate(data)

    # =========================================================================
    # Artifact & Feedback Methods
    # =========================================================================

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
        """Add an event to the execution history (ring buffer)."""
        self.history.append({
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "step": self.step_count
        })
        # Ring buffer: drop oldest entries if exceeding max
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    # =========================================================================
    # Context Generation (with history management)
    # =========================================================================

    def to_context_string(
        self,
        max_tokens: int = 12000,
        chars_per_token: int = 4,
        max_artifacts: int = 3,
        max_feedback: int = 5
    ) -> str:
        """
        Generate a token-aware context string for the supervisor LLM.
        
        This is what the Supervisor "sees" when deciding what to do next.
        Uses smart truncation to stay within token budget while prioritizing
        critical information (goal, status, feedback) over artifact content.
        
        Args:
            max_tokens: Maximum tokens for the entire context (default: 12000)
            chars_per_token: Approximate characters per token (default: 4)
            max_artifacts: Maximum number of recent artifacts to include
            max_feedback: Maximum number of recent feedback entries to include
            
        Returns:
            Context string within the token budget
        """
        max_chars = max_tokens * chars_per_token
        
        # =================================================================
        # Priority 1: Goal and Status (always included)
        # =================================================================
        base_context = f"## Goal\n{self.goal}\n\n## Status\n{self.status.value.upper()}\n"
        
        # Include context summary if available (for long sessions)
        if self.context_summary:
            base_context += f"\n## Previous Context Summary\n{self.context_summary}\n"
        
        base_context += f"\n## Step Count\n{self.step_count}\n"
        
        remaining_chars = max_chars - len(base_context)
        
        # =================================================================
        # Priority 2: Feedback (critical - supervisor needs to see errors)
        # =================================================================
        feedback_str = ""
        recent_feedback = list(reversed(self.feedback[-max_feedback:])) if self.feedback else []
        if recent_feedback:
            if len(self.feedback) > max_feedback:
                feedback_str = f"\n## Recent Feedback ({len(self.feedback)} total, showing last {max_feedback})\n"
            else:
                feedback_str = f"\n## Feedback ({len(self.feedback)} total)\n"
            
            for fb in recent_feedback:
                status = "PASSED" if fb.passed else "FAILED"
                feedback_str += f"- [{status}] {fb.source}: {fb.critique}\n"
        
        remaining_chars -= len(feedback_str)
        
        # =================================================================
        # Priority 3: Artifacts (variable - use smart truncation)
        # =================================================================
        artifacts_str = ""
        recent_artifacts = self.artifacts[-max_artifacts:] if self.artifacts else []
        
        if recent_artifacts:
            if len(self.artifacts) > max_artifacts:
                artifacts_str = f"\n## Artifacts ({len(self.artifacts)} total, showing last {max_artifacts})\n"
            else:
                artifacts_str = f"\n## Artifacts ({len(self.artifacts)} total)\n"
            
            remaining_chars -= len(artifacts_str)
            
            for artifact in recent_artifacts:
                header = f"\n### {artifact.type} (v{artifact.version}, ID: {artifact.id[:8]}) by {artifact.creator}\n"
                content_str = str(artifact.content)
                content_len = len(content_str)
                
                # Calculate how much space we have for this artifact's content
                # Reserve some space for subsequent artifacts
                artifact_budget = remaining_chars // max(1, len(recent_artifacts))
                content_budget = artifact_budget - len(header) - 50  # 50 char buffer
                
                if content_budget <= 0:
                    # No space - show metadata only
                    body = f"[Content too large to display: {content_len} chars]\n"
                elif content_len <= content_budget:
                    # Fits entirely
                    body = content_str + "\n"
                elif content_len > 2000 and content_budget >= 1000:
                    # Large content - show head/tail preview
                    head_size = min(400, content_budget // 3)
                    tail_size = min(400, content_budget // 3)
                    omitted = content_len - head_size - tail_size
                    body = (
                        content_str[:head_size] + 
                        f"\n... [{omitted} chars omitted] ...\n" + 
                        content_str[-tail_size:] + "\n"
                    )
                else:
                    # Truncate with ellipsis
                    body = content_str[:content_budget] + "...\n"
                
                artifact_entry = header + body
                artifacts_str += artifact_entry
                remaining_chars -= len(artifact_entry)
        
        return base_context + feedback_str + artifacts_str

    def get_context_summary(self) -> str:
        """
        Get a brief summary of the current state.
        
        Useful for condensed logging or status displays.
        """
        artifact_count = len(self.artifacts)
        feedback_count = len(self.feedback)
        last_passed = None
        if self.feedback:
            last_passed = self.feedback[-1].passed
        
        return (
            f"Status: {self.status.value} | "
            f"Steps: {self.step_count} | "
            f"Artifacts: {artifact_count} | "
            f"Feedback: {feedback_count} | "
            f"Last Review: {'Passed' if last_passed else 'Failed' if last_passed is not None else 'N/A'}"
        )

    def update_summary(self, summary: str) -> None:
        """
        Update the context summary for long-running sessions.
        
        Call this when the context grows too large to fit in the LLM window.
        The summary will be included in to_context_string() output.
        
        Args:
            summary: A compressed summary of earlier context
        """
        self.context_summary = summary
        self._log_event("summary_updated", {"length": len(summary)})
    
    def should_summarize(
        self,
        artifact_threshold: int = 10,
        feedback_threshold: int = 20,
        step_threshold: int = 50
    ) -> bool:
        """
        Check if the context should be summarized.
        
        Returns True if any threshold is exceeded.
        
        Args:
            artifact_threshold: Summarize if artifacts exceed this count
            feedback_threshold: Summarize if feedback exceeds this count
            step_threshold: Summarize if steps exceed this count
        """
        return (
            len(self.artifacts) > artifact_threshold or
            len(self.feedback) > feedback_threshold or
            self.step_count > step_threshold
        )
    
    def compact_history(self, keep_last: int = 20) -> int:
        """
        Compact the history log by keeping only recent entries.
        
        Args:
            keep_last: Number of recent history entries to keep
            
        Returns:
            Number of entries removed
        """
        if len(self.history) <= keep_last:
            return 0
        
        removed = len(self.history) - keep_last
        self.history = self.history[-keep_last:]
        return removed

