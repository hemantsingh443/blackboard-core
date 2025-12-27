"""
State Merging Utilities for Branch-Merge Pattern

Provides utilities to merge forked Blackboard states back together,
detecting and resolving conflicts between parallel sub-agent work.

Example:
    from blackboard.merging import merge_states, MergeConflict
    
    # Fork parent state for sub-agent work
    forked_state = parent_state.model_copy(deep=True)
    forked_state.goal = "Sub-task: Review auth.py"
    
    # Sub-agent does work, adds artifacts/feedback...
    
    # Merge back into parent
    merge_result = merge_states(parent_state, forked_state)
    if merge_result.conflicts:
        # Handle conflicts...

.. versionadded:: 1.8.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .state import Blackboard, Artifact, Feedback


class MergeStrategy(str, Enum):
    """Strategy for merging forked states."""
    THEIRS = "theirs"       # Child state wins on conflicts
    OURS = "ours"           # Parent state wins on conflicts
    FAIL = "fail"           # Fail on any conflict
    NEWEST = "newest"       # Prefer artifact with higher version


@dataclass
class MergeConflict:
    """
    Represents a conflict during state merge.
    
    Attributes:
        conflict_type: Type of conflict (artifact, feedback, metadata)
        artifact_id: Artifact ID if applicable
        description: Human-readable description of the conflict
        parent_value: Value in parent state
        child_value: Value in child state
        resolved: Whether the conflict was auto-resolved
        resolution: How it was resolved (if at all)
    """
    conflict_type: str
    artifact_id: Optional[str] = None
    description: str = ""
    parent_value: Optional[Any] = None
    child_value: Optional[Any] = None
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class MergeResult:
    """
    Result of merging two Blackboard states.
    
    Attributes:
        success: True if merge completed without unresolved conflicts
        merged_state: The resulting merged state
        artifacts_added: Number of new artifacts added from child
        artifacts_updated: Number of artifacts updated from child
        feedback_added: Number of feedback entries added from child
        conflicts: List of conflicts encountered during merge
    """
    success: bool
    merged_state: "Blackboard"
    artifacts_added: int = 0
    artifacts_updated: int = 0
    feedback_added: int = 0
    conflicts: List[MergeConflict] = field(default_factory=list)
    
    def has_conflicts(self) -> bool:
        """Check if there are any unresolved conflicts."""
        return any(not c.resolved for c in self.conflicts)


def merge_states(
    parent: "Blackboard",
    child: "Blackboard",
    strategy: MergeStrategy = MergeStrategy.THEIRS,
    artifact_filter: Optional[Callable[["Artifact"], bool]] = None,
    merge_feedback: bool = True,
    merge_metadata: bool = True,
) -> MergeResult:
    """
    Merge a forked child state back into the parent state.
    
    This is the core primitive for the "Branch-Merge" pattern, allowing
    sub-agents to work in isolation and then merge results back.
    
    The merge is performed IN-PLACE on the parent state.
    
    Args:
        parent: The parent state to merge into (modified in place)
        child: The child state to merge from
        strategy: How to handle conflicts (default: child wins)
        artifact_filter: Optional filter to only merge certain artifacts
        merge_feedback: Whether to merge feedback entries (default: True)
        merge_metadata: Whether to merge metadata (default: True)
        
    Returns:
        MergeResult with statistics and any conflicts
        
    Example:
        # Simple merge - child wins on conflicts
        result = merge_states(parent, child)
        
        # Only merge code artifacts
        result = merge_states(
            parent, child,
            artifact_filter=lambda a: a.type == "code"
        )
    """
    conflicts: List[MergeConflict] = []
    artifacts_added = 0
    artifacts_updated = 0
    feedback_added = 0
    
    # Build lookup of parent artifacts by ID
    parent_artifact_ids: Dict[str, int] = {
        a.id: idx for idx, a in enumerate(parent.artifacts)
    }
    
    # =================================================================
    # Merge Artifacts
    # =================================================================
    child_artifacts = child.artifacts
    if artifact_filter:
        child_artifacts = [a for a in child.artifacts if artifact_filter(a)]
    
    for child_artifact in child_artifacts:
        if child_artifact.id in parent_artifact_ids:
            # Artifact exists in parent - check for conflict
            parent_idx = parent_artifact_ids[child_artifact.id]
            parent_artifact = parent.artifacts[parent_idx]
            
            if parent_artifact.version == child_artifact.version:
                # Same version - no change needed
                continue
            
            if parent_artifact.content == child_artifact.content:
                # Content identical - just update version if needed
                if child_artifact.version > parent_artifact.version:
                    parent.artifacts[parent_idx] = child_artifact.model_copy()
                    artifacts_updated += 1
                continue
            
            # Content differs - this is a conflict
            conflict = MergeConflict(
                conflict_type="artifact",
                artifact_id=child_artifact.id,
                description=f"Artifact {child_artifact.id[:8]} modified in both parent and child",
                parent_value=parent_artifact.content[:100] if isinstance(parent_artifact.content, str) else str(parent_artifact.content)[:100],
                child_value=child_artifact.content[:100] if isinstance(child_artifact.content, str) else str(child_artifact.content)[:100],
            )
            
            # Apply resolution strategy
            if strategy == MergeStrategy.FAIL:
                conflicts.append(conflict)
            elif strategy == MergeStrategy.THEIRS:
                parent.artifacts[parent_idx] = child_artifact.model_copy()
                artifacts_updated += 1
                conflict.resolved = True
                conflict.resolution = "child_wins"
                conflicts.append(conflict)
            elif strategy == MergeStrategy.OURS:
                # Keep parent - no change
                conflict.resolved = True
                conflict.resolution = "parent_wins"
                conflicts.append(conflict)
            elif strategy == MergeStrategy.NEWEST:
                if child_artifact.version > parent_artifact.version:
                    parent.artifacts[parent_idx] = child_artifact.model_copy()
                    artifacts_updated += 1
                    conflict.resolved = True
                    conflict.resolution = "child_wins_newer"
                else:
                    conflict.resolved = True
                    conflict.resolution = "parent_wins_newer"
                conflicts.append(conflict)
        else:
            # New artifact from child - add to parent
            parent.add_artifact(child_artifact.model_copy())
            artifacts_added += 1
    
    # =================================================================
    # Merge Feedback
    # =================================================================
    if merge_feedback:
        # Get IDs of existing feedback to avoid duplicates
        parent_feedback_ids: Set[str] = {
            f.id for f in parent.feedback if hasattr(f, 'id') and f.id
        }
        
        for child_fb in child.feedback:
            fb_id = getattr(child_fb, 'id', None)
            if fb_id and fb_id in parent_feedback_ids:
                continue  # Skip duplicate
            
            parent.add_feedback(child_fb.model_copy())
            feedback_added += 1
    
    # =================================================================
    # Merge Metadata
    # =================================================================
    if merge_metadata and child.metadata:
        for key, value in child.metadata.items():
            if key not in parent.metadata:
                parent.metadata[key] = value
            elif parent.metadata[key] != value:
                # Metadata conflict
                conflict = MergeConflict(
                    conflict_type="metadata",
                    description=f"Metadata key '{key}' differs",
                    parent_value=parent.metadata[key],
                    child_value=value,
                )
                
                if strategy in (MergeStrategy.THEIRS, MergeStrategy.NEWEST):
                    parent.metadata[key] = value
                    conflict.resolved = True
                    conflict.resolution = "child_wins"
                elif strategy == MergeStrategy.OURS:
                    conflict.resolved = True
                    conflict.resolution = "parent_wins"
                # FAIL strategy leaves unresolved
                
                conflicts.append(conflict)
    
    # Determine overall success
    success = not any(not c.resolved for c in conflicts)
    
    return MergeResult(
        success=success,
        merged_state=parent,
        artifacts_added=artifacts_added,
        artifacts_updated=artifacts_updated,
        feedback_added=feedback_added,
        conflicts=conflicts,
    )


class StateMerger:
    """
    Utility class for merging multiple forked states.
    
    Useful when you have multiple sub-agents that all forked from
    the same parent and need to merge their results sequentially.
    
    Example:
        merger = StateMerger(parent_state, strategy=MergeStrategy.THEIRS)
        
        # Merge results from multiple sub-agents
        for child_state in sub_agent_results:
            result = merger.merge(child_state)
            print(f"Added {result.artifacts_added} artifacts")
        
        # Get final merged state
        final_state = merger.get_merged_state()
    """
    
    def __init__(
        self,
        base_state: "Blackboard",
        strategy: MergeStrategy = MergeStrategy.THEIRS,
        artifact_filter: Optional[Callable[["Artifact"], bool]] = None,
    ):
        """
        Initialize the merger.
        
        Args:
            base_state: The base state to merge into
            strategy: Default conflict resolution strategy
            artifact_filter: Optional filter for artifacts to merge
        """
        self._state = base_state
        self._strategy = strategy
        self._artifact_filter = artifact_filter
        self._merge_history: List[MergeResult] = []
    
    def merge(
        self,
        child_state: "Blackboard",
        strategy: Optional[MergeStrategy] = None,
    ) -> MergeResult:
        """
        Merge a child state into the base state.
        
        Args:
            child_state: The forked state to merge
            strategy: Override strategy for this merge (optional)
            
        Returns:
            MergeResult with merge statistics
        """
        result = merge_states(
            parent=self._state,
            child=child_state,
            strategy=strategy or self._strategy,
            artifact_filter=self._artifact_filter,
        )
        self._merge_history.append(result)
        return result
    
    def get_merged_state(self) -> "Blackboard":
        """Get the current merged state."""
        return self._state
    
    def get_merge_history(self) -> List[MergeResult]:
        """Get all merge results in order."""
        return self._merge_history
    
    def get_total_conflicts(self) -> int:
        """Get total number of conflicts across all merges."""
        return sum(len(r.conflicts) for r in self._merge_history)
    
    def get_unresolved_conflicts(self) -> List[MergeConflict]:
        """Get all unresolved conflicts across all merges."""
        unresolved = []
        for result in self._merge_history:
            unresolved.extend(c for c in result.conflicts if not c.resolved)
        return unresolved


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MergeStrategy",
    "MergeConflict",
    "MergeResult",
    "merge_states",
    "StateMerger",
]
