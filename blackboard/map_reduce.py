"""
Map-Reduce Pattern for Parallel Sub-Agent Execution

Enables "Scatter-Gather" workflows where a parent orchestrator can dispatch
work to multiple sub-agents in parallel, then merge the results.

Example:
    from blackboard.map_reduce import run_map_reduce, MapReduceWorker
    
    # Process multiple files in parallel
    result = await run_map_reduce(
        items=["auth.py", "db.py", "api.py"],
        worker=CodeReviewerWorker(),
        parent_state=state,
        llm=llm,
        max_concurrency=3
    )
    
    # Apply collected mutations to parent state
    for mr in result.results:
        for mutation in mr.mutations:
            parent_state.apply_mutation(mutation)

.. versionadded:: 1.8.0
"""

from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, 
    TypeVar, Generic, Union, TYPE_CHECKING
)
from enum import Enum
import asyncio
import logging

if TYPE_CHECKING:
    from .state import Blackboard, Artifact
    from .core import LLMClient
    from .protocols import Worker, WorkerOutput
    from .patching import ArtifactMutation

logger = logging.getLogger("blackboard.map_reduce")


# =============================================================================
# Result Types
# =============================================================================

class ConflictResolution(str, Enum):
    """Strategy for handling mutation conflicts."""
    FAIL = "fail"           # Fail if any conflicts detected
    FIRST_WINS = "first"    # First mutation to an artifact wins
    LAST_WINS = "last"      # Last mutation to an artifact wins
    MERGE_ALL = "merge"     # Try to apply all (may fail if overlapping)


@dataclass
class MutationConflict:
    """
    Represents a conflict between mutations targeting the same artifact.
    
    Attributes:
        artifact_id: The ID of the artifact with conflicting mutations
        conflicting_items: List of items that produced conflicting mutations
        resolution: How the conflict was resolved (if at all)
    """
    artifact_id: str
    conflicting_items: List[Any]
    resolution: Optional[str] = None


@dataclass
class MapResult:
    """
    Result from processing a single item in the map phase.
    
    Attributes:
        item: The input item that was processed
        success: Whether processing succeeded
        artifacts: New artifacts created by the worker
        mutations: Mutations to apply to existing artifacts
        feedback: Any feedback generated
        error: Error message if failed
        trace_id: Optional trace ID for debugging sub-agent execution
    """
    item: Any
    success: bool
    artifacts: List["Artifact"] = field(default_factory=list)
    mutations: List["ArtifactMutation"] = field(default_factory=list)
    feedback: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class MapReduceResult:
    """
    Aggregated result from a map-reduce operation.
    
    Attributes:
        success: True if all items processed successfully
        results: Individual results for each item
        total_items: Number of items processed
        successful_items: Number of items that succeeded
        failed_items: Number of items that failed
        all_artifacts: Combined list of all new artifacts
        all_mutations: Combined list of all mutations (may have conflicts)
        conflicts: List of detected mutation conflicts
    """
    success: bool
    results: List[MapResult]
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    all_artifacts: List["Artifact"] = field(default_factory=list)
    all_mutations: List["ArtifactMutation"] = field(default_factory=list)
    conflicts: List[MutationConflict] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.total_items:
            self.total_items = len(self.results)
        if not self.successful_items:
            self.successful_items = sum(1 for r in self.results if r.success)
        if not self.failed_items:
            self.failed_items = sum(1 for r in self.results if not r.success)
    
    def has_conflicts(self) -> bool:
        """Check if there are any mutation conflicts."""
        return len(self.conflicts) > 0
    
    def get_non_conflicting_mutations(self) -> List["ArtifactMutation"]:
        """Get mutations that don't have conflicts."""
        conflicting_ids = {c.artifact_id for c in self.conflicts}
        return [m for m in self.all_mutations if m.artifact_id not in conflicting_ids]


# =============================================================================
# Core Map-Reduce Function
# =============================================================================

async def run_map_reduce(
    items: Iterable[Any],
    worker: "Worker",
    parent_state: "Blackboard",
    llm: Optional["LLMClient"] = None,
    max_concurrency: int = 5,
    item_to_goal: Optional[Callable[[Any], str]] = None,
    artifact_filter: Optional[Callable[["Artifact", Any], bool]] = None,
    conflict_resolution: ConflictResolution = ConflictResolution.FAIL,
    timeout_per_item: Optional[float] = None,
) -> MapReduceResult:
    """
    Execute a worker on each item in parallel with concurrency control.
    
    This is the core "Scatter-Gather" primitive for swarm intelligence.
    Each item gets its own scoped context and runs independently.
    Results are aggregated and conflicts detected.
    
    Args:
        items: Iterable of items to process (e.g., filenames, tasks)
        worker: Worker to execute on each item
        parent_state: Parent blackboard state (used for context)
        llm: Optional LLM client (required if worker needs it)
        max_concurrency: Maximum parallel executions (default: 5)
        item_to_goal: Function to convert item to goal string for sub-task
        artifact_filter: Function(artifact, item) -> bool to filter context per item
        conflict_resolution: How to handle mutation conflicts
        timeout_per_item: Optional timeout in seconds per item
        
    Returns:
        MapReduceResult with all results and detected conflicts
        
    Example:
        result = await run_map_reduce(
            items=["auth.py", "db.py"],
            worker=code_reviewer,
            parent_state=state,
            max_concurrency=3,
            item_to_goal=lambda f: f"Review {f} for security issues"
        )
    """
    from .state import Blackboard
    from .protocols import WorkerInput
    
    items_list = list(items)
    if not items_list:
        return MapReduceResult(success=True, results=[])
    
    semaphore = asyncio.Semaphore(max_concurrency)
    results: List[MapResult] = []
    
    async def process_item(item: Any) -> MapResult:
        """Process a single item with semaphore control."""
        async with semaphore:
            try:
                # Create scoped goal
                goal = item_to_goal(item) if item_to_goal else f"Process: {item}"
                
                # Create a minimal scoped state for this item
                scoped_state = Blackboard(goal=goal)
                
                # Track artifact IDs being passed to sub-agent
                artifact_ids = []
                
                # Copy relevant artifacts based on filter
                if artifact_filter:
                    for artifact in parent_state.artifacts:
                        if artifact_filter(artifact, item):
                            copied = artifact.model_copy()
                            scoped_state.add_artifact(copied)
                            artifact_ids.append(artifact.id)
                else:
                    # Default: copy all artifacts (may want to be smarter here)
                    for artifact in parent_state.artifacts:
                        copied = artifact.model_copy()
                        scoped_state.add_artifact(copied)
                        artifact_ids.append(artifact.id)
                
                # Store artifact IDs in metadata for sub-agent to reference
                scoped_state.metadata["target_artifact_ids"] = artifact_ids
                scoped_state.metadata["map_reduce_item"] = item
                
                # Create input for the worker
                worker_input = WorkerInput(instructions=goal)
                
                # Execute worker
                if timeout_per_item:
                    output = await asyncio.wait_for(
                        _run_worker(worker, scoped_state, worker_input),
                        timeout=timeout_per_item
                    )
                else:
                    output = await _run_worker(worker, scoped_state, worker_input)
                
                # Extract results
                artifacts = [output.artifact] if output.artifact else []
                mutations = list(output.mutations) if output.mutations else []
                feedback = [output.feedback] if output.feedback else []
                
                return MapResult(
                    item=item,
                    success=True,
                    artifacts=artifacts,
                    mutations=mutations,
                    feedback=feedback,
                    trace_id=output.trace_id
                )
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout processing item: {item}")
                return MapResult(
                    item=item,
                    success=False,
                    error=f"Timeout after {timeout_per_item}s"
                )
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                return MapResult(
                    item=item,
                    success=False,
                    error=str(e)
                )
    
    # Execute all items in parallel
    tasks = [process_item(item) for item in items_list]
    results = await asyncio.gather(*tasks)
    
    # Aggregate results
    all_artifacts = []
    all_mutations = []
    
    for result in results:
        all_artifacts.extend(result.artifacts)
        all_mutations.extend(result.mutations)
    
    # Detect conflicts
    conflicts = _detect_mutation_conflicts(all_mutations, results, conflict_resolution)
    
    # Resolve conflicts based on strategy
    conflicts_resolved = False
    if conflicts and conflict_resolution != ConflictResolution.FAIL:
        all_mutations = _resolve_conflicts(all_mutations, conflicts, conflict_resolution)
        conflicts_resolved = True
    
    # Determine overall success:
    # - All items must have succeeded
    # - Either no conflicts, OR conflicts were resolved
    all_items_succeeded = all(r.success for r in results)
    conflicts_ok = not conflicts or conflicts_resolved
    
    return MapReduceResult(
        success=all_items_succeeded and conflicts_ok,
        results=results,
        all_artifacts=all_artifacts,
        all_mutations=all_mutations,
        conflicts=conflicts  # Keep conflicts for transparency even if resolved
    )


async def _run_worker(
    worker: "Worker",
    state: "Blackboard",
    inputs: "WorkerInput"
) -> "WorkerOutput":
    """Execute a worker (handles both sync and async workers)."""
    import inspect
    
    result = worker.run(state, inputs)
    if inspect.iscoroutine(result):
        return await result
    return result


def _detect_mutation_conflicts(
    mutations: List["ArtifactMutation"],
    results: List[MapResult],
    resolution: ConflictResolution
) -> List[MutationConflict]:
    """
    Detect conflicts where multiple mutations target the same artifact.
    """
    if resolution == ConflictResolution.MERGE_ALL:
        # In MERGE_ALL mode, we don't detect conflicts upfront
        return []
    
    # Group mutations by artifact_id
    artifact_mutations: Dict[str, List[tuple]] = {}
    for result in results:
        for mutation in result.mutations:
            if mutation.artifact_id not in artifact_mutations:
                artifact_mutations[mutation.artifact_id] = []
            artifact_mutations[mutation.artifact_id].append((result.item, mutation))
    
    # Find conflicts (same artifact from different items)
    conflicts = []
    for artifact_id, item_mutations in artifact_mutations.items():
        if len(item_mutations) > 1:
            # Multiple items modified the same artifact
            conflicting_items = [im[0] for im in item_mutations]
            conflicts.append(MutationConflict(
                artifact_id=artifact_id,
                conflicting_items=conflicting_items
            ))
    
    return conflicts


def _resolve_conflicts(
    mutations: List["ArtifactMutation"],
    conflicts: List[MutationConflict],
    resolution: ConflictResolution
) -> List["ArtifactMutation"]:
    """
    Apply conflict resolution strategy.
    """
    if resolution == ConflictResolution.FAIL:
        return mutations  # Don't modify - caller should check conflicts
    
    conflicting_ids = {c.artifact_id for c in conflicts}
    
    if resolution == ConflictResolution.FIRST_WINS:
        # Keep only the first mutation for each conflicting artifact
        seen = set()
        resolved = []
        for m in mutations:
            if m.artifact_id in conflicting_ids:
                if m.artifact_id not in seen:
                    seen.add(m.artifact_id)
                    resolved.append(m)
                    # Mark conflict as resolved
                    for c in conflicts:
                        if c.artifact_id == m.artifact_id:
                            c.resolution = "first_wins"
            else:
                resolved.append(m)
        return resolved
    
    elif resolution == ConflictResolution.LAST_WINS:
        # Keep only the last mutation for each conflicting artifact
        # Reverse, then do first_wins, then reverse back
        reversed_mutations = list(reversed(mutations))
        seen = set()
        resolved = []
        for m in reversed_mutations:
            if m.artifact_id in conflicting_ids:
                if m.artifact_id not in seen:
                    seen.add(m.artifact_id)
                    resolved.append(m)
                    for c in conflicts:
                        if c.artifact_id == m.artifact_id:
                            c.resolution = "last_wins"
            else:
                resolved.append(m)
        return list(reversed(resolved))
    
    return mutations


# =============================================================================
# MapReduceWorker - Wraps map-reduce as a Worker
# =============================================================================

class MapReduceWorker:
    """
    A Worker that internally runs map-reduce on a set of items.
    
    This wraps the map-reduce pattern as a Worker so it can be used
    within an Orchestrator like any other worker.
    
    Attributes:
        name: Worker name
        description: Worker description
        inner_worker: The worker to run on each item
        items_extractor: Function to extract items from state
        
    Example:
        # Create a MapReduceWorker that reviews all code artifacts
        reviewer = MapReduceWorker(
            name="ParallelCodeReviewer",
            description="Reviews all code files in parallel",
            inner_worker=CodeReviewWorker(),
            items_extractor=lambda state: [a for a in state.artifacts if a.type == "code"],
            max_concurrency=5
        )
        
        # Use in orchestrator
        orchestrator = Orchestrator(llm=llm, workers=[reviewer, ...])
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        inner_worker: "Worker",
        items_extractor: Callable[["Blackboard"], List[Any]],
        llm: Optional["LLMClient"] = None,
        max_concurrency: int = 5,
        item_to_goal: Optional[Callable[[Any], str]] = None,
        artifact_filter: Optional[Callable[["Artifact", Any], bool]] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.FIRST_WINS,
    ):
        self.name = name
        self.description = description
        self._inner_worker = inner_worker
        self._items_extractor = items_extractor
        self._llm = llm
        self._max_concurrency = max_concurrency
        self._item_to_goal = item_to_goal
        self._artifact_filter = artifact_filter
        self._conflict_resolution = conflict_resolution
        self.input_schema = None  # Optional structured input
    
    async def run(
        self,
        state: "Blackboard",
        inputs: Optional["WorkerInput"] = None
    ) -> "WorkerOutput":
        """Execute map-reduce and return aggregated results."""
        from .protocols import WorkerOutput
        from .state import Artifact, Feedback
        
        # Extract items from state
        items = self._items_extractor(state)
        
        if not items:
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    critique="No items to process",
                    passed=True
                )
            )
        
        # Run map-reduce
        result = await run_map_reduce(
            items=items,
            worker=self._inner_worker,
            parent_state=state,
            llm=self._llm,
            max_concurrency=self._max_concurrency,
            item_to_goal=self._item_to_goal,
            artifact_filter=self._artifact_filter,
            conflict_resolution=self._conflict_resolution,
        )
        
        # Build summary artifact
        summary_lines = [
            f"## Map-Reduce Results",
            f"- Total items: {result.total_items}",
            f"- Successful: {result.successful_items}",
            f"- Failed: {result.failed_items}",
        ]
        
        if result.conflicts:
            summary_lines.append(f"- Conflicts: {len(result.conflicts)}")
            for conflict in result.conflicts:
                summary_lines.append(f"  - Artifact {conflict.artifact_id[:8]}... modified by {len(conflict.conflicting_items)} items")
        
        if result.failed_items > 0:
            summary_lines.append("\n### Failures:")
            for r in result.results:
                if not r.success:
                    summary_lines.append(f"- {r.item}: {r.error}")
        
        summary_artifact = Artifact(
            type="report",
            content="\n".join(summary_lines),
            creator=self.name,
            metadata={
                "map_reduce": True,
                "total_items": result.total_items,
                "successful": result.successful_items,
                "failed": result.failed_items,
                "conflicts": len(result.conflicts)
            }
        )
        
        # Return output with mutations from successful items
        return WorkerOutput(
            artifact=summary_artifact,
            mutations=result.get_non_conflicting_mutations(),
            metadata={
                "all_mutations": result.all_mutations,
                "conflicts": [
                    {"artifact_id": c.artifact_id, "items": c.conflicting_items}
                    for c in result.conflicts
                ]
            }
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ConflictResolution",
    "MutationConflict",
    "MapResult",
    "MapReduceResult",
    "run_map_reduce",
    "MapReduceWorker",
]
