"""
Hierarchical Orchestration

Enables nested orchestrators for complex multi-agent workflows.
Sub-orchestrators manage their own workers and return consolidated results.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .protocols import Worker, WorkerOutput, WorkerInput
from .state import Blackboard, Artifact, Status

if TYPE_CHECKING:
    from .core import Orchestrator, LLMClient

logger = logging.getLogger("blackboard.hierarchy")

# Maximum recursion depth to prevent infinite loops
DEFAULT_MAX_DEPTH = 5


class RecursionLimitError(Exception):
    """Raised when sub-orchestrator recursion exceeds max_depth."""
    pass


class SubGoalInput(WorkerInput):
    """Input schema for sub-orchestrator workers."""
    sub_goal: str
    max_steps: int = 10
    context: Optional[str] = None
    _depth: int = 0  # Internal: tracks current recursion depth


class SubOrchestratorWorker(Worker):
    """
    A worker that delegates to a nested orchestrator.
    
    Use this for complex tasks that benefit from their own agent team.
    The sub-orchestrator has its own Blackboard and workers, keeping
    the main orchestrator's context clean.
    
    Args:
        name: Worker name visible to main supervisor
        description: Description for main supervisor
        llm: LLM client for the sub-orchestrator
        sub_workers: Workers managed by the sub-orchestrator
        goal_template: Template for sub-goal (uses {sub_goal} placeholder)
        max_sub_steps: Default max steps for sub-orchestration
        
    Example:
        research_team = SubOrchestratorWorker(
            name="ResearchTeam",
            description="Researches a topic using multiple specialists",
            llm=my_llm,
            sub_workers=[WebScraper(), Summarizer(), FactChecker()],
            goal_template="Research and summarize: {sub_goal}"
        )
        
        main_orchestrator = Orchestrator(
            llm=my_llm,
            workers=[Writer(), research_team]
        )
    """
    
    input_schema = SubGoalInput
    parallel_safe = False  # Sub-orchestrators manage their own parallel execution
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: "LLMClient",
        sub_workers: List[Worker],
        goal_template: str = "{sub_goal}",
        max_sub_steps: int = 10,
        max_depth: int = DEFAULT_MAX_DEPTH,
        verbose: bool = False
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.sub_workers = sub_workers
        self.goal_template = goal_template
        self.max_sub_steps = max_sub_steps
        self.max_depth = max_depth
        self.verbose = verbose
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[SubGoalInput] = None
    ) -> WorkerOutput:
        # Import here to avoid circular import
        from .core import Orchestrator
        
        if not inputs or not inputs.sub_goal:
            return WorkerOutput(
                metadata={"error": "sub_goal is required"}
            )
        
        # Check recursion depth
        current_depth = getattr(inputs, '_depth', 0)
        if current_depth >= self.max_depth:
            logger.warning(f"[{self.name}] Max recursion depth ({self.max_depth}) reached")
            return WorkerOutput(
                metadata={
                    "error": f"Recursion limit exceeded (depth={current_depth})",
                    "max_depth": self.max_depth
                }
            )
        
        # Format the sub-goal
        sub_goal = self.goal_template.format(
            sub_goal=inputs.sub_goal,
            main_goal=state.goal,
            context=inputs.context or ""
        )
        
        logger.info(f"[{self.name}] Starting sub-orchestration (depth={current_depth+1}): {sub_goal[:50]}...")
        
        # Create sub-orchestrator
        sub_orchestrator = Orchestrator(
            llm=self.llm,
            workers=self.sub_workers,
            verbose=self.verbose,
            enable_parallel=True
        )
        
        # Run with isolated blackboard
        max_steps = inputs.max_steps or self.max_sub_steps
        sub_result = await sub_orchestrator.run(
            goal=sub_goal,
            max_steps=max_steps
        )
        
        logger.info(f"[{self.name}] Sub-orchestration complete: {sub_result.status}")
        
        # Return consolidated result
        return self._consolidate_result(sub_result)
    
    def _consolidate_result(self, sub_blackboard: Blackboard) -> WorkerOutput:
        """
        Consolidate sub-orchestrator results into a single output.
        
        By default, returns the last artifact from the sub-blackboard.
        Override for custom consolidation logic.
        """
        # Get the final artifact
        final_artifact = sub_blackboard.get_last_artifact()
        
        if final_artifact:
            # Create a new artifact attributed to this worker
            return WorkerOutput(
                artifact=Artifact(
                    type=final_artifact.type,
                    content=final_artifact.content,
                    creator=self.name,
                    metadata={
                        "sub_orchestrator": True,
                        "sub_goal": sub_blackboard.goal,
                        "sub_steps": sub_blackboard.step_count,
                        "sub_status": sub_blackboard.status.value,
                        "original_creator": final_artifact.creator
                    }
                ),
                metadata={
                    "sub_artifacts_count": len(sub_blackboard.artifacts),
                    "sub_feedback_count": len(sub_blackboard.feedback)
                }
            )
        
        # No artifact - return summary as metadata
        return WorkerOutput(
            metadata={
                "sub_goal": sub_blackboard.goal,
                "sub_status": sub_blackboard.status.value,
                "sub_steps": sub_blackboard.step_count,
                "message": "Sub-orchestration completed without artifacts"
            }
        )


class DelegatorWorker(Worker):
    """
    A worker that can dynamically delegate to registered sub-teams.
    
    Unlike SubOrchestratorWorker which has fixed sub-workers,
    DelegatorWorker can route to different teams based on input.
    
    Args:
        name: Worker name
        description: Description for supervisor
        llm: LLM client
        teams: Dict mapping team names to their worker lists
        
    Example:
        delegator = DelegatorWorker(
            name="TaskRouter",
            description="Routes tasks to specialized teams",
            llm=my_llm,
            teams={
                "research": [WebScraper(), Summarizer()],
                "coding": [CodeGenerator(), CodeReviewer()],
                "writing": [Writer(), Editor()]
            }
        )
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: "LLMClient",
        teams: Dict[str, List[Worker]],
        max_depth: int = DEFAULT_MAX_DEPTH
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.teams = teams
        self.max_depth = max_depth
        self._current_depth = 0
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        from .core import Orchestrator
        
        if not inputs:
            return WorkerOutput(
                metadata={"error": "inputs required with 'team' and 'task' fields"}
            )
        
        # Check recursion depth
        depth = getattr(inputs, '_depth', self._current_depth)
        if depth >= self.max_depth:
            logger.warning(f"[{self.name}] Max recursion depth ({self.max_depth}) reached")
            return WorkerOutput(
                metadata={
                    "error": f"Recursion limit exceeded (depth={depth})",
                    "max_depth": self.max_depth
                }
            )
        
        # Get team and task from inputs
        team_name = getattr(inputs, 'team', None) or inputs.__dict__.get('team')
        task = getattr(inputs, 'task', None) or inputs.__dict__.get('task', inputs.instructions)
        
        if not team_name or team_name not in self.teams:
            return WorkerOutput(
                metadata={
                    "error": f"Unknown team: {team_name}",
                    "available_teams": list(self.teams.keys())
                }
            )
        
        logger.info(f"[{self.name}] Delegating to team '{team_name}' (depth={depth+1})")
        
        # Create and run sub-orchestrator for the team
        sub_orchestrator = Orchestrator(
            llm=self.llm,
            workers=self.teams[team_name]
        )
        
        result = await sub_orchestrator.run(goal=task, max_steps=10)
        
        # Return last artifact
        artifact = result.get_last_artifact()
        if artifact:
            return WorkerOutput(
                artifact=Artifact(
                    type=artifact.type,
                    content=artifact.content,
                    creator=self.name,
                    metadata={"team": team_name, "original_creator": artifact.creator}
                )
            )
        
        return WorkerOutput(metadata={"team": team_name, "status": result.status.value})
