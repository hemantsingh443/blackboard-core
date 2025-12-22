"""
Blueprints - Structured Workflows

Provides a State Machine layer to constrain the Orchestrator for Standard Operating Procedures (SOPs).

A Blueprint defines a sequence of Steps. Each Step specifies:
- Which workers are allowed
- Instructions for the LLM (injected into the system prompt)
- Exit conditions to advance to the next step

Example:
    from blackboard.flow import Blueprint, Step
    
    # Force a sequence: Research -> Outline -> Write -> Review
    blog_flow = Blueprint(
        name="Blog Writing",
        steps=[
            Step(
                name="research",
                description="Gather information",
                allowed_workers=["WebSearch", "Browser"],
                instructions="Focus ONLY on gathering information. Do not write yet."
            ),
            Step(
                name="outline",
                description="Create structure",
                allowed_workers=["Planner"],
                instructions="Create a detailed outline based on research."
            ),
            Step(
                name="write",
                description="Draft content",
                allowed_workers=["Writer"],
                instructions="Write the full content following the outline."
            ),
            Step(
                name="review",
                description="Quality check",
                allowed_workers=["Critic"],
                instructions="Review the draft and provide feedback.",
                exit_condition=lambda state: (
                    state.get_latest_feedback() and 
                    state.get_latest_feedback().passed
                )
            )
        ]
    )
    
    # Use with orchestrator
    result = await orchestrator.run(goal="Write a blog post", blueprint=blog_flow)

Simplified Patterns:
    from blackboard.flow import SequentialPipeline, Router
    
    # One-liner sequential execution
    pipeline = SequentialPipeline([Searcher(), Writer(), Critic()])
    
    # Supervisor chooses best worker
    router = Router([MathAgent, CodeAgent, ResearchAgent])
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from .state import Blackboard, Status

if TYPE_CHECKING:
    from .protocols import Worker

logger = logging.getLogger("blackboard.flow")



# =============================================================================
# Step Definition
# =============================================================================

@dataclass
class Step:
    """
    A single phase in a Blueprint workflow.
    
    Attributes:
        name: Unique identifier for the step (e.g., "research", "write")
        description: Human-readable description of this phase
        allowed_workers: List of worker names that can be called in this step.
                        Empty list means all workers are allowed.
        instructions: Text injected into the Supervisor system prompt.
                     Use this to guide LLM behavior for this phase.
        exit_condition: Optional callable that receives the Blackboard and returns
                       True if the step should advance. If None, step advances
                       after any worker completes successfully.
        max_iterations: Maximum number of worker calls in this step before
                       forcing advancement (prevents infinite loops)
        metadata: Additional step-specific data
    
    Example:
        research_step = Step(
            name="research",
            description="Information Gathering Phase",
            allowed_workers=["WebSearch", "Browser"],
            instructions="Only search and gather information. Do NOT write content yet.",
            max_iterations=5
        )
    """
    
    name: str
    description: str = ""
    allowed_workers: List[str] = field(default_factory=list)
    instructions: str = ""
    exit_condition: Optional[Callable[[Blackboard], bool]] = None
    max_iterations: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_advance(self, state: Blackboard) -> bool:
        """
        Check if the step should advance to the next phase.
        
        Returns True if:
        - exit_condition is defined and returns True
        - max_iterations reached
        - No exit_condition and at least one artifact was created
        """
        # Check iteration limit
        step_iterations = state.metadata.get(f"_step_{self.name}_iterations", 0)
        if step_iterations >= self.max_iterations:
            logger.warning(f"Step '{self.name}' hit max iterations ({self.max_iterations})")
            return True
        
        # Check custom exit condition
        if self.exit_condition is not None:
            try:
                return self.exit_condition(state)
            except Exception as e:
                logger.error(f"Exit condition error in step '{self.name}': {e}")
                return False
        
        # Default: advance if any artifact was created this step
        # This requires tracking which artifacts belong to this step
        return False
    
    def to_prompt_context(self) -> str:
        """Generate the context string to inject into the system prompt."""
        parts = [f"\n## Current Workflow Phase: {self.name}"]
        
        if self.description:
            parts.append(f"**Phase Description**: {self.description}")
        
        if self.instructions:
            parts.append(f"\n{self.instructions}")
        
        if self.allowed_workers:
            parts.append(f"\n**Available Workers in this Phase**: {', '.join(self.allowed_workers)}")
            parts.append("You may ONLY call workers from this list. Other workers are not available.")
        
        return "\n".join(parts)


# =============================================================================
# Blueprint Definition
# =============================================================================

@dataclass
class Blueprint:
    """
    A structured workflow definition with sequential steps.
    
    Blueprints constrain the Orchestrator by:
    1. Limiting available workers based on the current step
    2. Injecting phase-specific instructions into the LLM prompt
    3. Managing transitions between phases
    
    Attributes:
        name: Workflow name (e.g., "Blog Writing Pipeline")
        steps: Ordered list of Step objects
        allow_skip_to_done: If True, the orchestrator can call "done" from any step
        on_step_change: Optional callback when step advances
    
    Example:
        pipeline = Blueprint(
            name="Content Creation",
            steps=[
                Step(name="research", ...),
                Step(name="write", ...),
                Step(name="review", ...)
            ]
        )
    """
    
    name: str
    steps: List[Step]
    allow_skip_to_done: bool = True
    on_step_change: Optional[Callable[[str, str, Blackboard], None]] = None
    
    def __post_init__(self):
        if not self.steps:
            raise ValueError("Blueprint must have at least one step")
        
        # Validate unique step names
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("All step names must be unique")
    
    def get_current_step(self, state: Blackboard) -> Step:
        """Get the current step based on state metadata."""
        index = state.metadata.get("_blueprint_step_index", 0)
        index = min(index, len(self.steps) - 1)
        return self.steps[index]
    
    def get_current_step_index(self, state: Blackboard) -> int:
        """Get the current step index."""
        return state.metadata.get("_blueprint_step_index", 0)
    
    def is_complete(self, state: Blackboard) -> bool:
        """Check if the blueprint has completed all steps."""
        index = self.get_current_step_index(state)
        return index >= len(self.steps)
    
    def advance_step(self, state: Blackboard) -> Optional[Step]:
        """
        Advance to the next step if the current step's exit condition is met.
        
        Returns the new current step, or None if no advancement occurred.
        """
        current_index = self.get_current_step_index(state)
        
        if current_index >= len(self.steps):
            return None  # Already complete
        
        current_step = self.steps[current_index]
        
        if current_step.should_advance(state):
            old_step_name = current_step.name
            new_index = current_index + 1
            state.metadata["_blueprint_step_index"] = new_index
            
            # Reset iteration counter for new step
            state.metadata[f"_step_{current_step.name}_iterations"] = 0
            
            if new_index < len(self.steps):
                new_step = self.steps[new_index]
                logger.info(f"Blueprint advancing: {old_step_name} -> {new_step.name}")
                
                # Call callback if provided
                if self.on_step_change:
                    try:
                        self.on_step_change(old_step_name, new_step.name, state)
                    except Exception as e:
                        logger.error(f"on_step_change callback error: {e}")
                
                return new_step
            else:
                logger.info(f"Blueprint complete: all {len(self.steps)} steps finished")
                return None
        
        return None
    
    def increment_step_iteration(self, state: Blackboard) -> int:
        """Increment and return the iteration counter for the current step."""
        current_step = self.get_current_step(state)
        key = f"_step_{current_step.name}_iterations"
        count = state.metadata.get(key, 0) + 1
        state.metadata[key] = count
        return count
    
    def filter_workers(self, all_workers: List[str], state: Blackboard) -> List[str]:
        """
        Filter the available workers based on the current step.
        
        Args:
            all_workers: List of all registered worker names
            state: Current blackboard state
            
        Returns:
            List of worker names allowed in the current step
        """
        current_step = self.get_current_step(state)
        
        if not current_step.allowed_workers:
            return all_workers  # Empty list means all allowed
        
        return [w for w in all_workers if w in current_step.allowed_workers]
    
    def get_prompt_context(self, state: Blackboard) -> str:
        """
        Generate the context string to append to the Supervisor system prompt.
        
        This provides the LLM with awareness of the current workflow phase.
        """
        current_step = self.get_current_step(state)
        step_index = self.get_current_step_index(state)
        
        # Build progress indicator
        progress = f"Step {step_index + 1} of {len(self.steps)}"
        steps_overview = " -> ".join([
            f"**{s.name}**" if i == step_index else s.name
            for i, s in enumerate(self.steps)
        ])
        
        context_parts = [
            f"\n\n{'='*50}",
            f"## WORKFLOW: {self.name}",
            f"**Progress**: {progress}",
            f"**Pipeline**: {steps_overview}",
            current_step.to_prompt_context(),
            f"{'='*50}\n"
        ]
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize blueprint to dictionary (for API responses)."""
        return {
            "name": self.name,
            "steps": [
                {
                    "name": s.name,
                    "description": s.description,
                    "allowed_workers": s.allowed_workers,
                    "max_iterations": s.max_iterations
                }
                for s in self.steps
            ],
            "allow_skip_to_done": self.allow_skip_to_done
        }


# =============================================================================
# Utility Functions
# =============================================================================

def simple_blueprint(
    name: str,
    steps: List[Dict[str, Any]]
) -> Blueprint:
    """
    Create a blueprint from a simple list of step dictionaries.
    
    This is a convenience function for quickly defining workflows.
    
    Example:
        pipeline = simple_blueprint("Research & Write", [
            {"name": "research", "workers": ["WebSearch"], "instructions": "Research only"},
            {"name": "write", "workers": ["Writer"], "instructions": "Write the content"},
        ])
    
    Args:
        name: Blueprint name
        steps: List of dicts with keys: name, workers (optional), instructions (optional)
        
    Returns:
        Configured Blueprint instance
    """
    step_objects = []
    
    for step_dict in steps:
        step_objects.append(Step(
            name=step_dict["name"],
            description=step_dict.get("description", ""),
            allowed_workers=step_dict.get("workers", []),
            instructions=step_dict.get("instructions", ""),
            max_iterations=step_dict.get("max_iterations", 10)
        ))
    
    return Blueprint(name=name, steps=step_objects)


# =============================================================================
# Pre-built Blueprint Templates
# =============================================================================

def research_and_write_blueprint(
    research_workers: List[str] = None,
    writing_workers: List[str] = None,
    review_workers: List[str] = None
) -> Blueprint:
    """
    Create a common Research -> Write -> Review blueprint.
    
    Args:
        research_workers: Workers for research phase (default: WebSearch, Browser)
        writing_workers: Workers for writing phase (default: Writer)
        review_workers: Workers for review phase (default: Critic)
        
    Returns:
        Configured Blueprint for content creation
    """
    return Blueprint(
        name="Research & Write Pipeline",
        steps=[
            Step(
                name="research",
                description="Gather information and data",
                allowed_workers=research_workers or ["WebSearch", "Browser"],
                instructions="""Focus ONLY on gathering information. 
Do not attempt to write the final content yet.
Search for relevant sources, visit pages, and collect data.
Continue until you have enough information to write.""",
                max_iterations=5
            ),
            Step(
                name="write",
                description="Create the content",
                allowed_workers=writing_workers or ["Writer"],
                instructions="""Now write the content based on your research.
Use the gathered information to create high-quality output.
Be thorough and well-structured.""",
                max_iterations=3
            ),
            Step(
                name="review",
                description="Quality check",
                allowed_workers=review_workers or ["Critic"],
                instructions="""Review the written content for quality.
Check for accuracy, completeness, and clarity.
If issues are found, the writer may need to revise.""",
                exit_condition=lambda state: (
                    state.get_latest_feedback() is not None and
                    state.get_latest_feedback().passed
                ),
                max_iterations=3
            )
        ]
    )


# =============================================================================
# Simplified Pattern Functions
# =============================================================================

def SequentialPipeline(
    workers: List["Worker"],
    name: str = "Sequential Pipeline"
) -> Blueprint:
    """
    Create a sequential pipeline where each worker runs in strict order.
    
    This is the simplest way to force A → B → C execution. Each worker
    gets exactly one step, and must complete before the next can run.
    
    Args:
        workers: List of Worker instances to execute in order
        name: Optional name for the blueprint
        
    Returns:
        A Blueprint that enforces sequential execution
        
    Example:
        from blackboard.flow import SequentialPipeline
        
        pipeline = SequentialPipeline([
            SearchWorker(),
            WriterWorker(),
            CriticWorker()
        ])
        
        result = await orchestrator.run(
            goal="Research and write an article",
            blueprint=pipeline
        )
    """
    steps = []
    for i, worker in enumerate(workers):
        steps.append(Step(
            name=f"step_{i+1}_{worker.name}",
            description=f"Execute {worker.name}",
            allowed_workers=[worker.name],
            instructions=f"You MUST call {worker.name}. This is step {i+1} of {len(workers)}.",
            max_iterations=1  # Force single execution per step
        ))
    
    return Blueprint(name=name, steps=steps)


def Router(
    workers: List["Worker"],
    name: str = "Router",
    selection_prompt: str = ""
) -> Blueprint:
    """
    Create a router where the supervisor chooses the best worker for the task.
    
    All workers are available in a single step. The supervisor analyzes
    the goal and selects the most appropriate worker.
    
    Args:
        workers: List of Worker instances to choose from
        name: Optional name for the blueprint
        selection_prompt: Optional additional instructions for selection
        
    Returns:
        A Blueprint with a single routing step
        
    Example:
        from blackboard.flow import Router
        
        router = Router([
            MathAgent(),
            CodeAgent(),
            ResearchAgent()
        ], selection_prompt="Choose based on the user's query type")
        
        result = await orchestrator.run(
            goal="Solve this equation: 2x + 5 = 15",
            blueprint=router
        )
    """
    worker_names = [w.name for w in workers]
    worker_list = ", ".join(worker_names)
    
    instructions = f"""Analyze the goal and choose the SINGLE best worker for this task.

Available specialists: {worker_list}

{selection_prompt}

Choose wisely - you can only pick ONE worker."""
    
    return Blueprint(
        name=name,
        steps=[Step(
            name="route",
            description="Select and execute the best worker for this task",
            allowed_workers=worker_names,
            instructions=instructions,
            max_iterations=1
        )]
    )
