"""
Instruction Optimizer

Analyzes failed sessions and generates prompt patches to improve worker performance.

The optimizer:
1. Identifies steps where feedback.passed=False
2. Uses a meta-LLM to propose prompt improvements
3. Verifies candidates by forking sessions and re-running
4. Saves approved patches to a staging file for human review

Example:
    optimizer = Optimizer(llm, orchestrator)
    patches = await optimizer.run(session_id="session-001")
    optimizer.save_patches(patches, "blackboard.patches.json")
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LLMClient, Orchestrator
    from .state import Blackboard

logger = logging.getLogger("blackboard.optimize")


@dataclass
class FailureContext:
    """Context for a failed step."""
    session_id: str
    step_index: int
    worker_name: str
    instructions: str
    artifact_content: Optional[str]
    feedback_content: str
    feedback_reasoning: str


@dataclass
class PromptPatch:
    """A proposed prompt modification."""
    worker_name: str
    prompt_key: str
    original: str
    original_hash: str  # SHA256 for conflict detection
    proposed: str
    reasoning: str
    verification_score: float = 0.0
    verified: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptPatch":
        """Create from dictionary."""
        return cls(**data)


def hash_prompt(prompt: str) -> str:
    """Generate SHA256 hash of a prompt for conflict detection."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class Optimizer:
    """
    Analyzes failures and generates improved prompts.
    
    The optimizer follows this workflow:
    1. Load session history and find failures
    2. Extract context around each failure
    3. Use meta-LLM to generate improved prompts
    4. Fork session and verify improvements
    5. Save verified patches for human review
    
    Args:
        llm: LLM client for meta-prompting
        orchestrator: Orchestrator with persistence for forking
        
    Example:
        optimizer = Optimizer(llm, orchestrator)
        failures = await optimizer.analyze_failures("session-001")
        candidates = await optimizer.generate_candidates(failures[0], n=3)
        verified = await optimizer.verify_candidate(candidates[0], failures[0])
    """
    
    META_PROMPT = '''You are a prompt engineer analyzing why a worker failed.

## Worker
Name: {worker_name}
Current Prompt: {current_prompt}

## What Was Requested
{instructions}

## What The Worker Produced
{artifact}

## Feedback (FAILED)
{feedback}

## Your Task
Propose an improved prompt that would have avoided this failure.
Focus on:
1. Clearer constraints
2. Better examples
3. Edge case handling
4. More specific output format

Respond with JSON:
```json
{{
    "reasoning": "Why the current prompt failed",
    "proposed_prompt": "The full improved prompt text"
}}
```'''
    
    def __init__(
        self,
        llm: "LLMClient",
        orchestrator: Optional["Orchestrator"] = None
    ):
        self.llm = llm
        self.orchestrator = orchestrator
    
    async def analyze_failures(
        self,
        session_id: str,
        persistence = None
    ) -> List[FailureContext]:
        """
        Find all steps where feedback.passed=False.
        
        Args:
            session_id: Session to analyze
            persistence: Optional persistence layer (uses orchestrator's if not provided)
            
        Returns:
            List of FailureContext objects for each failure
        """
        persistence = persistence or (self.orchestrator.persistence if self.orchestrator else None)
        if not persistence:
            raise ValueError("Persistence layer required for analysis")
        
        # Load current session state
        state = await persistence.load(session_id)
        
        failures = []
        
        # Check history for failed feedback
        for event in state.history:
            if event.get("type") == "feedback_received":
                payload = event.get("payload", {})
                if payload.get("passed") is False:
                    failures.append(FailureContext(
                        session_id=session_id,
                        step_index=event.get("step", 0),
                        worker_name=payload.get("worker", "Unknown"),
                        instructions=payload.get("instructions", ""),
                        artifact_content=payload.get("artifact_content", "")[:500] if payload.get("artifact_content") else None,
                        feedback_content=payload.get("content", ""),
                        feedback_reasoning=payload.get("reasoning", "")
                    ))
        
        logger.info(f"Found {len(failures)} failures in session {session_id}")
        return failures
    
    async def generate_candidates(
        self,
        failure: FailureContext,
        current_prompt: str = "",
        n: int = 3
    ) -> List[PromptPatch]:
        """
        Generate prompt improvement candidates using meta-LLM.
        
        Args:
            failure: Context of the failure to fix
            current_prompt: Current prompt text for the worker
            n: Number of candidates to generate
            
        Returns:
            List of PromptPatch proposals
        """
        candidates = []
        
        for i in range(n):
            prompt = self.META_PROMPT.format(
                worker_name=failure.worker_name,
                current_prompt=current_prompt or "(no explicit prompt)",
                instructions=failure.instructions,
                artifact=failure.artifact_content or "(no artifact)",
                feedback=failure.feedback_content
            )
            
            try:
                response = await self.llm.generate(prompt)
                data = self._parse_json_response(response.content if hasattr(response, 'content') else str(response))
                
                if data and "proposed_prompt" in data:
                    patch = PromptPatch(
                        worker_name=failure.worker_name,
                        prompt_key=failure.worker_name,  # Default to worker name
                        original=current_prompt,
                        original_hash=hash_prompt(current_prompt),
                        proposed=data["proposed_prompt"],
                        reasoning=data.get("reasoning", "")
                    )
                    candidates.append(patch)
            except Exception as e:
                logger.warning(f"Failed to generate candidate {i+1}: {e}")
        
        logger.info(f"Generated {len(candidates)} candidates for {failure.worker_name}")
        return candidates
    
    async def verify_candidate(
        self,
        patch: PromptPatch,
        failure: FailureContext
    ) -> PromptPatch:
        """
        Verify a patch by forking the session and re-running.
        
        Args:
            patch: The proposed prompt patch
            failure: The failure context to verify against
            
        Returns:
            Updated PromptPatch with verification results
        """
        if not self.orchestrator or not self.orchestrator.persistence:
            logger.warning("Cannot verify without orchestrator with persistence")
            return patch
        
        if not hasattr(self.orchestrator.persistence, 'load_state_at_step'):
            logger.warning("Cannot verify: persistence doesn't support checkpoints")
            return patch
        
        try:
            # Fork session at the step before the failure
            fork_step = max(1, failure.step_index - 1)
            fork_id = await self.orchestrator.fork_session(
                failure.session_id,
                fork_step,
                fork_suffix=f"verify_{hash_prompt(patch.proposed)[:8]}"
            )
            
            # Load forked state
            forked_state = await self.orchestrator.persistence.load(fork_id)
            
            # Apply patch to prompt registry if available
            if hasattr(self.orchestrator, 'prompt_registry') and self.orchestrator.prompt_registry:
                self.orchestrator.prompt_registry.set(patch.prompt_key, patch.proposed)
            
            # Run for limited steps
            try:
                result = await self.orchestrator.run(
                    state=forked_state,
                    max_steps=3  # Limited verification run
                )
                
                # Check if the same failure occurred
                # (simplified: check if any feedback passed)
                passed_feedback = [
                    f for f in result.feedback 
                    if f.worker == failure.worker_name and f.passed
                ]
                
                patch.verified = len(passed_feedback) > 0
                patch.verification_score = 1.0 if patch.verified else 0.0
                
            except Exception as e:
                logger.warning(f"Verification run failed: {e}")
                patch.verification_score = -1.0
                
        except Exception as e:
            logger.error(f"Failed to verify patch: {e}")
        
        return patch
    
    def save_patches(
        self,
        patches: List[PromptPatch],
        path: str = "blackboard.patches.json"
    ) -> None:
        """
        Save patches to staging file for human review.
        
        Args:
            patches: List of patches to save
            path: Path to patches file
        """
        file_path = Path(path)
        
        # Load existing patches
        existing = []
        if file_path.exists():
            try:
                existing = json.loads(file_path.read_text())
            except Exception:
                pass
        
        # Add new patches
        new_patches = [p.to_dict() for p in patches]
        all_patches = existing + new_patches
        
        # Save
        file_path.write_text(json.dumps(all_patches, indent=2))
        logger.info(f"Saved {len(new_patches)} patches to {path} (total: {len(all_patches)})")
    
    def load_patches(self, path: str = "blackboard.patches.json") -> List[PromptPatch]:
        """Load pending patches from staging file."""
        file_path = Path(path)
        
        if not file_path.exists():
            return []
        
        try:
            data = json.loads(file_path.read_text())
            return [PromptPatch.from_dict(p) for p in data]
        except Exception as e:
            logger.error(f"Failed to load patches: {e}")
            return []
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        import re
        
        # Try code block first
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try raw JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None


async def run_optimization(
    session_id: str,
    llm: "LLMClient",
    orchestrator: "Orchestrator",
    output_path: str = "blackboard.patches.json"
) -> List[PromptPatch]:
    """
    Convenience function to run full optimization workflow.
    
    Args:
        session_id: Session to optimize
        llm: LLM client for meta-prompting
        orchestrator: Orchestrator with persistence
        output_path: Where to save patches
        
    Returns:
        List of verified patches
    """
    optimizer = Optimizer(llm, orchestrator)
    
    # 1. Find failures
    failures = await optimizer.analyze_failures(session_id)
    if not failures:
        logger.info("No failures found to optimize")
        return []
    
    all_patches = []
    
    for failure in failures:
        # 2. Generate candidates
        candidates = await optimizer.generate_candidates(failure, n=2)
        
        # 3. Verify each
        for patch in candidates:
            verified = await optimizer.verify_candidate(patch, failure)
            if verified.verified:
                all_patches.append(verified)
                break  # Use first verified patch
    
    # 4. Save to staging
    if all_patches:
        optimizer.save_patches(all_patches, output_path)
    
    return all_patches
