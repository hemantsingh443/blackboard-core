"""
Test Fixtures for Blackboard SDK

Provides pre-configured test environments for unit testing.
"""

import uuid
from typing import Any, List, Optional

from ..core import Orchestrator, LLMClient
from ..state import Blackboard
from ..protocols import Worker
from ..persistence import InMemoryPersistence
from .mock_llm import MockLLMClient 


class OrchestratorTestFixture:
    """
    Pre-configured test environment for orchestrator testing.
    
    Provides:
    - In-memory persistence (no file I/O)
    - Mock LLM (no API calls)
    - Deterministic ID generation
    - Easy state inspection
    
    Example:
        from blackboard.testing import OrchestratorTestFixture, MockLLMClient
        
        def test_my_agent():
            mock_llm = MockLLMClient(sequence=[
                "<response><action>call</action>...",
                "<response><action>done</action>...",
            ])
            
            fixture = OrchestratorTestFixture(
                llm=mock_llm,
                workers=[my_worker]
            )
            
            result = await fixture.run("Test goal")
            
            assert result.status.value == "done"
            assert fixture.llm.get_call_count() == 2
    """
    
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        workers: Optional[List[Worker]] = None,
        use_deterministic_ids: bool = True,
        max_steps: int = 10,
        **orchestrator_kwargs
    ):
        """
        Initialize test fixture.
        
        Args:
            llm: LLM client (defaults to MockLLMClient with done response)
            workers: List of workers (defaults to empty list with dummy worker)
            use_deterministic_ids: If True, use predictable IDs for testing
            max_steps: Maximum steps before stopping (safety limit)
            **orchestrator_kwargs: Additional args passed to Orchestrator
        """
        self.llm = llm or MockLLMClient()
        self.workers = workers or [self._create_dummy_worker()]
        self.persistence = InMemoryPersistence()
        self.max_steps = max_steps
        
        # Deterministic ID generation
        self._id_counter = 0
        self._use_deterministic_ids = use_deterministic_ids
        
        # Create orchestrator
        self.orchestrator = Orchestrator(
            llm=self.llm,
            workers=self.workers,
            **orchestrator_kwargs
        )
        self.orchestrator.set_persistence(self.persistence)
        
        # Track results
        self.last_result: Optional[Blackboard] = None
    
    def _create_dummy_worker(self) -> Worker:
        """Create a minimal worker for testing."""
        from ..decorators import worker
        
        @worker(name="DummyWorker", description="A dummy worker for testing")
        def dummy_worker(instructions: str) -> str:
            return f"Processed: {instructions}"
        
        return dummy_worker
    
    def generate_id(self, prefix: str = "test") -> str:
        """Generate a deterministic or random ID."""
        if self._use_deterministic_ids:
            self._id_counter += 1
            return f"{prefix}_{self._id_counter:04d}"
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    async def run(
        self,
        goal: str,
        state: Optional[Blackboard] = None,
        max_steps: Optional[int] = None
    ) -> Blackboard:
        """
        Run the orchestrator with the given goal.
        
        Args:
            goal: The goal to accomplish
            state: Optional initial state to resume from
            max_steps: Override max steps (defaults to fixture setting)
            
        Returns:
            The final Blackboard state
        """
        steps = max_steps or self.max_steps
        
        # Set session ID for persistence
        if state is None:
            state = Blackboard(goal=goal)
        
        state.metadata["session_id"] = self.generate_id("session")
        
        self.last_result = await self.orchestrator.run(
            goal=goal,
            state=state,
            max_steps=steps
        )
        
        return self.last_result
    
    def get_call_history(self) -> List[dict]:
        """Get LLM call history (if using MockLLMClient)."""
        if hasattr(self.llm, 'get_call_history'):
            return self.llm.get_call_history()
        return []
    
    def reset(self) -> None:
        """Reset fixture state for new test."""
        self._id_counter = 0
        if hasattr(self.llm, 'reset'):
            self.llm.reset()
        # Clear in-memory persistence
        if hasattr(self.persistence, '_store'):
            self.persistence._store.clear()
        self.last_result = None
