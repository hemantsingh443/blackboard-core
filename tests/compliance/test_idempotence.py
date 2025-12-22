"""
Idempotence Compliance Tests

These tests prove that the Blackboard SDK guarantees idempotent execution:
- Running N steps, serializing, and resuming produces the same result as running N+M steps continuously
- State hash is preserved across save/load cycles
- No "drift" or hallucination on resume

This is critical for:
- Spot instances / preemptible VMs
- Serverless functions (AWS Lambda)  
- Long-running agents with checkpointing
"""

import pytest
import asyncio
import json
import tempfile
import hashlib
from pathlib import Path
from typing import List, Optional

from blackboard import (
    Orchestrator, Worker, WorkerOutput, WorkerInput,
    Artifact, Blackboard, Status
)
from blackboard.protocols import Worker as WorkerProtocol


# =============================================================================
# Deterministic Mock LLM
# =============================================================================

class DeterministicLLM:
    """
    A mock LLM that returns a fixed sequence of responses.
    
    This ensures tests are reproducible - the same sequence of
    decisions will be made regardless of when the test runs.
    """
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_index = 0
        self.call_log: List[str] = []
    
    def generate(self, prompt: str) -> str:
        self.call_log.append(prompt[:100])  # Log first 100 chars
        
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        
        # Default: mark as done
        return '{"action": "done", "reasoning": "No more predefined responses"}'
    
    def reset(self):
        """Reset the LLM to start from the beginning."""
        self.call_index = 0
        self.call_log = []


# =============================================================================
# Test Workers
# =============================================================================

class CounterWorker(Worker):
    """A worker that appends a numbered artifact."""
    name = "Counter"
    description = "Appends a numbered artifact"
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        count = len(state.artifacts) + 1
        return WorkerOutput(
            artifact=Artifact(
                type="count",
                content=f"Artifact #{count}",
                creator=self.name
            )
        )


class AccumulatorWorker(Worker):
    """A worker that builds on previous artifacts."""
    name = "Accumulator"
    description = "Builds on previous work"
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        previous = state.get_last_artifact()
        if previous:
            content = f"{previous.content} + new"
        else:
            content = "Start"
        
        return WorkerOutput(
            artifact=Artifact(
                type="accumulated",
                content=content,
                creator=self.name
            )
        )


# =============================================================================
# Helper Functions
# =============================================================================

def compute_state_hash(state: Blackboard) -> str:
    """Compute a deterministic hash of the state for comparison."""
    # Serialize to JSON and hash
    # Exclude timestamps which may vary
    data = state.model_dump(exclude={"created_at", "updated_at"})
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def make_llm_sequence(n_steps: int, worker_name: str = "Counter") -> List[str]:
    """Generate a sequence of LLM responses for n steps."""
    responses = []
    for i in range(n_steps):
        responses.append(json.dumps({
            "action": "call",
            "worker": worker_name,
            "instructions": f"Execute step {i+1}",
            "reasoning": f"Performing step {i+1} of {n_steps}"
        }))
    # Final done response
    responses.append(json.dumps({
        "action": "done",
        "reasoning": "All steps completed"
    }))
    return responses


# =============================================================================
# Idempotence Tests
# =============================================================================

class TestOrchestratorIdempotence:
    """
    Tests that prove the orchestrator is idempotent.
    
    Key test: Run 5 steps, serialize, resume for 1 step == Run 6 steps continuously
    """
    
    @pytest.mark.asyncio
    async def test_interrupted_run_matches_continuous_run(self):
        """
        THE CORE IDEMPOTENCE TEST
        
        Mechanism:
        1. Run orchestrator for 5 steps with deterministic LLM
        2. Serialize state to JSON
        3. Create fresh orchestrator, load state
        4. Run for 1 more step  
        5. Compare final state against uninterrupted 6-step run
        
        This proves: pause/resume doesn't cause "drift"
        """
        # Create deterministic response sequence for 6 steps
        responses = make_llm_sequence(6)
        
        # --- INTERRUPTED RUN (5 + 1) ---
        llm_interrupted = DeterministicLLM(responses.copy())
        orch_interrupted = Orchestrator(
            llm=llm_interrupted,
            workers=[CounterWorker()]
        )
        
        # Run for 5 steps
        state_interrupted = await orch_interrupted.run(
            goal="Count to 6",
            max_steps=5
        )
        
        # Serialize
        state_json = state_interrupted.model_dump_json()
        
        # Create FRESH orchestrator and load state
        llm_resumed = DeterministicLLM(responses.copy())
        llm_resumed.call_index = 5  # Skip to where we left off
        
        orch_resumed = Orchestrator(
            llm=llm_resumed,
            workers=[CounterWorker()]
        )
        
        loaded_state = Blackboard.model_validate_json(state_json)
        
        # Run for 1 more step
        final_interrupted = await orch_resumed.run(
            state=loaded_state,
            max_steps=2  # 1 step + done
        )
        
        # --- CONTINUOUS RUN (6 steps) ---
        llm_continuous = DeterministicLLM(responses.copy())
        orch_continuous = Orchestrator(
            llm=llm_continuous,
            workers=[CounterWorker()]
        )
        
        final_continuous = await orch_continuous.run(
            goal="Count to 6",
            max_steps=7
        )
        
        # --- COMPARE ---
        # Same number of artifacts
        assert len(final_interrupted.artifacts) == len(final_continuous.artifacts), \
            f"Artifact count mismatch: {len(final_interrupted.artifacts)} vs {len(final_continuous.artifacts)}"
        
        # Same artifact contents
        for i, (a1, a2) in enumerate(zip(final_interrupted.artifacts, final_continuous.artifacts)):
            assert a1.content == a2.content, \
                f"Artifact {i} content mismatch: '{a1.content}' vs '{a2.content}'"
        
        # Same final status
        assert final_interrupted.status == final_continuous.status
    
    @pytest.mark.asyncio
    async def test_state_hash_preserved_across_save_load(self):
        """
        State hash before save equals state hash after load.
        
        This ensures no data is lost or corrupted during serialization.
        """
        # Create state with some content
        state = Blackboard(goal="Hash test")
        state.add_artifact(Artifact(type="test", content="content1", creator="Worker1"))
        state.add_artifact(Artifact(type="test", content="content2", creator="Worker2"))
        state.increment_step()
        state.increment_step()
        state.metadata["custom"] = {"key": "value", "list": [1, 2, 3]}
        
        # Compute hash before save
        hash_before = compute_state_hash(state)
        
        # Save and load
        json_str = state.model_dump_json()
        loaded = Blackboard.model_validate_json(json_str)
        
        # Compute hash after load
        hash_after = compute_state_hash(loaded)
        
        assert hash_before == hash_after, \
            f"State hash changed: {hash_before} -> {hash_after}"
    
    @pytest.mark.asyncio
    async def test_step_count_continues_correctly(self):
        """
        Step count continues from where it left off after resume.
        """
        responses = make_llm_sequence(3)
        
        # Run 2 steps
        llm = DeterministicLLM(responses.copy())
        orch = Orchestrator(llm=llm, workers=[CounterWorker()])
        state = await orch.run(goal="Test", max_steps=2)
        
        initial_step_count = state.step_count
        
        # Save and load
        loaded = Blackboard.model_validate_json(state.model_dump_json())
        
        assert loaded.step_count == initial_step_count
        
        # Resume
        llm_resumed = DeterministicLLM(responses.copy())
        llm_resumed.call_index = 2
        orch_resumed = Orchestrator(llm=llm_resumed, workers=[CounterWorker()])
        
        final = await orch_resumed.run(state=loaded, max_steps=2)
        
        # Step count should have increased
        assert final.step_count > initial_step_count
    
    @pytest.mark.asyncio
    async def test_file_based_persistence(self):
        """
        Test save to file and load from file produces identical results.
        """
        responses = make_llm_sequence(3)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Run and save to file
            llm = DeterministicLLM(responses.copy())
            orch = Orchestrator(llm=llm, workers=[CounterWorker()])
            state = await orch.run(goal="File test", max_steps=2)
            
            state.save_to_json(temp_path)
            
            # Load from file
            loaded = Blackboard.load_from_json(temp_path)
            
            # Verify
            assert loaded.goal == state.goal
            assert loaded.step_count == state.step_count
            assert len(loaded.artifacts) == len(state.artifacts)
            
            for a1, a2 in zip(loaded.artifacts, state.artifacts):
                assert a1.content == a2.content
                assert a1.id == a2.id
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_accumulator_idempotence(self):
        """
        Test with a worker that builds on previous state.
        
        This is a more realistic test - the Accumulator worker
        reads previous artifacts to build its output.
        """
        responses = make_llm_sequence(4, worker_name="Accumulator")
        
        # --- INTERRUPTED RUN (2 + 2) ---
        llm1 = DeterministicLLM(responses.copy())
        orch1 = Orchestrator(llm=llm1, workers=[AccumulatorWorker()])
        state1 = await orch1.run(goal="Accumulate", max_steps=2)
        
        json_str = state1.model_dump_json()
        
        # Resume
        llm2 = DeterministicLLM(responses.copy())
        llm2.call_index = 2
        orch2 = Orchestrator(llm=llm2, workers=[AccumulatorWorker()])
        loaded = Blackboard.model_validate_json(json_str)
        final_interrupted = await orch2.run(state=loaded, max_steps=3)
        
        # --- CONTINUOUS RUN ---
        llm3 = DeterministicLLM(responses.copy())
        orch3 = Orchestrator(llm=llm3, workers=[AccumulatorWorker()])
        final_continuous = await orch3.run(goal="Accumulate", max_steps=5)
        
        # --- COMPARE ---
        # The key assertion: accumulated content should be identical
        assert len(final_interrupted.artifacts) == len(final_continuous.artifacts)
        
        for i, (a1, a2) in enumerate(zip(final_interrupted.artifacts, final_continuous.artifacts)):
            assert a1.content == a2.content, \
                f"Accumulated artifact {i} differs: '{a1.content}' vs '{a2.content}'"
