"""
Unit tests for the Map-Reduce pattern module.

Tests cover parallel execution, conflict detection, and result aggregation.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass

from blackboard import Blackboard, Artifact, WorkerOutput
from blackboard.map_reduce import (
    MapResult,
    MapReduceResult,
    MutationConflict,
    ConflictResolution,
    run_map_reduce,
    MapReduceWorker,
)
from blackboard.patching import SearchReplacePatch, ArtifactMutation
from blackboard.protocols import Worker, WorkerInput


# =============================================================================
# Mock Workers for Testing
# =============================================================================

class SimpleWorker:
    """A simple worker that returns an artifact."""
    name = "SimpleWorker"
    description = "Returns a simple artifact"
    input_schema = None
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        return WorkerOutput(
            artifact=Artifact(
                type="result",
                content=f"Processed: {state.goal}",
                creator=self.name
            )
        )


class MutationWorker:
    """A worker that returns mutations instead of new artifacts."""
    name = "MutationWorker"
    description = "Returns mutations"
    input_schema = None
    
    def __init__(self, artifact_id: str):
        self._artifact_id = artifact_id
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        return WorkerOutput(
            mutations=[
                ArtifactMutation(
                    artifact_id=self._artifact_id,
                    patches=[SearchReplacePatch(
                        search="original",
                        replace="modified"
                    )]
                )
            ]
        )


class FailingWorker:
    """A worker that always fails."""
    name = "FailingWorker"
    description = "Always fails"
    input_schema = None
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        raise ValueError("Intentional failure")


class SlowWorker:
    """A worker that takes time to complete."""
    name = "SlowWorker"
    description = "Takes time"
    input_schema = None
    
    def __init__(self, delay: float = 0.1):
        self._delay = delay
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        await asyncio.sleep(self._delay)
        return WorkerOutput(
            artifact=Artifact(
                type="result",
                content=f"Slow result for: {state.goal}",
                creator=self.name
            )
        )


# =============================================================================
# Test MapResult and MapReduceResult
# =============================================================================

class TestMapResult:
    """Tests for MapResult dataclass."""
    
    def test_basic_creation(self):
        """Test creating a successful MapResult."""
        result = MapResult(
            item="test.py",
            success=True,
            artifacts=[Artifact(type="code", content="test", creator="Worker")]
        )
        assert result.success
        assert result.item == "test.py"
        assert len(result.artifacts) == 1
    
    def test_failed_result(self):
        """Test creating a failed MapResult."""
        result = MapResult(
            item="bad.py",
            success=False,
            error="File not found"
        )
        assert not result.success
        assert result.error == "File not found"


class TestMapReduceResult:
    """Tests for MapReduceResult dataclass."""
    
    def test_auto_counts(self):
        """Test that counts are calculated automatically."""
        results = [
            MapResult(item="a", success=True),
            MapResult(item="b", success=True),
            MapResult(item="c", success=False, error="Failed"),
        ]
        mr_result = MapReduceResult(success=False, results=results)
        
        assert mr_result.total_items == 3
        assert mr_result.successful_items == 2
        assert mr_result.failed_items == 1
    
    def test_has_conflicts(self):
        """Test conflict detection method."""
        result = MapReduceResult(
            success=False,
            results=[],
            conflicts=[MutationConflict(artifact_id="abc", conflicting_items=["a", "b"])]
        )
        assert result.has_conflicts()
    
    def test_get_non_conflicting_mutations(self):
        """Test filtering out conflicting mutations."""
        mutation1 = ArtifactMutation(artifact_id="abc", patches=[])
        mutation2 = ArtifactMutation(artifact_id="xyz", patches=[])
        
        result = MapReduceResult(
            success=False,
            results=[],
            all_mutations=[mutation1, mutation2],
            conflicts=[MutationConflict(artifact_id="abc", conflicting_items=["a", "b"])]
        )
        
        non_conflicting = result.get_non_conflicting_mutations()
        assert len(non_conflicting) == 1
        assert non_conflicting[0].artifact_id == "xyz"


# =============================================================================
# Test run_map_reduce
# =============================================================================

class TestRunMapReduce:
    """Tests for the run_map_reduce function."""
    
    @pytest.mark.asyncio
    async def test_empty_items(self):
        """Test with empty items list."""
        state = Blackboard(goal="Test")
        result = await run_map_reduce(
            items=[],
            worker=SimpleWorker(),
            parent_state=state
        )
        assert result.success
        assert result.total_items == 0
    
    @pytest.mark.asyncio
    async def test_single_item(self):
        """Test processing a single item."""
        state = Blackboard(goal="Parent goal")
        result = await run_map_reduce(
            items=["item1"],
            worker=SimpleWorker(),
            parent_state=state,
            item_to_goal=lambda x: f"Process {x}"
        )
        
        assert result.success
        assert result.total_items == 1
        assert result.successful_items == 1
        assert len(result.all_artifacts) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_items_parallel(self):
        """Test processing multiple items in parallel."""
        state = Blackboard(goal="Parent goal")
        items = ["file1.py", "file2.py", "file3.py"]
        
        result = await run_map_reduce(
            items=items,
            worker=SimpleWorker(),
            parent_state=state,
            max_concurrency=3,
            item_to_goal=lambda x: f"Process {x}"
        )
        
        assert result.success
        assert result.total_items == 3
        assert result.successful_items == 3
        assert len(result.all_artifacts) == 3
    
    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """Test that partial failure is handled gracefully."""
        state = Blackboard(goal="Test")
        
        # Create a worker that fails on specific items
        class SelectiveWorker:
            name = "SelectiveWorker"
            description = "Fails on 'bad' items"
            input_schema = None
            
            async def run(self, state, inputs=None):
                if "bad" in state.goal:
                    raise ValueError("Bad item!")
                return WorkerOutput(
                    artifact=Artifact(type="result", content="OK", creator=self.name)
                )
        
        result = await run_map_reduce(
            items=["good1", "bad1", "good2"],
            worker=SelectiveWorker(),
            parent_state=state,
            item_to_goal=lambda x: x
        )
        
        assert not result.success  # Overall failure due to partial
        assert result.total_items == 3
        assert result.successful_items == 2
        assert result.failed_items == 1
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that max_concurrency is respected."""
        state = Blackboard(goal="Test")
        
        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()
        
        class CountingWorker:
            name = "CountingWorker"
            description = "Counts concurrency"
            input_schema = None
            
            async def run(self, s, inputs=None):
                nonlocal concurrent_count, max_concurrent
                async with lock:
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                
                await asyncio.sleep(0.05)
                
                async with lock:
                    concurrent_count -= 1
                
                return WorkerOutput(
                    artifact=Artifact(type="result", content="OK", creator=self.name)
                )
        
        result = await run_map_reduce(
            items=list(range(10)),
            worker=CountingWorker(),
            parent_state=state,
            max_concurrency=3
        )
        
        assert result.success
        assert max_concurrent <= 3  # Never exceeded limit
    
    @pytest.mark.asyncio
    async def test_timeout_per_item(self):
        """Test that timeout_per_item works."""
        state = Blackboard(goal="Test")
        
        result = await run_map_reduce(
            items=["item1"],
            worker=SlowWorker(delay=0.5),
            parent_state=state,
            timeout_per_item=0.1
        )
        
        assert not result.success
        assert result.failed_items == 1
        assert "Timeout" in result.results[0].error


# =============================================================================
# Test Conflict Detection and Resolution
# =============================================================================

class TestConflictDetection:
    """Tests for mutation conflict detection."""
    
    @pytest.mark.asyncio
    async def test_no_conflicts(self):
        """Test when there are no conflicts."""
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="code", content="original", creator="Test"))
        artifact_id = state.artifacts[0].id
        
        # Each item targets a different artifact
        class UniqueTargetWorker:
            name = "UniqueTargetWorker"
            description = "Unique targets"
            input_schema = None
            
            async def run(self, s, inputs=None):
                unique_id = f"artifact-{s.goal}"
                return WorkerOutput(
                    mutations=[ArtifactMutation(
                        artifact_id=unique_id,
                        patches=[SearchReplacePatch(search="a", replace="b")]
                    )]
                )
        
        result = await run_map_reduce(
            items=["item1", "item2"],
            worker=UniqueTargetWorker(),
            parent_state=state,
            item_to_goal=lambda x: x
        )
        
        assert result.success
        assert not result.has_conflicts()
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        """Test that conflicts are detected when multiple items target same artifact."""
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="code", content="original", creator="Test"))
        artifact_id = state.artifacts[0].id
        
        # All items target the same artifact
        class ConflictingWorker:
            name = "ConflictingWorker"
            description = "Creates conflicts"
            input_schema = None
            
            def __init__(self, target_id):
                self._target_id = target_id
            
            async def run(self, s, inputs=None):
                return WorkerOutput(
                    mutations=[ArtifactMutation(
                        artifact_id=self._target_id,
                        patches=[SearchReplacePatch(search="x", replace="y")]
                    )]
                )
        
        result = await run_map_reduce(
            items=["item1", "item2", "item3"],
            worker=ConflictingWorker(artifact_id),
            parent_state=state,
            conflict_resolution=ConflictResolution.FAIL
        )
        
        assert not result.success
        assert result.has_conflicts()
        assert len(result.conflicts) == 1
        assert result.conflicts[0].artifact_id == artifact_id
        assert len(result.conflicts[0].conflicting_items) == 3
    
    @pytest.mark.asyncio
    async def test_first_wins_resolution(self):
        """Test FIRST_WINS conflict resolution."""
        state = Blackboard(goal="Test")
        artifact_id = "shared-artifact"
        
        class OrderedWorker:
            name = "OrderedWorker"
            description = "Creates ordered mutations"
            input_schema = None
            
            def __init__(self, target_id):
                self._target_id = target_id
            
            async def run(self, s, inputs=None):
                return WorkerOutput(
                    mutations=[ArtifactMutation(
                        artifact_id=self._target_id,
                        patches=[SearchReplacePatch(
                            search="x",
                            replace=f"from_{s.goal}"
                        )]
                    )]
                )
        
        result = await run_map_reduce(
            items=["first", "second", "third"],
            worker=OrderedWorker(artifact_id),
            parent_state=state,
            item_to_goal=lambda x: x,
            conflict_resolution=ConflictResolution.FIRST_WINS
        )
        
        # Should succeed with FIRST_WINS
        assert result.success
        assert len(result.all_mutations) == 1
        assert "from_first" in result.all_mutations[0].patches[0].replace


# =============================================================================
# Test MapReduceWorker
# =============================================================================

class TestMapReduceWorker:
    """Tests for the MapReduceWorker class."""
    
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Test basic MapReduceWorker execution."""
        state = Blackboard(goal="Process files")
        state.add_artifact(Artifact(type="file", content="file1.py", creator="Test"))
        state.add_artifact(Artifact(type="file", content="file2.py", creator="Test"))
        
        mr_worker = MapReduceWorker(
            name="FileProcessor",
            description="Processes all files in parallel",
            inner_worker=SimpleWorker(),
            items_extractor=lambda s: [a.content for a in s.artifacts if a.type == "file"],
            max_concurrency=2,
            item_to_goal=lambda x: f"Process {x}"
        )
        
        output = await mr_worker.run(state)
        
        assert output.artifact is not None
        assert "Map-Reduce Results" in output.artifact.content
        assert "Total items: 2" in output.artifact.content
    
    @pytest.mark.asyncio
    async def test_empty_items(self):
        """Test MapReduceWorker with no items."""
        state = Blackboard(goal="Process files")
        # No artifacts
        
        mr_worker = MapReduceWorker(
            name="FileProcessor",
            description="Processes all files",
            inner_worker=SimpleWorker(),
            items_extractor=lambda s: [a for a in s.artifacts if a.type == "file"]
        )
        
        output = await mr_worker.run(state)
        
        assert output.feedback is not None
        assert "No items" in output.feedback.critique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
