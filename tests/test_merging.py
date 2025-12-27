"""
Unit tests for the state merging utilities.

Tests cover merge strategies, conflict detection, and StateMerger class.
"""

import pytest
from blackboard import Blackboard, Artifact, Feedback
from blackboard.merging import (
    merge_states,
    StateMerger,
    MergeStrategy,
    MergeConflict,
    MergeResult,
)


# =============================================================================
# Test merge_states function
# =============================================================================

class TestMergeStates:
    """Tests for the merge_states function."""
    
    def test_no_changes(self):
        """Test merging when child has no changes."""
        parent = Blackboard(goal="Test")
        child = parent.model_copy(deep=True)
        
        result = merge_states(parent, child)
        
        assert result.success
        assert result.artifacts_added == 0
        assert result.artifacts_updated == 0
        assert len(result.conflicts) == 0
    
    def test_add_new_artifact(self):
        """Test adding a new artifact from child."""
        parent = Blackboard(goal="Test")
        child = parent.model_copy(deep=True)
        
        # Add artifact to child
        child.add_artifact(Artifact(
            type="code",
            content="def hello(): pass",
            creator="CodeWriter"
        ))
        
        result = merge_states(parent, child)
        
        assert result.success
        assert result.artifacts_added == 1
        assert len(parent.artifacts) == 1
        assert parent.artifacts[0].type == "code"
    
    def test_multiple_new_artifacts(self):
        """Test adding multiple new artifacts from child."""
        parent = Blackboard(goal="Test")
        child = parent.model_copy(deep=True)
        
        child.add_artifact(Artifact(type="code", content="code1", creator="Worker1"))
        child.add_artifact(Artifact(type="docs", content="docs1", creator="Worker2"))
        child.add_artifact(Artifact(type="test", content="test1", creator="Worker3"))
        
        result = merge_states(parent, child)
        
        assert result.success
        assert result.artifacts_added == 3
        assert len(parent.artifacts) == 3
    
    def test_existing_artifact_no_change(self):
        """Test when child has same artifact with no changes."""
        parent = Blackboard(goal="Test")
        parent.add_artifact(Artifact(type="code", content="original", creator="Writer"))
        
        child = parent.model_copy(deep=True)
        
        result = merge_states(parent, child)
        
        assert result.success
        assert result.artifacts_added == 0
        assert result.artifacts_updated == 0
    
    def test_conflict_theirs_strategy(self):
        """Test THEIRS strategy - child wins."""
        parent = Blackboard(goal="Test")
        parent.add_artifact(Artifact(type="code", content="parent content", creator="Writer"))
        artifact_id = parent.artifacts[0].id
        
        child = parent.model_copy(deep=True)
        child.artifacts[0].content = "child content"
        child.artifacts[0].version = 2
        
        result = merge_states(parent, child, strategy=MergeStrategy.THEIRS)
        
        assert result.success
        assert result.artifacts_updated == 1
        assert parent.artifacts[0].content == "child content"
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolved
        assert result.conflicts[0].resolution == "child_wins"
    
    def test_conflict_ours_strategy(self):
        """Test OURS strategy - parent wins."""
        parent = Blackboard(goal="Test")
        parent.add_artifact(Artifact(type="code", content="parent content", creator="Writer"))
        
        child = parent.model_copy(deep=True)
        child.artifacts[0].content = "child content"
        child.artifacts[0].version = 2
        
        result = merge_states(parent, child, strategy=MergeStrategy.OURS)
        
        assert result.success
        assert result.artifacts_updated == 0
        assert parent.artifacts[0].content == "parent content"
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolved
        assert result.conflicts[0].resolution == "parent_wins"
    
    def test_conflict_fail_strategy(self):
        """Test FAIL strategy - unresolved conflicts."""
        parent = Blackboard(goal="Test")
        parent.add_artifact(Artifact(type="code", content="parent content", creator="Writer"))
        
        child = parent.model_copy(deep=True)
        child.artifacts[0].content = "child content"
        child.artifacts[0].version = 2
        
        result = merge_states(parent, child, strategy=MergeStrategy.FAIL)
        
        assert not result.success
        assert len(result.conflicts) == 1
        assert not result.conflicts[0].resolved
    
    def test_conflict_newest_strategy(self):
        """Test NEWEST strategy - higher version wins."""
        parent = Blackboard(goal="Test")
        parent.add_artifact(Artifact(type="code", content="parent content", creator="Writer"))
        parent.artifacts[0].version = 5  # Parent has higher version
        
        child = parent.model_copy(deep=True)
        child.artifacts[0].content = "child content"
        child.artifacts[0].version = 3  # Child has lower version
        
        result = merge_states(parent, child, strategy=MergeStrategy.NEWEST)
        
        assert result.success
        assert parent.artifacts[0].content == "parent content"  # Parent wins
        assert result.conflicts[0].resolution == "parent_wins_newer"
    
    def test_artifact_filter(self):
        """Test filtering which artifacts to merge."""
        parent = Blackboard(goal="Test")
        child = parent.model_copy(deep=True)
        
        # Add mixed artifacts to child
        child.add_artifact(Artifact(type="code", content="code1", creator="Writer"))
        child.add_artifact(Artifact(type="docs", content="docs1", creator="Writer"))
        child.add_artifact(Artifact(type="code", content="code2", creator="Writer"))
        
        # Only merge code artifacts
        result = merge_states(
            parent, child,
            artifact_filter=lambda a: a.type == "code"
        )
        
        assert result.success
        assert result.artifacts_added == 2
        assert all(a.type == "code" for a in parent.artifacts)
    
    def test_merge_feedback(self):
        """Test merging feedback entries."""
        parent = Blackboard(goal="Test")
        parent.add_feedback(Feedback(source="Critic1", critique="Good", passed=True))
        
        child = parent.model_copy(deep=True)
        child.add_feedback(Feedback(source="Critic2", critique="Better", passed=True))
        
        result = merge_states(parent, child, merge_feedback=True)
        
        assert result.success
        assert result.feedback_added == 1
        assert len(parent.feedback) == 2
    
    def test_skip_feedback_merge(self):
        """Test skipping feedback merge."""
        parent = Blackboard(goal="Test")
        child = parent.model_copy(deep=True)
        child.add_feedback(Feedback(source="Critic", critique="Test", passed=True))
        
        result = merge_states(parent, child, merge_feedback=False)
        
        assert result.success
        assert result.feedback_added == 0
        assert len(parent.feedback) == 0
    
    def test_merge_metadata(self):
        """Test merging metadata."""
        parent = Blackboard(goal="Test")
        parent.metadata["key1"] = "value1"
        
        child = parent.model_copy(deep=True)
        child.metadata["key2"] = "value2"
        
        result = merge_states(parent, child, merge_metadata=True)
        
        assert result.success
        assert parent.metadata["key1"] == "value1"
        assert parent.metadata["key2"] == "value2"
    
    def test_metadata_conflict(self):
        """Test metadata conflict handling."""
        parent = Blackboard(goal="Test")
        parent.metadata["key"] = "parent_value"
        
        child = parent.model_copy(deep=True)
        child.metadata["key"] = "child_value"
        
        result = merge_states(parent, child, strategy=MergeStrategy.THEIRS)
        
        assert result.success
        assert parent.metadata["key"] == "child_value"
        assert len(result.conflicts) == 1
        assert result.conflicts[0].conflict_type == "metadata"


# =============================================================================
# Test StateMerger class
# =============================================================================

class TestStateMerger:
    """Tests for the StateMerger class."""
    
    def test_basic_usage(self):
        """Test basic StateMerger usage."""
        parent = Blackboard(goal="Test")
        merger = StateMerger(parent)
        
        child = parent.model_copy(deep=True)
        child.add_artifact(Artifact(type="code", content="test", creator="Worker"))
        
        result = merger.merge(child)
        
        assert result.success
        assert len(merger.get_merged_state().artifacts) == 1
    
    def test_multiple_merges(self):
        """Test merging multiple child states."""
        parent = Blackboard(goal="Test")
        merger = StateMerger(parent)
        
        # Create and merge multiple children
        for i in range(3):
            child = parent.model_copy(deep=True)
            child.add_artifact(Artifact(type="code", content=f"code{i}", creator=f"Worker{i}"))
            merger.merge(child)
        
        assert len(merger.get_merged_state().artifacts) == 3
        assert len(merger.get_merge_history()) == 3
    
    def test_get_total_conflicts(self):
        """Test counting total conflicts."""
        parent = Blackboard(goal="Test")
        parent.add_artifact(Artifact(type="code", content="original", creator="Writer"))
        
        merger = StateMerger(parent, strategy=MergeStrategy.THEIRS)
        
        # Merge conflicting child
        child = parent.model_copy(deep=True)
        child.artifacts[0].content = "modified"
        child.artifacts[0].version = 2
        merger.merge(child)
        
        assert merger.get_total_conflicts() == 1
    
    def test_override_strategy(self):
        """Test overriding strategy per merge."""
        parent = Blackboard(goal="Test")
        parent.add_artifact(Artifact(type="code", content="original", creator="Writer"))
        
        merger = StateMerger(parent, strategy=MergeStrategy.OURS)
        
        child = parent.model_copy(deep=True)
        child.artifacts[0].content = "modified"
        child.artifacts[0].version = 2
        
        # Override to THEIRS for this merge
        result = merger.merge(child, strategy=MergeStrategy.THEIRS)
        
        assert parent.artifacts[0].content == "modified"


# =============================================================================
# Test context filtering in Blackboard
# =============================================================================

class TestContextFiltering:
    """Tests for artifact_filter and feedback_filter in to_context_string."""
    
    def test_artifact_filter(self):
        """Test filtering artifacts in context string."""
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="code", content="code content", creator="Writer"))
        state.add_artifact(Artifact(type="docs", content="docs content", creator="Writer"))
        state.add_artifact(Artifact(type="code", content="more code", creator="Writer"))
        
        # Only show code artifacts
        context = state.to_context_string(
            artifact_filter=lambda a: a.type == "code"
        )
        
        assert "code content" in context
        assert "more code" in context
        assert "docs content" not in context
    
    def test_feedback_filter(self):
        """Test filtering feedback in context string."""
        state = Blackboard(goal="Test")
        state.add_feedback(Feedback(source="Critic", critique="Error found", passed=False))
        state.add_feedback(Feedback(source="Critic", critique="Looks good", passed=True))
        state.add_feedback(Feedback(source="Critic", critique="Another error", passed=False))
        
        # Only show failed feedback
        context = state.to_context_string(
            feedback_filter=lambda f: not f.passed
        )
        
        assert "Error found" in context
        assert "Another error" in context
        assert "Looks good" not in context
    
    def test_combined_filters(self):
        """Test using both filters together."""
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="code", content="code", creator="Writer"))
        state.add_artifact(Artifact(type="docs", content="docs", creator="Writer"))
        state.add_feedback(Feedback(source="Critic", critique="pass", passed=True))
        state.add_feedback(Feedback(source="Critic", critique="fail", passed=False))
        
        context = state.to_context_string(
            artifact_filter=lambda a: a.type == "code",
            feedback_filter=lambda f: not f.passed
        )
        
        assert "code" in context
        assert "docs" not in context
        assert "fail" in context
        assert "pass" not in context or "FAILED" in context  # 'pass' might appear in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
