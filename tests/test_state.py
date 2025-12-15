"""Tests for blackboard state models."""

import pytest
from datetime import datetime

from blackboard.state import Artifact, Feedback, Blackboard, Status


class TestArtifact:
    """Tests for the Artifact model."""
    
    def test_artifact_creation_minimal(self):
        """Test creating an artifact with minimal required fields."""
        artifact = Artifact(type="text", content="Hello", creator="TestWorker")
        
        assert artifact.type == "text"
        assert artifact.content == "Hello"
        assert artifact.creator == "TestWorker"
        assert artifact.version == 1
        assert artifact.id is not None
        assert artifact.created_at is not None
    
    def test_artifact_creation_full(self):
        """Test creating an artifact with all fields."""
        artifact = Artifact(
            id="custom-id",
            type="code",
            content="def foo(): pass",
            creator="CodeWriter",
            version=2,
            metadata={"language": "python"}
        )
        
        assert artifact.id == "custom-id"
        assert artifact.type == "code"
        assert artifact.version == 2
        assert artifact.metadata["language"] == "python"
    
    def test_artifact_with_complex_content(self):
        """Test artifact can hold complex content types."""
        artifact = Artifact(
            type="json",
            content={"nested": {"data": [1, 2, 3]}},
            creator="DataGenerator"
        )
        
        assert artifact.content["nested"]["data"] == [1, 2, 3]


class TestFeedback:
    """Tests for the Feedback model."""
    
    def test_feedback_creation(self):
        """Test creating feedback."""
        feedback = Feedback(
            source="Critic",
            critique="Looks good!",
            passed=True
        )
        
        assert feedback.source == "Critic"
        assert feedback.critique == "Looks good!"
        assert feedback.passed is True
        assert feedback.id is not None
    
    def test_feedback_with_artifact_reference(self):
        """Test feedback linked to an artifact."""
        feedback = Feedback(
            artifact_id="artifact-123",
            source="Reviewer",
            critique="Needs improvement",
            passed=False
        )
        
        assert feedback.artifact_id == "artifact-123"
        assert feedback.passed is False


class TestBlackboard:
    """Tests for the Blackboard model."""
    
    def test_blackboard_creation(self):
        """Test creating a new blackboard."""
        bb = Blackboard(goal="Test the system")
        
        assert bb.goal == "Test the system"
        assert bb.status == Status.PLANNING
        assert bb.artifacts == []
        assert bb.feedback == []
        assert bb.step_count == 0
    
    def test_add_artifact(self):
        """Test adding artifacts to the blackboard."""
        bb = Blackboard(goal="Generate content")
        
        artifact1 = Artifact(type="text", content="First", creator="Writer")
        artifact2 = Artifact(type="text", content="Second", creator="Writer")
        
        bb.add_artifact(artifact1)
        bb.add_artifact(artifact2)
        
        assert len(bb.artifacts) == 2
        assert bb.artifacts[0].version == 1
        assert bb.artifacts[1].version == 2
    
    def test_get_last_artifact(self):
        """Test retrieving the last artifact."""
        bb = Blackboard(goal="Test")
        
        assert bb.get_last_artifact() is None
        
        bb.add_artifact(Artifact(type="text", content="Hello", creator="A"))
        bb.add_artifact(Artifact(type="code", content="print()", creator="B"))
        
        assert bb.get_last_artifact().type == "code"
        assert bb.get_last_artifact("text").content == "Hello"
        assert bb.get_last_artifact("image") is None
    
    def test_add_feedback(self):
        """Test adding feedback to the blackboard."""
        bb = Blackboard(goal="Test")
        
        feedback = Feedback(source="Critic", critique="Good", passed=True)
        bb.add_feedback(feedback)
        
        assert len(bb.feedback) == 1
        assert bb.get_latest_feedback().passed is True
    
    def test_status_update(self):
        """Test updating the blackboard status."""
        bb = Blackboard(goal="Test")
        
        assert bb.status == Status.PLANNING
        
        bb.update_status(Status.GENERATING)
        assert bb.status == Status.GENERATING
        
        # Check history
        status_events = [h for h in bb.history if h["event"] == "status_changed"]
        assert len(status_events) == 1
    
    def test_increment_step(self):
        """Test step counter."""
        bb = Blackboard(goal="Test")
        
        assert bb.step_count == 0
        assert bb.increment_step() == 1
        assert bb.increment_step() == 2
        assert bb.step_count == 2
    
    def test_to_context_string(self):
        """Test generating context string for LLM."""
        bb = Blackboard(goal="Write a poem")
        bb.add_artifact(Artifact(type="text", content="Roses are red", creator="Poet"))
        bb.add_feedback(Feedback(source="Critic", critique="Too cliche", passed=False))
        
        context = bb.to_context_string()
        
        assert "Write a poem" in context
        assert "Roses are red" in context
        assert "Too cliche" in context
        assert "FAILED" in context


class TestStatus:
    """Tests for the Status enum."""
    
    def test_status_values(self):
        """Test all status values exist."""
        assert Status.PLANNING.value == "planning"
        assert Status.GENERATING.value == "generating"
        assert Status.CRITIQUING.value == "critiquing"
        assert Status.REFINING.value == "refining"
        assert Status.DONE.value == "done"
        assert Status.FAILED.value == "failed"
