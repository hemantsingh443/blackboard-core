"""Tests for state persistence and history management."""

import pytest
import tempfile
from pathlib import Path

from blackboard import Blackboard, Artifact, Feedback, Status


class TestPersistence:
    """Tests for save/load functionality."""
    
    def test_save_to_json(self, tmp_path):
        """Test saving state to JSON."""
        state = Blackboard(goal="Test goal")
        state.add_artifact(Artifact(type="text", content="Hello", creator="Test"))
        state.add_feedback(Feedback(source="Critic", critique="Good", passed=True))
        
        path = tmp_path / "state.json"
        state.save_to_json(path)
        
        assert path.exists()
        content = path.read_text()
        assert "Test goal" in content
        assert "Hello" in content
    
    def test_load_from_json(self, tmp_path):
        """Test loading state from JSON."""
        # Create and save state
        original = Blackboard(goal="Test goal")
        original.add_artifact(Artifact(type="text", content="Hello", creator="Test"))
        original.step_count = 5
        
        path = tmp_path / "state.json"
        original.save_to_json(path)
        
        # Load and verify
        loaded = Blackboard.load_from_json(path)
        
        assert loaded.goal == "Test goal"
        assert loaded.step_count == 5
        assert len(loaded.artifacts) == 1
        assert loaded.artifacts[0].content == "Hello"
    
    def test_round_trip_complex_state(self, tmp_path):
        """Test save/load with complex nested state."""
        state = Blackboard(goal="Complex test")
        state.status = Status.GENERATING
        state.metadata["key"] = {"nested": [1, 2, 3]}
        
        for i in range(3):
            state.add_artifact(Artifact(
                type="code",
                content=f"def func_{i}(): pass",
                creator=f"Worker{i}"
            ))
        
        state.add_feedback(Feedback(source="Critic", critique="Good", passed=True))
        state.add_feedback(Feedback(source="Critic", critique="Bad", passed=False))
        
        path = tmp_path / "complex.json"
        state.save_to_json(path)
        loaded = Blackboard.load_from_json(path)
        
        assert loaded.status == Status.GENERATING
        assert loaded.metadata["key"]["nested"] == [1, 2, 3]
        assert len(loaded.artifacts) == 3
        assert len(loaded.feedback) == 2
    
    def test_to_dict_from_dict(self):
        """Test dictionary serialization."""
        state = Blackboard(goal="Dict test")
        state.add_artifact(Artifact(type="text", content="Test", creator="A"))
        
        data = state.to_dict()
        restored = Blackboard.from_dict(data)
        
        assert restored.goal == "Dict test"
        assert len(restored.artifacts) == 1


class TestHistoryManagement:
    """Tests for sliding window context generation."""
    
    def test_context_string_with_limits(self):
        """Test that context respects max_artifacts limit."""
        state = Blackboard(goal="Test")
        
        # Add 5 artifacts
        for i in range(5):
            state.add_artifact(Artifact(
                type="text",
                content=f"Content {i}",
                creator=f"Worker{i}"
            ))
        
        # Request only last 2
        context = state.to_context_string(max_artifacts=2)
        
        assert "Content 3" in context
        assert "Content 4" in context
        assert "Content 0" not in context
        assert "5 total, showing last 2" in context
    
    def test_context_string_with_feedback_limit(self):
        """Test that context respects max_feedback limit."""
        state = Blackboard(goal="Test")
        state.add_artifact(Artifact(type="text", content="Test", creator="A"))
        
        # Add 5 feedback entries
        for i in range(5):
            state.add_feedback(Feedback(
                source=f"Critic{i}",
                critique=f"Feedback {i}",
                passed=(i % 2 == 0)
            ))
        
        context = state.to_context_string(max_feedback=2)
        
        assert "Feedback 3" in context
        assert "Feedback 4" in context
        assert "Feedback 0" not in context
    
    def test_content_length_truncation(self):
        """Test that long content is truncated."""
        state = Blackboard(goal="Test")
        long_content = "x" * 1000
        state.add_artifact(Artifact(type="text", content=long_content, creator="A"))
        
        context = state.to_context_string(max_content_length=100)
        
        assert "..." in context
        assert len(context) < 2000  # Reasonable limit
    
    def test_context_summary(self):
        """Test brief context summary."""
        state = Blackboard(goal="Test")
        state.step_count = 10
        state.add_artifact(Artifact(type="text", content="Test", creator="A"))
        state.add_feedback(Feedback(source="C", critique="Good", passed=True))
        
        summary = state.get_context_summary()
        
        assert "Steps: 10" in summary
        assert "Artifacts: 1" in summary
        assert "Passed" in summary
