"""
Tests for Blueprint Workflow System

Tests the flow.py module including Step, Blueprint, and Orchestrator integration.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from blackboard.state import Blackboard, Status, Artifact, Feedback
from blackboard.flow import Step, Blueprint, simple_blueprint, research_and_write_blueprint


# =============================================================================
# Step Tests
# =============================================================================

class TestStep:
    """Tests for Step dataclass."""
    
    def test_step_creation(self):
        """Test basic step creation."""
        step = Step(
            name="research",
            description="Gather information",
            allowed_workers=["WebSearch", "Browser"],
            instructions="Only search, don't write"
        )
        
        assert step.name == "research"
        assert step.description == "Gather information"
        assert step.allowed_workers == ["WebSearch", "Browser"]
        assert step.instructions == "Only search, don't write"
        assert step.max_iterations == 10  # Default
    
    def test_step_should_advance_no_condition(self):
        """Test should_advance with no exit condition."""
        step = Step(name="test")
        state = Blackboard(goal="test")
        
        # Should not advance by default without exit condition
        assert not step.should_advance(state)
    
    def test_step_should_advance_max_iterations(self):
        """Test should_advance when max iterations reached."""
        step = Step(name="test", max_iterations=3)
        state = Blackboard(goal="test")
        state.metadata["_step_test_iterations"] = 3
        
        # Should advance when max iterations hit
        assert step.should_advance(state)
    
    def test_step_should_advance_custom_condition(self):
        """Test should_advance with custom exit condition."""
        step = Step(
            name="review",
            exit_condition=lambda s: s.get_latest_feedback() and s.get_latest_feedback().passed
        )
        
        state = Blackboard(goal="test")
        
        # No feedback - should not advance
        assert not step.should_advance(state)
        
        # Add passing feedback
        state.add_feedback(Feedback(source="Critic", critique="Good", passed=True))
        assert step.should_advance(state)
    
    def test_step_to_prompt_context(self):
        """Test prompt context generation."""
        step = Step(
            name="research",
            description="Gather data",
            allowed_workers=["WebSearch"],
            instructions="Focus on searching only"
        )
        
        context = step.to_prompt_context()
        
        assert "research" in context
        assert "Gather data" in context
        assert "Focus on searching only" in context
        assert "WebSearch" in context


# =============================================================================
# Blueprint Tests
# =============================================================================

class TestBlueprint:
    """Tests for Blueprint class."""
    
    @pytest.fixture
    def sample_blueprint(self):
        """Create a sample blueprint for testing."""
        return Blueprint(
            name="Test Pipeline",
            steps=[
                Step(name="step1", allowed_workers=["Worker1"]),
                Step(name="step2", allowed_workers=["Worker2"]),
                Step(name="step3", allowed_workers=["Worker3"])
            ]
        )
    
    def test_blueprint_creation(self, sample_blueprint):
        """Test basic blueprint creation."""
        assert sample_blueprint.name == "Test Pipeline"
        assert len(sample_blueprint.steps) == 3
    
    def test_blueprint_empty_steps_raises(self):
        """Test that empty steps list raises error."""
        with pytest.raises(ValueError, match="at least one step"):
            Blueprint(name="Empty", steps=[])
    
    def test_blueprint_duplicate_names_raises(self):
        """Test that duplicate step names raise error."""
        with pytest.raises(ValueError, match="unique"):
            Blueprint(
                name="Duplicate",
                steps=[
                    Step(name="same"),
                    Step(name="same")
                ]
            )
    
    def test_get_current_step(self, sample_blueprint):
        """Test getting current step."""
        state = Blackboard(goal="test")
        
        # Initially at step 0
        current = sample_blueprint.get_current_step(state)
        assert current.name == "step1"
        
        # Move to step 1
        state.metadata["_blueprint_step_index"] = 1
        current = sample_blueprint.get_current_step(state)
        assert current.name == "step2"
    
    def test_filter_workers(self, sample_blueprint):
        """Test worker filtering."""
        state = Blackboard(goal="test")
        all_workers = ["Worker1", "Worker2", "Worker3", "OtherWorker"]
        
        # At step 0, only Worker1 allowed
        filtered = sample_blueprint.filter_workers(all_workers, state)
        assert filtered == ["Worker1"]
        
        # Move to step 1
        state.metadata["_blueprint_step_index"] = 1
        filtered = sample_blueprint.filter_workers(all_workers, state)
        assert filtered == ["Worker2"]
    
    def test_filter_workers_empty_allows_all(self):
        """Test that empty allowed_workers allows all workers."""
        blueprint = Blueprint(
            name="Open",
            steps=[Step(name="open", allowed_workers=[])]
        )
        state = Blackboard(goal="test")
        all_workers = ["A", "B", "C"]
        
        filtered = blueprint.filter_workers(all_workers, state)
        assert filtered == all_workers
    
    def test_advance_step(self):
        """Test step advancement."""
        blueprint = Blueprint(
            name="Test",
            steps=[
                Step(name="step1", max_iterations=1),
                Step(name="step2")
            ]
        )
        state = Blackboard(goal="test")
        state.metadata["_step_step1_iterations"] = 1
        
        # Should advance because max iterations reached
        new_step = blueprint.advance_step(state)
        assert new_step is not None
        assert new_step.name == "step2"
        assert blueprint.get_current_step_index(state) == 1
    
    def test_is_complete(self, sample_blueprint):
        """Test completion check."""
        state = Blackboard(goal="test")
        
        assert not sample_blueprint.is_complete(state)
        
        state.metadata["_blueprint_step_index"] = 3
        assert sample_blueprint.is_complete(state)
    
    def test_get_prompt_context(self, sample_blueprint):
        """Test prompt context generation."""
        state = Blackboard(goal="test")
        
        context = sample_blueprint.get_prompt_context(state)
        
        assert "WORKFLOW" in context
        assert "Test Pipeline" in context
        assert "Step 1 of 3" in context
        assert "step1" in context
    
    def test_to_dict(self, sample_blueprint):
        """Test serialization."""
        data = sample_blueprint.to_dict()
        
        assert data["name"] == "Test Pipeline"
        assert len(data["steps"]) == 3
        assert data["steps"][0]["name"] == "step1"


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_simple_blueprint(self):
        """Test simple_blueprint factory."""
        bp = simple_blueprint("Quick Flow", [
            {"name": "search", "workers": ["WebSearch"]},
            {"name": "write", "workers": ["Writer"]}
        ])
        
        assert bp.name == "Quick Flow"
        assert len(bp.steps) == 2
        assert bp.steps[0].name == "search"
    
    def test_research_and_write_blueprint(self):
        """Test pre-built research and write template."""
        bp = research_and_write_blueprint()
        
        assert bp.name == "Research & Write Pipeline"
        assert len(bp.steps) == 3
        assert bp.steps[0].name == "research"
        assert bp.steps[1].name == "write"
        assert bp.steps[2].name == "review"
