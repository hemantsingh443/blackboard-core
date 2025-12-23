"""Tests for Squad patterns."""

import pytest
from unittest.mock import MagicMock


class TestSquadPatterns:
    """Tests for Squad factory functions."""
    
    def test_create_squad_returns_agent(self):
        """Test that create_squad returns an Agent instance."""
        from blackboard.patterns import create_squad
        from blackboard.core import Agent
        
        mock_llm = MagicMock()
        
        squad = create_squad(
            name="TestSquad",
            description="A test squad",
            llm=mock_llm,
            workers=[]
        )
        
        assert isinstance(squad, Agent)
        assert squad.name == "TestSquad"
        assert squad.description == "A test squad"
    
    def test_research_squad_creation(self):
        """Test that research_squad can be created."""
        from blackboard.patterns import research_squad
        from blackboard.core import Agent
        
        mock_llm = MagicMock()
        
        squad = research_squad(llm=mock_llm)
        
        assert isinstance(squad, Agent)
        assert squad.name == "ResearchSquad"
        assert "research" in squad.description.lower()
    
    def test_code_squad_creation(self):
        """Test that code_squad can be created."""
        from blackboard.patterns import code_squad
        from blackboard.core import Agent
        
        mock_llm = MagicMock()
        
        squad = code_squad(llm=mock_llm)
        
        assert isinstance(squad, Agent)
        assert squad.name == "CodeSquad"
        assert "code" in squad.description.lower()
    
    def test_memory_squad_creation(self):
        """Test that memory_squad can be created."""
        from blackboard.patterns import memory_squad
        from blackboard.core import Agent
        
        mock_llm = MagicMock()
        
        squad = memory_squad(llm=mock_llm)
        
        assert isinstance(squad, Agent)
        assert squad.name == "MemorySquad"
        assert "memory" in squad.description.lower()
    
    def test_squad_type_alias(self):
        """Test that Squad is an alias for Agent."""
        from blackboard.patterns import Squad
        from blackboard.core import Agent
        
        assert Squad is Agent
    
    def test_create_squad_with_config(self):
        """Test that create_squad properly uses config."""
        from blackboard.patterns import create_squad
        from blackboard.config import BlackboardConfig
        
        mock_llm = MagicMock()
        config = BlackboardConfig(
            max_recursion_depth=5,
            max_steps=100
        )
        
        squad = create_squad(
            name="ConfiguredSquad",
            description="Test",
            llm=mock_llm,
            workers=[],
            config=config
        )
        
        assert squad.config.max_recursion_depth == 5
        assert squad.config.max_steps == 100
    
    def test_squad_with_additional_workers(self):
        """Test that squads accept additional workers."""
        from blackboard.patterns import research_squad
        from blackboard.protocols import Worker
        
        mock_llm = MagicMock()
        
        # Create a mock worker
        mock_worker = MagicMock(spec=Worker)
        mock_worker.name = "CustomWorker"
        
        squad = research_squad(
            llm=mock_llm,
            additional_workers=[mock_worker]
        )
        
        # The additional worker should be in the list
        assert any(w.name == "CustomWorker" for w in squad.workers)
    
    def test_squad_custom_name(self):
        """Test that squads can have custom names."""
        from blackboard.patterns import research_squad
        
        mock_llm = MagicMock()
        
        squad = research_squad(
            llm=mock_llm,
            name="MyResearchTeam",
            description="Custom research team"
        )
        
        assert squad.name == "MyResearchTeam"
        assert squad.description == "Custom research team"
