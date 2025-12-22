"""
Tests for reasoning strategies.

Verifies that both OneShotStrategy and ChainOfThoughtStrategy
correctly parse LLM responses into Decision objects.
"""

import pytest

from blackboard.reasoning import (
    OneShotStrategy,
    ChainOfThoughtStrategy,
    Decision
)


class TestOneShotStrategy:
    """Tests for the default one-shot JSON strategy."""
    
    def setup_method(self):
        self.strategy = OneShotStrategy()
    
    def test_parse_simple_call(self):
        """Parse a simple worker call."""
        response = '''
        {
            "action": "call",
            "worker": "Writer",
            "instructions": "Write a haiku about AI",
            "reasoning": "Need to generate content first"
        }
        '''
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "call"
        assert len(decision.calls) == 1
        assert decision.calls[0]["worker_name"] == "Writer"
        assert decision.calls[0]["instructions"] == "Write a haiku about AI"
        assert decision.reasoning == "Need to generate content first"
    
    def test_parse_done_action(self):
        """Parse a done action."""
        response = '{"action": "done", "reasoning": "Goal achieved"}'
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "done"
        assert decision.reasoning == "Goal achieved"
        assert len(decision.calls) == 0
    
    def test_parse_fail_action(self):
        """Parse a fail action."""
        response = '{"action": "fail", "reasoning": "Cannot complete task"}'
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "fail"
        assert decision.reasoning == "Cannot complete task"
    
    def test_parse_independent_calls(self):
        """Parse multiple independent worker calls."""
        response = '''
        {
            "action": "call_independent",
            "calls": [
                {"worker": "Researcher", "instructions": "Find data on topic A"},
                {"worker": "Analyst", "instructions": "Analyze metric B"}
            ],
            "reasoning": "These tasks are independent"
        }
        '''
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "call_independent"
        assert len(decision.calls) == 2
        assert decision.calls[0]["worker_name"] == "Researcher"
        assert decision.calls[1]["worker_name"] == "Analyst"
    
    def test_parse_json_with_markdown(self):
        """Parse JSON wrapped in markdown code blocks."""
        response = '''
        Here is my decision:
        
        ```json
        {
            "action": "call",
            "worker": "Coder",
            "instructions": "Fix the bug"
        }
        ```
        '''
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "call"
        assert decision.calls[0]["worker_name"] == "Coder"
    
    def test_parse_invalid_json(self):
        """Handle invalid JSON gracefully."""
        response = "This is not valid JSON at all"
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "fail"
        assert "parse" in decision.reasoning.lower() or "json" in decision.reasoning.lower()
    
    def test_parse_with_inputs(self):
        """Parse call with structured inputs."""
        response = '''
        {
            "action": "call",
            "worker": "Calculator",
            "inputs": {"a": 5, "b": 3, "operation": "add"}
        }
        '''
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "call"
        assert decision.calls[0]["inputs"]["a"] == 5
        assert decision.calls[0]["inputs"]["b"] == 3


class TestChainOfThoughtStrategy:
    """Tests for the Chain-of-Thought strategy."""
    
    def setup_method(self):
        self.strategy = ChainOfThoughtStrategy()
    
    def test_parse_with_thinking_block(self):
        """Parse response with thinking block."""
        response = '''
        <thinking>
        Let me analyze the current state:
        - The goal is to write a poem
        - No artifacts exist yet
        - I should call the Writer first
        </thinking>
        
        ```json
        {
            "action": "call",
            "worker": "Writer",
            "instructions": "Write a poem about nature",
            "reasoning": "Starting with content generation"
        }
        ```
        '''
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "call"
        assert decision.calls[0]["worker_name"] == "Writer"
        assert "analyze the current state" in decision.thinking
        assert "Writer first" in decision.thinking
    
    def test_parse_without_thinking_block(self):
        """Parse response that skipped thinking block."""
        response = '''
        {
            "action": "done",
            "reasoning": "Task complete"
        }
        '''
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "done"
        assert decision.thinking == ""
    
    def test_thinking_preserved_on_error(self):
        """Thinking is preserved even if JSON parsing fails."""
        response = '''
        <thinking>
        I'm confused about what to do next.
        </thinking>
        
        This is not valid JSON
        '''
        
        decision = self.strategy.parse_response(response)
        
        assert decision.action == "fail"
        assert "confused" in decision.thinking
    
    def test_build_prompt_includes_cot_instructions(self):
        """Built prompt includes CoT instructions."""
        context = "Goal: Test\nArtifacts: []"
        workers = {"Writer": "Writes content", "Critic": "Reviews content"}
        
        prompt = self.strategy.build_prompt(context, workers)
        
        assert "<thinking>" in prompt
        assert "think" in prompt.lower()


class TestDecisionDataclass:
    """Tests for the Decision dataclass."""
    
    def test_default_values(self):
        """Decision has sensible defaults."""
        decision = Decision(action="done")
        
        assert decision.action == "done"
        assert decision.calls == []
        assert decision.reasoning == ""
        assert decision.thinking == ""
    
    def test_full_decision(self):
        """Decision stores all fields."""
        decision = Decision(
            action="call",
            calls=[{"worker_name": "Test", "instructions": "Do it"}],
            reasoning="Because",
            thinking="Let me think..."
        )
        
        assert decision.action == "call"
        assert len(decision.calls) == 1
        assert decision.reasoning == "Because"
        assert decision.thinking == "Let me think..."
