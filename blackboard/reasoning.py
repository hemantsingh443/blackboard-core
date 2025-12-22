"""
Reasoning Strategies

Pluggable strategies for how the Supervisor LLM makes decisions.
Each strategy defines a different prompting approach to extract
the next action from the LLM.

The default (OneShot) requests JSON directly.
ChainOfThought allows the LLM to reason before deciding.
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Blackboard
    from .protocols import Worker

logger = logging.getLogger("blackboard.reasoning")


@dataclass
class Decision:
    """
    The parsed decision from the supervisor LLM.
    
    Attributes:
        action: The action to take ("call", "call_independent", "done", "fail")
        calls: List of worker calls to make
        reasoning: The supervisor's reasoning (visible to user/logs)
        thinking: Internal reasoning (from CoT, may be hidden)
    """
    action: str  # "call", "call_independent", "done", "fail"
    calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    thinking: str = ""  # Internal reasoning from CoT


class ReasoningStrategy(ABC):
    """
    Protocol for reasoning strategies.
    
    Implementations define how the supervisor prompt is constructed
    and how the LLM response is parsed into a Decision.
    """
    
    @abstractmethod
    def build_prompt(
        self,
        context: str,
        workers: Dict[str, str],
    ) -> str:
        """
        Build the full prompt to send to the LLM.
        
        Args:
            context: The current blackboard state as a string
            workers: Dict of worker names to descriptions
            
        Returns:
            The complete prompt string
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Decision:
        """
        Parse the LLM response into a Decision.
        
        Args:
            response: Raw LLM output
            
        Returns:
            Parsed Decision object
        """
        pass


# =============================================================================
# OneShot Strategy (Default)
# =============================================================================

class OneShotStrategy(ReasoningStrategy):
    """
    Simple one-shot JSON strategy.
    
    The LLM is asked to respond with JSON directly.
    This is the default and works well for capable models.
    """
    
    SYSTEM_PROMPT = '''You are a Supervisor managing a team of AI workers to accomplish a goal.

## Your Role
- You NEVER do the work yourself
- You ONLY decide which worker(s) to call next based on the current state
- You route tasks and provide specific instructions to workers

## Available Workers
{worker_list}

## Response Format
You MUST respond with valid JSON in one of these formats:

### Single Worker Call
```json
{{
    "reasoning": "Brief explanation of why you're making this decision",
    "action": "call",
    "worker": "WorkerName",
    "instructions": "Specific instructions for the worker"
}}
```

### Independent Worker Calls (parallel, NO dependencies)
IMPORTANT: Only use this when tasks do NOT depend on each other's outputs.
```json
{{
    "reasoning": "These tasks are fully independent",
    "action": "call_independent",
    "calls": [
        {{"worker": "Worker1", "instructions": "Task 1"}},
        {{"worker": "Worker2", "instructions": "Task 2"}}
    ]
}}
```

### Terminal Actions
```json
{{"action": "done", "reasoning": "Goal achieved"}}
{{"action": "fail", "reasoning": "Cannot complete"}}
```

## Rules
1. If there's no artifact yet, call a Generator/Writer worker
2. If there's an artifact but no feedback, call a Critic/Reviewer worker
3. If feedback says "passed: false", call the Generator again with the critique
4. If feedback says "passed: true", mark as "done"
5. Use "call_independent" ONLY for truly independent tasks
6. Don't call the same worker twice in a row without new information
'''
    
    def build_prompt(
        self,
        context: str,
        workers: Dict[str, str],
    ) -> str:
        worker_list = "\n".join(
            f"- **{name}**: {desc}" for name, desc in workers.items()
        )
        
        system_prompt = self.SYSTEM_PROMPT.format(worker_list=worker_list)
        return f"{system_prompt}\n\n## Current State\n{context}\n\n## Your Decision (JSON only):"
    
    def parse_response(self, response: str) -> Decision:
        """Parse JSON response."""
        # Try to extract JSON from code blocks first
        json_str = self._extract_json(response)
        if not json_str:
            logger.warning(f"No JSON found in response: {response[:200]}")
            return Decision(action="fail", reasoning="Could not parse LLM response")
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e}")
            return Decision(action="fail", reasoning=f"Invalid JSON: {e}")
        
        return self._parse_json_data(data)
    
    def _extract_json(self, response: str) -> Optional[str]:
        """Extract JSON from response, handling code blocks and raw JSON."""
        # Try code block first
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json_match.group()
        
        return None
    
    def _parse_json_data(self, data: Dict[str, Any]) -> Decision:
        """Parse JSON dict into Decision."""
        action = data.get("action", "fail")
        reasoning = data.get("reasoning", "")
        
        calls = []
        if action == "call":
            worker_name = data.get("worker", "")
            instructions = data.get("instructions", "")
            inputs = data.get("inputs", {})
            if worker_name:
                calls.append({
                    "worker_name": worker_name,
                    "instructions": instructions,
                    "inputs": inputs
                })
        elif action == "call_independent":
            for call_data in data.get("calls", []):
                calls.append({
                    "worker_name": call_data.get("worker", ""),
                    "instructions": call_data.get("instructions", ""),
                    "inputs": call_data.get("inputs", {})
                })
        
        return Decision(action=action, calls=calls, reasoning=reasoning)


# =============================================================================
# Chain-of-Thought Strategy
# =============================================================================

class ChainOfThoughtStrategy(ReasoningStrategy):
    """
    Chain-of-Thought reasoning strategy.
    
    Allows the LLM to output a <thinking> block before the JSON decision.
    This significantly improves reasoning quality for complex tasks.
    
    Models like Claude 3.5 Sonnet and GPT-4 benefit substantially from
    being allowed to "think out loud" before committing to a decision.
    """
    
    # Complete separate prompt - no string manipulation needed
    SYSTEM_PROMPT = '''You are a Supervisor managing a team of AI workers to accomplish a goal.

## Your Role
- You NEVER do the work yourself
- You ONLY decide which worker(s) to call next based on the current state
- You route tasks and provide specific instructions to workers

## Available Workers
{worker_list}

## Response Format

**IMPORTANT**: First, think through your decision inside <thinking> tags. Then provide your JSON decision.

### Step 1: Think
<thinking>
- What is the current state?
- What has been accomplished so far?
- What still needs to be done to achieve the goal?
- Which worker is best suited for the next step?
- What specific instructions should I give them?
</thinking>

### Step 2: Decide (JSON)

For a single worker call:
```json
{{
    "action": "call",
    "worker": "WorkerName",
    "instructions": "Specific instructions for the worker",
    "reasoning": "Brief summary of why"
}}
```

For independent parallel calls:
```json
{{
    "action": "call_independent",
    "calls": [
        {{"worker": "Worker1", "instructions": "Task 1"}},
        {{"worker": "Worker2", "instructions": "Task 2"}}
    ],
    "reasoning": "These tasks are fully independent"
}}
```

To finish:
```json
{{"action": "done", "reasoning": "Goal achieved"}}
```

To fail:
```json
{{"action": "fail", "reasoning": "Cannot complete because..."}}
```

## Rules
1. ALWAYS include a <thinking> block before your JSON - it helps you reason better
2. If there's no artifact yet, call a Generator/Writer worker
3. If there's an artifact but no feedback, call a Critic/Reviewer worker
4. If feedback says "passed: false", call the Generator again with the critique
5. If feedback says "passed: true", mark as "done"
6. Use "call_independent" ONLY for truly independent tasks
'''
    
    def build_prompt(
        self,
        context: str,
        workers: Dict[str, str],
    ) -> str:
        worker_list = "\n".join(
            f"- **{name}**: {desc}" for name, desc in workers.items()
        )
        
        system_prompt = self.SYSTEM_PROMPT.format(worker_list=worker_list)
        return f"{system_prompt}\n\n## Current State\n{context}\n\n## Your Response:"
    
    def parse_response(self, response: str) -> Decision:
        """Parse response with optional thinking block."""
        thinking = ""
        
        # Extract thinking block if present
        thinking_match = re.search(
            r'<thinking>([\s\S]*?)</thinking>',
            response,
            re.IGNORECASE
        )
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            logger.debug(f"CoT Thinking: {thinking[:200]}...")
        
        # Extract JSON
        json_str = self._extract_json(response)
        if not json_str:
            logger.warning("No JSON found in CoT response")
            return Decision(
                action="fail",
                reasoning="Could not parse LLM response",
                thinking=thinking
            )
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in CoT response: {e}")
            return Decision(
                action="fail",
                reasoning=f"Invalid JSON: {e}",
                thinking=thinking
            )
        
        decision = self._parse_json_data(data)
        decision.thinking = thinking
        return decision
    
    def _extract_json(self, response: str) -> Optional[str]:
        """Extract JSON from response."""
        # Try code block first
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json_match.group()
        
        return None
    
    def _parse_json_data(self, data: Dict[str, Any]) -> Decision:
        """Parse JSON dict into Decision."""
        action = data.get("action", "fail")
        reasoning = data.get("reasoning", "")
        
        calls = []
        if action == "call":
            worker_name = data.get("worker", "")
            instructions = data.get("instructions", "")
            inputs = data.get("inputs", {})
            if worker_name:
                calls.append({
                    "worker_name": worker_name,
                    "instructions": instructions,
                    "inputs": inputs
                })
        elif action == "call_independent":
            for call_data in data.get("calls", []):
                calls.append({
                    "worker_name": call_data.get("worker", ""),
                    "instructions": call_data.get("instructions", ""),
                    "inputs": call_data.get("inputs", {})
                })
        
        return Decision(action=action, calls=calls, reasoning=reasoning)


# =============================================================================
# Factory and Defaults
# =============================================================================

def get_strategy(name: str) -> ReasoningStrategy:
    """
    Get a reasoning strategy by name.
    
    Args:
        name: "oneshot" or "cot"
        
    Returns:
        The corresponding ReasoningStrategy instance
    """
    strategies = {
        "oneshot": OneShotStrategy,
        "cot": ChainOfThoughtStrategy,
        "chain_of_thought": ChainOfThoughtStrategy,
    }
    
    strategy_class = strategies.get(name.lower())
    if strategy_class is None:
        logger.warning(f"Unknown strategy '{name}', falling back to oneshot")
        return OneShotStrategy()
    
    return strategy_class()


# Default strategy
DEFAULT_STRATEGY = OneShotStrategy()
