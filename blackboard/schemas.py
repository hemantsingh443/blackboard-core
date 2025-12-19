"""
Pydantic Schema Models for Supervisor Responses

Provides structured validation for LLM responses using Pydantic,
enabling both strict parsing and JSON schema generation for prompts.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field

try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda *args, **kwargs: None
    ValidationError = Exception


# =============================================================================
# Pydantic Models (when pydantic is available)
# =============================================================================

if PYDANTIC_AVAILABLE:
    
    class WorkerCallSchema(BaseModel):
        """Schema for a single worker call."""
        worker: str = Field(description="Name of the worker to call")
        instructions: str = Field(default="", description="Instructions for the worker")
        inputs: Dict[str, Any] = Field(default_factory=dict, description="Structured inputs")
    
    class SingleCallAction(BaseModel):
        """Schema for calling a single worker."""
        action: Literal["call"] = Field(description="Action type: 'call'")
        worker: str = Field(description="Worker name to call")
        instructions: str = Field(default="", description="Instructions for the worker")
        inputs: Dict[str, Any] = Field(default_factory=dict, description="Optional structured inputs")
        reasoning: str = Field(default="", description="Brief explanation for this decision")
    
    class IndependentCallsAction(BaseModel):
        """Schema for calling multiple independent workers in parallel."""
        action: Literal["call_independent"] = Field(description="Action type: 'call_independent'")
        calls: List[WorkerCallSchema] = Field(description="List of worker calls to execute in parallel")
        reasoning: str = Field(default="", description="Brief explanation for parallel tasks")
    
    class DoneAction(BaseModel):
        """Schema for marking goal as complete."""
        action: Literal["done"] = Field(description="Action type: 'done'")
        reasoning: str = Field(default="Goal achieved", description="Reason for completion")
    
    class FailAction(BaseModel):
        """Schema for marking goal as failed."""
        action: Literal["fail"] = Field(description="Action type: 'fail'")
        reasoning: str = Field(default="Cannot complete", description="Reason for failure")
    
    # Union type for all possible supervisor actions
    SupervisorActionSchema = Union[SingleCallAction, IndependentCallsAction, DoneAction, FailAction]
    
    def get_supervisor_json_schema() -> Dict[str, Any]:
        """
        Generate JSON schema for supervisor responses.
        
        Useful for prompting weaker models with explicit schema.
        
        Returns:
            JSON schema dict that can be embedded in prompts
        """
        # Generate schemas for each action type
        single_call = SingleCallAction.model_json_schema()
        independent = IndependentCallsAction.model_json_schema() 
        done = DoneAction.model_json_schema()
        fail = FailAction.model_json_schema()
        
        return {
            "oneOf": [
                {"title": "Single Worker Call", "schema": single_call},
                {"title": "Independent Calls", "schema": independent},
                {"title": "Done", "schema": done},
                {"title": "Fail", "schema": fail}
            ],
            "examples": [
                {"action": "call", "worker": "Writer", "instructions": "Write a poem", "reasoning": "Starting task"},
                {"action": "call_independent", "calls": [{"worker": "A"}, {"worker": "B"}], "reasoning": "Parallel"},
                {"action": "done", "reasoning": "Goal achieved"},
                {"action": "fail", "reasoning": "Cannot complete"}
            ]
        }
    
    def validate_supervisor_response(data: Dict[str, Any]) -> Optional[str]:
        """
        Validate a parsed JSON response against supervisor schemas.
        
        Args:
            data: Parsed JSON dict from LLM response
            
        Returns:
            None if valid, error message string if invalid
        """
        action = data.get("action", "")
        
        try:
            if action == "call":
                SingleCallAction.model_validate(data)
            elif action == "call_independent":
                IndependentCallsAction.model_validate(data)
            elif action == "done":
                DoneAction.model_validate(data)
            elif action == "fail":
                FailAction.model_validate(data)
            else:
                return f"Unknown action: {action}"
            return None
        except ValidationError as e:
            return str(e)
    
    def get_simple_prompt_schema() -> str:
        """
        Generate a text-based schema for simpler prompts.
        
        Use this for weaker models that struggle with JSON schema.
        
        Returns:
            Human-readable schema description
        """
        return '''Response must be JSON with ONE of these structures:

1. Call single worker:
   {"action": "call", "worker": "WorkerName", "instructions": "...", "reasoning": "..."}

2. Call independent workers (parallel, no dependencies):
   {"action": "call_independent", "calls": [{"worker": "A", "instructions": "..."}, ...], "reasoning": "..."}

3. Mark complete:
   {"action": "done", "reasoning": "..."}

4. Mark failed:
   {"action": "fail", "reasoning": "..."}'''

else:
    # Fallback when pydantic is not installed
    
    def get_supervisor_json_schema() -> Dict[str, Any]:
        """Pydantic not available - returns empty schema."""
        return {}
    
    def validate_supervisor_response(data: Dict[str, Any]) -> Optional[str]:
        """Pydantic not available - no validation."""
        return None
    
    def get_simple_prompt_schema() -> str:
        """Return simple schema without pydantic."""
        return '''Response must be JSON with ONE of these structures:

1. Call single worker:
   {"action": "call", "worker": "WorkerName", "instructions": "..."}

2. Call independent workers (parallel):
   {"action": "call_independent", "calls": [{"worker": "A"}, ...]}

3. Mark complete:
   {"action": "done", "reasoning": "..."}

4. Mark failed:
   {"action": "fail", "reasoning": "..."}'''
