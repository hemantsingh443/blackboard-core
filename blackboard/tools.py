"""
Tool Calling Support

Native tool/function calling for LLMs that support it (OpenAI, Anthropic, Gemini).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable, Awaitable, Union
from abc import ABC, abstractmethod

from .protocols import Worker, WorkerInput


@dataclass
class ToolParameter:
    """A parameter for a tool definition."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None  # For string enums
    default: Any = None


@dataclass
class ToolDefinition:
    """
    A tool definition for LLM function calling.
    
    This matches the format expected by OpenAI/Anthropic/Gemini.
    
    Example:
        tool = ToolDefinition(
            name="Writer",
            description="Generates text content",
            parameters=[
                ToolParameter(name="instructions", type="string", description="Task instructions")
            ]
        )
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


@dataclass
class ToolCall:
    """
    A tool call from the LLM.
    
    This is what the LLM returns when using function calling.
    """
    id: str  # Unique ID for this call (for multi-turn)
    name: str  # Tool/function name
    arguments: Dict[str, Any]  # Parsed arguments
    
    def to_worker_inputs(self) -> Dict[str, Any]:
        """Convert tool call arguments to worker inputs."""
        return self.arguments


@dataclass
class ToolCallResponse:
    """
    Response to a tool call.
    
    The SDK generates this after executing a worker.
    """
    tool_call_id: str
    content: str  # Result content
    is_error: bool = False


@runtime_checkable
class ToolCallingLLMClient(Protocol):
    """
    Protocol for LLM clients that support native tool calling.
    
    If your LLM client implements this protocol, the Orchestrator
    will use tool calling instead of JSON parsing.
    
    Example:
        class OpenAIToolClient:
            def generate_with_tools(
                self,
                prompt: str,
                tools: List[ToolDefinition]
            ) -> Union[str, List[ToolCall]]:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[t.to_openai_format() for t in tools]
                )
                
                if response.choices[0].message.tool_calls:
                    return [
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments)
                        )
                        for tc in response.choices[0].message.tool_calls
                    ]
                return response.choices[0].message.content
    """
    
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition]
    ) -> Union[str, List[ToolCall], Awaitable[Union[str, List[ToolCall]]]]:
        """
        Generate a response, potentially with tool calls.
        
        Args:
            prompt: The prompt to send
            tools: Available tools
            
        Returns:
            Either a string response or a list of tool calls
        """
        ...


def worker_to_tool_definition(worker: Worker) -> ToolDefinition:
    """
    Convert a Worker to a ToolDefinition.
    
    Uses the worker's input_schema if available, otherwise creates
    a basic tool with just an 'instructions' parameter.
    """
    parameters = []
    
    if worker.input_schema is not None:
        # Get parameters from Pydantic schema
        schema = worker.input_schema.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        
        for name, prop in props.items():
            parameters.append(ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", f"The {name} parameter"),
                required=name in required,
                default=prop.get("default")
            ))
    else:
        # Default: just an instructions parameter
        parameters.append(ToolParameter(
            name="instructions",
            type="string",
            description="Specific instructions for this worker"
        ))
    
    return ToolDefinition(
        name=worker.name,
        description=worker.description,
        parameters=parameters
    )


def workers_to_tool_definitions(workers: List[Worker]) -> List[ToolDefinition]:
    """Convert a list of workers to tool definitions."""
    return [worker_to_tool_definition(w) for w in workers]


# Special tool definitions for built-in actions
DONE_TOOL = ToolDefinition(
    name="mark_done",
    description="Mark the task as complete. Call this when the goal has been achieved.",
    parameters=[
        ToolParameter(
            name="reason",
            type="string",
            description="Reason for marking as done"
        )
    ]
)

FAIL_TOOL = ToolDefinition(
    name="mark_failed",
    description="Mark the task as failed. Call this when the goal cannot be achieved.",
    parameters=[
        ToolParameter(
            name="reason",
            type="string",
            description="Reason for failure"
        )
    ]
)
