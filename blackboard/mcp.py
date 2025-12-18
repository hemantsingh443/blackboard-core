"""
Model Context Protocol (MCP) Integration

Provides a rock-solid MCP client that wraps MCP servers as Workers.
Enables connecting to external tools (Filesystem, GitHub, Postgres) without writing code.

DYNAMIC TOOL EXPANSION:
Each MCP tool is exposed as a separate Worker, giving the LLM direct access
to individual tool schemas (not a router pattern).

Example:
    from blackboard import Orchestrator
    from blackboard.mcp import MCPServerWorker
    
    # Connect to filesystem MCP server
    fs_server = await MCPServerWorker.create(
        name="Filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    )
    
    # DYNAMIC EXPANSION: Each MCP tool becomes a separate Worker
    tool_workers = fs_server.expand_to_workers()
    # -> [MCPToolWorker(read_file), MCPToolWorker(write_file), ...]
    
    orchestrator = Orchestrator(llm=llm, workers=tool_workers)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .protocols import Worker, WorkerOutput, WorkerInput
from .state import Artifact, Blackboard
from .tools import ToolDefinition, ToolParameter

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool

logger = logging.getLogger("blackboard.mcp")


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_tool_definition(self) -> ToolDefinition:
        """
        Convert this MCP tool to a ToolDefinition.
        
        This enables the LLM to see the full schema with proper types.
        """
        parameters = []
        
        # Parse JSON Schema format from MCP
        props = self.input_schema.get("properties", {})
        required_fields = self.input_schema.get("required", [])
        
        for param_name, param_info in props.items():
            # Map JSON Schema types to our types
            json_type = param_info.get("type", "string")
            if isinstance(json_type, list):
                json_type = json_type[0] if json_type else "string"
            
            parameters.append(ToolParameter(
                name=param_name,
                type=json_type,
                description=param_info.get("description", f"The {param_name} parameter"),
                required=param_name in required_fields,
                enum=param_info.get("enum"),
                default=param_info.get("default")
            ))
        
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters
        )


class MCPWorkerInput(WorkerInput):
    """Input for MCP tool worker."""
    # Dynamic arguments will be passed from LLM
    pass


class MCPToolWorker(Worker):
    """
    A single MCP tool exposed as a Worker.
    
    This is created by MCPServerWorker.expand_to_workers() and represents
    one specific tool from an MCP server. The LLM sees this tool directly
    with its full schema - no router pattern.
    
    Args:
        server: Parent MCPServerWorker
        tool: The specific MCPTool this worker represents
    """
    
    def __init__(self, server: "MCPServerWorker", tool: MCPTool):
        self._server = server
        self._tool = tool
    
    @property
    def name(self) -> str:
        # Use namespaced name to avoid collisions: "Filesystem:read_file"
        return f"{self._server.name}:{self._tool.name}"
    
    @property
    def description(self) -> str:
        return self._tool.description
    
    @property
    def parallel_safe(self) -> bool:
        return False  # MCP servers are stateful
    
    @property
    def tool_definition(self) -> ToolDefinition:
        """Get the ToolDefinition for direct LLM exposure."""
        return self._tool.to_tool_definition()
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """
        Protocol method: Return tool definitions for this worker.
        
        This is what the Orchestrator calls to get tools for the LLM.
        """
        return [self._tool.to_tool_definition()]
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        """
        Execute this specific MCP tool.
        
        Arguments are extracted from inputs and passed to the MCP server.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.types import TextContent
        except ImportError:
            return WorkerOutput(
                artifact=Artifact(
                    type="error",
                    content="MCP package not installed",
                    creator=self.name
                )
            )
        
        # Extract arguments from inputs
        arguments = {}
        if inputs:
            # Get all attributes from inputs that match tool schema
            for key in self._tool.input_schema.get("properties", {}).keys():
                if hasattr(inputs, key):
                    arguments[key] = getattr(inputs, key)
            
            # Also try instructions as fallback for simple tools
            if not arguments and hasattr(inputs, 'instructions'):
                instructions = getattr(inputs, 'instructions', '')
                if instructions:
                    # Check if tool has a single string parameter
                    props = self._tool.input_schema.get("properties", {})
                    if len(props) == 1:
                        param_name = list(props.keys())[0]
                        arguments[param_name] = instructions
        
        # Connect and call tool
        server_params = StdioServerParameters(
            command=self._server._command,
            args=self._server._args,
            env=self._server._env
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool(self._tool.name, arguments=arguments)
                    
                    # Extract content
                    content_parts = []
                    for content_block in result.content:
                        if isinstance(content_block, TextContent):
                            content_parts.append(content_block.text)
                        else:
                            content_parts.append(str(content_block))
                    
                    content = "\n".join(content_parts)
                    
                    return WorkerOutput(
                        artifact=Artifact(
                            type="mcp_result",
                            content=content,
                            creator=self.name,
                            metadata={
                                "tool": self._tool.name,
                                "server": self._server.name,
                                "structured": result.structuredContent
                            }
                        )
                    )
                    
        except Exception as e:
            logger.error(f"[{self.name}] Tool call failed: {e}")
            return WorkerOutput(
                artifact=Artifact(
                    type="error",
                    content=f"MCP tool '{self._tool.name}' failed: {str(e)}",
                    creator=self.name
                )
            )
    
    def __repr__(self) -> str:
        return f"MCPToolWorker({self.name})"


class MCPServerWorker(Worker):
    """
    Wraps an MCP server as a Worker.
    
    Connects to an MCP server via stdio transport, discovers its tools,
    and exposes them to the Orchestrator.
    
    Args:
        name: Worker name (used by Orchestrator)
        command: Command to start the MCP server
        args: Arguments for the command
        description: Optional description (auto-generated from tools if not provided)
        
    Example:
        # Create filesystem server worker
        fs = await MCPServerWorker.create(
            name="Filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        
        # Create GitHub server worker  
        github = await MCPServerWorker.create(
            name="GitHub",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]}
        )
    """
    
    input_schema = MCPWorkerInput
    
    def __init__(
        self,
        name: str,
        command: str,
        args: List[str],
        description: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        tools: Optional[List[MCPTool]] = None
    ):
        self._name = name
        self._command = command
        self._args = args
        self._env = env or {}
        self._tools = tools or []
        self._description = description or self._generate_description()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parallel_safe(self) -> bool:
        return False  # MCP servers are stateful
    
    @property
    def tools(self) -> List[MCPTool]:
        """Get the list of tools available from this MCP server."""
        return self._tools
    
    def _generate_description(self) -> str:
        """Generate description from available tools."""
        if not self._tools:
            return f"MCP Server: {self._name}"
        
        tool_names = [t.name for t in self._tools[:5]]
        suffix = f" (+{len(self._tools) - 5} more)" if len(self._tools) > 5 else ""
        return f"MCP Server with tools: {', '.join(tool_names)}{suffix}"
    
    def expand_to_workers(self) -> List["MCPToolWorker"]:
        """
        DYNAMIC TOOL EXPANSION: Create individual workers for each MCP tool.
        
        This is the production-grade approach. Instead of the LLM seeing
        one "router" worker, it sees each tool as a separate worker with
        its full schema.
        
        Returns:
            List of MCPToolWorker instances, one per MCP tool
            
        Example:
            fs_server = await MCPServerWorker.create(...)
            
            # OLD (router pattern - bad):
            # workers = [fs_server]  # LLM sees: Filesystem(tool_name=..., args=...)
            
            # NEW (dynamic expansion - good):
            workers = fs_server.expand_to_workers()
            # LLM sees: read_file(path=...), write_file(path=..., content=...), etc.
        """
        return [MCPToolWorker(self, tool) for tool in self._tools]
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """
        Get ToolDefinitions for all tools in this MCP server.
        
        Used by Orchestrator for native tool calling.
        """
        return [tool.to_tool_definition() for tool in self._tools]
    
    def to_tool_definitions(self) -> List[ToolDefinition]:
        """Alias for get_tool_definitions (for compatibility)."""
        return self.get_tool_definitions()
    
    @classmethod
    async def create(
        cls,
        name: str,
        command: str,
        args: List[str],
        description: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0
    ) -> "MCPServerWorker":
        """
        Create and initialize an MCP server worker.
        
        This connects to the server and discovers available tools.
        
        Args:
            name: Worker name
            command: Command to start MCP server
            args: Arguments for command
            description: Optional description
            env: Environment variables for server
            timeout: Connection timeout in seconds
            
        Returns:
            Initialized MCPServerWorker with discovered tools
            
        Raises:
            ImportError: If mcp package is not installed
            TimeoutError: If connection times out
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP package not installed. Install with: pip install 'blackboard-core[mcp]'"
            )
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        tools: List[MCPTool] = []
        
        try:
            async with asyncio.timeout(timeout):
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        # Discover tools
                        tools_result = await session.list_tools()
                        for tool in tools_result.tools:
                            tools.append(MCPTool(
                                name=tool.name,
                                description=tool.description or "",
                                input_schema=tool.inputSchema or {}
                            ))
                        
                        logger.info(f"[{name}] Discovered {len(tools)} tools")
                        
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP server '{name}' connection timed out after {timeout}s")
        except Exception as e:
            logger.error(f"[{name}] Failed to connect: {e}")
            raise
        
        return cls(
            name=name,
            command=command,
            args=args,
            description=description,
            env=env,
            tools=tools
        )
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[WorkerInput] = None
    ) -> WorkerOutput:
        """
        Execute an MCP tool based on inputs.
        
        Args:
            state: Current blackboard state
            inputs: Must contain tool_name and arguments
            
        Returns:
            WorkerOutput with tool result as artifact
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.types import TextContent
        except ImportError:
            return WorkerOutput(
                artifact=Artifact(
                    type="error",
                    content="MCP package not installed",
                    creator=self._name
                )
            )
        
        if not inputs:
            return WorkerOutput(
                artifact=Artifact(
                    type="error",
                    content="No inputs provided. Specify tool_name and arguments.",
                    creator=self._name
                )
            )
        
        # Extract tool name and arguments
        tool_name = getattr(inputs, 'tool_name', '') or self._infer_tool_from_instructions(inputs)
        arguments = getattr(inputs, 'arguments', {})
        
        if not tool_name:
            available = ", ".join(t.name for t in self._tools)
            return WorkerOutput(
                artifact=Artifact(
                    type="error",
                    content=f"No tool_name specified. Available tools: {available}",
                    creator=self._name
                )
            )
        
        # Connect and call tool
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    # Extract content from result
                    content_parts = []
                    for content_block in result.content:
                        if isinstance(content_block, TextContent):
                            content_parts.append(content_block.text)
                        else:
                            content_parts.append(str(content_block))
                    
                    content = "\n".join(content_parts)
                    
                    return WorkerOutput(
                        artifact=Artifact(
                            type="mcp_result",
                            content=content,
                            creator=self._name,
                            metadata={
                                "tool": tool_name,
                                "server": self._name,
                                "structured": result.structuredContent
                            }
                        )
                    )
                    
        except Exception as e:
            logger.error(f"[{self._name}] Tool call failed: {e}")
            return WorkerOutput(
                artifact=Artifact(
                    type="error",
                    content=f"MCP tool '{tool_name}' failed: {str(e)}",
                    creator=self._name
                )
            )
    
    def _infer_tool_from_instructions(self, inputs: WorkerInput) -> str:
        """Attempt to infer tool name from instructions."""
        instructions = getattr(inputs, 'instructions', '')
        if not instructions:
            return ""
        
        # Simple heuristic: check if any tool name appears in instructions
        instructions_lower = instructions.lower()
        for tool in self._tools:
            if tool.name.lower() in instructions_lower:
                return tool.name
        
        # Default to first tool if only one exists
        if len(self._tools) == 1:
            return self._tools[0].name
        
        return ""
    
    def __repr__(self) -> str:
        tool_count = len(self._tools)
        return f"MCPServerWorker({self._name}, {tool_count} tools)"


class MCPRegistry:
    """
    Registry for managing multiple MCP server connections.
    
    Example:
        registry = MCPRegistry()
        
        await registry.add(
            name="Filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        
        await registry.add(
            name="GitHub",
            command="npx", 
            args=["-y", "@modelcontextprotocol/server-github"]
        )
        
        workers = registry.get_workers()
        orchestrator = Orchestrator(llm=llm, workers=workers)
    """
    
    def __init__(self):
        self._servers: Dict[str, MCPServerWorker] = {}
    
    async def add(
        self,
        name: str,
        command: str,
        args: List[str],
        **kwargs
    ) -> MCPServerWorker:
        """
        Add and initialize an MCP server.
        
        Args:
            name: Worker name
            command: Command to start server
            args: Command arguments
            **kwargs: Additional arguments for MCPServerWorker.create
            
        Returns:
            The initialized MCPServerWorker
        """
        worker = await MCPServerWorker.create(
            name=name,
            command=command,
            args=args,
            **kwargs
        )
        self._servers[name] = worker
        return worker
    
    def get(self, name: str) -> Optional[MCPServerWorker]:
        """Get a server by name."""
        return self._servers.get(name)
    
    def get_workers(self) -> List[Worker]:
        """Get all servers as Workers."""
        return list(self._servers.values())
    
    def list_all_tools(self) -> Dict[str, List[MCPTool]]:
        """List all tools from all servers."""
        return {
            name: server.tools
            for name, server in self._servers.items()
        }
    
    def __len__(self) -> int:
        return len(self._servers)
    
    def __repr__(self) -> str:
        return f"MCPRegistry({len(self)} servers)"
