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
        
        If the parent server has a persistent connection (via connect()),
        uses it for maximum performance. Otherwise falls back to one-shot.
        """
        try:
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
        
        try:
            # FAST PATH: Use parent's persistent session if available
            if self._server.is_connected:
                content = await self._server.call_tool(self._tool.name, arguments)
                return WorkerOutput(
                    artifact=Artifact(
                        type="mcp_result",
                        content=content,
                        creator=self.name,
                        metadata={
                            "tool": self._tool.name,
                            "server": self._server.name,
                            "persistent": True
                        }
                    )
                )
            
            # SLOW PATH: One-shot connection (legacy behavior)
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            
            server_params = StdioServerParameters(
                command=self._server._command,
                args=self._server._args,
                env=self._server._env
            )
            
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
                                "persistent": False
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
        command: str = None,
        args: List[str] = None,
        description: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        tools: Optional[List[MCPTool]] = None,
        url: Optional[str] = None,  # SSE endpoint for remote MCP servers
        llm_client: Optional[Any] = None  # LLM client for sampling requests
    ):
        """
        Initialize an MCP server worker.
        
        Supports two transport modes:
        - **Stdio (local)**: Set command and args to spawn a local subprocess
        - **SSE (remote)**: Set url to connect to a remote MCP server
        
        Args:
            name: Worker name
            command: Command to start local MCP server (stdio mode)
            args: Arguments for the command
            description: Human-readable description
            env: Environment variables for the subprocess
            tools: Pre-discovered tools (usually set via create())
            url: SSE endpoint URL for remote MCP servers (e.g., "http://localhost:8080/sse")
            llm_client: LLM client for handling sampling requests from the MCP server
        """
        if not command and not url:
            raise ValueError("Either 'command' (for stdio) or 'url' (for SSE) must be provided")
        
        self._name = name
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._tools = tools or []
        self._url = url  # SSE endpoint
        self._llm_client = llm_client  # For sampling
        self._description = description or self._generate_description()
        
        # Transport mode
        self._transport_mode = "sse" if url else "stdio"
        
        # Persistent session management
        self._session: Optional["ClientSession"] = None
        self._stdio_context = None
        self._sse_context = None
        self._session_context = None
        self._connected = False
    
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
    
    @property
    def is_connected(self) -> bool:
        """Check if server has an active persistent connection."""
        return self._connected and self._session is not None
    
    async def connect(self) -> None:
        """
        Establish a persistent connection to the MCP server.
        
        Supports two transport modes:
        - **Stdio**: Spawns a local subprocess (command + args)
        - **SSE**: Connects to a remote HTTP endpoint (url)
        
        This keeps the connection alive for all subsequent tool calls,
        dramatically improving performance.
        
        Example:
            # Stdio (local)
            server = await MCPServerWorker.create(command="npx", args=[...])
            
            # SSE (remote)
            server = await MCPServerWorker.create(url="http://localhost:8080/sse")
            
            await server.connect()
            result = await server.call_tool("read_file", {"path": "a.txt"})
            await server.disconnect()
        """
        if self._connected:
            return
        
        try:
            from mcp import ClientSession
        except ImportError:
            raise ImportError(
                "MCP package not installed. Install with: pip install 'blackboard-core[mcp]'"
            )
        
        if self._transport_mode == "sse":
            await self._connect_sse(ClientSession)
        else:
            await self._connect_stdio(ClientSession)
        
        self._connected = True
        logger.info(f"[{self._name}] Persistent connection established ({self._transport_mode})")
    
    async def _connect_stdio(self, ClientSession) -> None:
        """Connect via stdio transport (local subprocess)."""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env
        )
        
        # Enter the stdio context (starts the process)
        self._stdio_context = stdio_client(server_params)
        read, write = await self._stdio_context.__aenter__()
        
        # Enter the session context
        self._session_context = ClientSession(read, write)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()
    
    async def _connect_sse(self, ClientSession) -> None:
        """Connect via SSE transport (remote HTTP endpoint)."""
        try:
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP SSE client not available. Ensure mcp package is up to date."
            )
        
        # Enter the SSE context
        self._sse_context = sse_client(self._url)
        read, write = await self._sse_context.__aenter__()
        
        # Create session with optional sampling handler
        if self._llm_client:
            self._session_context = ClientSession(
                read, write,
                sampling_callback=self._handle_sampling_request
            )
        else:
            self._session_context = ClientSession(read, write)
        
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()
    
    async def _handle_sampling_request(self, request) -> Any:
        """
        Handle sampling/createMessage requests from the MCP server.
        
        This enables "agentic tools" where the MCP server can ask
        the client's LLM for help during tool execution.
        
        Example: A code analyzer might ask "What does this regex do?"
        """
        if not self._llm_client:
            raise RuntimeError("Sampling requested but no llm_client provided")
        
        try:
            from mcp.types import CreateMessageResult, TextContent
        except ImportError:
            raise ImportError("MCP types not available")
        
        # Extract the prompt from the request
        if hasattr(request, 'messages') and request.messages:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content'):
                if hasattr(last_message.content, 'text'):
                    prompt = last_message.content.text
                else:
                    prompt = str(last_message.content)
            else:
                prompt = str(last_message)
        else:
            prompt = str(request)
        
        logger.debug(f"[{self._name}] Sampling request: {prompt[:100]}...")
        
        # Call the LLM
        import asyncio
        if asyncio.iscoroutinefunction(self._llm_client.generate):
            response = await self._llm_client.generate(prompt)
        else:
            response = self._llm_client.generate(prompt)
        
        # Handle LLMResponse objects
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        logger.debug(f"[{self._name}] Sampling response: {response_text[:100]}...")
        
        return CreateMessageResult(
            content=TextContent(type="text", text=response_text),
            model=getattr(self._llm_client, 'model', 'unknown'),
            role="assistant"
        )
    
    async def disconnect(self) -> None:
        """Close the persistent connection to the MCP server."""
        if not self._connected:
            return
        
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
            if self._sse_context:
                await self._sse_context.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"[{self._name}] Error during disconnect: {e}")
        finally:
            self._session = None
            self._session_context = None
            self._stdio_context = None
            self._sse_context = None
            self._connected = False
            logger.info(f"[{self._name}] Disconnected")
    
    async def __aenter__(self) -> "MCPServerWorker":
        """Async context manager entry - connect to server."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - disconnect from server."""
        await self.disconnect()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> str:
        """
        Call a tool directly on the persistent connection.
        
        This is the fastest way to call MCP tools after connect().
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool output as string
        """
        if not self.is_connected:
            raise RuntimeError(
                f"[{self._name}] Not connected. Call connect() or use 'async with server:'"
            )
        
        try:
            from mcp.types import TextContent
        except ImportError:
            raise ImportError("MCP package not installed")
        
        result = await self._session.call_tool(tool_name, arguments=arguments or {})
        
        content_parts = []
        for content_block in result.content:
            if isinstance(content_block, TextContent):
                content_parts.append(content_block.text)
            else:
                content_parts.append(str(content_block))
        
        return "\n".join(content_parts)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List all resources exposed by this MCP server.
        
        Resources are file-like data that the MCP server makes available.
        Use read_resource() to fetch the content.
        
        Returns:
            List of resource metadata dicts with uri, name, mimeType
            
        Example:
            async with server:
                resources = await server.list_resources()
                for r in resources:
                    print(f"{r['name']}: {r['uri']}")
        """
        if not self.is_connected:
            raise RuntimeError(
                f"[{self._name}] Not connected. Call connect() or use 'async with server:'"
            )
        
        try:
            result = await self._session.list_resources()
            return [
                {
                    "uri": str(r.uri),
                    "name": r.name,
                    "mimeType": getattr(r, 'mimeType', None),
                    "description": getattr(r, 'description', None)
                }
                for r in result.resources
            ]
        except Exception as e:
            logger.warning(f"[{self._name}] Failed to list resources: {e}")
            return []
    
    async def read_resource(self, uri: str) -> str:
        """
        Read a resource by URI.
        
        Args:
            uri: Resource URI (from list_resources)
            
        Returns:
            Resource content as string
            
        Example:
            async with server:
                resources = await server.list_resources()
                if resources:
                    content = await server.read_resource(resources[0]['uri'])
                    print(content)
        """
        if not self.is_connected:
            raise RuntimeError(
                f"[{self._name}] Not connected. Call connect() or use 'async with server:'"
            )
        
        try:
            from pydantic import AnyUrl
        except ImportError:
            # Fallback if pydantic not available
            AnyUrl = str
        
        result = await self._session.read_resource(AnyUrl(uri))
        
        # Extract text content
        parts = []
        for content_block in result.contents:
            if hasattr(content_block, 'text'):
                parts.append(content_block.text)
            else:
                parts.append(str(content_block))
        
        return "\n".join(parts)
    
    async def load_resources_to_state(self, state: Blackboard, max_resources: int = 10) -> int:
        """
        Load all resources from this MCP server into the Blackboard as artifacts.
        
        This is useful for injecting MCP resources into the agent's context.
        
        Args:
            state: Blackboard to add resources to
            max_resources: Maximum number of resources to load
            
        Returns:
            Number of resources loaded
            
        Example:
            async with server:
                loaded = await server.load_resources_to_state(state)
                print(f"Loaded {loaded} resources")
        """
        resources = await self.list_resources()
        loaded = 0
        
        for resource in resources[:max_resources]:
            try:
                content = await self.read_resource(resource['uri'])
                state.add_artifact(Artifact(
                    type="mcp_resource",
                    content=content,
                    creator=self._name,
                    metadata={
                        "uri": resource['uri'],
                        "name": resource.get('name'),
                        "mimeType": resource.get('mimeType')
                    }
                ))
                loaded += 1
                logger.debug(f"[{self._name}] Loaded resource: {resource['uri']}")
            except Exception as e:
                logger.warning(f"[{self._name}] Failed to load resource {resource['uri']}: {e}")
        
        return loaded
    
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
        command: str = None,
        args: List[str] = None,
        description: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        url: Optional[str] = None,
        llm_client: Optional[Any] = None
    ) -> "MCPServerWorker":
        """
        Create and initialize an MCP server worker.
        
        Supports two transport modes:
        - **Stdio (local)**: Set command and args to spawn a subprocess
        - **SSE (remote)**: Set url to connect via HTTP/SSE
        
        Args:
            name: Worker name
            command: Command to start MCP server (stdio mode)
            args: Arguments for command
            description: Optional description
            env: Environment variables for server
            timeout: Connection timeout in seconds
            url: SSE endpoint URL (e.g., "http://localhost:8080/sse")
            llm_client: LLM client for sampling requests
            
        Returns:
            Initialized MCPServerWorker with discovered tools
            
        Raises:
            ImportError: If mcp package is not installed
            TimeoutError: If connection times out
            ValueError: If neither command nor url provided
            
        Examples:
            # Stdio (local subprocess)
            server = await MCPServerWorker.create(
                name="Filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-fs", "/tmp"]
            )
            
            # SSE (remote HTTP)
            server = await MCPServerWorker.create(
                name="RemoteAPI",
                url="http://mcp-server:8080/sse",
                llm_client=my_llm  # For sampling
            )
        """
        if not command and not url:
            raise ValueError("Either 'command' (for stdio) or 'url' (for SSE) must be provided")
        
        tools: List[MCPTool] = []
        
        if url:
            # SSE transport
            tools = await cls._discover_tools_sse(name, url, timeout)
        else:
            # Stdio transport
            tools = await cls._discover_tools_stdio(name, command, args or [], env, timeout)
        
        return cls(
            name=name,
            command=command,
            args=args,
            description=description,
            env=env,
            tools=tools,
            url=url,
            llm_client=llm_client
        )
    
    @classmethod
    async def _discover_tools_stdio(
        cls, name: str, command: str, args: List[str], 
        env: Optional[Dict[str, str]], timeout: float
    ) -> List[MCPTool]:
        """Discover tools via stdio transport."""
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
                        
                        logger.info(f"[{name}] Discovered {len(tools)} tools via stdio")
                        
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP server '{name}' connection timed out after {timeout}s")
        except Exception as e:
            logger.error(f"[{name}] Failed to connect: {e}")
            raise
        
        return tools
    
    @classmethod
    async def _discover_tools_sse(
        cls, name: str, url: str, timeout: float
    ) -> List[MCPTool]:
        """Discover tools via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP SSE client not available. Install with: pip install 'blackboard-core[mcp]'"
            )
        
        tools: List[MCPTool] = []
        
        try:
            async with asyncio.timeout(timeout):
                async with sse_client(url) as (read, write):
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
                        
                        logger.info(f"[{name}] Discovered {len(tools)} tools via SSE")
                        
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP server '{name}' SSE connection timed out after {timeout}s")
        except Exception as e:
            logger.error(f"[{name}] Failed to connect via SSE: {e}")
            raise
        
        return tools
    
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
