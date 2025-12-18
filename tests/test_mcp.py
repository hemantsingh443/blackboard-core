"""Tests for MCP (Model Context Protocol) integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from blackboard import Blackboard
from blackboard.mcp import MCPServerWorker, MCPRegistry, MCPTool, MCPWorkerInput


class TestMCPTool:
    """Tests for MCPTool dataclass."""
    
    def test_mcp_tool_creation(self):
        """Test creating an MCPTool."""
        tool = MCPTool(
            name="read_file",
            description="Read a file from disk",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}}
        )
        
        assert tool.name == "read_file"
        assert tool.description == "Read a file from disk"
        assert "path" in tool.input_schema["properties"]


class TestMCPWorkerInput:
    """Tests for MCPWorkerInput."""
    
    def test_worker_input_creation(self):
        """Test creating MCPWorkerInput."""
        inputs = MCPWorkerInput(
            instructions="Read the config file",
            tool_name="read_file",
            arguments={"path": "/etc/config"}
        )
        
        assert inputs.tool_name == "read_file"
        assert inputs.arguments["path"] == "/etc/config"


class TestMCPServerWorker:
    """Tests for MCPServerWorker."""
    
    def test_worker_properties(self):
        """Test worker property accessors."""
        worker = MCPServerWorker(
            name="TestServer",
            command="npx",
            args=["-y", "test-server"],
            description="Test MCP server",
            tools=[
                MCPTool(name="tool1", description="First tool", input_schema={}),
                MCPTool(name="tool2", description="Second tool", input_schema={})
            ]
        )
        
        assert worker.name == "TestServer"
        assert worker.description == "Test MCP server"
        assert worker.parallel_safe is False
        assert len(worker.tools) == 2
    
    def test_generated_description(self):
        """Test auto-generated description from tools."""
        worker = MCPServerWorker(
            name="ToolServer",
            command="test",
            args=[],
            tools=[
                MCPTool(name="read", description="", input_schema={}),
                MCPTool(name="write", description="", input_schema={}),
            ]
        )
        
        assert "read" in worker.description
        assert "write" in worker.description
    
    def test_repr(self):
        """Test string representation."""
        worker = MCPServerWorker(
            name="Server",
            command="test",
            args=[],
            tools=[MCPTool(name="t1", description="", input_schema={})]
        )
        
        assert "Server" in repr(worker)
        assert "1 tools" in repr(worker)
    
    @pytest.mark.asyncio
    async def test_run_without_inputs(self):
        """Test run with no inputs returns error."""
        worker = MCPServerWorker(
            name="Test",
            command="test",
            args=[]
        )
        
        state = Blackboard(goal="Test")
        output = await worker.run(state, None)
        
        assert output.has_artifact()
        assert output.artifact.type == "error"
        # Error message depends on whether MCP is installed
        assert "No inputs" in output.artifact.content or "MCP package" in output.artifact.content
    
    @pytest.mark.asyncio
    async def test_run_without_tool_name(self):
        """Test run without tool_name returns available tools."""
        worker = MCPServerWorker(
            name="Test",
            command="test",
            args=[],
            tools=[
                MCPTool(name="read_file", description="", input_schema={}),
                MCPTool(name="write_file", description="", input_schema={})
            ]
        )
        
        state = Blackboard(goal="Test")
        inputs = MCPWorkerInput(instructions="do something")
        output = await worker.run(state, inputs)
        
        # Either lists tools or shows MCP not installed error
        content = output.artifact.content
        assert "read_file" in content or "MCP package" in content or output.artifact.type == "mcp_result"
    
    def test_infer_tool_from_instructions(self):
        """Test tool inference from instructions."""
        worker = MCPServerWorker(
            name="Test",
            command="test",
            args=[],
            tools=[
                MCPTool(name="read_file", description="", input_schema={}),
                MCPTool(name="write_file", description="", input_schema={})
            ]
        )
        
        inputs = MCPWorkerInput(instructions="Please read_file from disk")
        tool = worker._infer_tool_from_instructions(inputs)
        
        assert tool == "read_file"
    
    def test_infer_single_tool(self):
        """Test that single tool is auto-selected."""
        worker = MCPServerWorker(
            name="Test",
            command="test",
            args=[],
            tools=[MCPTool(name="only_tool", description="", input_schema={})]
        )
        
        inputs = MCPWorkerInput(instructions="do something")
        tool = worker._infer_tool_from_instructions(inputs)
        
        assert tool == "only_tool"


class TestMCPRegistry:
    """Tests for MCPRegistry."""
    
    def test_registry_creation(self):
        """Test creating empty registry."""
        registry = MCPRegistry()
        
        assert len(registry) == 0
        assert registry.get_workers() == []
    
    def test_registry_get(self):
        """Test getting server by name."""
        registry = MCPRegistry()
        worker = MCPServerWorker(name="Test", command="test", args=[])
        registry._servers["Test"] = worker
        
        assert registry.get("Test") is worker
        assert registry.get("NonExistent") is None
    
    def test_registry_len(self):
        """Test registry length."""
        registry = MCPRegistry()
        registry._servers["A"] = MCPServerWorker(name="A", command="a", args=[])
        registry._servers["B"] = MCPServerWorker(name="B", command="b", args=[])
        
        assert len(registry) == 2
    
    def test_list_all_tools(self):
        """Test listing tools from all servers."""
        registry = MCPRegistry()
        registry._servers["Server1"] = MCPServerWorker(
            name="Server1",
            command="s1",
            args=[],
            tools=[MCPTool(name="tool1", description="", input_schema={})]
        )
        registry._servers["Server2"] = MCPServerWorker(
            name="Server2",
            command="s2",
            args=[],
            tools=[MCPTool(name="tool2", description="", input_schema={})]
        )
        
        all_tools = registry.list_all_tools()
        
        assert "Server1" in all_tools
        assert "Server2" in all_tools
        assert len(all_tools["Server1"]) == 1
        assert all_tools["Server1"][0].name == "tool1"
    
    def test_repr(self):
        """Test string representation."""
        registry = MCPRegistry()
        registry._servers["A"] = MCPServerWorker(name="A", command="a", args=[])
        
        assert "1 servers" in repr(registry)


class TestMCPToolWorker:
    """Tests for MCPToolWorker dynamic expansion."""
    
    def test_expand_to_workers(self):
        """Test expanding server to individual tool workers."""
        from blackboard.mcp import MCPToolWorker
        
        server = MCPServerWorker(
            name="TestServer",
            command="test",
            args=[],
            tools=[
                MCPTool(name="read_file", description="Read a file", input_schema={}),
                MCPTool(name="write_file", description="Write a file", input_schema={})
            ]
        )
        
        workers = server.expand_to_workers()
        
        assert len(workers) == 2
        assert isinstance(workers[0], MCPToolWorker)
        assert workers[0].name == "TestServer:read_file"
        assert workers[1].name == "TestServer:write_file"
    
    def test_tool_worker_properties(self):
        """Test MCPToolWorker properties."""
        from blackboard.mcp import MCPToolWorker
        
        server = MCPServerWorker(
            name="FS",
            command="test",
            args=[],
            tools=[MCPTool(
                name="read",
                description="Read content",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}}
            )]
        )
        
        worker = server.expand_to_workers()[0]
        
        assert worker.name == "FS:read"
        assert worker.description == "Read content"
        assert worker.parallel_safe is False
    
    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        from blackboard.tools import ToolDefinition
        
        server = MCPServerWorker(
            name="Server",
            command="test",
            args=[],
            tools=[MCPTool(
                name="add",
                description="Add numbers",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            )]
        )
        
        definitions = server.get_tool_definitions()
        
        assert len(definitions) == 1
        assert isinstance(definitions[0], ToolDefinition)
        assert definitions[0].name == "add"
        assert len(definitions[0].parameters) == 2
    
    def test_mcp_tool_to_definition(self):
        """Test MCPTool.to_tool_definition conversion."""
        from blackboard.tools import ToolDefinition
        
        tool = MCPTool(
            name="search",
            description="Search for files",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10}
                },
                "required": ["query"]
            }
        )
        
        definition = tool.to_tool_definition()
        
        assert definition.name == "search"
        assert definition.description == "Search for files"
        assert len(definition.parameters) == 2
        
        query_param = next(p for p in definition.parameters if p.name == "query")
        assert query_param.type == "string"
        assert query_param.required is True
        
        limit_param = next(p for p in definition.parameters if p.name == "limit")
        assert limit_param.type == "integer"
        assert limit_param.required is False


class TestMCPImport:
    """Tests for MCP module imports."""
    
    def test_import(self):
        """Test that MCP module can be imported."""
        from blackboard.mcp import MCPServerWorker, MCPRegistry, MCPTool
        
        assert MCPServerWorker is not None
        assert MCPRegistry is not None
        assert MCPTool is not None
