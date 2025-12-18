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


class TestMCPImport:
    """Tests for MCP module imports."""
    
    def test_import(self):
        """Test that MCP module can be imported."""
        from blackboard.mcp import MCPServerWorker, MCPRegistry, MCPTool
        
        assert MCPServerWorker is not None
        assert MCPRegistry is not None
        assert MCPTool is not None
