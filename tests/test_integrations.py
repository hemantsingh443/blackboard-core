"""Tests for Phase 4: Interactive TUI and Ecosystem Adapters."""

import pytest
import sys
import warnings
from unittest.mock import MagicMock, AsyncMock

# Check if textual is available
try:
    import textual
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


class TestTextualTUI:
    """Tests for Textual-based TUI."""
    
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_import_with_textual(self):
        """Test that TUI imports successfully when textual is installed."""
        from blackboard.ui import BlackboardApp, create_tui, is_headless
        
        assert BlackboardApp is not None
        assert create_tui is not None
        assert is_headless is not None
    
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_headless_detection(self):
        """Test headless environment detection."""
        from blackboard.ui import is_headless
        import os
        
        # Store original
        original_ci = os.environ.get("CI")
        
        # Test with CI=true
        os.environ["CI"] = "true"
        assert is_headless() is True
        
        # Restore
        if original_ci:
            os.environ["CI"] = original_ci
        else:
            os.environ.pop("CI", None)
    
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_create_tui_factory(self):
        """Test TUI factory function."""
        from blackboard.ui import create_tui
        
        app = create_tui(orchestrator=None)
        assert app is not None
        assert hasattr(app, "paused")
        assert hasattr(app, "action_toggle_pause")
    
    def test_legacy_tui_deprecation_warning(self):
        """Test legacy tui.py emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib
            # Reload to trigger warning
            import blackboard.tui
            importlib.reload(blackboard.tui)
            
            # Check deprecation warning was issued
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0


class TestLangChainAdapter:
    """Tests for LangChain integration."""
    
    def test_import(self):
        """Test adapter import."""
        from blackboard.integrations.langchain import wrap_tool, wrap_tools
        
        assert wrap_tool is not None
        assert wrap_tools is not None
    
    def test_wrap_mock_tool(self):
        """Test wrapping a mock LangChain tool."""
        from blackboard.integrations.langchain import wrap_tool
        from blackboard import Worker
        
        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.name = "MockSearch"
        mock_tool.description = "Mock search tool"
        mock_tool.args_schema = None
        mock_tool.invoke = MagicMock(return_value="Search results")
        
        worker = wrap_tool(mock_tool)
        
        assert isinstance(worker, Worker)
        assert worker.name == "MockSearch"
        assert worker.description == "Mock search tool"
    
    @pytest.mark.asyncio
    async def test_wrapped_tool_execution(self):
        """Test executing a wrapped tool."""
        from blackboard.integrations.langchain import wrap_tool
        from blackboard import Blackboard
        
        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.name = "Calculator"
        mock_tool.description = "Does math"
        mock_tool.args_schema = None
        mock_tool.invoke = MagicMock(return_value="42")
        
        worker = wrap_tool(mock_tool)
        
        state = Blackboard(goal="Calculate something")
        output = await worker.run(state, {"query": "2+2"})
        
        assert output.has_artifact()
        assert "42" in output.artifact.content
        assert output.artifact.metadata["source"] == "langchain"
    
    @pytest.mark.skipif(
        "langchain_core" not in sys.modules,
        reason="langchain-core not installed"
    )
    def test_real_langchain_tool(self):
        """Test with actual LangChain tool (if installed)."""
        try:
            from langchain_core.tools import tool
            from blackboard.integrations.langchain import wrap_tool
            
            @tool
            def add(a: int, b: int) -> int:
                """Add two numbers."""
                return a + b
            
            worker = wrap_tool(add)
            assert worker.name == "add"
            assert worker.input_schema is not None
        except ImportError:
            pytest.skip("langchain-core not available")


class TestLlamaIndexAdapter:
    """Tests for LlamaIndex integration."""
    
    def test_import(self):
        """Test adapter import."""
        from blackboard.integrations.llamaindex import wrap_query_engine
        
        assert wrap_query_engine is not None
    
    def test_wrap_mock_engine(self):
        """Test wrapping a mock QueryEngine."""
        from blackboard.integrations.llamaindex import wrap_query_engine
        from blackboard import Worker
        
        # Create mock engine
        mock_engine = MagicMock()
        mock_engine.query = MagicMock(return_value="RAG response")
        
        worker = wrap_query_engine(mock_engine, name="DocSearch")
        
        assert isinstance(worker, Worker)
        assert worker.name == "DocSearch"
    
    @pytest.mark.asyncio
    async def test_wrapped_engine_execution(self):
        """Test executing a wrapped engine."""
        from blackboard.integrations.llamaindex import wrap_query_engine
        from blackboard import Blackboard
        
        # Create mock engine with source nodes
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value="Answer from docs")
        mock_response.source_nodes = []
        
        mock_engine = MagicMock()
        mock_engine.query = MagicMock(return_value=mock_response)
        
        worker = wrap_query_engine(mock_engine, name="RAG")
        
        state = Blackboard(goal="Find information about X")
        output = await worker.run(state, {"query": "What is X?"})
        
        assert output.has_artifact()
        assert output.artifact.metadata["source"] == "llamaindex"


class TestFastAPIAdapter:
    """Tests for FastAPI dependency."""
    
    def test_import(self):
        """Test adapter import."""
        from blackboard.integrations.fastapi_dep import get_orchestrator_session, OrchestratorSession
        
        assert get_orchestrator_session is not None
        assert OrchestratorSession is not None
    
    def test_session_container(self):
        """Test OrchestratorSession container."""
        from blackboard.integrations.fastapi_dep import OrchestratorSession
        
        mock_orch = MagicMock()
        session = OrchestratorSession(
            orchestrator=mock_orch,
            session_id="test-123"
        )
        
        assert session.orchestrator == mock_orch
        assert session.session_id == "test-123"
        assert session.state is None
    
    @pytest.mark.asyncio
    async def test_session_run_with_goal(self):
        """Test running session with goal."""
        from blackboard.integrations.fastapi_dep import OrchestratorSession
        from blackboard import Blackboard
        
        mock_result = Blackboard(goal="test")
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(return_value=mock_result)
        
        session = OrchestratorSession(orchestrator=mock_orch, session_id=None)
        result = await session.run(goal="Do something")
        
        mock_orch.run.assert_called_once_with(goal="Do something")
        assert result == mock_result


class TestIntegrationsPackage:
    """Tests for integrations package as a whole."""
    
    def test_package_import(self):
        """Test that integrations package imports without errors."""
        import blackboard.integrations
        
        assert hasattr(blackboard.integrations, "__all__")
    
    def test_optional_imports_dont_fail(self):
        """Test that missing optional deps don't break package."""
        # This should never raise even if langchain/llamaindex not installed
        from blackboard import integrations
        
        # Package should still be importable
        assert integrations is not None
    
    def test_exports_when_installed(self):
        """Test exports when modules are available."""
        from blackboard.integrations import wrap_tool, wrap_query_engine, get_orchestrator_session
        
        assert wrap_tool is not None
        assert wrap_query_engine is not None
        assert get_orchestrator_session is not None
