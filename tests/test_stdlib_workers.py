"""
Tests for Standard Library Workers

Comprehensive tests for WebSearchWorker, BrowserWorker, 
CodeInterpreterWorker, and HumanProxyWorker.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

from blackboard.state import Blackboard, Status, Artifact
from blackboard.protocols import WorkerInput, WorkerOutput


# ============================================================================
# WebSearchWorker Tests
# ============================================================================

class TestWebSearchWorker:
    """Tests for WebSearchWorker."""
    
    @pytest.fixture
    def mock_httpx_response(self):
        """Create a mock httpx response."""
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        return mock
    
    @pytest.fixture
    def blackboard(self):
        """Create a test blackboard."""
        return Blackboard(goal="Test search goal")
    
    @pytest.mark.asyncio
    async def test_search_tavily_success(self, blackboard):
        """Test successful Tavily search."""
        from blackboard.stdlib.workers.search import WebSearchWorker, WebSearchInput
        
        mock_response_data = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is the first result snippet.",
                    "score": 0.95
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "This is the second result snippet.",
                    "score": 0.85
                }
            ]
        }
        
        worker = WebSearchWorker(provider="tavily", api_key="test-key")
        
        # Create mock httpx module and client
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
        
        import sys
        original_httpx = sys.modules.get("httpx")
        sys.modules["httpx"] = mock_httpx
        
        try:
            inputs = WebSearchInput(query="test query", num_results=5)
            result = await worker.run(blackboard, inputs)
            
            assert result.artifact is not None
            assert result.artifact.type == "search_results"
            assert "Test Result 1" in result.artifact.content
            assert result.artifact.metadata["num_results"] == 2
        finally:
            # Restore original httpx
            if original_httpx is not None:
                sys.modules["httpx"] = original_httpx
            else:
                sys.modules.pop("httpx", None)
    
    @pytest.mark.asyncio
    async def test_search_no_api_key(self, blackboard):
        """Test error when no API key is configured."""
        from blackboard.stdlib.workers.search import WebSearchWorker, WebSearchInput
        
        # Need to ensure env vars are cleared AND mock the error path
        import os
        
        # Create worker with no API key and ensure env vars are cleared
        saved_tavily = os.environ.pop("TAVILY_API_KEY", None)
        saved_serper = os.environ.pop("SERPER_API_KEY", None)
        
        try:
            worker = WebSearchWorker()  # No api_key provided
            inputs = WebSearchInput(query="test query")
            
            result = await worker.run(blackboard, inputs)
            
            # The error should be caught and returned in metadata
            assert result.artifact is None
            assert "error" in result.metadata
            assert "API" in result.metadata["error"] or "configured" in result.metadata["error"]
        finally:
            # Restore env vars
            if saved_tavily:
                os.environ["TAVILY_API_KEY"] = saved_tavily
            if saved_serper:
                os.environ["SERPER_API_KEY"] = saved_serper
    
    @pytest.mark.asyncio
    async def test_search_missing_query(self, blackboard):
        """Test error when query is missing."""
        from blackboard.stdlib.workers.search import WebSearchWorker
        
        worker = WebSearchWorker(api_key="test-key")
        
        # Test with None inputs
        result = await worker.run(blackboard, None)
        
        assert result.artifact is None
        assert "error" in result.metadata


# ============================================================================
# BrowserWorker Tests
# ============================================================================

class TestBrowserWorker:
    """Tests for BrowserWorker."""
    
    @pytest.fixture
    def blackboard(self):
        """Create a test blackboard."""
        return Blackboard(goal="Test browser goal")
    
    @pytest.mark.asyncio
    async def test_browser_missing_url(self, blackboard):
        """Test error when URL is missing."""
        from blackboard.stdlib.workers.browser import BrowserWorker, BrowserInput
        
        worker = BrowserWorker()
        inputs = BrowserInput(url="")
        
        result = await worker.run(blackboard, inputs)
        
        assert result.artifact is None
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_browser_extract_content(self, blackboard):
        """Test content extraction (mocked)."""
        pytest.importorskip("playwright", reason="playwright not installed")
        
        from blackboard.stdlib.workers.browser import BrowserWorker, BrowserInput
        
        worker = BrowserWorker()
        
        # Mock playwright
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Test Page Title")
        mock_page.url = "https://example.com"
        mock_page.evaluate = AsyncMock(return_value="Test page content here")
        mock_page.close = AsyncMock()
        
        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_browser.close = AsyncMock()
        
        mock_playwright = AsyncMock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_playwright.stop = AsyncMock()
        
        with patch("playwright.async_api.async_playwright") as mock_async_pw:
            mock_context = AsyncMock()
            mock_context.start = AsyncMock(return_value=mock_playwright)
            mock_async_pw.return_value = mock_context
            
            # Manually set browser state
            worker._playwright = mock_playwright
            worker._browser = mock_browser
            
            inputs = BrowserInput(url="https://example.com")
            result = await worker.run(blackboard, inputs)
            
            assert result.artifact is not None
            assert result.artifact.type == "web_content"
            assert "Test page content" in result.artifact.content
            assert result.artifact.metadata["title"] == "Test Page Title"


# ============================================================================
# CodeInterpreterWorker Tests
# ============================================================================

class TestCodeInterpreterWorker:
    """Tests for CodeInterpreterWorker."""
    
    @pytest.fixture
    def blackboard(self):
        """Create a test blackboard."""
        return Blackboard(goal="Test code execution")
    
    @pytest.mark.asyncio
    async def test_code_interpreter_missing_code(self, blackboard):
        """Test error when code is missing."""
        from blackboard.stdlib.workers.code import CodeInterpreterWorker, CodeInterpreterInput
        
        worker = CodeInterpreterWorker()
        inputs = CodeInterpreterInput(code="")
        
        result = await worker.run(blackboard, inputs)
        
        assert result.artifact is None
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_code_interpreter_unsupported_language(self, blackboard):
        """Test error for unsupported language."""
        from blackboard.stdlib.workers.code import CodeInterpreterWorker, CodeInterpreterInput
        
        worker = CodeInterpreterWorker()
        inputs = CodeInterpreterInput(code="console.log('hello')", language="javascript")
        
        result = await worker.run(blackboard, inputs)
        
        assert result.artifact is None
        assert "error" in result.metadata
        assert "not supported" in result.metadata["error"]
    
    @pytest.mark.asyncio
    async def test_code_interpreter_success(self, blackboard):
        """Test successful code execution (mocked sandbox)."""
        from blackboard.stdlib.workers.code import CodeInterpreterWorker, CodeInterpreterInput
        from blackboard.sandbox import SandboxResult
        
        worker = CodeInterpreterWorker(use_docker=False)
        
        # Mock the sandbox
        mock_sandbox = AsyncMock()
        mock_sandbox.execute = AsyncMock(return_value=SandboxResult(
            success=True,
            stdout="Hello, World!\n",
            stderr="",
            execution_time=0.5
        ))
        worker._sandbox = mock_sandbox
        
        with patch.dict("os.environ", {"BLACKBOARD_ALLOW_UNSAFE_EXECUTION": "1"}):
            inputs = CodeInterpreterInput(code="print('Hello, World!')")
            result = await worker.run(blackboard, inputs)
            
            assert result.artifact is not None
            assert result.artifact.type == "code_output"
            assert "Hello, World!" in result.artifact.content
            assert result.artifact.metadata["success"] is True


# ============================================================================
# HumanProxyWorker Tests
# ============================================================================

class TestHumanProxyWorker:
    """Tests for HumanProxyWorker."""
    
    @pytest.fixture
    def blackboard(self):
        """Create a test blackboard."""
        return Blackboard(goal="Test human input")
    
    @pytest.mark.asyncio
    async def test_human_proxy_missing_question(self, blackboard):
        """Test error when question is missing."""
        from blackboard.stdlib.workers.human import HumanProxyWorker, HumanProxyInput
        
        worker = HumanProxyWorker()
        inputs = HumanProxyInput(question="")
        
        result = await worker.run(blackboard, inputs)
        
        assert result.artifact is None
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_human_proxy_pause_mode(self, blackboard):
        """Test that worker pauses execution and sets pending_input."""
        from blackboard.stdlib.workers.human import HumanProxyWorker, HumanProxyInput
        
        worker = HumanProxyWorker()
        inputs = HumanProxyInput(
            question="What is your name?",
            context="For testing purposes"
        )
        
        result = await worker.run(blackboard, inputs)
        
        # Should have set status to PAUSED
        assert blackboard.status == Status.PAUSED
        
        # Should have stored the question in pending_input
        assert blackboard.pending_input is not None
        assert blackboard.pending_input["question"] == "What is your name?"
        assert blackboard.pending_input["context"] == "For testing purposes"
        assert blackboard.pending_input["answer"] is None
        
        # Result should indicate waiting
        assert result.metadata.get("status") == "waiting_for_input"
    
    @pytest.mark.asyncio
    async def test_human_proxy_resume_with_answer(self, blackboard):
        """Test resuming with a pending answer."""
        from blackboard.stdlib.workers.human import HumanProxyWorker, HumanProxyInput
        
        # Simulate resumed state with answer
        blackboard.pending_input = {
            "question": "What is your name?",
            "answer": "Alice"
        }
        
        worker = HumanProxyWorker()
        inputs = HumanProxyInput(question="What is your name?")
        
        result = await worker.run(blackboard, inputs)
        
        # Should return the answer as artifact
        assert result.artifact is not None
        assert result.artifact.type == "human_input"
        assert result.artifact.content == "Alice"
        assert result.artifact.metadata["resumed"] is True
        
        # Should have cleared pending_input
        assert blackboard.pending_input is None
    
    @pytest.mark.asyncio
    async def test_human_proxy_callback_mode(self, blackboard):
        """Test using callback for immediate input."""
        from blackboard.stdlib.workers.human import HumanProxyWorker, HumanProxyInput
        
        async def mock_callback(question, context, options):
            return "Callback answer"
        
        worker = HumanProxyWorker(input_callback=mock_callback)
        inputs = HumanProxyInput(question="Test question?")
        
        result = await worker.run(blackboard, inputs)
        
        # Should return immediately with callback result
        assert result.artifact is not None
        assert result.artifact.content == "Callback answer"
        assert result.artifact.metadata["via_callback"] is True
        
        # Status should NOT be paused
        assert blackboard.status != Status.PAUSED
    
    @pytest.mark.asyncio
    async def test_human_proxy_with_options(self, blackboard):
        """Test providing options to the human."""
        from blackboard.stdlib.workers.human import HumanProxyWorker, HumanProxyInput
        
        worker = HumanProxyWorker()
        inputs = HumanProxyInput(
            question="Choose a color:",
            options=["Red", "Green", "Blue"]
        )
        
        result = await worker.run(blackboard, inputs)
        
        assert blackboard.pending_input is not None
        assert blackboard.pending_input["options"] == ["Red", "Green", "Blue"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestStdlibIntegration:
    """Integration tests for stdlib workers."""
    
    @pytest.mark.asyncio
    async def test_human_proxy_full_flow(self):
        """Test full pause/resume flow with HumanProxyWorker."""
        from blackboard.stdlib.workers.human import HumanProxyWorker, HumanProxyInput
        
        # Step 1: Initial call - should pause
        blackboard = Blackboard(goal="Test human flow")
        worker = HumanProxyWorker()
        
        inputs = HumanProxyInput(question="What is 2+2?")
        result1 = await worker.run(blackboard, inputs)
        
        assert blackboard.status == Status.PAUSED
        assert blackboard.pending_input is not None
        
        # Step 2: External system provides answer
        blackboard.pending_input["answer"] = "4"
        blackboard.status = Status.GENERATING  # Reset status
        
        # Step 3: Resume - should return answer
        result2 = await worker.run(blackboard, inputs)
        
        assert result2.artifact is not None
        assert result2.artifact.content == "4"
        assert blackboard.pending_input is None
