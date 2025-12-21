"""
Tests for Blackboard Serve API

Comprehensive tests for the FastAPI endpoints and SessionManager.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

from blackboard.state import Blackboard, Status, Artifact


# ============================================================================
# SessionManager Tests
# ============================================================================

class TestSessionManager:
    """Tests for SessionManager."""
    
    @pytest.fixture
    def mock_orchestrator_factory(self):
        """Create a mock orchestrator factory."""
        def factory():
            mock_orch = MagicMock()
            mock_orch.event_bus = MagicMock()
            mock_orch.event_bus.subscribe_all_async = MagicMock()
            
            async def mock_run(goal=None, state=None, max_steps=20):
                bb = state or Blackboard(goal=goal or "test")
                bb.update_status(Status.DONE)
                return bb
            
            mock_orch.run = mock_run
            return mock_orch
        
        return factory
    
    @pytest.mark.asyncio
    async def test_create_run(self, mock_orchestrator_factory):
        """Test creating a new run."""
        from blackboard.serve.manager import SessionManager, RunStatus
        
        manager = SessionManager(mock_orchestrator_factory, max_sessions=10)
        
        session = await manager.create_run("Test goal", start_immediately=False)
        
        assert session.id is not None
        assert session.goal == "Test goal"
        assert session.status == RunStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_session(self, mock_orchestrator_factory):
        """Test retrieving a session by ID."""
        from blackboard.serve.manager import SessionManager
        
        manager = SessionManager(mock_orchestrator_factory)
        session = await manager.create_run("Test goal", start_immediately=False)
        
        retrieved = manager.get_session(session.id)
        
        assert retrieved is not None
        assert retrieved.id == session.id
    
    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_orchestrator_factory):
        """Test retrieving non-existent session."""
        from blackboard.serve.manager import SessionManager
        
        manager = SessionManager(mock_orchestrator_factory)
        
        retrieved = manager.get_session("nonexistent-id")
        
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, mock_orchestrator_factory):
        """Test listing sessions."""
        from blackboard.serve.manager import SessionManager
        
        manager = SessionManager(mock_orchestrator_factory)
        
        await manager.create_run("Goal 1", start_immediately=False)
        await manager.create_run("Goal 2", start_immediately=False)
        await manager.create_run("Goal 3", start_immediately=False)
        
        sessions = manager.list_sessions(limit=2)
        
        assert len(sessions) == 2
    
    @pytest.mark.asyncio
    async def test_max_sessions_limit(self, mock_orchestrator_factory):
        """Test max sessions limit enforcement."""
        from blackboard.serve.manager import SessionManager
        
        manager = SessionManager(mock_orchestrator_factory, max_sessions=2)
        
        await manager.create_run("Goal 1", start_immediately=False)
        await manager.create_run("Goal 2", start_immediately=False)
        
        with pytest.raises(RuntimeError, match="Maximum number of sessions"):
            await manager.create_run("Goal 3", start_immediately=False)
    
    @pytest.mark.asyncio
    async def test_cancel_run(self, mock_orchestrator_factory):
        """Test cancelling a run."""
        from blackboard.serve.manager import SessionManager, RunStatus
        
        manager = SessionManager(mock_orchestrator_factory)
        session = await manager.create_run("Test goal", start_immediately=False)
        
        cancelled = await manager.cancel_run(session.id)
        
        assert cancelled.status == RunStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_session_to_dict(self, mock_orchestrator_factory):
        """Test session serialization."""
        from blackboard.serve.manager import SessionManager
        
        manager = SessionManager(mock_orchestrator_factory)
        session = await manager.create_run("Test goal", start_immediately=False)
        
        data = session.to_dict()
        
        assert data["id"] == session.id
        assert data["goal"] == "Test goal"
        assert "status" in data
        assert "created_at" in data


# ============================================================================
# FastAPI App Tests
# ============================================================================

class TestBlackboardAPI:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def mock_orchestrator_factory(self):
        """Create a mock orchestrator factory."""
        def factory():
            mock_orch = MagicMock()
            mock_orch.event_bus = MagicMock()
            mock_orch.event_bus.subscribe_all_async = MagicMock()
            
            async def mock_run(goal=None, state=None, max_steps=20):
                bb = state or Blackboard(goal=goal or "test")
                bb.update_status(Status.DONE)
                bb.add_artifact(Artifact(
                    type="text",
                    content="Test output",
                    creator="TestWorker"
                ))
                return bb
            
            mock_orch.run = mock_run
            return mock_orch
        
        return factory
    
    @pytest.fixture
    def test_client(self, mock_orchestrator_factory):
        """Create a test client with mocked dependencies."""
        # This requires FastAPI to be installed
        pytest.importorskip("fastapi")
        pytest.importorskip("sse_starlette")
        
        from fastapi.testclient import TestClient
        from blackboard.serve.app import create_app
        
        # Create app with test module path
        with patch("blackboard.serve.app.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.create_orchestrator = mock_orchestrator_factory
            mock_import.return_value = mock_module
            
            app = create_app("test_module:create_orchestrator")
            
            # Use TestClient as context manager to trigger lifespan
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_create_run(self, test_client):
        """Test creating a run via POST /runs."""
        response = test_client.post(
            "/runs",
            json={"goal": "Test goal", "max_steps": 10}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["goal"] == "Test goal"
    
    def test_create_run_validation(self, test_client):
        """Test validation on create run."""
        # Missing goal
        response = test_client.post(
            "/runs",
            json={"max_steps": 10}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_list_runs(self, test_client):
        """Test listing runs via GET /runs."""
        # Create a few runs first
        test_client.post("/runs", json={"goal": "Goal 1"})
        test_client.post("/runs", json={"goal": "Goal 2"})
        
        response = test_client.get("/runs")
        
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "total" in data
    
    def test_get_run(self, test_client):
        """Test getting a specific run."""
        # Create a run
        create_response = test_client.post("/runs", json={"goal": "Test goal"})
        run_id = create_response.json()["id"]
        
        response = test_client.get(f"/runs/{run_id}")
        
        assert response.status_code == 200
        assert response.json()["id"] == run_id
    
    def test_get_run_not_found(self, test_client):
        """Test getting non-existent run."""
        response = test_client.get("/runs/nonexistent-id")
        
        assert response.status_code == 404
    
    def test_cancel_run(self, test_client):
        """Test cancelling a run."""
        # Create a run
        create_response = test_client.post("/runs", json={"goal": "Test goal"})
        run_id = create_response.json()["id"]
        
        response = test_client.delete(f"/runs/{run_id}")
        
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"


# ============================================================================
# Resume Flow Tests
# ============================================================================

class TestResumeFlow:
    """Tests for pause/resume flow."""
    
    @pytest.mark.asyncio
    async def test_pause_and_resume_session(self):
        """Test full pause and resume flow."""
        from blackboard.serve.manager import SessionManager, RunStatus, RunSession
        from blackboard.events import EventBus
        
        paused_state = None
        
        def factory():
            mock_orch = MagicMock()
            mock_orch.event_bus = EventBus()
            
            async def mock_run(goal=None, state=None, max_steps=20):
                nonlocal paused_state
                
                if state and state.pending_input and state.pending_input.get("answer"):
                    # Resuming - complete
                    state.update_status(Status.DONE)
                    state.add_artifact(Artifact(
                        type="answer",
                        content=state.pending_input["answer"],
                        creator="TestWorker"
                    ))
                    return state
                else:
                    # First run - pause
                    bb = state or Blackboard(goal=goal or "test")
                    bb.pending_input = {"question": "What is your name?", "answer": None}
                    bb.update_status(Status.PAUSED)
                    paused_state = bb
                    return bb
            
            mock_orch.run = mock_run
            return mock_orch
        
        manager = SessionManager(factory)
        
        # Start session (will pause)
        session = await manager.create_run("Test goal")
        
        # Wait for task to complete
        await asyncio.sleep(0.1)
        
        assert session.status == RunStatus.PAUSED
        assert session.state is not None
        assert session.state.pending_input is not None
        
        # Resume with answer
        await manager.resume_run(
            session.id,
            {"answer": "Alice"},
            max_steps=10
        )
        
        # Wait for resume task
        await asyncio.sleep(0.1)
        
        assert session.status == RunStatus.COMPLETED
        assert session.state.get_last_artifact().content == "Alice"
