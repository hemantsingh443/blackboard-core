"""
Blackboard API Application

FastAPI application for exposing Orchestrator over HTTP.
"""

import asyncio
import importlib
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional 
import os

from pydantic import BaseModel, Field

logger = logging.getLogger("blackboard.serve.app")


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateRunRequest(BaseModel):
    """Request to create a new run."""
    goal: str = Field(..., description="The objective for the orchestrator")
    max_steps: int = Field(default=20, ge=1, le=1000, description="Maximum execution steps")


class ResumeRunRequest(BaseModel):
    """Request to resume a paused run."""
    answer: str = Field(..., description="The user's answer to the pending question")
    max_steps: int = Field(default=20, ge=1, le=1000, description="Maximum additional steps")


class RunResponse(BaseModel):
    """Response containing run information."""
    id: str
    goal: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    step_count: Optional[int] = None
    artifacts_count: Optional[int] = None
    feedback_count: Optional[int] = None
    pending_input: Optional[Dict[str, Any]] = None
    last_artifact: Optional[Dict[str, Any]] = None
    event_count: Optional[int] = None
    error: Optional[str] = None


class RunListResponse(BaseModel):
    """Response containing list of runs."""
    runs: List[RunResponse]
    total: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


# ============================================================================
# Application Factory
# ============================================================================

def create_app(
    orchestrator_path: str,
    title: str = "Blackboard API",
    description: str = "Multi-agent orchestration API",
    version: str = "1.0.0",
    max_sessions: int = 100,
    sessions_dir: Optional[str] = None
) -> "FastAPI":
    """
    Create a FastAPI application for the Blackboard API.
    
    Args:
        orchestrator_path: Module path to orchestrator factory ("module:attribute")
        title: API title
        description: API description
        version: API version
        max_sessions: Maximum concurrent sessions
        sessions_dir: Directory for session persistence (autodetected if None)
        
    Returns:
        Configured FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException, BackgroundTasks
        from fastapi.responses import JSONResponse
        from sse_starlette.sse import EventSourceResponse
    except ImportError:
        raise ImportError(
            "FastAPI dependencies not installed. "
            "Install with: pip install blackboard-core[serve]"
        )
    
    from blackboard.serve.manager import SessionManager, RunStatus
    
    # Parse orchestrator path
    module_path, _, attr_name = orchestrator_path.partition(":")
    
    def load_orchestrator_factory() -> Callable:
        """Load the orchestrator factory from module path."""
        try:
            module = importlib.import_module(module_path)
            factory = getattr(module, attr_name)
            return factory
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Failed to load orchestrator from '{orchestrator_path}': {e}")
    
    # Session manager (created on startup)
    manager: Optional[SessionManager] = None
    
    # Determine default sessions dir if not provided
    final_sessions_dir = sessions_dir or os.getenv("BLACKBOARD_SESSIONS_DIR")
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        nonlocal manager
        
        # Startup
        logger.info(f"Loading orchestrator from {orchestrator_path}")
        factory = load_orchestrator_factory()
        
        manager = SessionManager(
            factory, 
            max_sessions=max_sessions,
            sessions_dir=final_sessions_dir
        )
        await manager.start()
        logger.info(f"Blackboard API started (sessions: {final_sessions_dir or 'in-memory'})")
        
        yield
        
        # Shutdown
        await manager.stop()
        logger.info("Blackboard API stopped")
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan
    )
    
    # ========================================================================
    # Routes
    # ========================================================================
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.post("/runs", response_model=RunResponse, status_code=201)
    async def create_run(request: CreateRunRequest):
        """
        Start a new orchestration run.
        
        Creates a new session and begins execution in the background.
        """
        try:
            session = await manager.create_run(
                goal=request.goal,
                max_steps=request.max_steps
            )
            return RunResponse(**session.to_dict())
        except RuntimeError as e:
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            logger.error(f"Create run error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/runs", response_model=RunListResponse)
    async def list_runs(limit: int = 100, offset: int = 0):
        """
        List all runs.
        
        Returns runs sorted by creation time (newest first).
        """
        sessions = manager.list_sessions(limit=limit, offset=offset)
        return RunListResponse(
            runs=[RunResponse(**s.to_dict()) for s in sessions],
            total=len(manager._sessions)
        )
    
    @app.get("/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str):
        """
        Get the status of a specific run.
        """
        session = manager.get_session(run_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return RunResponse(**session.to_dict())
    
    @app.get("/runs/{run_id}/full")
    async def get_run_full(run_id: str):
        """
        Get the full state of a run including all artifacts and feedback.
        """
        session = manager.get_session(run_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return session.to_full_dict()
    
    @app.post("/runs/{run_id}/resume", response_model=RunResponse)
    async def resume_run(run_id: str, request: ResumeRunRequest):
        """
        Resume a paused run with user input.
        
        Injects the user's answer and continues execution.
        """
        try:
            session = await manager.resume_run(
                session_id=run_id,
                input_data={"answer": request.answer},
                max_steps=request.max_steps
            )
            return RunResponse(**session.to_dict())
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Resume run error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/runs/{run_id}", response_model=RunResponse)
    async def cancel_run(run_id: str):
        """
        Cancel a running or paused run.
        """
        try:
            session = await manager.cancel_run(run_id)
            return RunResponse(**session.to_dict())
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        except Exception as e:
            logger.error(f"Cancel run error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/runs/{run_id}/events")
    async def get_events(run_id: str, limit: int = 100, offset: int = 0):
        """
        Get historical events for a run.
        """
        session = manager.get_session(run_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        
        events = session.events[offset:offset + limit]
        return {
            "events": events,
            "total": len(session.events)
        }
    
    @app.get("/runs/{run_id}/stream")
    async def stream_events(run_id: str):
        """
        Stream events in real-time via Server-Sent Events (SSE).
        
        Connect to this endpoint to receive live updates as the
        orchestration progresses.
        """
        session = manager.get_session(run_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        
        async def event_generator():
            """Generate SSE events from session queue."""
            import json
            
            # Send initial status
            yield {
                "event": "status",
                "data": json.dumps(session.to_dict())
            }
            
            # Stream events
            while True:
                try:
                    # Wait for next event with timeout
                    event = await asyncio.wait_for(
                        session._event_queue.get(),
                        timeout=30.0
                    )
                    yield {
                        "event": event.get("type", "message"),
                        "data": json.dumps(event)
                    }
                    
                    # Check if session is complete
                    if session.status in (
                        RunStatus.COMPLETED,
                        RunStatus.FAILED,
                        RunStatus.CANCELLED
                    ):
                        yield {
                            "event": "complete",
                            "data": json.dumps(session.to_dict())
                        }
                        break
                
                except asyncio.CancelledError:
                    # Client disconnected
                    logger.info(f"Client disconnected from stream {run_id}")
                    break
                        
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "ping",
                        "data": "{}"
                    }
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    break
        
        return EventSourceResponse(event_generator())
    
    return app


class BlackboardAPI:
    """
    Convenience wrapper for creating and running the API.
    
    Example:
        api = BlackboardAPI("my_app:create_orchestrator")
        api.run(port=8000)
    """
    
    def __init__(
        self,
        orchestrator_path: str,
        title: str = "Blackboard API",
        max_sessions: int = 100,
        sessions_dir: Optional[str] = None
    ):
        self.orchestrator_path = orchestrator_path
        self.title = title
        self.max_sessions = max_sessions
        self.sessions_dir = sessions_dir
        self._app = None
    
    @property
    def app(self):
        """Get or create the FastAPI app."""
        if self._app is None:
            self._app = create_app(
                orchestrator_path=self.orchestrator_path,
                title=self.title,
                max_sessions=self.max_sessions,
                sessions_dir=self.sessions_dir
            )
        return self._app
    
    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        log_level: str = "info",
        reload: bool = False
    ):
        """Run the API server."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn not installed. Install with: pip install blackboard-core[serve]"
            )
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=log_level,
            reload=reload
        )
