"""
FastAPI Dependencies

Helper dependencies for using Blackboard in FastAPI applications.

Example:
    from fastapi import FastAPI, Depends
    from blackboard.integrations.fastapi_dep import get_orchestrator_session
    
    app = FastAPI()
    
    @app.post("/run")
    async def run_agent(
        goal: str,
        session = Depends(get_orchestrator_session(llm=my_llm, workers=my_workers))
    ):
        result = await session.orchestrator.run(goal=goal)
        return {"status": result.status.value}

Requirements:
    pip install fastapi
"""

import logging
from typing import Optional, Callable, Any, TYPE_CHECKING

logger = logging.getLogger("blackboard.integrations.fastapi")

if TYPE_CHECKING:
    from blackboard import Orchestrator
    from blackboard.persistence import PersistenceLayer
    from blackboard.state import Blackboard


class OrchestratorSession:
    """
    Container for an Orchestrator with its associated session state.
    
    Attributes:
        orchestrator: The Orchestrator instance
        session_id: Current session ID (from request)
        state: Loaded Blackboard state (if session exists)
    """
    
    def __init__(
        self,
        orchestrator: "Orchestrator",
        session_id: Optional[str] = None,
        state: Optional["Blackboard"] = None
    ):
        self.orchestrator = orchestrator
        self.session_id = session_id
        self.state = state
    
    async def run(self, goal: Optional[str] = None, **kwargs) -> "Blackboard":
        """Run the orchestrator with the current session state."""
        if self.state:
            return await self.orchestrator.run(state=self.state, **kwargs)
        elif goal:
            return await self.orchestrator.run(goal=goal, **kwargs)
        else:
            raise ValueError("Either goal or existing state required")


def get_orchestrator_session(
    llm: Any = None,
    workers: list = None,
    persistence_factory: Optional[Callable] = None,
    session_header: str = "X-Session-ID",
    session_query_param: str = "session_id",
):
    """
    Create a FastAPI dependency that provides an OrchestratorSession.
    
    Args:
        llm: LLM client instance
        workers: List of workers
        persistence_factory: Async factory function that returns PersistenceLayer
        session_header: Header name for session ID
        session_query_param: Query param name for session ID
        
    Returns:
        FastAPI Depends-compatible dependency function
        
    Example:
        from blackboard.integrations.fastapi_dep import get_orchestrator_session
        
        # Create dependency
        get_session = get_orchestrator_session(
            llm=my_llm,
            workers=[MyWorker()],
            persistence_factory=lambda: SQLitePersistence("./db.sqlite")
        )
        
        @app.post("/run")
        async def run(goal: str, session = Depends(get_session)):
            result = await session.run(goal=goal)
            return {"session_id": session.session_id}
    """
    from fastapi import Request, Depends
    
    async def dependency(request: Request):
        from blackboard import Orchestrator
        
        # Extract session ID from request
        session_id = (
            request.headers.get(session_header) or
            request.query_params.get(session_query_param)
        )
        
        # Create orchestrator
        orchestrator = Orchestrator(llm=llm, workers=workers or [])
        
        state = None
        
        # Attach persistence if factory provided
        if persistence_factory:
            try:
                persistence = await persistence_factory()
                # Ensure initialized
                if hasattr(persistence, "initialize"):
                    await persistence.initialize()
                    
                orchestrator.set_persistence(persistence)
                
                # Load existing session if ID provided
                if session_id:
                    try:
                        state = await persistence.load(session_id)
                        logger.info(f"Loaded session: {session_id}")
                    except Exception as e:
                        logger.debug(f"No existing session: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize persistence: {e}")
        
        yield OrchestratorSession(
            orchestrator=orchestrator,
            session_id=session_id,
            state=state
        )
    
    return Depends(dependency)
