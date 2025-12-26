"""
Blackboard Integrations Module

Adapters for integrating external frameworks with Blackboard:

- LangChain: wrap_tool() - Wrap LangChain BaseTool as Worker
- LlamaIndex: wrap_query_engine() - Wrap QueryEngine as Worker
- FastAPI: get_orchestrator_session() - Dependency for FastAPI
"""

__all__ = []

# LangChain adapter
try:
    from .langchain import wrap_tool
    __all__.append("wrap_tool")
except ImportError:
    pass

# LlamaIndex adapter
try:
    from .llamaindex import wrap_query_engine
    __all__.append("wrap_query_engine")
except ImportError:
    pass

# FastAPI dependency
try:
    from .fastapi_dep import get_orchestrator_session
    __all__.append("get_orchestrator_session")
except ImportError:
    pass
