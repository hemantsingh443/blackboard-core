"""
LangChain Adapter

Wrap LangChain tools as Blackboard Workers.

Example:
    from langchain_community.tools import TavilySearchResults
    from blackboard.integrations.langchain import wrap_tool
    
    tool = TavilySearchResults()
    worker = wrap_tool(tool)
    
    orchestrator = Orchestrator(llm=llm, workers=[worker])

Requirements:
    pip install langchain-core>=0.3.0
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

logger = logging.getLogger("blackboard.integrations.langchain")

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def wrap_tool(
    tool: "BaseTool",
    artifact_type: str = "text",
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> "Worker":
    """
    Wrap a LangChain BaseTool as a Blackboard Worker.
    
    Args:
        tool: LangChain BaseTool instance
        artifact_type: Type for the output artifact
        name: Optional override for worker name
        description: Optional override for description
        
    Returns:
        A Blackboard Worker that wraps the tool
        
    Example:
        from langchain_community.tools import WikipediaQueryRun
        from blackboard.integrations.langchain import wrap_tool
        
        wiki = WikipediaQueryRun()
        worker = wrap_tool(wiki, artifact_type="research")
    """
    from blackboard import Worker, WorkerOutput
    from blackboard.state import Artifact
    
    # Get schema using Pydantic v2 model_json_schema()
    input_schema = None
    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        try:
            input_schema = tool.args_schema.model_json_schema()
        except AttributeError:
            # Fallback for Pydantic v1 compatibility
            try:
                input_schema = tool.args_schema.schema()
            except Exception:
                pass
    
    worker_name = name or tool.name
    worker_description = description or tool.description or f"LangChain tool: {tool.name}"
    
    class LangChainWorker(Worker):
        name = worker_name
        description = worker_description
        parallel_safe = True  # Most tools are stateless
        
        def __init__(self):
            super().__init__()
            self._tool = tool
            self._input_schema = input_schema
        
        @property
        def input_schema(self) -> Optional[Dict[str, Any]]:
            return self._input_schema
        
        async def run(self, state, inputs=None) -> WorkerOutput:
            """Execute the LangChain tool."""
            try:
                # LangChain tools have sync invoke() - run in thread
                if inputs:
                    result = await asyncio.to_thread(self._tool.invoke, inputs)
                else:
                    # Use goal as default input if no specific inputs
                    result = await asyncio.to_thread(self._tool.invoke, {"query": state.goal})
                
                return WorkerOutput(
                    artifact=Artifact(
                        type=artifact_type,
                        content=str(result),
                        creator=self.name,
                        metadata={"source": "langchain", "tool": tool.name}
                    )
                )
            except Exception as e:
                logger.error(f"LangChain tool {tool.name} failed: {e}")
                return WorkerOutput(
                    artifact=Artifact(
                        type="error",
                        content=f"Tool error: {str(e)}",
                        creator=self.name,
                        metadata={"error": str(e)}
                    )
                )
    
    return LangChainWorker()


def wrap_tools(tools: list, **kwargs) -> list:
    """Wrap multiple LangChain tools as Workers.
    
    Args:
        tools: List of LangChain BaseTool instances
        **kwargs: Passed to each wrap_tool call
        
    Returns:
        List of Blackboard Workers
    """
    return [wrap_tool(t, **kwargs) for t in tools]
