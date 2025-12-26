"""
LlamaIndex Adapter

Wrap LlamaIndex QueryEngines as Blackboard Workers.

Example:
    from llama_index.core import VectorStoreIndex
    from blackboard.integrations.llamaindex import wrap_query_engine
    
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine()
    worker = wrap_query_engine(engine, name="RAG")
    
    orchestrator = Orchestrator(llm=llm, workers=[worker])

Requirements:
    pip install llama-index-core>=0.14.0
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

logger = logging.getLogger("blackboard.integrations.llamaindex")

if TYPE_CHECKING:
    from llama_index.core.query_engine import BaseQueryEngine


def wrap_query_engine(
    engine: "BaseQueryEngine",
    name: str = "RAG",
    description: str = "",
    artifact_type: str = "rag_response",
) -> "Worker":
    """
    Wrap a LlamaIndex QueryEngine as a Blackboard Worker.
    
    Args:
        engine: LlamaIndex QueryEngine instance
        name: Worker name
        description: Worker description
        artifact_type: Type for output artifacts
        
    Returns:
        A Blackboard Worker that queries the engine
        
    Example:
        from llama_index.core import VectorStoreIndex
        from blackboard.integrations.llamaindex import wrap_query_engine
        
        index = VectorStoreIndex.from_documents(docs)
        engine = index.as_query_engine()
        
        worker = wrap_query_engine(
            engine,
            name="DocumentSearch",
            description="Searches internal documents for relevant information"
        )
    """
    from blackboard import Worker, WorkerOutput
    from blackboard.state import Artifact
    
    # Store in closure to avoid class attribute shadowing
    worker_name = name
    worker_description = description or f"Query indexed documents via {name}"
    worker_artifact_type = artifact_type
    
    class LlamaIndexWorker(Worker):
        name = worker_name
        description = worker_description
        parallel_safe = True
        input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
        
        def __init__(self):
            super().__init__()
            self._engine = engine
        
        async def run(self, state, inputs=None) -> WorkerOutput:
            """Query the LlamaIndex engine."""
            try:
                # Get query from inputs or use goal
                query = state.goal
                if inputs:
                    query = inputs.get("query", query)
                
                # QueryEngine.query() is sync - run in thread
                response = await asyncio.to_thread(self._engine.query, query)
                
                # Extract response text
                response_text = str(response)
                
                # Try to get source nodes for metadata
                metadata = {"source": "llamaindex"}
                if hasattr(response, "source_nodes"):
                    sources = []
                    for node in response.source_nodes[:3]:  # Limit to 3
                        if hasattr(node, "node"):
                            sources.append({
                                "score": getattr(node, "score", None),
                                "text": node.node.get_content()[:200]
                            })
                    metadata["sources"] = sources
                
                return WorkerOutput(
                    artifact=Artifact(
                        type=worker_artifact_type,
                        content=response_text,
                        creator=self.name,
                        metadata=metadata
                    )
                )
            except Exception as e:
                logger.error(f"LlamaIndex query failed: {e}")
                return WorkerOutput(
                    artifact=Artifact(
                        type="error",
                        content=f"Query error: {str(e)}",
                        creator=self.name,
                        metadata={"error": str(e)}
                    )
                )
    
    return LlamaIndexWorker()
