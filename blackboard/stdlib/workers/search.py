"""
Web Search Worker

Wraps Tavily or Serper APIs for web search functionality.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import Field

from blackboard.protocols import Worker, WorkerInput, WorkerOutput
from blackboard.state import Blackboard, Artifact

logger = logging.getLogger("blackboard.stdlib.search")


class WebSearchInput(WorkerInput):
    """Input schema for WebSearchWorker."""
    query: str = Field(..., description="The search query")
    num_results: int = Field(default=5, description="Number of results to return")
    search_depth: str = Field(default="basic", description="Search depth: 'basic' or 'advanced' (Tavily only)")


class SearchResult:
    """A single search result."""
    def __init__(self, title: str, url: str, snippet: str, score: Optional[float] = None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score
        }


class WebSearchWorker(Worker):
    """
    Web search worker using Tavily or Serper APIs.
    
    Supports both Tavily (preferred, AI-optimized) and Serper (Google-based) APIs.
    Set the appropriate environment variable to enable:
    - TAVILY_API_KEY for Tavily
    - SERPER_API_KEY for Serper
    
    If both are set, Tavily is used by default.
    
    Args:
        name: Worker name (default: "WebSearch")
        description: Worker description
        provider: Force a specific provider ("tavily" or "serper")
        api_key: Override API key (otherwise uses env var)
        timeout: HTTP request timeout in seconds
        
    Example:
        search = WebSearchWorker()
        orchestrator = Orchestrator(llm=my_llm, workers=[search])
    """
    
    name = "WebSearch"
    description = "Searches the web and returns relevant results"
    input_schema = WebSearchInput
    parallel_safe = True  # Read-only, safe for parallel execution
    
    def __init__(
        self,
        name: str = "WebSearch",
        description: str = "Searches the web and returns relevant results",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.name = name
        self.description = description
        self.provider = provider
        self._api_key = api_key
        self.timeout = timeout
        self._client = None
    
    def _get_provider(self) -> str:
        """Determine which provider to use."""
        if self.provider:
            return self.provider
        
        if self._api_key:
            # Can't auto-detect, default to tavily
            return "tavily"
        
        if os.getenv("TAVILY_API_KEY"):
            return "tavily"
        elif os.getenv("SERPER_API_KEY"):
            return "serper"
        else:
            raise ValueError(
                "No search API configured. Set TAVILY_API_KEY or SERPER_API_KEY environment variable."
            )
    
    def _get_api_key(self, provider: str) -> str:
        """Get the API key for the provider."""
        if self._api_key:
            return self._api_key
        
        if provider == "tavily":
            key = os.getenv("TAVILY_API_KEY")
            if not key:
                raise ValueError("TAVILY_API_KEY environment variable not set")
            return key
        elif provider == "serper":
            key = os.getenv("SERPER_API_KEY")
            if not key:
                raise ValueError("SERPER_API_KEY environment variable not set")
            return key
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _search_tavily(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic"
    ) -> List[SearchResult]:
        """Search using Tavily API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for WebSearchWorker. Install with: pip install httpx")
        
        api_key = self._get_api_key("tavily")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": num_results,
                    "include_answer": False,
                    "include_raw_content": False
                }
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                score=item.get("score")
            ))
        
        return results
    
    async def _search_serper(
        self,
        query: str,
        num_results: int = 5
    ) -> List[SearchResult]:
        """Search using Serper API (Google)."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for WebSearchWorker. Install with: pip install httpx")
        
        api_key = self._get_api_key("serper")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": num_results
                }
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                score=None
            ))
        
        return results
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[WebSearchInput] = None
    ) -> WorkerOutput:
        """Execute web search and return results."""
        if not inputs or not inputs.query:
            return WorkerOutput(
                metadata={"error": "query is required"}
            )
        
        try:
            provider = self._get_provider()
            logger.info(f"[{self.name}] Searching '{inputs.query}' via {provider}")
            
            if provider == "tavily":
                results = await self._search_tavily(
                    query=inputs.query,
                    num_results=inputs.num_results,
                    search_depth=inputs.search_depth
                )
            elif provider == "serper":
                results = await self._search_serper(
                    query=inputs.query,
                    num_results=inputs.num_results
                )
            else:
                return WorkerOutput(
                    metadata={"error": f"Unknown provider: {provider}"}
                )
            
            # Format results as content
            content_lines = [f"Search results for: {inputs.query}\n"]
            for i, result in enumerate(results, 1):
                content_lines.append(f"{i}. {result.title}")
                content_lines.append(f"   URL: {result.url}")
                content_lines.append(f"   {result.snippet}")
                content_lines.append("")
            
            content = "\n".join(content_lines)
            
            return WorkerOutput(
                artifact=Artifact(
                    type="search_results",
                    content=content,
                    creator=self.name,
                    metadata={
                        "query": inputs.query,
                        "provider": provider,
                        "num_results": len(results),
                        "results": [r.to_dict() for r in results]
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Search failed: {e}")
            return WorkerOutput(
                metadata={
                    "error": str(e),
                    "query": inputs.query
                }
            )

