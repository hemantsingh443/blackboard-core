"""
Squad Patterns - Pre-configured Agent Factories

Provides convenient factory functions for creating pre-configured Agent instances
for common multi-agent patterns.

.. versionadded:: 1.6.0

Example:
    from blackboard.patterns import research_squad, code_squad
    
    # Create a research squad with web search and browser workers
    researcher = research_squad(llm, config=parent_config.for_child_agent())
    
    # Use as a worker in parent orchestrator
    orchestrator = Orchestrator(llm=llm, workers=[researcher, writer])
"""

from typing import Any, List, Optional

from .core import Agent, LLMClient
from .config import BlackboardConfig
from .events import EventBus
from .protocols import Worker


def create_squad(
    name: str,
    description: str,
    llm: LLMClient,
    workers: List[Worker],
    config: Optional[BlackboardConfig] = None,
    current_depth: int = 0,
    persistence: Optional[Any] = None,
    event_bus: Optional[EventBus] = None,
    parent_event_bus: Optional[EventBus] = None,
    verbose: bool = False
) -> Agent:
    """
    Create a Squad (pre-configured Agent).
    
    A Squad is simply a factory function that returns an Agent with
    specific workers and configuration. This is the base factory that
    all other squad patterns use.
    
    Args:
        name: Name of the squad (for tool definitions)
        description: Description for parent orchestrator
        llm: LLM client for internal orchestration
        workers: Workers available to this squad
        config: BlackboardConfig (creates default if not provided)
        current_depth: Current recursion depth
        persistence: Optional shared persistence layer
        event_bus: Optional dedicated event bus
        parent_event_bus: Optional parent bus for event bubbling
        verbose: Enable verbose logging
        
    Returns:
        Configured Agent instance
        
    Example:
        squad = create_squad(
            name="ResearchTeam",
            description="Researches topics using web search",
            llm=llm,
            workers=[WebSearchWorker(), BrowserWorker()]
        )
    """
    return Agent(
        name=name,
        description=description,
        llm=llm,
        workers=workers,
        config=config or BlackboardConfig(),
        current_depth=current_depth,
        persistence=persistence,
        event_bus=event_bus,
        parent_event_bus=parent_event_bus,
        verbose=verbose
    )


def research_squad(
    llm: LLMClient,
    config: Optional[BlackboardConfig] = None,
    name: str = "ResearchSquad",
    description: str = "Performs comprehensive research on a topic using web search and browsing",
    additional_workers: Optional[List[Worker]] = None,
    **kwargs
) -> Agent:
    """
    Create a Research Squad with web search and browser capabilities.
    
    This squad is designed for tasks that require gathering information
    from the web, reading pages, and synthesizing findings.
    
    Note: The WebSearchWorker and BrowserWorker must be available in
    the stdlib to use this pattern. If not installed, you can provide
    your own workers via additional_workers.
    
    Args:
        llm: LLM client for orchestration
        config: Optional configuration (inherits from parent if not provided)
        name: Squad name (default: "ResearchSquad")
        description: Squad description
        additional_workers: Extra workers to include
        **kwargs: Additional arguments passed to Agent
        
    Returns:
        Configured Agent for research tasks
    """
    workers = []
    
    # Try to import stdlib workers
    try:
        from .stdlib.workers.websearch import WebSearchWorker
        workers.append(WebSearchWorker())
    except ImportError:
        pass
    
    try:
        from .stdlib.workers.browser import BrowserWorker
        workers.append(BrowserWorker())
    except ImportError:
        pass
    
    if additional_workers:
        workers.extend(additional_workers)
    
    return create_squad(
        name=name,
        description=description,
        llm=llm,
        workers=workers,
        config=config,
        **kwargs
    )


def code_squad(
    llm: LLMClient,
    config: Optional[BlackboardConfig] = None,
    name: str = "CodeSquad",
    description: str = "Writes, executes, and debugs code",
    additional_workers: Optional[List[Worker]] = None,
    **kwargs
) -> Agent:
    """
    Create a Code Squad with code interpretation capabilities.
    
    This squad is designed for tasks that require writing and executing
    code, debugging, and providing code-based solutions.
    
    Note: The CodeInterpreterWorker must be available in the stdlib.
    Requires a configured sandbox (InsecureLocalExecutor or DockerSandbox).
    
    Args:
        llm: LLM client for orchestration
        config: Optional configuration
        name: Squad name (default: "CodeSquad")
        description: Squad description
        additional_workers: Extra workers to include
        **kwargs: Additional arguments passed to Agent
        
    Returns:
        Configured Agent for code tasks
    """
    workers = []
    
    try:
        from .stdlib.workers.code import CodeInterpreterWorker
        workers.append(CodeInterpreterWorker())
    except ImportError:
        pass
    
    if additional_workers:
        workers.extend(additional_workers)
    
    return create_squad(
        name=name,
        description=description,
        llm=llm,
        workers=workers,
        config=config,
        **kwargs
    )


def memory_squad(
    llm: LLMClient,
    config: Optional[BlackboardConfig] = None,
    name: str = "MemorySquad",
    description: str = "Manages and searches long-term memory",
    additional_workers: Optional[List[Worker]] = None,
    **kwargs
) -> Agent:
    """
    Create a Memory Squad with long-term memory capabilities.
    
    This squad is designed for tasks that require storing, retrieving,
    and managing information in a vector memory store.
    
    Args:
        llm: LLM client for orchestration
        config: Optional configuration
        name: Squad name (default: "MemorySquad")
        description: Squad description
        additional_workers: Extra workers to include
        **kwargs: Additional arguments passed to Agent
        
    Returns:
        Configured Agent for memory tasks
    """
    workers = []
    
    try:
        from .memory import MemoryWorker, SimpleVectorMemory
        workers.append(MemoryWorker(memory=SimpleVectorMemory()))
    except ImportError:
        pass
    
    if additional_workers:
        workers.extend(additional_workers)
    
    return create_squad(
        name=name,
        description=description,
        llm=llm,
        workers=workers,
        config=config,
        **kwargs
    )


# Type aliases for convenience
Squad = Agent  # A Squad is just an Agent with a specific configuration

__all__ = [
    "create_squad",
    "research_squad",
    "code_squad",
    "memory_squad",
    "Squad",
]
