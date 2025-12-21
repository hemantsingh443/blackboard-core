# Blackboard-Core

A Python SDK for building **LLM-powered multi-agent systems** using the Blackboard Pattern.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/blackboard-core.svg)](https://pypi.org/project/blackboard-core/)

![Blackboard TUI Demo](public/demo.gif)

## What is Blackboard-Core?

Blackboard-Core provides a **centralized state architecture** for multi-agent AI systems. Instead of agents messaging each other directly, all agents read from and write to a shared **Blackboard** (state), while a **Supervisor LLM** orchestrates which agent runs next.

```
┌─────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR                          │
│  ┌─────────────┐    ┌──────────────────────────────────┐    │
│  │  Supervisor │──▶│          BLACKBOARD              │    │
│  │    (LLM)    │    │  • Goal      • Artifacts         │    │
│  └─────────────┘    │  • Status    • Feedback          │    │
│         │           │  • History   • Metadata          │    │
│         ▼           └──────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                       WORKERS                       │    │
│  │  [Writer]  [Critic]  [Refiner]  [Researcher]  ...   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Centralized State** - All agents share a typed Pydantic state model
- **LLM Orchestration** - A supervisor LLM decides which worker runs next
- **Async-First** - Built for high-performance async/await patterns
- **LiteLLM Integration** - 100+ LLM providers via `LiteLLMClient`
- **Model Context Protocol** - Connect to MCP servers for external tools
- **OpenTelemetry** - Distributed tracing with span hierarchy
- **Session Replay** - Record and replay for debugging
- **Middleware System** - Budget tracking, logging, human approval
- **Tool Calling** - Native support for OpenAI-style function calling
- **Memory System** - Vector memory with pluggable embeddings
- **Live TUI** - Real-time terminal visualization with markdown support
- **Blueprints** - Structured workflows with step-by-step control
- **Graph Memory** - GraphRAG with hybrid vector+graph search
- **Reference UI** - Streamlit dashboard for API interaction

## Installation

```bash
pip install blackboard-core

# Optional extras
pip install blackboard-core[mcp]        # Model Context Protocol
pip install blackboard-core[telemetry]  # OpenTelemetry
pip install blackboard-core[chroma]     # ChromaDB for memory
pip install blackboard-core[graph]      # Graph Memory (NetworkX)
pip install blackboard-core[ui]         # Streamlit Reference UI
pip install blackboard-core[serve]      # FastAPI server
pip install blackboard-core[all]        # Everything
```

## Quick Start

```python
from blackboard import Orchestrator, worker
from blackboard.llm import LiteLLMClient

# Define workers with decorators
@worker(name="Writer", description="Writes content")
def write(topic: str) -> str:
    return f"Article about {topic}..."

@worker(name="Critic", description="Reviews content")  
def critique(content: str) -> str:
    return "Approved!" if len(content) > 50 else "Needs more detail"

# Create orchestrator
llm = LiteLLMClient(model="gpt-4o")  # Auto-detects API key
orchestrator = Orchestrator(llm=llm, workers=[write, critique])

# Run
result = orchestrator.run_sync(goal="Write about AI safety")
print(result.artifacts[-1].content)
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Blackboard** | Shared state containing goal, artifacts, feedback, and metadata |
| **Worker** | An agent that reads state and produces artifacts or feedback |
| **Orchestrator** | Manages the control loop and calls the supervisor LLM |
| **Supervisor** | The LLM that decides which worker to call next |
| **Artifact** | Versioned output produced by a worker |
| **Feedback** | Review/critique of an artifact |

## Advanced Features

### Middleware

```python
from blackboard.middleware import BudgetMiddleware, HumanApprovalMiddleware

orchestrator = Orchestrator(
    llm=my_llm,
    workers=[...],
    middleware=[
        BudgetMiddleware(max_tokens=100000),
        HumanApprovalMiddleware(require_approval_for=["Deployer"])
    ]
)
```

### Memory System

```python
from blackboard.memory import SimpleVectorMemory, MemoryWorker
from blackboard.embeddings import OpenAIEmbedder

memory = SimpleVectorMemory(embedder=OpenAIEmbedder())
worker = MemoryWorker(memory=memory)
```

### Persistence

```python
# Save session
result.save_to_json("session.json")

# Resume later
state = Blackboard.load_from_json("session.json")
await orchestrator.run(state=state)
```

## Documentation

See [DOCS.md](DOCS.md) for the complete API reference and advanced usage guide.

## License

MIT License - see [LICENSE](LICENSE) for details.
