# Blackboard-Core

A Python SDK for building **LLM-powered multi-agent systems** using the Blackboard Pattern.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/blackboard-core.svg)](https://pypi.org/project/blackboard-core/)

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
- **Middleware System** - Budget tracking, logging, human approval, auto-summarization
- **Tool Calling** - Native support for OpenAI-style function calling
- **Persistence** - Save/resume sessions with optimistic locking
- **Memory System** - Vector memory with pluggable embeddings
- **Parallel Execution** - Run independent workers concurrently

## Installation

```bash
pip install blackboard-core
```

## Quick Start

```python
import asyncio
from blackboard import Orchestrator, Worker, WorkerOutput, Artifact, Blackboard

# 1. Define your workers
class Writer(Worker):
    name = "Writer"
    description = "Generates text content based on the goal"
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        # Your LLM call here
        content = f"Generated content for: {state.goal}"
        return WorkerOutput(
            artifact=Artifact(type="text", content=content, creator=self.name)
        )

class Critic(Worker):
    name = "Critic"
    description = "Reviews content and provides feedback"
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        artifact = state.get_last_artifact()
        # Your review logic here
        return WorkerOutput(
            feedback=Feedback(
                source=self.name,
                critique="Looks good!",
                passed=True,
                artifact_id=artifact.id
            )
        )

# 2. Create an LLM client (implement the LLMClient protocol)
class MyLLM:
    async def generate(self, prompt: str) -> str:
        # Your LLM API call (OpenAI, Anthropic, etc.)
        return '{"action": "call", "worker": "Writer", "reasoning": "Start writing"}'

# 3. Run the orchestrator
async def main():
    orchestrator = Orchestrator(
        llm=MyLLM(),
        workers=[Writer(), Critic()],
        verbose=True
    )
    
    result = await orchestrator.run(
        goal="Write a haiku about programming",
        max_steps=10
    )
    
    print(f"Final status: {result.status}")
    print(f"Artifacts: {[a.content for a in result.artifacts]}")

asyncio.run(main())
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

### Tool Calling (OpenAI-style)

```python
from blackboard.tools import ToolCallingLLMClient

class MyToolLLM(ToolCallingLLMClient):
    async def generate_with_tools(self, prompt, tools):
        # Use OpenAI function calling
        ...
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
