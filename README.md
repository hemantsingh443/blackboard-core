# Blackboard-Core SDK

A Python SDK implementing the **Blackboard Pattern** for centralized state multi-agent systems.

## The Paradigm Shift

| Feature | Old Way (LangChain/Graphs) | New Way (Blackboard) |
|---------|---------------------------|---------------------|
| Communication | Hot Potato: Agent A → Agent B → Agent C | Bulletin Board: All agents read/write to center |
| State | Transient: Lost after function returns | Persistent: Structured JSON ledger |
| Error Handling | Fragile try/catch | Semantic: Supervisor interprets feedback |
| Orchestration | Hard-coded scripts | Dynamic LLM-driven routing |

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from blackboard import Orchestrator, Worker, WorkerOutput, Artifact, Feedback, Blackboard

# 1. Define your workers
class TextWriter(Worker):
    name = "Writer"
    description = "Generates text content"
    
    def run(self, state: Blackboard) -> WorkerOutput:
        content = f"Content for: {state.goal}"
        return WorkerOutput(
            artifact=Artifact(type="text", content=content, creator=self.name)
        )

class TextReviewer(Worker):
    name = "Reviewer"
    description = "Reviews and approves content"
    
    def run(self, state: Blackboard) -> WorkerOutput:
        last = state.get_last_artifact()
        return WorkerOutput(
            feedback=Feedback(source=self.name, critique="Looks good!", passed=True)
        )

# 2. Create your LLM client (implements generate(prompt) -> str)
class MyLLM:
    def generate(self, prompt: str) -> str:
        # Your LLM call here (OpenAI, Anthropic, etc.)
        return '{"action": "call", "worker": "Writer", "instructions": "Generate"}'

# 3. Run the orchestrator
orchestrator = Orchestrator(
    llm=MyLLM(),
    workers=[TextWriter(), TextReviewer()]
)

result = orchestrator.run(
    goal="Write a greeting message",
    max_steps=10
)

print(f"Status: {result.status}")
print(f"Final artifact: {result.get_last_artifact().content}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
│                   (LLM Supervisor)                      │
│                                                         │
│   Observe → Reason → Act → Check                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    BLACKBOARD                           │
│                 (Shared State Ledger)                   │
│                                                         │
│   goal │ status │ artifacts │ feedback │ history       │
└────────┬─────────────────────────────┬──────────────────┘
         │                             │
    ┌────▼────┐                   ┌────▼────┐
    │ Writer  │                   │ Critic  │
    │ Worker  │                   │ Worker  │
    └─────────┘                   └─────────┘
```

## Core Concepts

### The Blackboard
The central state store. All agents read from and write to this shared ledger.

```python
state = Blackboard(goal="Design a logo")
state.add_artifact(Artifact(type="image", content="...", creator="Designer"))
state.add_feedback(Feedback(source="Critic", critique="Too busy", passed=False))
```

### Workers
Specialized agents that perform one job. They never talk to each other directly.

```python
class MyWorker(Worker):
    name = "MyWorker"
    description = "Does something specific"
    
    def run(self, state: Blackboard) -> WorkerOutput:
        # Read state, do work, return result
        return WorkerOutput(artifact=...)
```

### The Orchestrator
The LLM-driven supervisor. It reads state, decides which worker to call, and manages the loop.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
