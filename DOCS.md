# Blackboard-Core Documentation

Complete API reference and usage guide for building LLM-powered multi-agent systems.

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Creating Workers](#creating-workers)
4. [Implementing an LLM Client](#implementing-an-llm-client)
5. [Running the Orchestrator](#running-the-orchestrator)
6. [Middleware](#middleware)
7. [Tool Calling](#tool-calling)
8. [Memory System](#memory-system)
9. [Persistence](#persistence)
10. [Events & Observability](#events--observability)
11. [Error Handling](#error-handling)
12. [Best Practices](#best-practices)

---

## Installation

```bash
pip install blackboard-core
```

Optional dependencies:
```bash
pip install sentence-transformers  # For LocalEmbedder
pip install openai                 # For OpenAIEmbedder
```

---

## Core Concepts

### The Blackboard Pattern

The Blackboard Pattern is an architectural style where multiple specialized agents collaborate through a shared workspace:

1. **Blackboard** - Shared state that all agents can read/write
2. **Workers** - Specialized agents that perform specific tasks
3. **Supervisor** - An LLM that decides which worker to call next

### State Model

```python
from blackboard import Blackboard, Artifact, Feedback, Status

# Create a new session
state = Blackboard(goal="Write a technical blog post")

# State contains:
# - goal: The objective (immutable)
# - status: Current phase (PLANNING, GENERATING, DONE, etc.)
# - artifacts: List of versioned outputs
# - feedback: List of reviews/critiques
# - metadata: Arbitrary key-value data
# - history: Execution log
```

---

## Creating Workers

Workers are the agents that perform actual work. Each worker:
- Reads from the Blackboard
- Performs a task (often using an LLM)
- Returns artifacts and/or feedback

```python
from blackboard import Worker, WorkerOutput, Artifact, Feedback, Blackboard

class ResearchWorker(Worker):
    name = "Researcher"
    description = "Gathers information on a topic"
    
    # Optional: Define expected input schema
    input_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Topic to research"}
        }
    }
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        topic = inputs.get("topic", state.goal) if inputs else state.goal
        
        # Your research logic here (API calls, LLM, etc.)
        research_result = f"Research findings about {topic}..."
        
        return WorkerOutput(
            artifact=Artifact(
                type="research",
                content=research_result,
                creator=self.name,
                metadata={"topic": topic}
            )
        )


class CriticWorker(Worker):
    name = "Critic"
    description = "Reviews artifacts and provides feedback"
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        # Get the most recent artifact to review
        artifact = state.get_last_artifact()
        
        if not artifact:
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    critique="No artifact to review",
                    passed=False
                )
            )
        
        # Your review logic here
        is_good = len(artifact.content) > 100
        
        return WorkerOutput(
            feedback=Feedback(
                source=self.name,
                artifact_id=artifact.id,
                critique="Content is detailed enough" if is_good else "Needs more detail",
                passed=is_good
            )
        )
```

### Worker Best Practices

1. **Single Responsibility** - Each worker should do one thing well
2. **Stateless** - Workers should not store state between calls
3. **Descriptive Names** - The supervisor LLM uses names to choose workers
4. **Input Schema** - Define schemas for complex inputs

---

## Implementing an LLM Client

The Orchestrator needs an LLM to act as the supervisor. Implement the `LLMClient` protocol:

### Basic LLM Client (JSON Response)

```python
import openai
from blackboard import LLMClient

class OpenAILLM(LLMClient):
    def __init__(self, model="gpt-4"):
        self.client = openai.AsyncOpenAI()
        self.model = model
    
    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
```

### Tool Calling LLM Client (Recommended)

For better reliability, use native tool calling:

```python
from blackboard.tools import ToolCallingLLMClient, ToolCall

class OpenAIToolLLM(ToolCallingLLMClient):
    def __init__(self, model="gpt-4"):
        self.client = openai.AsyncOpenAI()
        self.model = model
    
    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    async def generate_with_tools(self, prompt: str, tools: list) -> list:
        # Convert to OpenAI format
        openai_tools = [t.to_openai_format() for t in tools]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=openai_tools
        )
        
        # Parse tool calls
        result = []
        for tc in response.choices[0].message.tool_calls or []:
            result.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments)
            ))
        return result
```

---

## Running the Orchestrator

```python
import asyncio
from blackboard import Orchestrator

async def main():
    orchestrator = Orchestrator(
        llm=OpenAILLM(),
        workers=[ResearchWorker(), CriticWorker()],
        
        # Options
        verbose=True,              # Enable logging
        enable_parallel=True,      # Allow parallel worker execution
        use_tool_calling=True,     # Use native tool calling
        strict_tools=False,        # Crash on tool config errors
        allow_json_fallback=True,  # Fallback to JSON if tools fail
    )
    
    result = await orchestrator.run(
        goal="Research AI safety and write a summary",
        max_steps=20
    )
    
    print(f"Status: {result.status}")
    for artifact in result.artifacts:
        print(f"- {artifact.type}: {artifact.content[:100]}...")

asyncio.run(main())
```

### Resuming a Session

```python
# Resume from existing state
result = await orchestrator.run(state=existing_state, max_steps=10)
```

---

## Middleware

Middleware intercepts the orchestration flow for cross-cutting concerns.

### Budget Middleware

```python
from blackboard.middleware import BudgetMiddleware

budget = BudgetMiddleware(
    max_tokens=100000,
    max_cost=5.0,
    cost_per_1k_input=0.01,
    cost_per_1k_output=0.03
)

orchestrator = Orchestrator(llm=llm, workers=workers, middleware=[budget])
```

### Human Approval Middleware

```python
from blackboard.middleware import HumanApprovalMiddleware, ApprovalRequired

# Option 1: Catch the exception
approval = HumanApprovalMiddleware(require_approval_for=["Deployer"])

try:
    await orchestrator.run(goal="Deploy to production")
except ApprovalRequired as e:
    print(f"Approval needed for: {e.worker_name}")
    # Save state, wait for human, resume later

# Option 2: Async callback
async def check_approval(worker, instructions):
    return await database.get_approval_flag(worker)

approval = HumanApprovalMiddleware(
    require_approval_for=["Deployer"],
    approval_callback=check_approval
)
```

### Auto-Summarization Middleware

Automatically compresses context when it grows too large:

```python
from blackboard.middleware import AutoSummarizationMiddleware

summarizer = AutoSummarizationMiddleware(
    llm=my_llm,
    artifact_threshold=10,
    step_threshold=50
)
```

### Custom Middleware

```python
from blackboard.middleware import Middleware, StepContext

class MetricsMiddleware(Middleware):
    name = "MetricsMiddleware"
    
    async def before_step(self, ctx: StepContext) -> None:
        ctx.state.metadata["step_start"] = time.time()
    
    async def after_step(self, ctx: StepContext) -> None:
        duration = time.time() - ctx.state.metadata["step_start"]
        print(f"Step {ctx.step_number} took {duration:.2f}s")
```

---

## Tool Calling

Workers are automatically converted to tools for LLMs that support function calling.

```python
from blackboard.tools import workers_to_tool_definitions

tools = workers_to_tool_definitions([ResearchWorker(), CriticWorker()])
# Returns ToolDefinition objects with name, description, parameters
```

---

## Memory System

For long-running agents that need to remember past interactions:

```python
from blackboard.memory import SimpleVectorMemory, MemoryWorker, MemoryInput
from blackboard.embeddings import TFIDFEmbedder

# Create memory with embedder
memory = SimpleVectorMemory(embedder=TFIDFEmbedder())

# Create memory worker
memory_worker = MemoryWorker(memory=memory)

# Use in orchestrator
orchestrator = Orchestrator(
    llm=llm,
    workers=[ResearchWorker(), memory_worker]
)
```

### Embedder Options

| Embedder | Description | Requirements |
|----------|-------------|--------------|
| `NoOpEmbedder` | No embeddings (keyword search) | None |
| `TFIDFEmbedder` | TF-IDF based | None |
| `LocalEmbedder` | Sentence Transformers | `sentence-transformers` |
| `OpenAIEmbedder` | OpenAI Embeddings API | `openai` |

---

## Persistence

### Save and Load (File-Based)

```python
# Save state
result.save_to_json("session.json")

# Load state
state = Blackboard.load_from_json("session.json")

# Resume
await orchestrator.run(state=state)
```

### Distributed Persistence (v1.0.1)

For serverless and multi-container deployments:

```python
from blackboard.persistence import RedisPersistence, JSONFilePersistence

# Redis backend (requires: pip install blackboard-core[redis])
persistence = RedisPersistence(
    redis_url="redis://localhost:6379",
    prefix="myapp:",
    ttl=86400  # Optional: expire after 24 hours
)

# Set on orchestrator
orchestrator.set_persistence(persistence)

# Save and resume sessions
await persistence.save(state, "session-123")
state = await persistence.load("session-123")

# List all sessions
sessions = await persistence.list_sessions()
```

### Pause/Resume (v1.0.1)

Handle long-running tasks with human approval:

```python
# Pause and save state
await orchestrator.pause(state, "session-123", reason="Needs approval")

# Later, resume with user input
result = await orchestrator.resume(
    "session-123",
    user_input={"approved": True}
)
```

### Optimistic Locking

Prevents data loss from concurrent updates:

```python
from blackboard import StateConflictError

try:
    state.save_to_json("session.json")
except StateConflictError:
    # Another process updated the file
    # Reload and merge changes
    pass
```

---

## Sandbox Code Execution (v1.0.1)

Safely execute LLM-generated code:

```python
from blackboard.sandbox import SubprocessSandbox, DockerSandbox

# Lightweight isolation
sandbox = SubprocessSandbox(timeout=30)
result = await sandbox.execute("print('hello')")
print(result.stdout)  # "hello"

# Full isolation (requires Docker)
docker_sandbox = DockerSandbox(
    memory_limit="256m",
    network_disabled=True
)
result = await docker_sandbox.execute(untrusted_code)
```

---

## Hierarchical Orchestration (v1.0.1)

Delegate complex tasks to sub-teams:

```python
from blackboard.hierarchy import SubOrchestratorWorker

research_team = SubOrchestratorWorker(
    name="ResearchTeam",
    description="Researches topics using specialists",
    llm=my_llm,
    sub_workers=[WebScraper(), Summarizer(), FactChecker()],
    goal_template="Research: {sub_goal}"
)

# Main orchestrator delegates to team
orchestrator = Orchestrator(
    llm=my_llm,
    workers=[Writer(), research_team]
)
```

---

## Streaming (v1.0.1)

Token-by-token streaming for responsive UIs:

```python
from blackboard.streaming import StreamingLLMClient, BufferedStream

class MyStreamingLLM(StreamingLLMClient):
    async def generate_stream(self, prompt):
        async for token in openai_stream(prompt):
            yield token

# Collect and emit events
from blackboard.streaming import StreamCollector

collector = StreamCollector(event_bus, source="supervisor")
response = await collector.collect(llm.generate_stream(prompt))
```

---

## Production Vector DB (v1.0.1)

Scalable memory with ChromaDB:

```python
from blackboard.vectordb import ChromaMemory, HybridSearchMemory

# ChromaDB backend (requires: pip install blackboard-core[chroma])
memory = ChromaMemory(
    collection_name="agent_memory",
    persist_directory="./chroma_data"
)

# Hybrid search (semantic + keyword)
hybrid = HybridSearchMemory(memory, alpha=0.7)
results = await hybrid.search("AI safety research", k=10)
```

---

## Evaluation Framework (v1.0.1)

Test and score agent performance:

```python
from blackboard.evals import Evaluator, LLMJudge, RuleBasedJudge, EvalCase

# LLM-based judging
judge = LLMJudge(my_llm, threshold=0.7)

# Or rule-based
judge = RuleBasedJudge([
    ("has_artifacts", lambda bb: len(bb.artifacts) > 0),
    ("is_done", lambda bb: bb.status == Status.DONE),
])

# Run evaluation
evaluator = Evaluator(orchestrator, judge)
report = await evaluator.run([
    EvalCase(id="1", goal="Write a haiku", expected_criteria=["Has 3 lines"]),
    EvalCase(id="2", goal="Summarize article", expected_criteria=["Under 100 words"]),
])

print(f"Pass rate: {report.pass_rate:.1%}")
```

---

## Dynamic Worker Loading (v1.0.1)

Load heavy workers on-demand:

```python
from blackboard.protocols import WorkerFactory, LazyWorkerRegistry

class MyFactory:
    def get_worker(self, name):
        if name == "DataAnalyzer":
            from .heavy import DataAnalyzer  # Lazy import
            return DataAnalyzer()
        return None
    
    def list_available(self):
        return ["DataAnalyzer", "ImageProcessor"]
    
    def get_description(self, name):
        return "Analyzes data" if name == "DataAnalyzer" else "Processes images"

registry = LazyWorkerRegistry(factory=MyFactory())
```

---

## Events & Observability

Subscribe to events for monitoring and debugging:

```python
from blackboard.events import EventBus, EventType, get_event_bus

bus = get_event_bus()

@bus.subscribe(EventType.STEP_STARTED)
def on_step(event):
    print(f"Step {event.data['step']} started")

@bus.subscribe(EventType.WORKER_COMPLETED)
def on_worker(event):
    print(f"Worker {event.data['worker']} completed")
```

---

## Error Handling

### Retry Policy

```python
from blackboard.retry import RetryPolicy

retry = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0
)

orchestrator = Orchestrator(llm=llm, workers=workers, retry_policy=retry)
```

### Strict Mode

```python
orchestrator = Orchestrator(
    llm=llm,
    workers=workers,
    strict_tools=True,         # Crash on tool definition errors
    allow_json_fallback=False  # Crash if tool calling fails
)
```

---

## Best Practices

### 1. Keep Workers Focused

```python
# Good: Single responsibility
class ResearchWorker(Worker): ...
class WriterWorker(Worker): ...
class CriticWorker(Worker): ...

# Bad: One worker doing everything
class DoEverythingWorker(Worker): ...
```

### 2. Use Descriptive Names

The supervisor LLM sees worker names and descriptions to decide which to call.

```python
# Good: Clear purpose
name = "TechnicalWriter"
description = "Writes technical documentation with code examples"

# Bad: Vague
name = "Worker1"
description = "Does stuff"
```

### 3. Handle Errors Gracefully

```python
async def run(self, state, inputs=None):
    try:
        result = await self.call_api()
    except APIError as e:
        return WorkerOutput(
            feedback=Feedback(
                source=self.name,
                critique=f"API error: {e}",
                passed=False
            )
        )
```

### 4. Use Middleware for Cross-Cutting Concerns

Don't put budget tracking, logging, or approval logic in workers.

### 5. Test Workers in Isolation

```python
async def test_research_worker():
    state = Blackboard(goal="Test goal")
    worker = ResearchWorker()
    result = await worker.run(state, {"topic": "AI"})
    assert result.artifact is not None
```
