# Blackboard-Core Documentation

Complete API reference and usage guide for building LLM-powered multi-agent systems.

## Table of Contents

**Getting Started**

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)

**Building Agents** 4. [Creating Workers](#creating-workers) 5. [LLM Clients](#llm-clients) 6. [Running the Orchestrator](#running-the-orchestrator)

**State & Persistence** 7. [Persistence](#persistence) 8. [Memory System](#memory-system)

**Orchestration** 9. [Middleware](#middleware) 10. [Reasoning Strategies](#reasoning-strategies) 11. [Blueprints](#blueprints) 12. [Tool Calling](#tool-calling)

**Advanced Patterns** 13. [Fractal Agents](#fractal-agents) 14. [Model Context Protocol](#model-context-protocol) 15. [Swarm Intelligence](#swarm-intelligence)

**Developer Experience** 15. [Interactive TUI](#interactive-tui) 16. [CLI Commands](#cli-commands) 17. [Session Replay](#session-replay)

**Production** 18. [Runtime Security](#runtime-security) 19. [OpenTelemetry](#opentelemetry) 20. [Config Propagation](#config-propagation) 21. [Error Handling](#error-handling)

**Ecosystem** 22. [Ecosystem Adapters](#ecosystem-adapters) 23. [Standard Library Workers](#standard-library-workers) 24. [Blackboard Serve](#blackboard-serve)

**Testing & Optimization** 25. [Prompt Registry](#prompt-registry) 26. [Instruction Optimizer](#instruction-optimizer) 27. [Evaluation Framework](#evaluation-framework)

**Reference** 28. [Best Practices](#best-practices) 29. [Deprecated APIs](#deprecated-apis)

---

## Installation

```bash
pip install blackboard-core
```

Optional dependencies:

```bash
pip install blackboard-core[redis]       # Redis persistence
pip install blackboard-core[chroma]      # ChromaDB vector memory
pip install blackboard-core[hybrid]      # Hybrid BM25 search
pip install blackboard-core[serve]       # FastAPI server
pip install blackboard-core[stdlib]      # Standard library workers
pip install blackboard-core[browser]     # Playwright browser worker
pip install blackboard-core[textual-tui] # Interactive TUI
pip install blackboard-core[langchain]   # LangChain adapter
pip install blackboard-core[llamaindex]  # LlamaIndex adapter
pip install blackboard-core[all]         # Everything
```

---

## Core Concepts

### The Blackboard Pattern

The Blackboard Pattern is an architectural style where multiple specialized agents collaborate through a shared workspace:

1. **Blackboard** - Shared state that all agents can read/write
2. **Workers** - Specialized agents that perform specific tasks
3. **Supervisor** - An LLM that decides which worker to call next

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

## Quick Start

```python
from blackboard import Orchestrator, worker
from blackboard.llm import LiteLLMClient

# Define workers with simple type hints
@worker
def write(topic: str) -> str:
    """Writes content about a topic."""
    return f"Article about {topic}..."

@worker
def critique(content: str) -> str:
    """Reviews content for quality."""
    return "Approved!" if len(content) > 50 else "Needs more detail"

# Create orchestrator
llm = LiteLLMClient(model="gpt-4o")
orchestrator = Orchestrator(llm=llm, workers=[write, critique])

# Run
result = orchestrator.run_sync(goal="Write about AI safety")
print(result.artifacts[-1].content)
```

---

## Creating Workers

Workers are the agents that perform actual work. Each worker reads from the Blackboard, performs a task, and returns artifacts and/or feedback.

### The @worker Decorator (Recommended)

The simplest way to create workers - just add type hints:

```python
from blackboard import worker
from blackboard.state import Blackboard

# Simple function - schema auto-generated from type hints
@worker
def calculate(a: int, b: int, operation: str = "add") -> str:
    """Performs basic math operations."""
    if operation == "add":
        return str(a + b)
    return str(a - b)

# With state access - just add a 'state' parameter
@worker
def summarize(state: Blackboard) -> str:
    """Summarizes the current artifacts."""
    return f"Goal: {state.goal}, Artifacts: {len(state.artifacts)}"

# Mixed: user inputs + state
@worker
def analyze(topic: str, depth: int, state: Blackboard) -> str:
    """Analyzes a topic at specified depth."""
    return f"Analyzing {topic} at depth {depth} for goal: {state.goal}"

# Async support
@worker
async def research(topic: str) -> str:
    """Researches a topic online."""
    await asyncio.sleep(0.1)
    return f"Research on {topic}..."

# With explicit options
@worker(artifact_type="code", parallel_safe=True)
def generate_code(language: str = "python") -> str:
    """Generates boilerplate code."""
    return f"# {language} code here"
```

### The @critic Decorator

For workers that provide feedback instead of artifacts:

```python
from blackboard import critic

@critic(name="CodeReviewer", description="Reviews code quality")
def review_code(state: Blackboard) -> tuple[bool, str]:
    last = state.get_last_artifact()
    if not last or "def " not in last.content:
        return False, "No function definitions found"
    return True, "Code looks good!"
```

### Class-Based Workers (Advanced)

For complex workers that need initialization or internal state:

```python
from blackboard import Worker, WorkerOutput, Artifact, Feedback, Blackboard

class ResearchWorker(Worker):
    name = "Researcher"
    description = "Gathers information on a topic"

    input_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Topic to research"}
        }
    }

    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        topic = inputs.get("topic", state.goal) if inputs else state.goal
        research_result = f"Research findings about {topic}..."

        return WorkerOutput(
            artifact=Artifact(
                type="research",
                content=research_result,
                creator=self.name,
                metadata={"topic": topic}
            )
        )
```

---

## LLM Clients

The Orchestrator needs an LLM to act as the supervisor.

### Using LiteLLMClient (Recommended)

```python
from blackboard.llm import LiteLLMClient

# Supports 100+ providers via LiteLLM
llm = LiteLLMClient(
    model="gpt-4o",
    fallback_models=["gpt-4o-mini", "claude-3-5-sonnet-20241022"],
    temperature=0.7
)
```

### Implementing LLMClient Protocol

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

### Tool Calling LLM Client

For native function calling support:

```python
from blackboard.tools import ToolCallingLLMClient, ToolCall

class OpenAIToolLLM(ToolCallingLLMClient):
    async def generate_with_tools(self, prompt: str, tools: list) -> list:
        openai_tools = [t.to_openai_format() for t in tools]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=openai_tools
        )
        return [
            ToolCall(id=tc.id, name=tc.function.name,
                    arguments=json.loads(tc.function.arguments))
            for tc in response.choices[0].message.tool_calls or []
        ]
```

---

## Running the Orchestrator

```python
import asyncio
from blackboard import Orchestrator, BlackboardConfig

async def main():
    config = BlackboardConfig(
        max_steps=50,
        reasoning_strategy="cot",  # Chain-of-Thought
        enable_parallel=True,
        verbose=True
    )

    orchestrator = Orchestrator(
        llm=llm,
        workers=[write, critique],
        config=config
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

### Resuming Sessions

```python
# Resume from existing state
result = await orchestrator.run(state=existing_state, max_steps=10)
```

---

## Persistence

### SQLite Persistence (Recommended)

```python
from blackboard.persistence import SQLitePersistence

persistence = SQLitePersistence("./blackboard.db")
await persistence.initialize()

# Attach to orchestrator
orchestrator.set_persistence(persistence)

# Save session
await persistence.save(state, "session-123", parent_session_id="parent-001")

# Load session
state = await persistence.load("session-123")

# List sessions
sessions = await persistence.list_sessions()
```

### PostgreSQL (Distributed)

```python
from blackboard.persistence import PostgresPersistence

persistence = PostgresPersistence("postgresql://user:pass@localhost/db")
await persistence.initialize()
```

### Redis (High-Throughput)

```python
from blackboard.persistence import RedisPersistence

persistence = RedisPersistence(
    redis_url="redis://localhost:6379",
    prefix="myapp:",
    ttl=86400  # Expire after 24 hours
)
```

### Pause/Resume Pattern

```python
# Pause: Save state and stop
await persistence.save(state, "session-123")

# Resume: Load and run
loaded_state = await persistence.load("session-123")
result = await orchestrator.run(state=loaded_state)
```

### Optimistic Locking

```python
from blackboard.persistence import SessionConflictError

try:
    await persistence.save(state, "session-123")
except SessionConflictError:
    latest_state = await persistence.load("session-123")
    # Merge or retry...
```

### Time-Travel Debugging

Fork sessions at any checkpoint to experiment:

```python
# Run original session (checkpoints saved automatically)
result = await orchestrator.run(goal="Write an article")

# Fork at step 5 to try different approach
session_id = result.metadata["session_id"]
fork_id = await orchestrator.fork_session(session_id, step_index=5)

# Continue from forked state
forked = await persistence.load(fork_id)
new_result = await orchestrator.run(state=forked)
```

---

## Memory System

For agents that need to remember past interactions:

```python
from blackboard.memory import SimpleVectorMemory, MemoryWorker
from blackboard.embeddings import TFIDFEmbedder, OpenAIEmbedder

# Create memory with embedder
memory = SimpleVectorMemory(embedder=TFIDFEmbedder())

# Or use OpenAI embeddings
memory = SimpleVectorMemory(embedder=OpenAIEmbedder())

# Create memory worker
memory_worker = MemoryWorker(memory=memory)

orchestrator = Orchestrator(llm=llm, workers=[ResearchWorker(), memory_worker])
```

### Embedder Options

| Embedder         | Description                    | Requirements            |
| ---------------- | ------------------------------ | ----------------------- |
| `NoOpEmbedder`   | No embeddings (keyword search) | None                    |
| `TFIDFEmbedder`  | TF-IDF based                   | None                    |
| `LocalEmbedder`  | Sentence Transformers          | `sentence-transformers` |
| `OpenAIEmbedder` | OpenAI Embeddings API          | `openai`                |

### Production Vector DB

```python
from blackboard.vectordb import ChromaMemory, HybridSearchMemory

# ChromaDB backend
memory = ChromaMemory(
    collection_name="agent_memory",
    persist_directory="./chroma_data"
)

# Hybrid search (semantic + keyword)
hybrid = HybridSearchMemory(memory, alpha=0.7)
results = await hybrid.search("AI safety research", k=10)
```

---

## Middleware

Middleware intercepts the orchestration flow for cross-cutting concerns.

### Budget Middleware

```python
from blackboard.middleware import BudgetMiddleware

budget = BudgetMiddleware(
    max_cost_usd=5.0,
    max_tokens=100000
)

orchestrator = Orchestrator(llm=llm, workers=workers, middleware=[budget])
```

### Human Approval Middleware

```python
from blackboard.middleware import HumanApprovalMiddleware, ApprovalRequired

approval = HumanApprovalMiddleware(require_approval_for=["Deployer"])

try:
    await orchestrator.run(goal="Deploy to production")
except ApprovalRequired as e:
    print(f"Approval needed for: {e.worker_name}")
```

### Auto-Summarization Middleware

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

## Reasoning Strategies

Control how the supervisor LLM reasons about decisions.

### OneShot (Default)

Fast, single-pass decision making:

```python
from blackboard import BlackboardConfig

config = BlackboardConfig(reasoning_strategy="oneshot")
```

### Chain-of-Thought

Explicit reasoning before decisions:

```python
config = BlackboardConfig(reasoning_strategy="cot")
```

---

## Blueprints

Blueprints constrain the orchestrator to follow specific patterns.

### Sequential Pipeline

```python
from blackboard.flow import SequentialPipeline

pipeline = SequentialPipeline([
    SearchWorker(),
    WriterWorker(),
    CriticWorker()
])

result = await orchestrator.run(goal="Research and write", blueprint=pipeline)
```

### Router

```python
from blackboard.flow import Router

router = Router([
    MathAgent(),
    CodeAgent(),
    ResearchAgent()
], selection_prompt="Choose based on the query type")

result = await orchestrator.run(goal="Solve 2x + 5 = 15", blueprint=router)
```

---

## Tool Calling

Workers are automatically converted to tools for LLMs that support function calling:

```python
from blackboard.tools import workers_to_tool_definitions

tools = workers_to_tool_definitions([ResearchWorker(), CriticWorker()])
```

---

## Fractal Agents

The Fractal Agent Architecture enables nested multi-agent systems where Agents can delegate to sub-agents.

### Agent Class (Agent-as-Worker)

```python
from blackboard import Orchestrator, Agent, BlackboardConfig

config = BlackboardConfig(max_recursion_depth=3, max_steps=50)

# Create sub-agent
research_agent = Agent(
    name="ResearchAgent",
    description="Performs web research",
    llm=llm,
    workers=[WebSearchWorker(), BrowserWorker()],
    config=config.for_child_agent()
)

# Use as worker in parent
parent = Orchestrator(
    llm=llm,
    workers=[research_agent, WriterWorker()],
    config=config
)

result = await parent.run(goal="Research and write about AI safety")
```

### Squad Patterns

```python
from blackboard.patterns import research_squad, code_squad, memory_squad

researcher = research_squad(llm, config=config.for_child_agent())
coder = code_squad(llm, config=config.for_child_agent())
memory_agent = memory_squad(llm)
```

---

## Model Context Protocol

Connect to external tools via MCP servers:

```python
from blackboard.mcp import MCPServerWorker

# Local via stdio
server = await MCPServerWorker.create(
    name="Filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-fs", "/tmp"]
)

# Remote via SSE
remote = await MCPServerWorker.create(
    name="RemoteAPI",
    url="http://mcp-server:8080/sse"
)

# Each tool becomes a worker
workers = server.expand_to_workers()
orchestrator = Orchestrator(llm=llm, workers=workers)
```

---

## Swarm Intelligence

_Added in v1.8.0_

Enable parallel sub-agent execution with the "Git-like Branch-Merge" pattern.

### Delta Protocol (Artifact Patching)

Return incremental changes instead of full artifact replacements:

```python
from blackboard import WorkerOutput, ArtifactMutation, SearchReplacePatch

@worker
def refactor(state: Blackboard) -> WorkerOutput:
    """Refactors code using patches."""
    artifact = state.artifacts[0]

    return WorkerOutput(
        mutations=[
            ArtifactMutation(
                artifact_id=artifact.id,
                patches=[
                    SearchReplacePatch(
                        search="def old_name(",
                        replace="def new_name("
                    )
                ]
            )
        ]
    )
```

### Map-Reduce Pattern

Process multiple items in parallel with conflict handling:

```python
from blackboard.map_reduce import run_map_reduce, ConflictResolution

result = await run_map_reduce(
    items=["auth.py", "db.py", "api.py"],
    worker=CodeReviewerWorker(),
    parent_state=state,
    max_concurrency=3,
    item_to_goal=lambda f: f"Review {f} for security issues",
    conflict_resolution=ConflictResolution.FIRST_WINS
)

# Apply non-conflicting mutations
for mutation in result.get_non_conflicting_mutations():
    artifact = state.get_artifact(mutation.artifact_id)
    # Apply patches...
```

### MapReduceWorker

Wrap map-reduce as a Worker for use in Orchestrator:

```python
from blackboard.map_reduce import MapReduceWorker

parallel_reviewer = MapReduceWorker(
    name="ParallelCodeReviewer",
    description="Reviews all code files in parallel",
    inner_worker=CodeReviewWorker(),
    items_extractor=lambda s: [a for a in s.artifacts if a.type == "code"],
    max_concurrency=5,
    conflict_resolution=ConflictResolution.FIRST_WINS
)

orchestrator = Orchestrator(llm=llm, workers=[parallel_reviewer])
```

### State Merging

Merge forked sub-agent states back into parent:

```python
from blackboard.merging import merge_states, StateMerger, MergeStrategy

# Simple merge - child wins on conflicts
result = merge_states(parent_state, child_state, strategy=MergeStrategy.THEIRS)

# Sequential merging of multiple children
merger = StateMerger(parent_state, strategy=MergeStrategy.THEIRS)
for child in sub_agent_results:
    merger.merge(child)

final_state = merger.get_merged_state()
```

### Context Scoping

Filter artifacts/feedback for sub-agents:

```python
# Only show code artifacts to sub-agent
context = state.to_context_string(
    artifact_filter=lambda a: a.type == "code",
    feedback_filter=lambda f: not f.passed  # Only failed feedback
)
```

---

## Interactive TUI

A Textual-based Mission Control dashboard for real-time debugging:

```python
from blackboard.ui import create_tui, is_headless

async def main():
    orchestrator = Orchestrator(...)

    if is_headless():
        await orchestrator.run(goal="...")
    else:
        app = create_tui(orchestrator)
        await app.run_async()
```

### Key Bindings

| Key     | Action                      |
| ------- | --------------------------- |
| `Space` | Pause/Resume execution      |
| `I`     | Inject intervention command |
| `Q`     | Quit                        |
| `Tab`   | Switch panels               |

### Panels

- **Activity Log**: Real-time event ticker
- **Artifacts**: Live view of generated artifacts
- **State**: JSON view of current Blackboard state

---

## CLI Commands

### Initialize Project

```bash
blackboard init
# Creates prompts/ directory and blackboard.prompts.json
```

### Optimize Prompts

```bash
blackboard optimize run --session-id my-session --db-path ./blackboard.db
blackboard optimize review --patches-file blackboard.patches.json
```

### Run Server

```bash
blackboard serve my_app:create_orchestrator --port 8000
```

---

## Session Replay

Record and replay sessions for debugging:

```python
from blackboard.replay import SessionRecorder, RecordingLLMClient, ReplayOrchestrator

# Record
recorder = SessionRecorder()
recording_llm = RecordingLLMClient(llm, recorder)
orchestrator = Orchestrator(llm=recording_llm, workers=workers)
result = await orchestrator.run(goal="Write a poem")
recorder.save("session.json")

# Replay (no API calls!)
replay = ReplayOrchestrator.from_file("session.json", workers=workers)
replayed_result = await replay.run()
```

---

## Runtime Security

Secure code execution environments:

```python
from blackboard.runtime import LocalRuntime, DockerRuntime, RuntimeSecurityError

# LocalRuntime requires explicit acknowledgment
runtime = LocalRuntime(dangerously_allow_execution=True)

# Production: Use Docker
runtime = DockerRuntime(
    image="python:3.11-slim",
    memory_limit="256m",
    network_disabled=True
)

result = await runtime.execute("print('hello')")
```

---

## OpenTelemetry

Distributed tracing with span hierarchy:

```python
from blackboard.telemetry import OpenTelemetryMiddleware

otel = OpenTelemetryMiddleware(service_name="my-agent")
orchestrator = Orchestrator(llm=llm, workers=workers, middleware=[otel])

# Creates span hierarchy:
# orchestrator.run
# ├── step.1
# │   └── worker.Writer
# └── step.2
#     └── worker.Critic
```

---

## Config Propagation

```python
from blackboard.config import BlackboardConfig

parent = BlackboardConfig(
    max_recursion_depth=3,
    max_steps=100,
    allow_unsafe_execution=True
)

child = parent.for_child_agent()
print(child.max_recursion_depth)  # 2
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

---

## Ecosystem Adapters

### LangChain Adapter

```python
from langchain_community.tools import TavilySearchResults
from blackboard.integrations.langchain import wrap_tool

tool = TavilySearchResults()
worker = wrap_tool(tool, artifact_type="search_results")

orchestrator = Orchestrator(llm=llm, workers=[worker])
```

### LlamaIndex Adapter

```python
from llama_index.core import VectorStoreIndex
from blackboard.integrations.llamaindex import wrap_query_engine

index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine()

worker = wrap_query_engine(engine, name="DocumentSearch")
```

### FastAPI Dependency

```python
from fastapi import FastAPI, Depends
from blackboard.integrations.fastapi_dep import get_orchestrator_session

app = FastAPI()

@app.post("/run")
async def run(
    goal: str,
    session = Depends(get_orchestrator_session(llm=my_llm, workers=workers))
):
    result = await session.run(goal=goal)
    return {"status": result.status.value}
```

---

## Standard Library Workers

Install with `pip install blackboard-core[stdlib]`.

### WebSearchWorker

```python
from blackboard.stdlib import WebSearchWorker

search = WebSearchWorker()  # Uses TAVILY_API_KEY or SERPER_API_KEY
```

### BrowserWorker

```python
from blackboard.stdlib import BrowserWorker

browser = BrowserWorker(headless=True, browser_type="chromium")
```

### CodeInterpreterWorker

```python
from blackboard.stdlib import CodeInterpreterWorker

interpreter = CodeInterpreterWorker(
    docker_image="python:3.11-slim",
    memory_limit="512m"
)
```

### HumanProxyWorker

```python
from blackboard.stdlib import HumanProxyWorker

human = HumanProxyWorker()
# Sets status to PAUSED, stores question in pending_input
```

---

## Blackboard Serve

Deploy as a REST API:

```bash
blackboard serve my_app:create_orchestrator --port 8000
```

### API Endpoints

| Endpoint            | Method | Description       |
| ------------------- | ------ | ----------------- |
| `/runs`             | POST   | Start a new run   |
| `/runs/{id}`        | GET    | Get run status    |
| `/runs/{id}/stream` | GET    | SSE event stream  |
| `/runs/{id}/resume` | POST   | Resume paused run |

---

## Prompt Registry

Externalize prompts for customization:

```python
from blackboard import PromptRegistry

registry = PromptRegistry(
    prompts_dir="prompts/",
    config_path="blackboard.prompts.json"
)

prompt = registry.get("Writer", {"topic": "AI safety", "style": "formal"})
```

### Template Files

Create `prompts/Writer.jinja2`:

```jinja2
You are a professional writer.
{% if style %}Style: {{ style }}{% endif %}
Topic: {{ topic }}
```

---

## Instruction Optimizer

Auto-analyze failures and generate improved prompts:

```python
from blackboard import Optimizer

optimizer = Optimizer(llm=meta_llm, orchestrator=orchestrator)

failures = await optimizer.analyze_failures("session-001")
for failure in failures:
    candidates = await optimizer.generate_candidates(failure, n=3)
    for patch in candidates:
        verified = await optimizer.verify_candidate(patch, failure)
        if verified.verified:
            print(f"Found fix for {patch.worker_name}")
            break

optimizer.save_patches(verified_patches, "blackboard.patches.json")
```

---

## Evaluation Framework

Test and score agent performance:

```python
from blackboard.evals import Evaluator, LLMJudge, RuleBasedJudge, EvalCase

judge = LLMJudge(my_llm, threshold=0.7)

evaluator = Evaluator(orchestrator, judge)
report = await evaluator.run([
    EvalCase(id="1", goal="Write a haiku", expected_criteria=["Has 3 lines"]),
    EvalCase(id="2", goal="Summarize article", expected_criteria=["Under 100 words"]),
])

print(f"Pass rate: {report.pass_rate:.1%}")
```

---

## Best Practices

1. **Use persistence** for production deployments
2. **Set recursion limits** when using fractal agents
3. **Enable CoT** for complex decision-making
4. **Use SQLite** for single-node, PostgreSQL for distributed
5. **Externalize prompts** for easy iteration
6. **Fork sessions** to debug failures
7. **Use ecosystem adapters** to leverage existing tools
8. **Add middleware** for cross-cutting concerns

---

## Deprecated APIs

> [!WARNING]
> The following APIs are deprecated and will be removed in a future version.

### JSONFilePersistence

**Status**: Deprecated - for debugging only

**Migration**: Use `SQLitePersistence` for production.

```python
# Old (deprecated)
from blackboard.persistence import JSONFilePersistence
persistence = JSONFilePersistence("./data")

# New (recommended)
from blackboard.persistence import SQLitePersistence
persistence = SQLitePersistence("./blackboard.db")
```

### blackboard.tui

**Status**: Deprecated - use `blackboard.ui` instead

**Migration**: The new Textual-based TUI provides an interactive dashboard.

```python
# Old (deprecated)
from blackboard.tui import BlackboardTUI, watch
result = watch(orchestrator, goal="...")

# New (recommended)
from blackboard.ui import create_tui, is_headless
app = create_tui(orchestrator)
await app.run_async()
```

### blackboard.hierarchy

**Status**: Deprecated - use `Agent` class instead

**Migration**: The `Agent` class provides agent-as-worker functionality.

```python
# Old (deprecated)
from blackboard.hierarchy import HierarchicalOrchestrator

# New (recommended)
from blackboard import Agent
agent = Agent(name="SubAgent", llm=llm, workers=[...])
```
