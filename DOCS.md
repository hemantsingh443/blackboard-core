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
10. [Fractal Agent Architecture (v1.6.0)](#fractal-agent-architecture-v160)
11. [SQLite Persistence (v1.5.1)](#sqlite-persistence-v151)
12. [Runtime Security (v1.5.2)](#runtime-security-v152)
13. [Config Propagation](#config-propagation)
14. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
15. [Blueprints (Structured Workflows)](#blueprints-structured-workflows)
16. [Reasoning Strategies](#reasoning-strategies)
17. [OpenTelemetry](#opentelemetry)
18. [Session Replay](#session-replay)
19. [Terminal UI](#terminal-ui)
20. [Standard Library Workers](#standard-library-workers)
21. [Blackboard Serve (API Deployment)](#blackboard-serve-api-deployment)
22. [Events & Observability](#events--observability)
23. [Error Handling](#error-handling)
24. [Time-Travel Debugging (v1.6.3)](#time-travel-debugging-v163)
25. [Prompt Registry (v1.6.3)](#prompt-registry-v163)
26. [Instruction Optimizer (v1.6.3)](#instruction-optimizer-v163)
27. [CLI Commands (v1.6.3)](#cli-commands-v163)
28. [Best Practices](#best-practices)

---

## Installation

```bash
pip install blackboard-core
```

Optional dependencies:
```bash
pip install blackboard-core[redis]   # RedisPersistence
pip install blackboard-core[chroma]  # ChromaMemory (vector DB)
pip install blackboard-core[hybrid]  # HybridSearchMemory (BM25)
pip install blackboard-core[serve]   # FastAPI API server
pip install blackboard-core[stdlib]  # Standard library workers
pip install blackboard-core[browser] # BrowserWorker (Playwright)
pip install blackboard-core[all]     # All optional features

# Or install embedders directly:
pip install sentence-transformers    # For LocalEmbedder
pip install openai                   # For OpenAI embeddings
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

### The Magic Decorator (Recommended)

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
    await asyncio.sleep(0.1)  # Simulated async work
    return f"Research on {topic}..."

# With explicit options
@worker(artifact_type="code", parallel_safe=True)
def generate_code(language: str = "python") -> str:
    """Generates boilerplate code."""
    return f"# {language} code here"
```

### Critic Decorator

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
```

### Worker Best Practices

1. **Single Responsibility** - Each worker should do one thing well
2. **Stateless** - Workers should not store state between calls
3. **Descriptive Names** - The supervisor LLM uses names to choose workers
4. **Type Hints** - Always use type hints for auto-schema generation


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
from blackboard import Orchestrator, BlackboardConfig

async def main():
    # Option 1: Direct configuration
    orchestrator = Orchestrator(
        llm=OpenAILLM(),
        workers=[write, critique],  # Magic decorator workers
        
        # Options
        verbose=True,              # Enable logging
        enable_parallel=True,      # Allow parallel worker execution
        use_tool_calling=True,     # Use native tool calling
    )
    
    # Option 2: Using BlackboardConfig (recommended for production)
    config = BlackboardConfig(
        max_steps=50,
        reasoning_strategy="cot",  # Chain-of-Thought for smarter decisions
        enable_parallel=True,
        verbose=True
    )
    orchestrator = Orchestrator(llm=llm, workers=workers, config=config)
    
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

# Reactive budget enforcement (v1.6.2)
budget = BudgetMiddleware(
    max_cost_usd=5.0,     # Hard limit in USD
    max_tokens=100000,    # Optional token limit
)

orchestrator = Orchestrator(llm=llm, workers=workers, middleware=[budget])

# Uses LiteLLM pricing automatically, or configure custom:
from blackboard import configure_pricing
configure_pricing({"my-azure-model": (0.02, 0.04)})
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

## Fractal Agent Architecture (v1.6.0)

The **Fractal Agent Architecture** enables nested multi-agent systems where Agents can delegate to sub-agents, forming hierarchical teams. This replaces and supersedes the older `hierarchy` module.

> [!IMPORTANT]
> This is the recommended pattern for complex multi-step tasks. Sub-agents have bounded context (via compression), trace linking for debugging, and security flag propagation.

### Agent Class (Agent-as-Worker)

The `Agent` class implements `Worker`, so it can be used as a worker within a parent orchestrator:

```python
from blackboard import Orchestrator, Agent, BlackboardConfig

# Parent config with recursion depth limit
config = BlackboardConfig(max_recursion_depth=3, max_steps=50)

# Create a sub-agent with its own workers
research_agent = Agent(
    name="ResearchAgent",
    description="Performs web research on a topic",
    llm=llm,
    workers=[WebSearchWorker(), BrowserWorker()],
    config=config.for_child_agent()  # Decrements recursion depth
)

# Use as a worker in parent orchestrator
parent = Orchestrator(
    llm=llm,
    workers=[research_agent, WriterWorker()],
    config=config
)

result = await parent.run(goal="Research and write about AI safety")
```

### Recursion Depth Limits

The `max_recursion_depth` setting prevents infinite agent loops:

```python
from blackboard.core import RecursionDepthExceededError

config = BlackboardConfig(max_recursion_depth=2)

# Agents at depth 2 will raise RecursionDepthExceededError
# if they try to spawn another sub-agent
```

### Context Compression

Sub-agents automatically compress their execution history before returning to the parent, preventing context window explosion:

```python
# Instead of full history, parent receives:
# "Task: Research AI safety
#  Status: done
#  Steps taken: 5
#  Final result: [truncated summary]"
```

### Trace Linking (Observability)

Each sub-agent run returns a `trace_id` in its `WorkerOutput`, linking to the full session in persistence:

```python
# WorkerOutput from sub-agent includes:
output.trace_id  # e.g., "researchagent-abc12345"
output.has_trace()  # True

# The TUI shows trace links:
# ✓ ResearchAgent completed [trace: researchagent-abc12345]
```

### Squad Factory Functions

Pre-configured agent factories for common patterns:

```python
from blackboard.patterns import research_squad, code_squad, memory_squad

# Web research squad
researcher = research_squad(llm, config=parent_config.for_child_agent())

# Code execution squad
coder = code_squad(llm, config=parent_config.for_child_agent())

# Memory management squad
memory_agent = memory_squad(llm)

# Custom squad
from blackboard.patterns import create_squad
custom = create_squad(
    name="AnalysisTeam",
    description="Analyzes data and generates reports",
    llm=llm,
    workers=[DataLoader(), Analyzer(), ReportWriter()]
)
```

---

## SQLite Persistence (v1.5.1)

Production-grade persistence with parent-child session tracking:

```python
from blackboard.persistence import SQLitePersistence

persistence = SQLitePersistence("./data/sessions.db")
await persistence.initialize()

# Save with parent link (for fractal agents)
await persistence.save(state, "session-123", parent_session_id="parent-001")

# Load
state = await persistence.load("session-123")

# List child sessions
children = await persistence.list_sessions(parent_id="parent-001")

# Event logging for debugging
await persistence.log_event(
    session_id="session-123",
    event_type="WORKER_COMPLETED",
    payload={"worker": "ResearchAgent", "trace_id": "..."},
    step_index=5
)
events = await persistence.get_events("session-123")
```

### Features

- **WAL Mode**: Concurrent reads/writes
- **Parent-Child Tracking**: `parent_session_id` for fractal agents
- **Event Logging**: Full step-by-step replay capability
- **Shared Connections**: Sub-agents can share DB connection
- **Optimistic Locking**: Prevents concurrent update conflicts

---

## Runtime Security (v1.5.2)

The `runtime` module provides secure code execution environments:

```python
from blackboard.runtime import LocalRuntime, DockerRuntime, RuntimeSecurityError

# ⚠️ LocalRuntime requires EXPLICIT acknowledgment
try:
    runtime = LocalRuntime()  # Raises RuntimeSecurityError!
except RuntimeSecurityError as e:
    print("Security: Must acknowledge unsafe execution")

# Development use (explicit acknowledgment)
runtime = LocalRuntime(dangerously_allow_execution=True)

# Or via environment variable
# BLACKBOARD_ALLOW_UNSAFE_EXECUTION=1

# Production: Use Docker
runtime = DockerRuntime(
    image="python:3.11-slim",
    memory_limit="256m",
    network_disabled=True
)

result = await runtime.execute("print('hello')")
print(result.stdout)  # "hello"
```

### Security Features

- **Explicit Acknowledgment**: `LocalRuntime` requires `dangerously_allow_execution=True`
- **Env Var Sanitization**: Sensitive keys (API keys, DB URLs) are removed from subprocess
- **Docker Isolation**: Full container isolation with resource limits
- **Import Restrictions**: Optional whitelist of allowed imports

---

## Config Propagation

`BlackboardConfig` supports child config creation for fractal agents:

```python
from blackboard.config import BlackboardConfig

parent = BlackboardConfig(
    max_recursion_depth=3,
    max_steps=100,
    allow_unsafe_execution=True  # Propagates to children
)

# Create child config with decremented depth
child = parent.for_child_agent()
print(child.max_recursion_depth)  # 2
print(child.allow_unsafe_execution)  # True (inherited)

# Serialize for network transfer
data = parent.to_dict()
restored = BlackboardConfig.from_dict(data)
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

## Model Context Protocol

Connect to external tools via MCP servers (v1.2.0):

```python
from blackboard.mcp import MCPServerWorker

# Connect to filesystem MCP server
fs = await MCPServerWorker.create(
    name="Filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
)

# DYNAMIC EXPANSION: Each MCP tool becomes a separate Worker
workers = fs.expand_to_workers()
# -> [MCPToolWorker(read_file), MCPToolWorker(write_file), ...]

orchestrator = Orchestrator(llm=llm, workers=workers)
# LLM sees: Filesystem:read_file(path), Filesystem:write_file(path, content), etc.
```

### MCPRegistry

```python
from blackboard.mcp import MCPRegistry

registry = MCPRegistry()
await registry.add("fs", command="npx", args=["@mcp/server-filesystem", "/"])
await registry.add("github", command="npx", args=["@mcp/server-github"])

# Get all workers
all_workers = registry.get_workers()
orchestrator = Orchestrator(llm=llm, workers=all_workers)
```

---

## OpenTelemetry

Distributed tracing with span hierarchy (v1.2.0):

```python
from blackboard.telemetry import OpenTelemetryMiddleware

# Basic usage
otel = OpenTelemetryMiddleware(service_name="my-agent")
orchestrator = Orchestrator(llm=llm, workers=workers, middleware=[otel])

# Creates span hierarchy:
# orchestrator.run
# ├── step.1
# │   └── worker.Writer
# ├── step.2
# │   └── worker.Critic
# └── ...
```

### With Jaeger/Zipkin

```python
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry import trace

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(JaegerExporter()))
trace.set_tracer_provider(provider)

otel = OpenTelemetryMiddleware(service_name="my-agent")
```

### Metrics Collector (no OTEL)

```python
from blackboard.telemetry import MetricsCollector

collector = MetricsCollector()
orchestrator.event_bus.subscribe_all(collector.on_event)

result = await orchestrator.run(goal="...")
print(collector.get_summary())
# {"duration_seconds": 5.2, "total_steps": 3, "total_worker_calls": 5}
```

---

## Session Replay

Record and replay sessions for debugging (v1.2.0):

```python
from blackboard.replay import SessionRecorder, RecordingLLMClient, ReplayOrchestrator

# 1. Record a session
recorder = SessionRecorder()
recording_llm = RecordingLLMClient(llm, recorder)

orchestrator = Orchestrator(llm=recording_llm, workers=workers)
result = await orchestrator.run(goal="Write a poem")
recorder.save("session.json")

# 2. Replay (no API calls!)
replay = ReplayOrchestrator.from_file("session.json", workers=workers)
replayed_result = await replay.run()
```

### Compare Sessions

```python
from blackboard.replay import compare_sessions, RecordedSession

original = RecordedSession.load("session.json")
diff = compare_sessions(original, replayed_result)

print(f"Status match: {diff.status_match}")
print(f"Differences: {diff.differences}")
```

---

## Terminal UI

Real-time terminal visualization with animated progress and markdown rendering (v1.3.0):

```python
from blackboard.tui import BlackboardTUI, watch

# Option 1: Quick start with watch()
result = watch(orchestrator, goal="Research and write about AI")

# Option 2: Manual control
tui = BlackboardTUI(orchestrator.event_bus)
with tui.live():
    result = await orchestrator.run(goal="...")
    tui.print_summary(result)
```

### TUI Features

- **Activity Log**: Real-time event ticker with timestamps
- **Animated Spinners**: Visual worker progress indicators
- **Live Markdown**: Artifacts rendered with proper formatting
- **Responsive Layout**: Adapts to terminal size automatically
- **Dynamic Borders**: Colors change based on state (thinking/running/done/failed)

---

## Production Fixes

Critical production fixes introduced in v1.3.0:

### Event Loop Safety

The SDK now safely executes inside existing event loops (FastAPI, Jupyter, nested async):

```python
# Works in FastAPI routes, Jupyter notebooks, etc.
result = orchestrator.run_sync(goal="...")  # No RuntimeError

# Also safe in TUI
from blackboard.tui import watch
result = watch(orchestrator, goal="...")
```

### Smart Context Management

Token-aware truncation prevents LLM context overflow:

```python
# Automatically truncates large states to fit context window
context = state.to_context_string(max_tokens=4000, chars_per_token=4)

# Priority: Goal > Feedback > Artifacts (with head/tail preview)
```

### MCP Persistent Sessions

MCP servers now maintain persistent connections for 90% faster tool calls:

```python
async with MCPServerWorker.create(...) as server:
    # All tool calls reuse the same session
    workers = server.expand_to_workers()
    orchestrator = Orchestrator(llm=llm, workers=workers)
    await orchestrator.run(goal="...")
```

### UsageTracker Memory Management

Prevent unbounded memory growth in long-running agents:

```python
from blackboard.usage import UsageTracker

tracker = UsageTracker(
    max_records=10000,  # Auto-eviction after 10k records
    on_flush=lambda records: save_to_db(records)  # Callback before eviction
)
```

---

## Standard Library Workers

Pre-built workers for common tasks. Install with `pip install blackboard-core[stdlib]`.

### WebSearchWorker

Search the web using Tavily or Serper APIs:

```python
from blackboard.stdlib import WebSearchWorker

# Uses TAVILY_API_KEY or SERPER_API_KEY from environment
search = WebSearchWorker()

# Or specify explicitly
search = WebSearchWorker(provider="tavily", api_key="...")

orchestrator = Orchestrator(llm=llm, workers=[search])
```

### BrowserWorker

Web scraping with Playwright. Install with `pip install blackboard-core[browser]`:

```python
from blackboard.stdlib import BrowserWorker

browser = BrowserWorker(headless=True, browser_type="chromium")

# Extracts page content, can take screenshots, extract links
orchestrator = Orchestrator(llm=llm, workers=[browser])
```

### CodeInterpreterWorker

Safe code execution using Docker sandbox:

```python
from blackboard.stdlib import CodeInterpreterWorker

# Uses Docker for isolation (falls back to local with warning)
interpreter = CodeInterpreterWorker(
    docker_image="python:3.11-slim",
    memory_limit="512m",
    network_enabled=False
)

orchestrator = Orchestrator(llm=llm, workers=[interpreter])
```

### HumanProxyWorker

Pause execution and wait for human input:

```python
from blackboard.stdlib import HumanProxyWorker

# API Mode: Sets status to PAUSED, stores question in pending_input
human = HumanProxyWorker()

# Resume with: state.pending_input["answer"] = "user response"
# Then: await orchestrator.run(state=state)

# Callback Mode: Get input immediately via callback
async def get_input(question, context, options):
    return input(f"{question}: ")

human = HumanProxyWorker(input_callback=get_input)
```

For CLI usage, use `CLIHumanProxyWorker`:

```python
from blackboard.stdlib.workers.human import CLIHumanProxyWorker

human = CLIHumanProxyWorker()  # Prompts via stdin/stdout
```

---

## Blackboard Serve (API Deployment)

Deploy your orchestrator as a REST API with one command. Install with `pip install blackboard-core[serve]`.

### CLI Usage

```bash
# Start API server
python -m pip install -e .[serve]  # Install serve dependencies
python -m blackboard.cli serve my_app:create_orchestrator --port 8000
```

Where `my_app:create_orchestrator` is a module path to a function that returns an Orchestrator:

```python
# my_app.py
from blackboard import Orchestrator
from blackboard.llm import LiteLLMClient

def create_orchestrator():
    return Orchestrator(
        llm=LiteLLMClient(model="gpt-4o"),
        workers=[...]
    )
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/runs` | POST | Start a new run |
| `/runs` | GET | List all runs |
| `/runs/{id}` | GET | Get run status |
| `/runs/{id}/full` | GET | Get full state with artifacts |
| `/runs/{id}/resume` | POST | Resume paused run |
| `/runs/{id}/stream` | GET | SSE event stream |
| `/runs/{id}/events` | GET | Historical events |
| `/health` | GET | Health check |

### Programmatic Usage

```python
from blackboard.serve import create_app, BlackboardAPI

# Option 1: Create FastAPI app directly
app = create_app("my_app:create_orchestrator")

# Option 2: Use convenience wrapper
api = BlackboardAPI("my_app:create_orchestrator")
api.run(port=8000)
```

### Example API Calls

```bash
# Start a run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"goal": "Write a haiku about AI"}'

# Check status
curl http://localhost:8000/runs/{run_id}

# Resume paused run
curl -X POST http://localhost:8000/runs/{run_id}/resume \
  -H "Content-Type: application/json" \
  -d '{"answer": "Yes, proceed"}'

# Stream events (SSE)
curl http://localhost:8000/runs/{run_id}/stream
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

---

## Model Context Protocol (MCP)

MCP enables connecting to external tools without writing code.

### Basic Usage (Stdio)

```python
from blackboard.mcp import MCPServerWorker

# Connect to local MCP server
server = await MCPServerWorker.create(
    name="Filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-fs", "/tmp"]
)

# Expand to individual workers (recommended)
workers = server.expand_to_workers()
orchestrator = Orchestrator(llm=llm, workers=workers)
```

### SSE Transport (Remote Servers)

Connect to remote/Dockerized MCP servers via HTTP:

```python
# Connect via SSE endpoint
server = await MCPServerWorker.create(
    name="RemoteAPI",
    url="http://mcp-server:8080/sse"
)
```

### Sampling Support

Let MCP tools ask the LLM for help:

```python
server = await MCPServerWorker.create(
    name="CodeAnalyzer",
    url="http://localhost:8080/sse",
    llm_client=my_llm  # For sampling requests
)
```

### MCP Resources

Access file-like data from MCP servers:

```python
async with server:
    # List available resources
    resources = await server.list_resources()
    
    # Read a resource
    content = await server.read_resource(resources[0]['uri'])
    
    # Load all resources into Blackboard
    loaded = await server.load_resources_to_state(state)
```

---

## Blueprints (Structured Workflows)

Blueprints constrain the orchestrator to follow specific patterns.

### SequentialPipeline

Force strict A → B → C execution:

```python
from blackboard.flow import SequentialPipeline

pipeline = SequentialPipeline([
    SearchWorker(),
    WriterWorker(),
    CriticWorker()
])

result = await orchestrator.run(
    goal="Research and write an article",
    blueprint=pipeline
)
```

### Router

Supervisor chooses the best worker for the task:

```python
from blackboard.flow import Router

router = Router([
    MathAgent(),
    CodeAgent(),
    ResearchAgent()
], selection_prompt="Choose based on the query type")

result = await orchestrator.run(
    goal="Solve 2x + 5 = 15",
    blueprint=router
)
```

### Custom Blueprints

Define multi-step workflows with detailed control:

```python
from blackboard.flow import Blueprint, Step

blog_flow = Blueprint(
    name="Blog Writing",
    steps=[
        Step(
            name="research",
            allowed_workers=["WebSearch", "Browser"],
            instructions="Gather information only"
        ),
        Step(
            name="write",
            allowed_workers=["Writer"],
            instructions="Write based on research"
        ),
        Step(
            name="review",
            allowed_workers=["Critic"],
            exit_condition=lambda s: s.get_latest_feedback().passed
        )
    ]
)
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

The supervisor will produce structured thinking before each decision.

---

## Time-Travel Debugging (v1.6.3)

Fork sessions at any checkpoint to experiment with different approaches:

```python
from blackboard import Orchestrator, SQLitePersistence

persistence = SQLitePersistence("./sessions.db")
await persistence.initialize()

orchestrator = Orchestrator(llm=llm, workers=workers)
orchestrator.set_persistence(persistence)

# Run original session (checkpoints saved automatically)
result = await orchestrator.run(goal="Write an article")

# Later: fork at step 5 to try different approach
session_id = result.metadata["session_id"]
fork_id = await orchestrator.fork_session(session_id, step_index=5)

# Load and continue from forked state
forked = await persistence.load(fork_id)
new_result = await orchestrator.run(state=forked)
```

### Checkpoint API

```python
# Manual checkpoint operations
await persistence.save_checkpoint(session_id, step_index, state)
state_at_step = await persistence.load_state_at_step(session_id, 5)
checkpoints = await persistence.list_checkpoints(session_id)
await persistence.delete_checkpoints(session_id)
```

---

## Prompt Registry (v1.6.3)

Externalize prompts for easier customization without code changes:

```python
from blackboard import PromptRegistry

# Initialize registry
registry = PromptRegistry(
    prompts_dir="prompts/",           # Jinja2 templates
    config_path="blackboard.prompts.json"  # JSON overrides
)

# Get rendered prompt
prompt = registry.get("Writer", {"topic": "AI safety", "style": "formal"})

# Runtime override (for testing/optimization)
registry.set("Writer", "New prompt: {{ topic }}")

# List available prompts
keys = registry.list_keys()  # {"Writer": "config", "Critic": "template"}
```

### Template Files

Create `prompts/Writer.jinja2`:

```jinja2
You are a professional writer.

{% if style %}
Style: {{ style }}
{% endif %}

Topic: {{ topic }}
```

---

## Instruction Optimizer (v1.6.3)

Automatically analyze failures and generate improved prompts:

```python
from blackboard import Optimizer, PromptPatch

optimizer = Optimizer(llm=meta_llm, orchestrator=orchestrator)

# Find failures in a session
failures = await optimizer.analyze_failures("session-001")

# Generate improved prompts
for failure in failures:
    candidates = await optimizer.generate_candidates(failure, n=3)
    
    # Verify by forking and re-running
    for patch in candidates:
        verified = await optimizer.verify_candidate(patch, failure)
        if verified.verified:
            print(f"Found fix for {patch.worker_name}")
            break

# Save patches for human review
optimizer.save_patches(verified_patches, "blackboard.patches.json")
```

### PromptPatch Structure

```python
@dataclass
class PromptPatch:
    worker_name: str
    prompt_key: str
    original: str
    original_hash: str  # SHA256 for conflict detection
    proposed: str
    reasoning: str
    verification_score: float
    verified: bool
```

---

## CLI Commands (v1.6.3)

### Initialize Project

```bash
blackboard init
# Creates:
#   prompts/           - Directory for Jinja2 templates
#   prompts/example.jinja2  - Example template
#   blackboard.prompts.json - Config file
```

### Optimize Prompts

```bash
# Analyze failures and generate patches
blackboard optimize run --session-id my-session --db-path ./blackboard.db

# Review pending patches
blackboard optimize review --patches-file blackboard.patches.json
```

---

## Interactive TUI (v1.7.0)

A Textual-based Mission Control for real-time debugging:

```python
from blackboard.ui import create_tui, is_headless

# Check if running in CI
if not is_headless():
    app = create_tui(orchestrator)
    await app.run_async()
```

### Key Bindings

| Key | Action |
|-----|--------|
| `Space` | Pause/Resume execution |
| `I` | Inject intervention command |
| `Q` | Quit |

### Features
- **3-pane dashboard**: Log, Artifacts, State
- **Pause/Resume**: Stop execution at any point
- **Intervention**: Inject commands mid-execution
- **Headless detection**: Auto-detects CI environments

---

## Ecosystem Adapters (v1.7.0)

### LangChain Adapter

```python
from langchain_community.tools import TavilySearchResults
from blackboard.integrations.langchain import wrap_tool

# Wrap any LangChain tool
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

## Best Practices

1. **Use persistence** for production deployments
2. **Set recursion limits** when using fractal agents
3. **Enable CoT** for complex decision-making
4. **Use SQLite** for single-node, PostgreSQL for distributed
5. **Externalize prompts** for easy iteration without code changes
6. **Fork sessions** to debug failures without rerunning from scratch
7. **Use ecosystem adapters** to leverage existing LangChain/LlamaIndex tools
