import asyncio
import re
from pathlib import Path

import pytest

from blackboard import Blackboard, Orchestrator, Status
from blackboard.core import Agent
from blackboard.decorators import worker
from blackboard.persistence import InMemoryPersistence
from blackboard.protocols import WorkerInput
from blackboard.serve.manager import RunStatus, SessionManager


class StaticDecisionLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt: str) -> str:
        if self.responses:
            return self.responses.pop(0)
        return '{"action": "done", "reasoning": "done"}'


@pytest.mark.asyncio
async def test_agent_success_path_returns_summary_artifact():
    @worker
    def sub_writer() -> str:
        return "child result"

    agent = Agent(
        name="ChildAgent",
        description="Runs a child orchestrator",
        llm=StaticDecisionLLM([
            '{"action": "call", "worker": "SubWriter", "reasoning": "write"}',
            '{"action": "done", "reasoning": "finished"}',
        ]),
        workers=[sub_writer],
    )

    output = await agent.run(
        Blackboard(goal="parent"),
        WorkerInput(instructions="complete delegated task"),
    )

    assert output.feedback is None
    assert output.artifact is not None
    assert output.artifact.type == "agent_result"
    assert "child result" in output.artifact.content


@pytest.mark.asyncio
async def test_recover_session_preserves_original_status():
    persistence = InMemoryPersistence()
    state = Blackboard(goal="recover me", status=Status.GENERATING)
    await persistence.save(state, "session-1")

    recovered = await Orchestrator.recover_session(
        persistence=persistence,
        session_id="session-1",
        llm=StaticDecisionLLM([]),
        workers=[],
    )

    assert recovered.status == Status.PAUSED
    assert recovered.metadata["recovered_from_status"] == Status.GENERATING.value
    assert recovered.metadata["recovery_mode"] is True


@pytest.mark.asyncio
async def test_sqlite_save_load_stays_fresh_when_checkpoint_interval_is_greater_than_one(tmp_path):
    aiosqlite = pytest.importorskip("aiosqlite")
    assert aiosqlite is not None

    from blackboard import Artifact
    from blackboard.persistence import SQLitePersistence

    persistence = SQLitePersistence(str(tmp_path / "fresh.db"), checkpoint_interval=3)
    state = Blackboard(goal="freshness")
    state.add_artifact(Artifact(type="text", content="first", creator="Writer"))

    await persistence.save(state, "session-1")
    await persistence.save_checkpoint("session-1", 1, state)
    loaded = await persistence.load("session-1")
    assert loaded.get_last_artifact().content == "first"

    state.add_artifact(Artifact(type="text", content="second", creator="Writer"))
    await persistence.save(state, "session-1")
    await persistence.save_checkpoint("session-1", 2, state)
    loaded = await persistence.load("session-1")
    assert loaded.get_last_artifact().content == "second"
    assert await persistence.list_checkpoints("session-1") == []

    state.add_artifact(Artifact(type="text", content="third", creator="Writer"))
    await persistence.save(state, "session-1")
    await persistence.save_checkpoint("session-1", 3, state)
    loaded = await persistence.load("session-1")
    assert loaded.get_last_artifact().content == "third"
    assert await persistence.list_checkpoints("session-1") == [3]

    state.update_status(Status.PAUSED)
    await persistence.save(state, "session-1")
    await persistence.save_checkpoint("session-1", 4, state)
    assert await persistence.list_checkpoints("session-1") == [3, 4]

    await persistence.close()


@pytest.mark.asyncio
async def test_session_manager_uses_persistence_and_restores_sessions():
    persistence = InMemoryPersistence()

    def factory():
        class StubOrchestrator:
            def __init__(self):
                self.event_bus = type("Bus", (), {"subscribe_all_async": lambda self, cb: None})()
                self.persistence = None

            def set_persistence(self, persistence_backend):
                self.persistence = persistence_backend

            async def run(self, goal=None, state=None, max_steps=20):
                assert self.persistence is persistence
                assert state is not None
                assert state.metadata["session_id"]
                state.update_status(Status.PAUSED)
                state.pending_input = {"question": "name", "answer": None}
                await self.persistence.save(state, state.metadata["session_id"])
                return state

        return StubOrchestrator()

    manager = SessionManager(factory, persistence=persistence)
    session = await manager.create_run("Persist me")
    await asyncio.sleep(0.05)

    assert session.status == RunStatus.PAUSED
    assert session.state is not None
    assert session.state.metadata["session_id"] == session.id

    restored_manager = SessionManager(factory, persistence=persistence)
    await restored_manager.start()
    restored = restored_manager.get_session(session.id)

    assert restored is not None
    assert restored.status == RunStatus.PAUSED
    assert restored.state is not None
    assert restored.state.pending_input == {"question": "name", "answer": None}

    await restored_manager.stop()


def _parse_optional_dependency_group(pyproject_text: str, group: str) -> set[str]:
    pattern = rf"(?ms)^{re.escape(group)}\s*=\s*\[(.*?)^\]"
    match = re.search(pattern, pyproject_text)
    assert match is not None, f"Missing optional dependency group: {group}"
    return {
        dep
        for dep in re.findall(r'"([^"]+)"', match.group(1))
    }


def test_all_extra_is_superset_of_documented_runtime_extras():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    all_extra = _parse_optional_dependency_group(pyproject_text, "all")

    required = set()
    for group in ("serve", "browser", "stdlib", "ui"):
        required.update(_parse_optional_dependency_group(pyproject_text, group))

    assert required.issubset(all_extra)
