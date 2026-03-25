"""Focused tests for the Streamlit API client helpers."""

from types import SimpleNamespace

from blackboard.ui.app import BlackboardClient, extract_run_id, load_run_view_data


class FakeResponse:
    """Minimal HTTP response test double."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class RecordingClient:
    """Record requests made by BlackboardClient."""

    def __init__(self):
        self.calls = []

    def post(self, url, json):
        self.calls.append(("POST", url, json))
        return FakeResponse({"id": "run-123"})

    def get(self, url):
        self.calls.append(("GET", url, None))
        if url.endswith("/full"):
            return FakeResponse({
                "state": {
                    "artifacts": [{"id": "art-1", "type": "text", "content": "hello"}]
                }
            })
        return FakeResponse({"id": "run-123", "status": "completed"})

    def close(self):
        return None


def test_extract_run_id_prefers_api_id():
    """The UI should read the modern API identifier."""
    assert extract_run_id({"id": "run-123", "run_id": "legacy"}) == "run-123"


def test_resume_run_sends_answer_payload(monkeypatch):
    """Resume calls should follow the HTTP API contract."""
    recorder = RecordingClient()
    fake_httpx = SimpleNamespace(Client=lambda timeout: recorder)
    monkeypatch.setattr("blackboard.ui.app.httpx", fake_httpx)

    client = BlackboardClient("http://localhost:8000")
    response = client.resume_run("run-123", "Alice", max_steps=15)

    assert response["id"] == "run-123"
    assert recorder.calls == [
        (
            "POST",
            "http://localhost:8000/runs/run-123/resume",
            {"answer": "Alice", "max_steps": 15},
        )
    ]


def test_load_run_view_data_merges_artifacts_from_full_endpoint():
    """Artifact rendering should pull from the full run endpoint."""

    class FakeClient:
        def __init__(self):
            self.calls = []

        def get_run(self, run_id):
            self.calls.append(("summary", run_id))
            return {"id": run_id, "status": "completed"}

        def get_run_full(self, run_id):
            self.calls.append(("full", run_id))
            return {
                "state": {
                    "artifacts": [{"id": "art-1", "type": "text", "content": "hello"}]
                }
            }

    client = FakeClient()

    run_data = load_run_view_data(client, "run-123")

    assert client.calls == [("summary", "run-123"), ("full", "run-123")]
    assert run_data["artifacts"] == [{"id": "art-1", "type": "text", "content": "hello"}]
