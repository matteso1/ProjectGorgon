from fastapi.testclient import TestClient

from backend.app.main import app


def test_generate_stream_payload():
    client = TestClient(app)
    with client.websocket_connect("/generate_stream") as ws:
        data = ws.receive_json()
        assert "text" in data
        assert "tree_debug" in data
        assert "candidates" in data["tree_debug"]
        assert "accepted" in data["tree_debug"]
        assert "speedup" in data["tree_debug"]