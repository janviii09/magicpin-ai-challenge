"""
Tests for all 5 HTTP endpoints exposed by Vera 2.0:
- POST /v1/context
- POST /v1/tick
- POST /v1/reply
- GET /v1/healthz
- GET /v1/metadata
"""

import pytest
from fastapi.testclient import TestClient
from server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_healthz(client):
    response = client.get("/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "contexts_loaded" in data


def test_get_metadata(client):
    response = client.get("/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "team_name" in data
    assert "team_members" in data
    assert "model" in data
    assert "approach" in data
    assert "version" in data


def test_post_context(client):
    # Push category context
    payload = {
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": {
            "slug": "dentists",
            "voice": {"tone": "peer_clinical"},
            "metrics": {"avg_ticket": 1200}
        }
    }
    response = client.post("/v1/context", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] is True

    # Push merchant context
    merchant_payload = {
        "scope": "merchant",
        "context_id": "m_test_001",
        "version": 1,
        "payload": {
            "merchant_id": "m_test_001",
            "category_slug": "dentists",
            "identity": {
                "name": "Dr. Smile Dental",
                "owner_first_name": "Smile"
            }
        }
    }
    resp = client.post("/v1/context", json=merchant_payload)
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


def test_post_tick(client):
    tick_payload = {
        "now": "2026-09-27T10:00:00Z",
        "available_triggers": ["trg_001_research_digest_dentists"]
    }
    response = client.post("/v1/tick", json=tick_payload)
    assert response.status_code == 200
    data = response.json()
    assert "actions" in data
    assert isinstance(data["actions"], list)


def test_post_reply(client):
    reply_payload = {
        "conversation_id": "conv_reply_test_001",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Yes, please activate it.",
        "received_at": "2026-09-27T10:05:00Z"
    }
    response = client.post("/v1/reply", json=reply_payload)
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
