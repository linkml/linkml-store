"""Tests for the FastAPI web API."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("linkml_renderer")

from fastapi.testclient import TestClient  # noqa: E402

from linkml_store import Client  # noqa: E402
from linkml_store.webapi.main import app, get_client  # noqa: E402

BASE = "/databases/test/collections/persons"


@pytest.fixture
def api(tmp_path):
    # The server handles requests on another thread, which an in-memory DuckDB database would not share,
    # so the test uses a database file.
    client = Client()
    db = client.attach_database(f"duckdb:///{tmp_path / 'test.ddb'}", alias="test")
    collection = db.create_collection("Person", alias="persons")
    collection.insert([{"id": f"P{i}", "name": "a" if i < 3 else "b", "age": 30 + i} for i in range(5)])
    app.dependency_overrides[get_client] = lambda: client
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("where", ['{"name": "a"}', "name: a"])
def test_objects_where_filters_and_counts(api, where):
    response = api.get(f"{BASE}/objects", params={"where": where, "limit": 2})
    assert response.status_code == 200
    body = response.json()
    assert [item["data"]["name"] for item in body["items"]] == ["a", "a"]
    assert body["meta"]["item_count"] == 3
    next_link = next(link["href"] for link in body["links"] if link["rel"] == "next")
    page_two = api.get(next_link).json()
    assert [item["data"]["id"] for item in page_two["items"]] == ["P2"]


@pytest.mark.parametrize("route", ["facets", "attributes", "attributes/name"])
def test_facet_routes_accept_where(api, route):
    response = api.get(f"{BASE}/{route}", params={"where": '{"name": "a"}'})
    assert response.status_code == 200


def test_equals_route_applies_where(api):
    response = api.get(f"{BASE}/attributes/name/equals/a", params={"where": '{"id": "P1"}'})
    assert response.status_code == 200
    body = response.json()
    assert [item["data"]["id"] for item in body["items"]] == ["P1"]
    assert body["meta"]["item_count"] == 1


@pytest.mark.parametrize("where", ["{name: [", "just a string"])
def test_bad_where_is_a_client_error(api, where):
    assert api.get(f"{BASE}/objects", params={"where": where}).status_code == 400
