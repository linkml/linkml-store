"""Tests for the FastAPI web API."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("linkml_renderer")

from fastapi.testclient import TestClient  # noqa: E402

from linkml_store import Client  # noqa: E402
from linkml_store.webapi.main import app, get_client  # noqa: E402

BASE = "/databases/test/collections/persons"
NAME_A = {"where": '{"name": "a"}'}


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


def link(body, rel):
    return next(link["href"] for link in body["links"] if link["rel"] == rel)


@pytest.mark.parametrize("where", ['{"name": "a"}', "name: a"])
def test_objects_where_filters_and_counts(api, where):
    response = api.get(f"{BASE}/objects", params={"where": where, "limit": 2})
    assert response.status_code == 200
    body = response.json()
    assert [item["data"]["name"] for item in body["items"]] == ["a", "a"]
    assert body["meta"]["item_count"] == 3
    page_two = api.get(link(body, "next")).json()
    assert [item["data"]["id"] for item in page_two["items"]] == ["P2"]


def test_object_link_without_identifier_resolves(api):
    body = api.get(f"{BASE}/objects", params={**NAME_A, "offset": 1}).json()
    item = body["items"][0]
    target = api.get(item["links"][0]["href"]).json()
    assert [i["data"] for i in target["items"]] == [item["data"]]


def test_objects_with_no_match_link_to_valid_pages(api):
    body = api.get(f"{BASE}/objects", params={"where": '{"name": "zzz"}'}).json()
    assert body["meta"]["item_count"] == 0
    assert api.get(link(body, "last")).status_code == 200


def test_blank_where_means_no_filter(api):
    assert api.get(f"{BASE}/objects", params={"where": " "}).json()["meta"]["item_count"] == 5


def test_facets_route_counts_the_filtered_set(api):
    body = api.get(f"{BASE}/facets", params=NAME_A).json()
    assert body["data"]["items"]["name"] == [["a", 3]]
    assert body["data"]["total_count"] == 3
    assert "where=" in link(body, "self")


def test_attribute_routes_apply_where(api):
    attributes = api.get(f"{BASE}/attributes", params=NAME_A).json()
    name_facet = next(item for item in attributes["items"] if item["name"] == "name")
    assert name_facet["data"] == [{"value": "a", "count": 3}]
    attribute = api.get(f"{BASE}/attributes/name", params=NAME_A).json()
    assert [(item["name"], item["data"]) for item in attribute["items"]] == [("a", {"count": 3})]


def test_equals_route_applies_where(api):
    response = api.get(f"{BASE}/attributes/name/equals/a", params={"where": '{"id": "P1"}'})
    assert response.status_code == 200
    body = response.json()
    assert [item["data"]["id"] for item in body["items"]] == ["P1"]
    assert body["meta"]["item_count"] == 1


def test_equals_route_names_items_by_position(api):
    body = api.get(f"{BASE}/attributes/name/equals/a", params={"limit": 2, "offset": 2}).json()
    assert [item["name"] for item in body["items"]] == ["2"]


def test_object_by_id_without_identifier_is_not_found(api):
    assert api.get(f"{BASE}/objects/P1").status_code == 404


@pytest.mark.parametrize("where", ["{name: [", "just a string"])
def test_bad_where_is_a_client_error(api, where):
    assert api.get(f"{BASE}/objects", params={"where": where}).status_code == 400


def test_where_is_documented():
    paths = app.openapi()["paths"]
    where_params = [
        param
        for operation in paths.values()
        for param in operation.get("get", {}).get("parameters", [])
        if param["name"] == "where"
    ]
    assert len(where_params) == 5
    assert all("YAML or JSON" in param["description"] for param in where_params)
