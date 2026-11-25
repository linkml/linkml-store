"""Tests for behavior of the DuckDB adapter and of the generic Collection it builds on."""

import logging
import math

import pytest

from linkml_store import Client
from linkml_store.api.queries import Query
from tests import COUNTRIES_SCHEMA


@pytest.fixture
def collection():
    client = Client()
    db = client.attach_database("duckdb", alias="test")
    return db.create_collection("Person")


def test_attribute_first_seen_late_is_stored(collection):
    """An attribute that first appears after the type-inference sample is still stored."""
    objs = [{"id": i} for i in range(20)] + [{"id": 20, "nickname": "Al"}]
    collection.insert(objs)
    assert collection.find({"id": 20}).rows == [{"id": 20, "nickname": "Al"}]


def test_attribute_null_in_first_objects_is_typed(collection):
    """The range is inferred from non-null values, wherever they are in the list."""
    objs = [{"id": i, "age": None} for i in range(20)] + [{"id": 20, "age": 33}]
    collection.insert(objs)
    assert collection.class_definition().attributes["age"].range == "integer"
    assert collection.find({"age": 33}).num_rows == 1


def test_insert_warns_about_fields_that_are_not_columns(collection, caplog):
    collection.insert([{"id": 1, "name": "a"}])
    with caplog.at_level(logging.WARNING):
        collection.insert([{"id": 2, "name": "b", "age": 5}])
    assert "age" in caplog.text
    assert collection.find({}).num_rows == 2


def test_insert_rejects_non_dict(collection):
    with pytest.raises(ValueError, match="dict"):
        collection.insert(["not an object"])


@pytest.mark.parametrize(
    "where,expected_ids",
    [
        ({"id": {"$in": [1, 3]}}, [1, 3]),
        ({"id": {"$nin": [1, 3]}}, [2]),
        ({"id": {"$gt": 1}}, [2, 3]),
        ({"id": {"$gte": 2, "$lt": 3}}, [2]),
        ({"name": {"$ne": "a"}}, [2]),
        ({"name": None}, [3]),
        ({"name": {"$ne": None}}, [1, 2]),
        ({"name": "a", "id": 1}, [1]),
    ],
)
def test_where_operators(collection, where, expected_ids):
    collection.insert([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": None}])
    assert sorted(row["id"] for row in collection.find(where).rows) == expected_ids


def test_get_with_integer_ids(collection):
    collection.insert([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}])
    collection.set_identifier_attribute_name("id")
    assert sorted(row["id"] for row in collection.get([1, 3]).rows) == [1, 3]


def test_select_cols_omitting_inlined_column(collection):
    collection.insert([{"id": 1, "address": {"city": "Paris"}}])
    assert collection.find({}).rows == [{"id": 1, "address": {"city": "Paris"}}]
    assert collection.find({}, select_cols=["id"]).rows == [{"id": 1}]


def test_database_query_before_collections_are_initialized(tmp_path):
    handle = f"duckdb:///{tmp_path / 'test.db'}"
    db = Client().attach_database(handle, alias="a")
    db.create_collection("Person").insert([{"id": 1}])
    db.commit()
    db.close()
    db2 = Client().attach_database(handle, alias="b")
    assert db2.query(Query(from_table="Person")).rows == [{"id": 1}]


def test_facet_limit(collection):
    collection.insert([{"id": i, "name": f"n{i}"} for i in range(20)])
    assert len(collection.query_facets(facet_columns=["name"], facet_limit=5)["name"]) == 5
    assert len(collection.query_facets(facet_columns=["name"])["name"]) == 20


def test_group_by_empty_collection(collection):
    assert collection.group_by(["name"]).rows == []


def test_no_indexers(collection):
    assert collection.list_index_names() == []
    with pytest.raises(ValueError, match="No index named"):
        collection.index_objects([], "simple")


@pytest.mark.parametrize("hidden", [True, False])
def test_hidden(hidden):
    db = Client().attach_database("duckdb", alias="test")
    c = db.create_collection("Person")
    c.metadata.hidden = hidden
    assert c.hidden is hidden


def test_search_query_shorter_than_a_trigram(collection):
    """The simple indexer has no trigrams for a short query; scores must still be numbers."""
    collection.insert([{"id": 1, "name": "New Zealand"}, {"id": 2, "name": "France"}])
    scores = [score for score, _ in collection.search("NZ").ranked_rows]
    assert len(scores) == 2
    assert not any(math.isnan(score) for score in scores)


def test_search_mmr_without_limit(collection):
    collection.insert([{"id": 1, "name": "New Zealand"}, {"id": 2, "name": "France"}])
    ranked_rows = collection.search("France", mmr_relevance_factor=0.5).ranked_rows
    assert [obj["name"] for _, obj in ranked_rows] == ["France", "New Zealand"]


def test_index_from_another_session_is_invalidated(tmp_path):
    """
    An index is stored in the database, but the indexer that made it is not.

    Changes made through a new client must still invalidate the stored index.
    """
    handle = f"duckdb:///{tmp_path / 'test.db'}"

    def open_collection():
        return Client().attach_database(handle, alias="test").get_collection("persons")

    c = open_collection()
    c.insert([{"id": 1, "name": "alpha scientist"}, {"id": 2, "name": "beta plumber"}])
    c.attach_indexer("simple")
    assert c.search("scientist").ranked_rows[0][1]["id"] == 1
    c.parent.close()

    c = open_collection()
    c.insert([{"id": 3, "name": "gamma astronaut"}])
    c.parent.close()

    c = open_collection()
    c.attach_indexer("simple", auto_index=False)
    ranked_rows = c.search("astronaut").ranked_rows
    assert len(ranked_rows) == 3
    assert ranked_rows[0][1]["id"] == 3


def test_attach_database_with_schema_location():
    db = Client().attach_database("duckdb", alias="test", schema_location=str(COUNTRIES_SCHEMA))
    assert db.schema_view.get_class("Country") is not None
