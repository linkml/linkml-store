# test_mongodb_adapter.py

import pytest
import yaml
from linkml_store.api.stores.mongodb.mongodb_database import MongoDBDatabase
from pymongo import MongoClient


@pytest.fixture(scope="module")
def mongodb_client():
    client = MongoClient("mongodb://localhost:27017")
    yield client
    client.close()


@pytest.fixture(scope="function")
def mongodb_database(mongodb_client):
    db = mongodb_client["test_db"]
    yield db
    db.drop_collection("test_collection")

@pytest.fixture(scope="function")
def mongodb_collection(mongodb_client):
    db = mongodb_client["test_db"]
    collection = db["test_collection"]
    collection.delete_many({})  # Ensure a clean slate
    yield MongoDBCollection(collection)
    collection.drop()

def test_upsert_insert(mongodb_collection):
    """
    Test that the upsert method inserts a new document if it does not exist.
    """
    obj = {"_id": "1", "name": "Alice", "age": 25, "occupation": "Engineer"}

    # Upsert operation: should insert because no document with _id=1 exists
    mongodb_collection.upsert(obj, filter_fields=["_id"])

    # Check if the document exists in the collection
    result = mongodb_collection.mongo_collection.find_one({"_id": "1"})
    assert result is not None
    assert result["name"] == "Alice"
    assert result["age"] == 25
    assert result["occupation"] == "Engineer"

def test_upsert_update(mongodb_collection):
    """
    Test that the upsert method updates an existing document while preserving unchanged fields.
    """
    # Insert initial document
    initial_obj = {"_id": "2", "name": "Bob", "age": 30, "occupation": "Builder"}
    mongodb_collection.mongo_collection.insert_one(initial_obj)

    # Upsert with an update (modifying age only)
    updated_obj = {"_id": "2", "age": 35}
    mongodb_collection.upsert(updated_obj, filter_fields=["_id"], update_fields=["age"])

    # Verify that the document was updated correctly
    result = mongodb_collection.mongo_collection.find_one({"_id": "2"})
    assert result is not None
    assert result["_id"] == "2"
    assert result["age"] == 35  # Should be updated
    assert result["name"] == "Bob"  # Should remain unchanged
    assert result["occupation"] == "Builder"  # Should remain unchanged


@pytest.mark.parametrize("handle", ["mongodb://localhost:27017/test_db", None, "mongodb"])
@pytest.mark.integration
def test_insert_and_query(handle):
    # Create a MongoDBDatabase instance
    db = MongoDBDatabase(handle=handle)

    # Create a collection
    collection = db.create_collection("test_collection", recreate_if_exists=True)

    # Insert a few documents
    documents = [
        {
            "name": "Alice",
            "age": 25,
            "occupation": "Architect",
            "foods": ["apple", "banana"],
            "relationships": [
                {"person": "Bob", "relation": "friend"},
                {"person": "Charlie", "relation": "brother"},
            ],
            "meta": {"date": "2021-01-01", "notes": "likes fruit"},
        },
        {
            "name": "Bob",
            "age": 30,
            "occupation": "Builder",
            "foods": ["carrot", "date"],
            "relationships": [
                {"person": "Alice", "relation": "friend"},
            ],
            "meta": {"date": "2021-01-01"},
        },
        {
            "name": "Charlie",
            "age": 35,
            "occupation": "Lawyer",
            "foods": ["eggplant", "fig", "banana"],
            "meta": {"date": "2021-01-03", "notes": "likes fruit", "curator": "Ziggy"},
        },
        {
            "name": "Jie",
            "age": 27,
            "occupation": "Architect",
            "foods": ["grape", "honey", "apple"],
            "meta": {"date": "2021-01-03"},
        },
    ]
    collection.insert(documents)

    query_result = collection.find()
    assert query_result.num_rows == len(documents)

    # Query the collection
    query_result = collection.find({"age": {"$gte": 30}})

    # Assert the query results
    assert query_result.num_rows == 2
    assert len(query_result.rows) == 2
    assert query_result.rows[0]["name"] == "Bob"
    assert query_result.rows[1]["name"] == "Charlie"
    assert set(query_result.rows[0].keys()) == {"name", "age", "occupation", "foods", "meta", "relationships"}
    cases = [
        ({}, "occupation", {("Architect", 2), ("Lawyer", 1), ("Builder", 1)}),
        ({"occupation": "Architect"}, "occupation", {("Architect", 2)}),
        # test unwinding multivalued
        (
            {},
            "foods",
            {
                ("fig", 1),
                ("banana", 2),
                ("eggplant", 1),
                ("apple", 2),
                ("carrot", 1),
                ("date", 1),
                ("grape", 1),
                ("honey", 1),
            },
        ),
        ({}, "relationships.relation", {("friend", 2), ("brother", 1)}),
        ({}, "meta.date", {("2021-01-01", 2), ("2021-01-03", 2)}),
        (
            {},
            "meta",
            {
                ("curator: Ziggy\ndate: '2021-01-03'\nnotes: likes fruit\n", 1),
                ("date: '2021-01-01'\nnotes: likes fruit\n", 1),
                ("date: '2021-01-01'\n", 1),
                ("date: '2021-01-03'\n", 1),
            },
        ),
        (
            {},
            ("occupation", "foods"),
            {
                (("Architect", "apple"), 2),
                (("Architect", "banana"), 1),
                (("Architect", "grape"), 1),
                (("Architect", "honey"), 1),
                (("Builder", "carrot"), 1),
                (("Builder", "date"), 1),
                (("Lawyer", "banana"), 1),
                (("Lawyer", "eggplant"), 1),
                (("Lawyer", "fig"), 1),
            },
        ),
    ]
    for where, fc, expected in cases:
        fr = collection.query_facets(where, facet_columns=[fc])
        val_count_tuples = fr[fc]
        if any(isinstance(v, dict) for v, _ in val_count_tuples):
            val_count_tuples = {(yaml.dump(v), c) for v, c in val_count_tuples}
        results = set(val_count_tuples)
        assert results == expected
        # test re-querying with facets
        for v, c in fr[fc]:
            if isinstance(fc, tuple):
                where = {fc[i]: v[i] for i in range(len(fc))}
            else:
                where = {fc: v}
            results = collection.find(where, limit=-1)
            assert results.num_rows == c, f"where {where} failed to find expected"
