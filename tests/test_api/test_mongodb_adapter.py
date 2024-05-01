# test_mongodb_adapter.py

import pytest
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


@pytest.mark.integration
def test_insert_and_query(mongodb_database):
    # Create a MongoDBDatabase instance
    db = MongoDBDatabase(handle="mongodb://localhost:27017/test_db")

    # Create a collection
    collection = db.create_collection("test_collection", recreate_if_exists=True)

    # Insert a few documents
    documents = [
        {"name": "Alice", "age": 25, "occupation": "Architect"},
        {"name": "Bob", "age": 30, "occupation": "Builder"},
        {"name": "Charlie", "age": 35, "occupation": "Lawyer"},
        {"name": "Jie", "age": 27, "occupation": "Architect"},
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
    cases = [
        ({}, "occupation", {("Architect", 2), ("Lawyer", 1), ("Builder", 1)}),
        ({"occupation": "Architect"}, "occupation", {("Architect", 2)}),
    ]
    for where, fc, expected in cases:
        fr = collection.query_facets(where, facet_columns=[fc])
        results = set(fr[fc])
        assert results == expected
