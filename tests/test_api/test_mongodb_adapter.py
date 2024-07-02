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


@pytest.mark.integration
def test_insert_and_query(mongodb_database):
    # Create a MongoDBDatabase instance
    db = MongoDBDatabase(handle="mongodb://localhost:27017/test_db")

    # Create a collection
    collection = db.create_collection("test_collection", recreate_if_exists=True)

    # Insert a few documents
    documents = [
        {"name": "Alice", "age": 25, "occupation": "Architect", "foods": ["apple", "banana"],
         "relationships": [
                {"person": "Bob", "relation": "friend"},
                {"person": "Charlie", "relation": "brother"},
            ],
         "meta": {"date": "2021-01-01", "notes": "likes fruit"}},
        {"name": "Bob", "age": 30, "occupation": "Builder", "foods": ["carrot", "date"],
         "relationships": [
                {"person": "Alice", "relation": "friend"},
            ],
         "meta": {"date": "2021-01-01"}},
        {"name": "Charlie", "age": 35, "occupation": "Lawyer", "foods": ["eggplant", "fig", "banana"],
         "meta": {"date": "2021-01-03", "notes": "likes fruit", "curator": "Ziggy"}},
        {"name": "Jie", "age": 27, "occupation": "Architect", "foods": ["grape", "honey", "apple"],
         "meta": {"date": "2021-01-03"}},
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
        ({}, "foods", {('fig', 1), ('banana', 2), ('eggplant', 1),
                       ('apple', 2), ('carrot', 1), ('date', 1), ('grape', 1), ('honey', 1)}),
        ({}, "relationships.relation", {('friend', 2), ('brother', 1)}),
        ({}, "meta.date", {("2021-01-01", 2), ("2021-01-03", 2)}),
        ({}, "meta", {("curator: Ziggy\ndate: '2021-01-03'\nnotes: likes fruit\n", 1),
                      ("date: '2021-01-01'\nnotes: likes fruit\n", 1),
                      ("date: '2021-01-01'\n", 1), ("date: '2021-01-03'\n", 1)}),
        ({}, ("occupation", "foods"), {(('Architect', 'apple'), 2),
             (('Architect', 'banana'), 1),
             (('Architect', 'grape'), 1),
             (('Architect', 'honey'), 1),
             (('Builder', 'carrot'), 1),
             (('Builder', 'date'), 1),
             (('Lawyer', 'banana'), 1),
             (('Lawyer', 'eggplant'), 1),
             (('Lawyer', 'fig'), 1)}),

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
