# test_mongodb_adapter.py

import pytest
from linkml_store.api.stores.mongodb.mongodb_database import MongoDBDatabase
from linkml_store.api.stores.mongodb.mongodb_collection import MongoDBCollection
from pymongo import MongoClient

@pytest.fixture(scope="module")
def mongodb_client():
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)  # 2s timeout
        client.admin.command("ping")  # Check MongoDB connectivity
        yield client
        client.close()
    except Exception:
        pytest.skip("Skipping tests: MongoDB is not available.")


@pytest.fixture(scope="function")
def mongodb_database(mongodb_client):
    db = mongodb_client["test_db"]
    yield db
    db.drop_collection("test_collection")


@pytest.fixture(scope="function")
def mongodb_collection(mongodb_client):
    """Fixture to provide a MongoDB test collection, ensuring database and collection creation if necessary."""
    if mongodb_client is None:
        pytest.skip("Skipping tests: MongoDB client is not available.")

    db_name = "test_db"
    collection_name = "test_collection"

    # Ensure database exists by creating a temporary collection
    existing_dbs = mongodb_client.list_database_names()
    if db_name not in existing_dbs:
        temp_db = mongodb_client[db_name]
        temp_db.create_collection("temp_init_collection")
        temp_db.drop_collection("temp_init_collection")  # Clean up temp collection

    # Now attach to the database and collection
    db = mongodb_client[db_name]

    # Ensure the test collection exists
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    # Correctly initialize MongoDBCollection by passing collection name
    collection = MongoDBCollection(name=collection_name, parent=db)

    yield collection

    # Cleanup: Drop the test collection after each test
    db.drop_collection(collection_name)

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
    """
    Test inserting and querying documents in MongoDB.
    """
    # Create a MongoDBDatabase instance
    db = MongoDBDatabase(handle=handle)

    # Create a collection
    collection = db.create_collection("test_collection", recreate_if_exists=True)

    # Insert multiple documents
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
            "relationships": [{"person": "Alice", "relation": "friend"}],
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
