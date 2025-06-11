# test_mongodb_client.py

import pytest
from pymongo import MongoClient

from linkml_store.api.client import Client
from linkml_store.api.stores.mongodb.mongodb_database import MongoDBDatabase


@pytest.fixture(scope="module")
def mongodb_client():
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)  # 2s timeout
        client.admin.command("ping")  # Check MongoDB connectivity
        yield client
        client.close()
    except Exception:
        pytest.skip("Skipping tests: MongoDB is not available.")


def test_attach_mongodb_client(mongodb_client):
    """
    Test attaching an existing MongoClient instance directly.
    """
    # Setup
        
    # Setup
    db_name = "test_direct_client_db"
    test_collection = "test_collection"
    
    # Create a Client
    client = Client()
    
    # Attach MongoDB client directly
    db = client.attach_mongodb_client(
        mongo_client=mongodb_client,
        db_name=db_name,
        alias="direct_mongo"
    )
    
    # Verify the database was attached
    assert "direct_mongo" in client.databases
    assert db.alias == "direct_mongo"
    assert db._db_name == db_name
    assert isinstance(db, MongoDBDatabase)
    
    # Test basic operations with the database
    collection = db.create_collection(test_collection, recreate_if_exists=True)
    test_data = [
        {"id": "1", "name": "Test Item 1", "value": 100},
        {"id": "2", "name": "Test Item 2", "value": 200}
    ]
    
    # Insert data
    collection.insert(test_data)
    
    # Query data
    result = collection.find()
    assert result.num_rows == len(test_data)
    
    # Cleanup
    mongodb_client.drop_database(db_name)


def test_attach_native_connection_with_mongodb(mongodb_client):
    """
    Test the generic attach_native_connection method with a MongoDB client.
    """
    # Skip if no MongoDB available
    if mongodb_client is None:
        pytest.skip("Skipping test: MongoDB client is not available.")
        
    # Setup
    db_name = "test_native_connection_db"
    test_collection = "test_collection"
    
    # Create a Client
    client = Client()
    
    # Attach MongoDB client via the generic method
    db = client.attach_native_connection(
        connection_type="mongodb",
        connection_object=mongodb_client,
        db_name=db_name,
        alias="generic_mongo"
    )
    
    # Verify the database was attached
    assert "generic_mongo" in client.databases
    assert db.alias == "generic_mongo"
    assert db._db_name == db_name
    assert isinstance(db, MongoDBDatabase)
    
    # Test basic operations with the database
    collection = db.create_collection(test_collection, recreate_if_exists=True)
    test_data = [
        {"id": "1", "name": "Test Item 1", "value": 100},
        {"id": "2", "name": "Test Item 2", "value": 200}
    ]
    
    # Insert data
    collection.insert(test_data)
    
    # Query data
    result = collection.find()
    assert result.num_rows == len(test_data)
    
    # Cleanup
    mongodb_client.drop_database(db_name)