"""
Test the Ibis backend adapter.

This test suite verifies that the Ibis backend works correctly for basic CRUD operations
and integrates properly with the LinkML Store API.
"""

import logging
from pathlib import Path

import pytest

from linkml_store.api.client import Client
from linkml_store.api.queries import Query
from tests import OUTPUT_DIR

logger = logging.getLogger(__name__)

TEMP_DB_PATH = OUTPUT_DIR / "temp_ibis.duckdb"

# Test with different Ibis backends
IBIS_SCHEMES = [
    "ibis+duckdb:///:memory:",  # In-memory DuckDB via Ibis
    f"ibis+duckdb:///{TEMP_DB_PATH}",  # File-based DuckDB via Ibis
]

PERSONS = [
    {"id": "P1", "name": "Alice", "age": 30},
    {"id": "P2", "name": "Bob", "age": 25},
    {"id": "P3", "name": "Charlie", "age": 35},
]


@pytest.fixture(autouse=True)
def cleanup_temp_db():
    """Cleanup temp files after each test."""
    yield
    if TEMP_DB_PATH.exists():
        TEMP_DB_PATH.unlink()


class TestIbisAdapter:
    """Test suite for Ibis backend adapter."""

    @pytest.mark.parametrize("handle", IBIS_SCHEMES)
    def test_basic_insert_and_find(self, handle):
        """Test basic insert and find operations."""
        try:
            # Skip if ibis not installed
            import ibis
        except ImportError:
            pytest.skip("ibis-framework not installed")

        client = Client()
        db = client.attach_database(handle, alias="test", recreate_if_exists=True)

        # Create a collection
        collection = db.create_collection("Person")

        # Insert objects
        collection.insert(PERSONS)

        # Find all
        result = collection.find()
        assert result.num_rows == 3, f"Expected 3 persons, got {result.num_rows}"

        # Find by ID
        result = collection.find({"id": "P1"})
        assert result.num_rows == 1
        assert result.rows[0]["name"] == "Alice"

        # Find by name
        result = collection.find({"name": "Bob"})
        assert result.num_rows == 1
        assert result.rows[0]["age"] == 25

        # Clean up
        db.drop(missing_ok=True)

    @pytest.mark.parametrize("handle", IBIS_SCHEMES)
    def test_query(self, handle):
        """Test query operations."""
        try:
            import ibis
        except ImportError:
            pytest.skip("ibis-framework not installed")

        client = Client()
        db = client.attach_database(handle, alias="test", recreate_if_exists=True)
        collection = db.create_collection("Person")
        collection.insert(PERSONS)

        # Query with limit - num_rows returns total count for pagination
        query = Query(limit=2)
        result = collection.query(query)
        assert len(result.rows) == 2

        # Query with where clause
        query = Query(where_clause={"name": "Alice"})
        result = collection.query(query)
        assert result.num_rows == 1
        assert result.rows[0]["age"] == 30

        # Query with sorting
        query = Query(sort_by=["age"])
        result = collection.query(query)
        assert result.rows[0]["name"] == "Bob"  # Youngest
        assert result.rows[-1]["name"] == "Charlie"  # Oldest

        # Clean up
        db.drop(missing_ok=True)

    @pytest.mark.parametrize("handle", IBIS_SCHEMES)
    def test_delete(self, handle):
        """Test delete operations."""
        try:
            import ibis
        except ImportError:
            pytest.skip("ibis-framework not installed")

        client = Client()
        db = client.attach_database(handle, alias="test", recreate_if_exists=True)
        collection = db.create_collection("Person")
        collection.insert(PERSONS)

        # Delete by where clause
        collection.delete_where({"id": "P1"})

        # Verify deletion
        result = collection.find()
        assert len(result.rows) == 2
        assert all(r["id"] != "P1" for r in result.rows)

        # Clean up
        db.drop(missing_ok=True)

    @pytest.mark.parametrize("handle", IBIS_SCHEMES)
    def test_peek(self, handle):
        """Test peek operation."""
        try:
            import ibis
        except ImportError:
            pytest.skip("ibis-framework not installed")

        client = Client()
        db = client.attach_database(handle, alias="test", recreate_if_exists=True)
        collection = db.create_collection("Person")
        collection.insert(PERSONS)

        # Peek at data - returns QueryResult
        results = collection.peek(limit=2)
        assert len(results.rows) == 2

        # Clean up
        db.drop(missing_ok=True)

    @pytest.mark.parametrize("handle", IBIS_SCHEMES)
    def test_list_collections(self, handle):
        """Test listing collections."""
        try:
            import ibis
        except ImportError:
            pytest.skip("ibis-framework not installed")

        client = Client()
        db = client.attach_database(handle, alias="test", recreate_if_exists=True)

        # Create collections and insert data (tables are created on insert)
        person_coll = db.create_collection("Person")
        person_coll.insert([{"id": "P1", "name": "Test"}])

        org_coll = db.create_collection("Organization")
        org_coll.insert([{"id": "O1", "name": "TestOrg"}])

        # List collections
        collections = db.list_collections()
        collection_names = [c.alias for c in collections]
        assert "Person" in collection_names
        assert "Organization" in collection_names

        # Clean up
        db.drop(missing_ok=True)

    def test_handle_parsing(self):
        """Test Ibis handle parsing."""
        try:
            import ibis
        except ImportError:
            pytest.skip("ibis-framework not installed")

        client = Client()

        # Test various handle formats
        test_handles = [
            ("ibis+duckdb:///:memory:", "duckdb"),
            ("ibis+sqlite:///test.db", "sqlite"),
            ("ibis+postgres://localhost/test", "postgres"),
        ]

        for handle, expected_backend in test_handles:
            db = client.attach_database(handle, alias=f"test_{expected_backend}")
            parsed = db._parse_handle(handle)
            assert parsed["backend"] == expected_backend
            db.drop(missing_ok=True)

    def test_short_form_handles(self):
        """Test short form Ibis handles."""
        try:
            import ibis
        except ImportError:
            pytest.skip("ibis-framework not installed")

        client = Client()

        # Test short forms
        db = client.attach_database("ibis://", alias="test")
        assert db.handle == "ibis+duckdb:///:memory:" or db.handle == "ibis://"

        collection = db.create_collection("TestCollection")
        collection.insert([{"id": "1", "name": "test"}])
        result = collection.find()
        assert result.num_rows == 1

        db.drop(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
