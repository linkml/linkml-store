"""Tests for the Dremio adapter.

These tests require a running Dremio instance. They will be skipped
if Dremio is not available.

To run these tests:
1. Start a Dremio instance (e.g., using Docker):
   docker run -p 9047:9047 -p 31010:31010 -p 32010:32010 dremio/dremio-oss

2. Set up a test data source in Dremio

3. Run the tests:
   pytest tests/test_api/test_dremio_adapter.py -v
"""

import pytest

# Check if pyarrow is available
try:
    import pyarrow
    import pyarrow.flight as flight

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


# Skip all tests if pyarrow is not installed
pytestmark = pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")


def dremio_available(host="localhost", port=32010):
    """Check if Dremio is available at the specified location."""
    if not PYARROW_AVAILABLE:
        return False
    try:
        client = flight.FlightClient(f"grpc://{host}:{port}")
        # Try to list flights to check connectivity
        list(client.list_flights())
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def dremio_client():
    """Fixture to check Dremio availability and provide connection info."""
    if not dremio_available():
        pytest.skip("Skipping tests: Dremio is not available at localhost:32010")

    yield {"host": "localhost", "port": 32010}


@pytest.fixture(scope="function")
def dremio_database(dremio_client):
    """Fixture to provide a DremioDatabase instance."""
    from linkml_store.api.stores.dremio.dremio_database import DremioDatabase

    handle = f"dremio://{dremio_client['host']}:{dremio_client['port']}?useEncryption=false"
    db = DremioDatabase(handle=handle)
    yield db
    db.close()


class TestDremioHandleParsing:
    """Tests for Dremio handle/URL parsing."""

    def test_parse_simple_handle(self):
        """Test parsing a simple handle without credentials."""
        from linkml_store.api.stores.dremio.dremio_database import DremioDatabase

        db = DremioDatabase.__new__(DremioDatabase)
        info = db._parse_handle("dremio://localhost:32010")

        assert info["host"] == "localhost"
        assert info["port"] == 32010
        assert info["username"] is None
        assert info["password"] is None
        assert info["use_encryption"] is True  # Default

    def test_parse_handle_with_credentials(self):
        """Test parsing a handle with username and password."""
        from linkml_store.api.stores.dremio.dremio_database import DremioDatabase

        db = DremioDatabase.__new__(DremioDatabase)
        info = db._parse_handle("dremio://user:pass@dremio.example.com:32010")

        assert info["host"] == "dremio.example.com"
        assert info["port"] == 32010
        assert info["username"] == "user"
        assert info["password"] == "pass"

    def test_parse_handle_with_path(self):
        """Test parsing a handle with a path (schema)."""
        from linkml_store.api.stores.dremio.dremio_database import DremioDatabase

        db = DremioDatabase.__new__(DremioDatabase)
        info = db._parse_handle("dremio://localhost:32010/Samples")

        assert info["path"] == "Samples"

    def test_parse_handle_with_params(self):
        """Test parsing a handle with query parameters."""
        from linkml_store.api.stores.dremio.dremio_database import DremioDatabase

        db = DremioDatabase.__new__(DremioDatabase)
        info = db._parse_handle(
            "dremio://localhost:32010?useEncryption=false&disableCertificateVerification=true"
        )

        assert info["use_encryption"] is False
        assert info["disable_cert_verify"] is True

    def test_default_port(self):
        """Test that default port is used when not specified."""
        from linkml_store.api.stores.dremio.dremio_database import DremioDatabase

        db = DremioDatabase.__new__(DremioDatabase)
        info = db._parse_handle("dremio://localhost")

        assert info["port"] == 32010


class TestDremioMappings:
    """Tests for type mappings."""

    def test_linkml_to_arrow_mapping(self):
        """Test LinkML to Arrow type conversion."""
        from linkml_store.api.stores.dremio.mappings import get_arrow_type

        import pyarrow as pa

        assert get_arrow_type("string") == pa.string()
        assert get_arrow_type("integer") == pa.int64()
        assert get_arrow_type("float") == pa.float64()
        assert get_arrow_type("boolean") == pa.bool_()
        assert get_arrow_type("date") == pa.date32()

    def test_arrow_to_linkml_mapping(self):
        """Test Arrow to LinkML type conversion."""
        from linkml_store.api.stores.dremio.mappings import get_linkml_type_from_arrow

        import pyarrow as pa

        assert get_linkml_type_from_arrow(pa.string()) == "string"
        assert get_linkml_type_from_arrow(pa.int64()) == "integer"
        assert get_linkml_type_from_arrow(pa.float64()) == "float"
        assert get_linkml_type_from_arrow(pa.bool_()) == "boolean"


class TestDremioCollection:
    """Tests for DremioCollection SQL building."""

    def test_build_where_conditions_simple(self):
        """Test building simple WHERE conditions."""
        from linkml_store.api.stores.dremio.dremio_collection import DremioCollection

        collection = DremioCollection.__new__(DremioCollection)
        result = collection._build_where_conditions({"name": "Alice", "age": 25})

        assert '"name" = \'Alice\'' in result
        assert '"age" = 25' in result
        assert " AND " in result

    def test_build_where_conditions_operators(self):
        """Test building WHERE conditions with operators."""
        from linkml_store.api.stores.dremio.dremio_collection import DremioCollection

        collection = DremioCollection.__new__(DremioCollection)

        # Greater than
        result = collection._build_where_conditions({"age": {"$gt": 30}})
        assert '"age" > 30' in result

        # Greater than or equal
        result = collection._build_where_conditions({"age": {"$gte": 30}})
        assert '"age" >= 30' in result

        # Less than
        result = collection._build_where_conditions({"age": {"$lt": 30}})
        assert '"age" < 30' in result

        # Not equal
        result = collection._build_where_conditions({"status": {"$ne": "inactive"}})
        assert '"status" != \'inactive\'' in result

        # IN operator
        result = collection._build_where_conditions({"status": {"$in": ["active", "pending"]}})
        assert '"status" IN' in result
        assert "'active'" in result
        assert "'pending'" in result

    def test_build_where_conditions_null(self):
        """Test building WHERE conditions with NULL."""
        from linkml_store.api.stores.dremio.dremio_collection import DremioCollection

        collection = DremioCollection.__new__(DremioCollection)

        result = collection._build_where_conditions({"deleted": None})
        assert '"deleted" IS NULL' in result

        result = collection._build_where_conditions({"deleted": {"$ne": None}})
        assert '"deleted" IS NOT NULL' in result

    def test_sql_value_escaping(self):
        """Test SQL value escaping."""
        from linkml_store.api.stores.dremio.dremio_collection import DremioCollection

        collection = DremioCollection.__new__(DremioCollection)

        # String with quotes
        result = collection._sql_value("O'Brien")
        assert result == "'O''Brien'"

        # Boolean
        assert collection._sql_value(True) == "TRUE"
        assert collection._sql_value(False) == "FALSE"

        # Numbers
        assert collection._sql_value(42) == "42"
        assert collection._sql_value(3.14) == "3.14"

        # NULL
        assert collection._sql_value(None) == "NULL"


@pytest.mark.integration
class TestDremioIntegration:
    """Integration tests requiring a running Dremio instance."""

    def test_connection(self, dremio_database):
        """Test basic connection to Dremio."""
        # Just accessing the flight_client should work if Dremio is up
        client = dremio_database.flight_client
        assert client is not None

    def test_init_collections(self, dremio_database):
        """Test discovering collections from Dremio."""
        dremio_database.init_collections()
        # Just verify it doesn't error - actual collections depend on Dremio setup

    def test_induce_schema_view(self, dremio_database):
        """Test inducing schema from Dremio tables."""
        schema_view = dremio_database.induce_schema_view()
        assert schema_view is not None


@pytest.mark.integration
class TestDremioQueries:
    """Query tests requiring Dremio with sample data."""

    @pytest.fixture
    def sample_collection(self, dremio_database):
        """Get a sample collection if available."""
        # Try to find a collection with data
        dremio_database.init_collections()
        collections = dremio_database.list_collections()

        if not collections:
            pytest.skip("No collections available in Dremio for testing")

        return collections[0]

    def test_find_all(self, sample_collection):
        """Test finding all records."""
        result = sample_collection.find({}, limit=10)
        assert result is not None
        assert hasattr(result, "rows")
        assert hasattr(result, "num_rows")

    def test_find_with_limit(self, sample_collection):
        """Test finding with limit."""
        result = sample_collection.find({}, limit=5)
        assert len(result.rows) <= 5

    def test_find_with_where(self, sample_collection):
        """Test finding with WHERE clause."""
        # Get a sample row to build a query
        all_rows = sample_collection.find({}, limit=1)
        if not all_rows.rows:
            pytest.skip("No data in collection")

        # Get first column and value for filtering
        first_row = all_rows.rows[0]
        if not first_row:
            pytest.skip("Empty row")

        first_col = list(first_row.keys())[0]
        first_val = first_row[first_col]

        # Query with that filter
        result = sample_collection.find({first_col: first_val})
        assert result.num_rows >= 1


class TestClientIntegration:
    """Test integration with the linkml-store Client."""

    def test_attach_dremio_database(self, dremio_client):
        """Test attaching a Dremio database through the Client."""
        from linkml_store.api.client import Client

        client = Client()
        handle = f"dremio://{dremio_client['host']}:{dremio_client['port']}?useEncryption=false"

        db = client.attach_database(handle, alias="test_dremio")
        assert db is not None
        assert "test_dremio" in client.databases

        db.close()

    def test_dremio_scheme_in_handle_map(self):
        """Test that dremio scheme is registered in HANDLE_MAP."""
        from linkml_store.api.client import HANDLE_MAP

        assert "dremio" in HANDLE_MAP
        assert "DremioDatabase" in HANDLE_MAP["dremio"]
