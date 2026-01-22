"""Tests for the Dremio REST API adapter.

Unit tests run without network access.
Integration tests use vcr.py to record/replay HTTP interactions.

To record new cassettes against the JGI lakehouse:
1. Set environment variables: DREMIO_USER, DREMIO_PASSWORD, CF_AUTHORIZATION
2. Delete the cassette file you want to re-record
3. Run: pytest tests/test_api/test_dremio_rest_adapter.py -v --vcr-record=new_episodes
"""

import os
from pathlib import Path

import pytest

# VCR cassettes directory
CASSETTES_DIR = Path(__file__).parent / "cassettes" / "dremio_rest"


@pytest.fixture
def vcr_config():
    """VCR configuration for recording HTTP interactions."""
    return {
        "cassette_library_dir": str(CASSETTES_DIR),
        "filter_headers": ["authorization", "cookie"],
        "filter_post_data_parameters": ["password"],
        "record_mode": "none",  # Only use existing cassettes; change to "new_episodes" to record
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }


class TestDremioRestHandleParsing:
    """Tests for Dremio REST handle/URL parsing."""

    def test_parse_simple_handle(self):
        """Test parsing a simple handle without credentials."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        db._connection_info = None
        info = db._parse_handle("dremio-rest://lakehouse.example.com")

        assert info["host"] == "lakehouse.example.com"
        assert info["port"] == 443
        assert info["username"] is None
        assert info["password"] is None
        assert info["verify_ssl"] is True

    def test_parse_handle_with_credentials(self):
        """Test parsing a handle with username and password."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        info = db._parse_handle("dremio-rest://user:pass@dremio.example.com")

        assert info["host"] == "dremio.example.com"
        assert info["port"] == 443
        assert info["username"] == "user"
        assert info["password"] == "pass"

    def test_parse_handle_with_custom_port(self):
        """Test parsing a handle with custom port."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        info = db._parse_handle("dremio-rest://localhost:9047")

        assert info["host"] == "localhost"
        assert info["port"] == 9047

    def test_parse_handle_with_schema(self):
        """Test parsing a handle with schema parameter."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        info = db._parse_handle("dremio-rest://localhost?schema=gold.study")

        assert info["default_schema"] == "gold.study"

    def test_parse_handle_with_verify_ssl_false(self):
        """Test parsing a handle with SSL verification disabled."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        info = db._parse_handle("dremio-rest://localhost?verify_ssl=false")

        assert info["verify_ssl"] is False

    def test_parse_handle_with_env_var_names(self):
        """Test parsing with custom environment variable names."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        info = db._parse_handle(
            "dremio-rest://localhost?username_env=MY_USER&password_env=MY_PASS&cf_token_env=MY_CF"
        )

        # These just change which env vars are read, not the values
        assert "username" in info
        assert "password" in info


class TestDremioRestCollection:
    """Tests for DremioRestCollection SQL building."""

    def test_build_where_conditions_simple(self):
        """Test building simple WHERE conditions."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)
        result = collection._build_where_conditions({"name": "Alice", "age": 25})

        assert '"name" = \'Alice\'' in result
        assert '"age" = 25' in result
        assert " AND " in result

    def test_build_where_conditions_operators(self):
        """Test building WHERE conditions with MongoDB-style operators."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)

        # Greater than
        result = collection._build_where_conditions({"age": {"$gt": 30}})
        assert '"age" > 30' in result

        # Greater than or equal
        result = collection._build_where_conditions({"age": {"$gte": 30}})
        assert '"age" >= 30' in result

        # Less than
        result = collection._build_where_conditions({"age": {"$lt": 30}})
        assert '"age" < 30' in result

        # Less than or equal
        result = collection._build_where_conditions({"age": {"$lte": 30}})
        assert '"age" <= 30' in result

        # Not equal
        result = collection._build_where_conditions({"status": {"$ne": "inactive"}})
        assert '"status" != \'inactive\'' in result

    def test_build_where_conditions_in_operator(self):
        """Test building WHERE conditions with IN operator."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)

        result = collection._build_where_conditions({"status": {"$in": ["active", "pending"]}})
        assert '"status" IN' in result
        assert "'active'" in result
        assert "'pending'" in result

        # NOT IN
        result = collection._build_where_conditions({"status": {"$nin": ["deleted"]}})
        assert '"status" NOT IN' in result

    def test_build_where_conditions_like(self):
        """Test building WHERE conditions with LIKE operator."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)

        result = collection._build_where_conditions({"name": {"$like": "%methane%"}})
        assert '"name" LIKE' in result
        assert "'%methane%'" in result

    def test_build_where_conditions_ilike(self):
        """Test building WHERE conditions with case-insensitive ILIKE operator."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)

        result = collection._build_where_conditions({"name": {"$ilike": "%Methane%"}})
        assert "LOWER" in result
        assert "'%Methane%'" in result

    def test_build_where_conditions_regex(self):
        """Test building WHERE conditions with regex operator."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)

        result = collection._build_where_conditions({"name": {"$regex": "^meth.*"}})
        assert "REGEXP_LIKE" in result

    def test_build_where_conditions_null(self):
        """Test building WHERE conditions with NULL."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)

        result = collection._build_where_conditions({"deleted": None})
        assert '"deleted" IS NULL' in result

        result = collection._build_where_conditions({"deleted": {"$ne": None}})
        assert '"deleted" IS NOT NULL' in result

    def test_sql_value_escaping(self):
        """Test SQL value escaping."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)

        # String with quotes - should be escaped
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

    def test_build_select_sql(self):
        """Test building SELECT SQL statements."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

        collection = DremioRestCollection.__new__(DremioRestCollection)
        collection._parent = None

        # Mock _get_table_path
        collection._get_table_path = lambda: '"schema"."table"'

        sql = collection._build_select_sql(
            select_cols=["id", "name"],
            where_clause={"active": True},
            sort_by=["name"],
            limit=10,
            offset=5,
        )

        assert 'SELECT "id", "name"' in sql
        assert 'FROM "schema"."table"' in sql
        assert '"active" = TRUE' in sql
        assert 'ORDER BY "name"' in sql
        assert "LIMIT 10" in sql
        assert "OFFSET 5" in sql


class TestDremioRestTablePath:
    """Tests for table path handling."""

    def test_get_table_path_simple(self):
        """Test getting table path for simple table name."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        db._connection_info = {"default_schema": None, "path": None}

        result = db._get_table_path("mytable")
        assert result == '"mytable"'

    def test_get_table_path_with_default_schema(self):
        """Test getting table path with default schema."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        db._connection_info = {"default_schema": "gold.tables", "path": None}

        result = db._get_table_path("study")
        assert result == "gold.tables.study"

    def test_get_table_path_already_qualified(self):
        """Test getting table path when already fully qualified."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        db = DremioRestDatabase.__new__(DremioRestDatabase)
        db._connection_info = {"default_schema": "other", "path": None}

        # Already has dots - should be returned as-is
        result = db._get_table_path('"gold-db".gold.study')
        assert result == '"gold-db".gold.study'


class TestClientIntegration:
    """Test integration with the linkml-store Client."""

    def test_dremio_rest_scheme_in_handle_map(self):
        """Test that dremio-rest scheme is registered in HANDLE_MAP."""
        from linkml_store.api.client import HANDLE_MAP

        assert "dremio-rest" in HANDLE_MAP
        assert "DremioRestDatabase" in HANDLE_MAP["dremio-rest"]

    def test_attach_dremio_rest_database(self):
        """Test attaching a Dremio REST database through the Client."""
        from linkml_store import Client

        client = Client()
        # This won't actually connect until we try to query
        db = client.attach_database("dremio-rest://example.com", alias="test_dremio")

        assert db is not None
        assert "test_dremio" in client.databases
        assert db.handle == "dremio-rest://example.com"


# Integration tests using VCR cassettes
@pytest.mark.vcr()
class TestDremioRestIntegration:
    """Integration tests that use recorded HTTP interactions.

    To record new cassettes:
    1. Set DREMIO_USER, DREMIO_PASSWORD, CF_AUTHORIZATION env vars
    2. Run with --vcr-record=new_episodes
    """

    @pytest.fixture
    def jgi_database(self):
        """Fixture to provide a DremioRestDatabase for JGI lakehouse."""
        from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

        # Skip if no cassette and no credentials
        if not CASSETTES_DIR.exists():
            CASSETTES_DIR.mkdir(parents=True, exist_ok=True)

        db = DremioRestDatabase(handle="dremio-rest://lakehouse.jgi.lbl.gov")
        yield db
        db.close()

    @pytest.mark.skip(reason="Requires recorded cassette or live credentials")
    def test_authentication(self, jgi_database):
        """Test authentication to Dremio."""
        token = jgi_database._authenticate()
        assert token is not None
        assert token.startswith("_dremio")

    @pytest.mark.skip(reason="Requires recorded cassette or live credentials")
    def test_query_study_table(self, jgi_database):
        """Test querying the GOLD study table."""
        collection = jgi_database.get_collection('"gold-db-2 postgresql".gold.study')
        result = collection.find({"is_public": "Yes"}, limit=5)

        assert result is not None
        assert len(result.rows) <= 5
        assert "gold_id" in result.rows[0] or "study_id" in result.rows[0]

    @pytest.mark.skip(reason="Requires recorded cassette or live credentials")
    def test_query_with_ilike(self, jgi_database):
        """Test case-insensitive search with $ilike."""
        collection = jgi_database.get_collection('"gold-db-2 postgresql".gold.study')
        result = collection.find({"study_name": {"$ilike": "%methane%"}}, limit=5)

        assert result is not None
        for row in result.rows:
            assert "methane" in row.get("study_name", "").lower()
