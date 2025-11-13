"""Ibis database adapter for linkml-store."""

import logging
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import pandas as pd
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.ibis.ibis_collection import IbisCollection
from linkml_store.utils.format_utils import Format

logger = logging.getLogger(__name__)

TYPE_MAP = {
    "string": "string",
    "integer": "int64",
    "int64": "int64",
    "boolean": "boolean",
    "date": "date",
    "datetime": "timestamp",
    "float": "float64",
    "double": "float64",
}

MEMORY_HANDLE = "ibis+duckdb:///:memory:"


class IbisDatabase(Database):
    """
    An adapter for databases using Ibis as an abstraction layer.

    Ibis provides a unified interface across multiple database backends including:
    - DuckDB (default)
    - PostgreSQL
    - SQLite
    - BigQuery
    - Snowflake
    - And many more

    Connection strings should be in the format:
    - ibis+duckdb:///:memory: (in-memory DuckDB)
    - ibis+duckdb:///path/to/db.duckdb
    - ibis+postgres://user:pass@host:port/dbname
    - ibis+sqlite:///path/to/db.sqlite
    - ibis+bigquery://project/dataset

    For convenience, you can also use short forms:
    - ibis:// defaults to ibis+duckdb:///:memory:
    - ibis:///path.duckdb uses DuckDB
    """

    _connection = None
    collection_class = IbisCollection

    def __init__(self, handle: Optional[str] = None, recreate_if_exists: bool = False, **kwargs):
        if handle is None:
            handle = MEMORY_HANDLE
        if recreate_if_exists and handle != MEMORY_HANDLE:
            # For file-based databases, delete the file if it exists
            parsed = self._parse_handle(handle)
            if parsed.get("path") and Path(parsed["path"]).exists():
                Path(parsed["path"]).unlink()
        super().__init__(handle=handle, **kwargs)

    def _parse_handle(self, handle: str) -> dict:
        """
        Parse an Ibis handle into components.

        Returns a dict with keys:
        - backend: The Ibis backend name (duckdb, postgres, etc.)
        - connection_string: The connection string for the backend
        - path: File path for file-based backends
        """
        if not handle:
            handle = MEMORY_HANDLE

        # Handle short forms
        if handle == "ibis://" or handle == "ibis":
            handle = MEMORY_HANDLE
        elif handle.startswith("ibis:///") and not handle.startswith("ibis+"):
            # Assume DuckDB for file paths
            path = handle.replace("ibis:///", "")
            handle = f"ibis+duckdb:///{path}"

        # Parse the handle
        if handle.startswith("ibis+"):
            # Format: ibis+backend://rest
            rest = handle[5:]  # Remove 'ibis+'
            parsed = urlparse(rest)
            backend = parsed.scheme

            # Reconstruct connection string for the specific backend
            if backend == "duckdb":
                if parsed.netloc == "" and parsed.path == "/:memory:":
                    connection_string = ":memory:"
                    path = None
                else:
                    path = parsed.path.lstrip("/") if parsed.path else None
                    connection_string = path or ":memory:"
            elif backend == "sqlite":
                path = parsed.path.lstrip("/") if parsed.path else None
                connection_string = f"{backend}:///{path}"
            elif backend in ["postgres", "postgresql"]:
                # postgres://user:pass@host:port/dbname
                connection_string = f"{backend}://{parsed.netloc}{parsed.path}"
                path = None
            elif backend == "bigquery":
                # bigquery://project/dataset
                connection_string = f"{parsed.netloc}{parsed.path}"
                path = None
            else:
                # Generic backend
                connection_string = rest.replace(f"{backend}://", "")
                path = None

            return {
                "backend": backend,
                "connection_string": connection_string,
                "path": path,
            }
        else:
            raise ValueError(
                f"Invalid Ibis handle: {handle}. "
                f"Expected format: ibis+backend://connection_string "
                f"(e.g., ibis+duckdb:///:memory:, ibis+postgres://host/db)"
            )

    @property
    def connection(self):
        """Get or create the Ibis connection."""
        if not self._connection:
            try:
                import ibis
            except ImportError:
                raise ImportError(
                    "Ibis is not installed. Install it with: pip install 'linkml-store[ibis]' "
                    "or pip install 'ibis-framework[duckdb]'"
                )

            parsed = self._parse_handle(self.handle)
            backend = parsed["backend"]
            connection_string = parsed["connection_string"]

            logger.info(f"Connecting to Ibis backend: {backend} with connection: {connection_string}")

            try:
                if backend == "duckdb":
                    self._connection = ibis.duckdb.connect(connection_string)
                elif backend == "sqlite":
                    self._connection = ibis.sqlite.connect(connection_string)
                elif backend in ["postgres", "postgresql"]:
                    self._connection = ibis.postgres.connect(connection_string)
                elif backend == "bigquery":
                    self._connection = ibis.bigquery.connect(connection_string)
                else:
                    # Try generic connect
                    self._connection = ibis.connect(f"{backend}://{connection_string}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Ibis backend {backend}: {e}")

        return self._connection

    def commit(self, **kwargs):
        """Commit changes (no-op for most Ibis backends)."""
        # Most Ibis backends auto-commit, but we keep this for interface compatibility
        pass

    def close(self, **kwargs):
        """Close the Ibis connection."""
        if self._connection:
            # Ibis connections may not have an explicit close method in all backends
            # but we set to None to allow garbage collection
            self._connection = None

    def drop(self, missing_ok=True, **kwargs):
        """Drop the database."""
        self.close()
        if self.handle == MEMORY_HANDLE:
            return

        parsed = self._parse_handle(self.handle)
        path = parsed.get("path")
        if path:
            path_obj = Path(path)
            if path_obj.exists():
                path_obj.unlink()
            elif not missing_ok:
                raise FileNotFoundError(f"Database file not found: {path}")

    def _table_exists(self, table: str) -> bool:
        """Check if a table exists in the database."""
        try:
            return table in self.connection.list_tables()
        except Exception as e:
            logger.warning(f"Error checking if table {table} exists: {e}")
            return False

    def list_collections(self) -> List[str]:
        """List all collections (tables) in the database."""
        try:
            return self.connection.list_tables()
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []

    def query(self, query: Union[str, Query], **kwargs) -> QueryResult:
        """
        Execute a query against the database.

        For Ibis, we support both:
        - SQL strings (executed directly)
        - Query objects (converted to Ibis operations)
        """
        if isinstance(query, str):
            # Direct SQL query
            try:
                result = self.connection.sql(query)
                df = result.to_pandas()
                return QueryResult(
                    num_rows=len(df),
                    rows=df.to_dict("records"),
                    rows_dataframe=df,
                )
            except Exception as e:
                logger.error(f"Error executing SQL query: {e}")
                raise
        else:
            # Delegate to collection
            collection_name = query.from_table
            if not collection_name:
                raise ValueError("Query must specify a from_table")
            collection = self.get_collection(collection_name)
            return collection.query(query, **kwargs)

    def induce_schema_view(self) -> SchemaView:
        """
        Induce a LinkML schema from the database structure.

        For Ibis, we introspect the database schema and convert it to LinkML.
        """
        sb = SchemaBuilder()
        tables = self.list_collections()

        for table_name in tables:
            try:
                table = self.connection.table(table_name)
                schema = table.schema()

                # Create a class for this table
                class_def = ClassDefinition(name=table_name, description=f"Table: {table_name}")

                # Add attributes from columns
                for col_name, col_type in schema.items():
                    ibis_type = str(col_type)
                    # Map Ibis types to LinkML types
                    linkml_type = self._map_ibis_type_to_linkml(ibis_type)

                    slot_def = SlotDefinition(name=col_name, range=linkml_type)
                    sb.add_slot(slot_def)
                    class_def.attributes[col_name] = slot_def

                sb.add_class(class_def)
            except Exception as e:
                logger.warning(f"Error introspecting table {table_name}: {e}")

        schema = sb.schema
        return SchemaView(schema)

    def _map_ibis_type_to_linkml(self, ibis_type: str) -> str:
        """Map an Ibis type string to a LinkML type."""
        ibis_type_lower = ibis_type.lower()

        if "int" in ibis_type_lower:
            return "integer"
        elif "float" in ibis_type_lower or "double" in ibis_type_lower or "decimal" in ibis_type_lower:
            return "float"
        elif "bool" in ibis_type_lower:
            return "boolean"
        elif "date" in ibis_type_lower and "time" not in ibis_type_lower:
            return "date"
        elif "timestamp" in ibis_type_lower or "datetime" in ibis_type_lower:
            return "datetime"
        elif "string" in ibis_type_lower or "varchar" in ibis_type_lower or "text" in ibis_type_lower:
            return "string"
        else:
            return "string"  # Default to string for unknown types
