"""Dremio database adapter using Arrow Flight SQL.

This module provides a Database implementation for connecting to Dremio
data lakehouse using the Arrow Flight SQL protocol for high-performance
data access.
"""

import logging
import re
from typing import List, Optional, Union
from urllib.parse import parse_qs, urlparse

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.dremio.dremio_collection import DremioCollection
from linkml_store.api.stores.dremio.mappings import get_linkml_type_from_arrow
from linkml_store.utils.format_utils import Format

logger = logging.getLogger(__name__)


class DremioDatabase(Database):
    """
    An adapter for Dremio data lakehouse using Arrow Flight SQL.

    This adapter connects to Dremio using the Arrow Flight SQL protocol,
    which provides high-performance data transfer using Apache Arrow.

    Handle format:
        dremio://[username:password@]host[:port][/path][?params]

    Examples:
        - dremio://localhost:32010
        - dremio://user:pass@localhost:32010
        - dremio://localhost:32010?useEncryption=false
        - dremio://user:pass@dremio.example.com:32010/Samples

    Parameters (query string):
        - useEncryption: Whether to use TLS (default: true)
        - disableCertificateVerification: Skip cert verification (default: false)
        - schema: Default schema/space to use

    Note:
        Requires pyarrow with Flight SQL support. Install with:
        pip install pyarrow
    """

    _flight_client = None
    _connection_info: dict = None
    collection_class = DremioCollection

    def __init__(
        self,
        handle: Optional[str] = None,
        recreate_if_exists: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """Initialize a Dremio database connection.

        Args:
            handle: Connection string in format dremio://host:port
            recreate_if_exists: Not applicable for Dremio (ignored)
            username: Optional username (can also be in handle)
            password: Optional password (can also be in handle)
            **kwargs: Additional arguments passed to parent
        """
        if handle is None:
            handle = "dremio://localhost:32010"

        self._connection_info = self._parse_handle(handle)

        # Override with explicit credentials if provided
        if username:
            self._connection_info["username"] = username
        if password:
            self._connection_info["password"] = password

        super().__init__(handle=handle, **kwargs)

    def _parse_handle(self, handle: str) -> dict:
        """Parse a Dremio connection handle.

        Args:
            handle: Connection string like dremio://user:pass@host:port/path?params

        Returns:
            Dictionary with connection parameters.
        """
        # Ensure scheme is present
        if not handle.startswith("dremio://"):
            handle = f"dremio://{handle}"

        parsed = urlparse(handle)

        # Parse query parameters
        params = parse_qs(parsed.query)

        # Extract single values from query params
        use_encryption = params.get("useEncryption", ["true"])[0].lower() == "true"
        disable_cert_verify = params.get("disableCertificateVerification", ["false"])[0].lower() == "true"
        default_schema = params.get("schema", [None])[0]

        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 32010,
            "username": parsed.username,
            "password": parsed.password,
            "path": parsed.path.lstrip("/") if parsed.path else None,
            "use_encryption": use_encryption,
            "disable_cert_verify": disable_cert_verify,
            "default_schema": default_schema,
        }

    @property
    def flight_client(self):
        """Get or create the Arrow Flight SQL client.

        Returns:
            FlightSqlClient connected to Dremio.

        Raises:
            ImportError: If pyarrow is not installed.
            ConnectionError: If connection to Dremio fails.
        """
        if self._flight_client is None:
            try:
                import pyarrow.flight as flight
            except ImportError as e:
                raise ImportError(
                    "pyarrow with Flight support is required for Dremio adapter. "
                    "Install with: pip install pyarrow"
                ) from e

            info = self._connection_info
            host = info["host"]
            port = info["port"]

            # Build location
            if info["use_encryption"]:
                location = f"grpc+tls://{host}:{port}"
            else:
                location = f"grpc://{host}:{port}"

            logger.info(f"Connecting to Dremio at {location}")

            # Build connection options
            client_options = []

            if info["disable_cert_verify"]:
                client_options.append(("disable_server_verification", "true"))

            try:
                client = flight.FlightClient(location)

                # Authenticate if credentials provided
                if info["username"] and info["password"]:
                    # Get auth token using basic auth
                    bearer_token = self._authenticate(client, info["username"], info["password"])
                    # Store token for subsequent requests
                    self._bearer_token = bearer_token
                else:
                    self._bearer_token = None

                self._flight_client = client

            except Exception as e:
                raise ConnectionError(f"Failed to connect to Dremio at {location}: {e}") from e

        return self._flight_client

    def _authenticate(self, client, username: str, password: str) -> bytes:
        """Authenticate with Dremio and get bearer token.

        Args:
            client: Flight client
            username: Dremio username
            password: Dremio password

        Returns:
            Bearer token for subsequent requests.
        """
        import pyarrow.flight as flight

        # Use basic authentication
        auth_handler = flight.BasicAuth(username, password)
        token_pair = client.authenticate_basic_token(username, password)
        return token_pair[1]  # Return the token value

    def _get_call_options(self):
        """Get Flight call options with authentication headers.

        Returns:
            FlightCallOptions with bearer token if authenticated.
        """
        import pyarrow.flight as flight

        if hasattr(self, "_bearer_token") and self._bearer_token:
            return flight.FlightCallOptions(headers=[(b"authorization", self._bearer_token)])
        return flight.FlightCallOptions()

    def _execute_query(self, sql: str) -> "pyarrow.Table":
        """Execute a SQL query and return results as Arrow Table.

        Args:
            sql: SQL query string.

        Returns:
            PyArrow Table with query results.
        """
        import pyarrow.flight as flight

        logger.debug(f"Executing SQL: {sql}")

        client = self.flight_client
        options = self._get_call_options()

        # Create Flight SQL command
        flight_desc = flight.FlightDescriptor.for_command(sql.encode("utf-8"))

        # Get flight info
        try:
            # For Dremio, we use the execute method directly
            # Prepare the SQL statement
            info = client.get_flight_info(flight_desc, options)

            # Get the data from all endpoints
            tables = []
            for endpoint in info.endpoints:
                reader = client.do_get(endpoint.ticket, options)
                tables.append(reader.read_all())

            if not tables:
                import pyarrow as pa

                return pa.table({})

            # Concatenate all tables
            import pyarrow as pa

            return pa.concat_tables(tables)

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def _execute_update(self, sql: str) -> int:
        """Execute a SQL update/insert/delete statement.

        Args:
            sql: SQL statement.

        Returns:
            Number of affected rows (-1 if unknown).
        """
        import pyarrow.flight as flight

        logger.debug(f"Executing update: {sql}")

        client = self.flight_client
        options = self._get_call_options()

        try:
            # For DML statements, use do_action
            action = flight.Action("query", sql.encode("utf-8"))
            results = list(client.do_action(action, options))
            # Try to parse affected rows from result
            if results:
                try:
                    return int(results[0].body.to_pybytes().decode("utf-8"))
                except (ValueError, AttributeError):
                    pass
            return -1
        except Exception as e:
            logger.warning(f"Update execution failed, trying alternative method: {e}")
            # Some Dremio versions may not support do_action for DML
            # Fall back to regular query
            try:
                self._execute_query(sql)
                return -1
            except Exception as e2:
                logger.error(f"Update failed: {e2}")
                raise

    def commit(self, **kwargs):
        """Commit pending changes.

        Note: Dremio auto-commits, this is a no-op.
        """
        pass

    def close(self, **kwargs):
        """Close the Dremio connection."""
        if self._flight_client:
            self._flight_client.close()
            self._flight_client = None

    def drop(self, missing_ok=True, **kwargs):
        """Drop the database.

        Note: This is not supported for Dremio as it's typically a read/query layer.
        Individual tables can be dropped if you have permissions.
        """
        self.close()
        logger.warning("Dremio does not support dropping databases through this adapter")

    def query(self, query: Query, **kwargs) -> QueryResult:
        """Execute a query against Dremio.

        Args:
            query: Query object specifying the query parameters.
            **kwargs: Additional arguments.

        Returns:
            QueryResult with matching rows.
        """
        from_table = query.from_table
        if not from_table:
            raise ValueError("Query must specify from_table")

        # Check if collection exists
        collection = self.get_collection(from_table, create_if_not_exists=False)
        if collection:
            return collection.query(query, **kwargs)
        else:
            return QueryResult(query=query, num_rows=0, rows=[])

    def _get_table_path(self, table_name: str) -> str:
        """Get the full table path including schema if configured.

        Args:
            table_name: Table name.

        Returns:
            Full table path.
        """
        default_schema = self._connection_info.get("default_schema")
        path = self._connection_info.get("path")

        if "." in table_name:
            # Already qualified
            return table_name

        if default_schema:
            return f'"{default_schema}"."{table_name}"'
        elif path:
            return f'"{path}"."{table_name}"'
        else:
            return f'"{table_name}"'

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in Dremio.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists.
        """
        try:
            # Try to get table info by querying with LIMIT 0
            full_path = self._get_table_path(table_name)
            sql = f"SELECT * FROM {full_path} LIMIT 0"
            self._execute_query(sql)
            return True
        except Exception:
            return False

    def init_collections(self):
        """Initialize collections from Dremio tables.

        This queries the INFORMATION_SCHEMA to discover available tables.
        """
        if self._collections is None:
            self._collections = {}

        try:
            # Query information schema for tables
            path = self._connection_info.get("path")
            default_schema = self._connection_info.get("default_schema")

            # Build query for tables
            sql = """
                SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                FROM INFORMATION_SCHEMA."TABLES"
                WHERE TABLE_TYPE IN ('TABLE', 'VIEW')
            """

            if path:
                sql += f" AND TABLE_SCHEMA = '{path}'"
            elif default_schema:
                sql += f" AND TABLE_SCHEMA = '{default_schema}'"

            result = self._execute_query(sql)

            for i in range(result.num_rows):
                schema_name = result.column("TABLE_SCHEMA")[i].as_py()
                table_name = result.column("TABLE_NAME")[i].as_py()

                # Use simple name if in default schema, otherwise qualified name
                if schema_name in (path, default_schema):
                    collection_name = table_name
                else:
                    collection_name = f"{schema_name}.{table_name}"

                if collection_name not in self._collections:
                    collection = DremioCollection(name=collection_name, parent=self)
                    collection.metadata.is_prepopulated = True
                    self._collections[collection_name] = collection

            logger.info(f"Discovered {len(self._collections)} tables in Dremio")

        except Exception as e:
            logger.warning(f"Could not query INFORMATION_SCHEMA: {e}")
            # Collections will be created on-demand

    def induce_schema_view(self) -> SchemaView:
        """Induce a schema view from Dremio table structures.

        Returns:
            SchemaView representing the database schema.
        """
        logger.info(f"Inducing schema view for {self.metadata.handle}")
        sb = SchemaBuilder()

        # Ensure collections are initialized
        if not self._collections:
            self.init_collections()

        path = self._connection_info.get("path")
        default_schema = self._connection_info.get("default_schema")

        try:
            # Query columns from INFORMATION_SCHEMA
            sql = """
                SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE,
                       IS_NULLABLE, ORDINAL_POSITION
                FROM INFORMATION_SCHEMA."COLUMNS"
            """

            if path:
                sql += f" WHERE TABLE_SCHEMA = '{path}'"
            elif default_schema:
                sql += f" WHERE TABLE_SCHEMA = '{default_schema}'"

            sql += " ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION"

            result = self._execute_query(sql)

            # Group columns by table
            current_table = None
            for i in range(result.num_rows):
                schema_name = result.column("TABLE_SCHEMA")[i].as_py()
                table_name = result.column("TABLE_NAME")[i].as_py()
                column_name = result.column("COLUMN_NAME")[i].as_py()
                data_type = result.column("DATA_TYPE")[i].as_py()
                is_nullable = result.column("IS_NULLABLE")[i].as_py()

                # Get class name
                if schema_name in (path, default_schema):
                    class_name = table_name
                else:
                    class_name = f"{schema_name}_{table_name}"

                # Add class if new
                if class_name != current_table:
                    sb.add_class(class_name)
                    current_table = class_name

                # Map Dremio type to LinkML type
                from linkml_store.api.stores.dremio.mappings import DREMIO_SQL_TO_LINKML

                # Extract base type (before any parentheses)
                base_type = re.split(r"[\(\[]", data_type)[0].upper()
                linkml_type = DREMIO_SQL_TO_LINKML.get(base_type, "string")

                # Create slot definition
                sd = SlotDefinition(column_name, required=is_nullable == "NO", range=linkml_type)
                sb.schema.classes[class_name].attributes[sd.name] = sd
                logger.debug(f"Introspected slot: {class_name}.{sd.name}: {sd.range}")

        except Exception as e:
            logger.warning(f"Could not introspect schema from INFORMATION_SCHEMA: {e}")

        sb.add_defaults()
        return SchemaView(sb.schema)

    def export_database(self, location: str, target_format: Optional[Union[str, Format]] = None, **kwargs):
        """Export database to a file.

        Args:
            location: Output file path.
            target_format: Output format.
            **kwargs: Additional arguments.
        """
        # Use default export logic from parent
        super().export_database(location, target_format=target_format, **kwargs)

    def import_database(self, location: str, source_format: Optional[str] = None, **kwargs):
        """Import data into Dremio.

        Note: Direct import is limited in Dremio. Data typically needs to be
        loaded through Dremio's data sources or uploaded to connected storage.

        Args:
            location: Source file path.
            source_format: Source format.
            **kwargs: Additional arguments.
        """
        super().import_database(location, source_format=source_format, **kwargs)
