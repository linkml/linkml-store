"""Dremio database adapter using Arrow Flight SQL.

This module provides a Database implementation for connecting to Dremio
data lakehouse using the Arrow Flight SQL protocol for high-performance
data access.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import pandas as pd
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.dremio.dremio_collection import DremioCollection
from linkml_store.api.stores.dremio.mappings import get_linkml_type_from_arrow
from linkml_store.utils.format_utils import Format

logger = logging.getLogger(__name__)


@dataclass
class ForeignKeyInfo:
    """Information about a foreign key constraint."""

    constraint_name: str
    source_table: str
    source_columns: List[str]
    target_table: str
    target_columns: List[str]
    source_schema: Optional[str] = None
    target_schema: Optional[str] = None


@dataclass
class ColumnInfo:
    """Information about a column including comments and nested structure."""

    name: str
    data_type: str
    is_nullable: bool = True
    comment: Optional[str] = None
    ordinal_position: int = 0
    nested_fields: List["ColumnInfo"] = field(default_factory=list)


@dataclass
class TableInfo:
    """Information about a table including comments."""

    name: str
    schema_name: Optional[str] = None
    comment: Optional[str] = None
    columns: List[ColumnInfo] = field(default_factory=list)


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
        - username_env: Environment variable for username (default: DREMIO_USER)
        - password_env: Environment variable for password (default: DREMIO_PASSWORD)

    Environment variables:
        - DREMIO_USER: Default username (if not in URL)
        - DREMIO_PASSWORD: Default password (if not in URL)

    Note:
        Requires pyarrow with Flight SQL support. Install with:
        pip install pyarrow
    """

    _flight_client = None
    _adbc_connection = None
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

        # Get env var names (configurable via query params)
        username_env = params.get("username_env", ["DREMIO_USER"])[0]
        password_env = params.get("password_env", ["DREMIO_PASSWORD"])[0]

        # Get credentials from URL or environment variables
        username = parsed.username or os.environ.get(username_env)
        password = parsed.password or os.environ.get(password_env)

        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 32010,
            "username": username,
            "password": password,
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

        Uses ADBC Flight SQL driver when available (faster), falls back to
        raw Flight RPC otherwise.

        Args:
            sql: SQL query string.

        Returns:
            PyArrow Table with query results.
        """
        logger.debug(f"Executing SQL: {sql}")

        # Try ADBC first (much faster for Flight SQL)
        try:
            return self._execute_query_adbc(sql)
        except ImportError:
            logger.debug("ADBC not available, falling back to raw Flight")
            return self._execute_query_flight(sql)

    @property
    def adbc_connection(self):
        """Get or create cached ADBC Flight SQL connection.

        Returns:
            ADBC connection to Dremio.

        Raises:
            ImportError: If ADBC driver is not installed.
        """
        if self._adbc_connection is None:
            import adbc_driver_flightsql.dbapi as flightsql

            info = self._connection_info
            host = info["host"]
            port = info["port"]

            # Build URI
            if info["use_encryption"]:
                uri = f"grpc+tls://{host}:{port}"
            else:
                uri = f"grpc://{host}:{port}"

            # Build connection kwargs
            connect_kwargs = {"uri": uri}

            # Add auth if available
            if info["username"] and info["password"]:
                connect_kwargs["db_kwargs"] = {
                    "username": info["username"],
                    "password": info["password"],
                }

            logger.info(f"Establishing ADBC Flight SQL connection to {uri}")
            self._adbc_connection = flightsql.connect(**connect_kwargs)

        return self._adbc_connection

    def _execute_query_adbc(self, sql: str) -> "pyarrow.Table":
        """Execute query using ADBC Flight SQL driver (fast path).

        Args:
            sql: SQL query string.

        Returns:
            PyArrow Table with query results.

        Raises:
            ImportError: If ADBC driver is not installed.
        """
        import pyarrow as pa

        conn = self.adbc_connection
        sql_upper = sql.strip().upper()

        # Handle context-setting statements (USE, SET, ALTER SESSION, etc.)
        # These don't return meaningful results but set session state
        if sql_upper.startswith(("USE ", "SET ", "ALTER SESSION")):
            with conn.cursor() as cursor:
                cursor.execute(sql)
                # Don't try to fetch results - just execute for side effect
                return pa.table({})

        with conn.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetch_arrow_table()

    def _execute_query_flight(self, sql: str) -> "pyarrow.Table":
        """Execute query using raw Flight RPC (fallback path).

        Args:
            sql: SQL query string.

        Returns:
            PyArrow Table with query results.
        """
        import pyarrow.flight as flight

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
        if self._adbc_connection:
            self._adbc_connection.close()
            self._adbc_connection = None

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

    @property
    def supports_sql(self) -> bool:
        """Return True - Dremio supports raw SQL queries."""
        return True

    def _qualify_table_names(self, sql: str) -> str:
        """Qualify unqualified table names in SQL using configured schema.

        Handles FROM and JOIN clauses, qualifying table names that don't
        already contain dots or quotes.

        Args:
            sql: SQL query string.

        Returns:
            SQL with qualified table names.
        """
        default_schema = self._connection_info.get("default_schema")
        if not default_schema:
            return sql

        # Pattern matches FROM/JOIN followed by an unqualified table name
        # Unqualified = no dots, no quotes, just a simple identifier
        # Captures: (FROM|JOIN) (tablename) (optional: AS? alias | WHERE | ORDER | LIMIT | GROUP | ; | end)
        pattern = r'(?i)((?:FROM|JOIN)\s+)([a-zA-Z_][a-zA-Z0-9_]*)(\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*|\s+(?:WHERE|ORDER|GROUP|LIMIT|HAVING|UNION|INTERSECT|EXCEPT|ON|LEFT|RIGHT|INNER|OUTER|CROSS|FULL|;)|$)'

        def replace_table(match):
            prefix = match.group(1)  # "FROM " or "JOIN "
            table = match.group(2)   # table name
            suffix = match.group(3)  # rest (alias, WHERE, etc.)

            # Check if this looks like a keyword (not a table name)
            keywords = {'WHERE', 'ORDER', 'GROUP', 'LIMIT', 'HAVING', 'UNION',
                       'INTERSECT', 'EXCEPT', 'SELECT', 'AS', 'ON', 'AND', 'OR',
                       'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'FULL', 'JOIN'}
            if table.upper() in keywords:
                return match.group(0)

            qualified = self._get_table_path(table)
            return f"{prefix}{qualified}{suffix}"

        return re.sub(pattern, replace_table, sql)

    def execute_sql(self, sql: str, **kwargs) -> QueryResult:
        """
        Execute a raw SQL query against Dremio.

        If a default schema is configured in the connection URL, unqualified
        table names in FROM/JOIN clauses will be automatically qualified.

        :param sql: SQL query string
        :param kwargs: Additional arguments
        :return: QueryResult containing the results
        """
        sql = self._qualify_table_names(sql)
        logger.debug(f"Qualified SQL: {sql}")
        result = self._execute_query(sql)
        df = result.to_pandas()
        return QueryResult(num_rows=len(df), rows=df.to_dict("records"))

    def _needs_quoting(self, identifier: str) -> bool:
        """Check if an identifier needs quoting in SQL.

        Identifiers need quoting if they contain special characters
        like hyphens, spaces, or start with a digit.
        """
        if not identifier:
            return False
        # Needs quoting if contains non-alphanumeric/underscore or starts with digit
        if not identifier[0].isalpha() and identifier[0] != "_":
            return True
        return not all(c.isalnum() or c == "_" for c in identifier)

    def _quote_if_needed(self, identifier: str) -> str:
        """Quote an identifier if it contains special characters."""
        if self._needs_quoting(identifier):
            return f'"{identifier}"'
        return identifier

    def _get_table_path(self, table_name: str) -> str:
        """Get the full table path including schema if configured.

        Args:
            table_name: Table name.

        Returns:
            Full table path.
        """
        default_schema = self._connection_info.get("default_schema")
        path = self._connection_info.get("path")

        if "." in table_name or '"' in table_name:
            # Already qualified
            return table_name

        if default_schema:
            # Schema like "gold-db-2 postgresql.gold" needs proper quoting
            # Split into source and schema.table parts
            parts = default_schema.split(".")
            if len(parts) >= 2:
                # Source name may have spaces/hyphens - quote if needed
                source = self._quote_if_needed(parts[0])
                schema = ".".join(parts[1:])
                return f'{source}.{schema}.{table_name}'
            else:
                return f'{self._quote_if_needed(default_schema)}.{table_name}'
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

    def _detect_source_type(self, source_name: str) -> Optional[str]:
        """Detect the type of a Dremio data source.

        Args:
            source_name: Name of the source (e.g., 'gold-db-2 postgresql').

        Returns:
            Source type ('postgresql', 'mysql', 'mongodb', etc.) or None.
        """
        source_lower = source_name.lower()
        if "postgresql" in source_lower or "postgres" in source_lower:
            return "postgresql"
        elif "mysql" in source_lower:
            return "mysql"
        elif "mongo" in source_lower:
            return "mongodb"
        elif "iceberg" in source_lower:
            return "iceberg"
        elif "hive" in source_lower:
            return "hive"
        return None

    def _get_source_from_schema(self, schema_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract source name and schema from a full schema path.

        Args:
            schema_name: Full schema path like 'gold-db-2 postgresql.gold'.

        Returns:
            Tuple of (source_name, schema_within_source).
        """
        if "." in schema_name:
            parts = schema_name.split(".", 1)
            return parts[0], parts[1] if len(parts) > 1 else None
        return schema_name, None

    def get_foreign_keys(
        self, schema_name: Optional[str] = None, table_name: Optional[str] = None
    ) -> List[ForeignKeyInfo]:
        """Get foreign key constraints from PostgreSQL sources via pg_catalog.

        This method queries PostgreSQL's pg_catalog.pg_constraint to retrieve
        foreign key information. Only works for PostgreSQL-backed sources.

        Args:
            schema_name: Full schema path (e.g., 'gold-db-2 postgresql.gold').
                        If None, uses the default schema from connection.
            table_name: Optional table name to filter results.

        Returns:
            List of ForeignKeyInfo objects describing FK relationships.

        Example:
            >>> db = DremioDatabase("dremio://lakehouse:32010")
            >>> fks = db.get_foreign_keys("gold-db-2 postgresql.gold")
            >>> for fk in fks[:3]:
            ...     print(f"{fk.source_table}.{fk.source_columns} -> {fk.target_table}")
        """
        if schema_name is None:
            schema_name = self._connection_info.get("default_schema") or self._connection_info.get("path")

        if not schema_name:
            logger.warning("No schema specified for FK introspection")
            return []

        source_name, pg_schema = self._get_source_from_schema(schema_name)
        source_type = self._detect_source_type(source_name)

        if source_type != "postgresql":
            logger.info(f"FK introspection only supported for PostgreSQL sources, not {source_type}")
            return []

        # Query FK constraints from pg_catalog
        fk_sql = f'''
            SELECT
                con.conname as constraint_name,
                src_class.relname as source_table,
                tgt_class.relname as target_table,
                con.conkey as source_col_nums,
                con.confkey as target_col_nums,
                con.conrelid as source_oid,
                con.confrelid as target_oid
            FROM "{source_name}".pg_catalog.pg_constraint con
            JOIN "{source_name}".pg_catalog.pg_class src_class ON con.conrelid = src_class.oid
            JOIN "{source_name}".pg_catalog.pg_class tgt_class ON con.confrelid = tgt_class.oid
            JOIN "{source_name}".pg_catalog.pg_namespace nsp ON src_class.relnamespace = nsp.oid
            WHERE con.contype = 'f'
        '''

        if pg_schema:
            fk_sql += f" AND nsp.nspname = '{pg_schema}'"
        if table_name:
            fk_sql += f" AND src_class.relname = '{table_name}'"

        # Query column info for resolving column numbers to names
        col_sql = f'''
            SELECT
                c.oid as table_oid,
                a.attnum as col_num,
                a.attname as col_name
            FROM "{source_name}".pg_catalog.pg_class c
            JOIN "{source_name}".pg_catalog.pg_attribute a ON a.attrelid = c.oid
            JOIN "{source_name}".pg_catalog.pg_namespace nsp ON c.relnamespace = nsp.oid
            WHERE a.attnum > 0 AND NOT a.attisdropped
        '''

        if pg_schema:
            col_sql += f" AND nsp.nspname = '{pg_schema}'"

        try:
            fk_result = self._execute_query(fk_sql)
            col_result = self._execute_query(col_sql)

            # Build column lookup: (table_oid, col_num) -> col_name
            col_df = col_result.to_pandas()
            col_lookup = {}
            for _, row in col_df.iterrows():
                key = (row["table_oid"], row["col_num"])
                col_lookup[key] = row["col_name"]

            # Build FK info list
            fk_df = fk_result.to_pandas()
            fk_list = []

            for _, fk in fk_df.iterrows():
                # Parse array strings like '{1}' or '{1,2}'
                src_nums = [int(x) for x in str(fk["source_col_nums"]).strip("{}").split(",") if x]
                tgt_nums = [int(x) for x in str(fk["target_col_nums"]).strip("{}").split(",") if x]

                src_cols = [col_lookup.get((fk["source_oid"], n), f"col_{n}") for n in src_nums]
                tgt_cols = [col_lookup.get((fk["target_oid"], n), f"col_{n}") for n in tgt_nums]

                fk_list.append(
                    ForeignKeyInfo(
                        constraint_name=fk["constraint_name"],
                        source_table=fk["source_table"],
                        source_columns=src_cols,
                        target_table=fk["target_table"],
                        target_columns=tgt_cols,
                        source_schema=pg_schema,
                        target_schema=pg_schema,  # Assumes same schema
                    )
                )

            logger.info(f"Found {len(fk_list)} foreign key constraints in {schema_name}")
            return fk_list

        except Exception as e:
            logger.warning(f"Could not retrieve FK constraints: {e}")
            return []

    def get_table_comments(
        self, schema_name: Optional[str] = None, table_name: Optional[str] = None
    ) -> Dict[str, TableInfo]:
        """Get table and column comments from PostgreSQL sources via pg_description.

        Args:
            schema_name: Full schema path (e.g., 'gold-db-2 postgresql.gold').
            table_name: Optional table name to filter results.

        Returns:
            Dictionary mapping table names to TableInfo with comments.
        """
        if schema_name is None:
            schema_name = self._connection_info.get("default_schema") or self._connection_info.get("path")

        if not schema_name:
            return {}

        source_name, pg_schema = self._get_source_from_schema(schema_name)
        source_type = self._detect_source_type(source_name)

        if source_type != "postgresql":
            logger.info(f"Comment introspection only supported for PostgreSQL sources")
            return {}

        sql = f'''
            SELECT
                c.relname as table_name,
                a.attname as column_name,
                a.attnum as col_num,
                td.description as table_comment,
                cd.description as column_comment
            FROM "{source_name}".pg_catalog.pg_class c
            JOIN "{source_name}".pg_catalog.pg_namespace nsp ON c.relnamespace = nsp.oid
            LEFT JOIN "{source_name}".pg_catalog.pg_attribute a
                ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
            LEFT JOIN "{source_name}".pg_catalog.pg_description td
                ON td.objoid = c.oid AND td.objsubid = 0
            LEFT JOIN "{source_name}".pg_catalog.pg_description cd
                ON cd.objoid = c.oid AND cd.objsubid = a.attnum
            WHERE c.relkind IN ('r', 'v')
        '''

        if pg_schema:
            sql += f" AND nsp.nspname = '{pg_schema}'"
        if table_name:
            sql += f" AND c.relname = '{table_name}'"

        sql += " ORDER BY c.relname, a.attnum"

        try:
            result = self._execute_query(sql)
            df = result.to_pandas()

            tables = {}
            for tbl_name in df["table_name"].unique():
                tbl_df = df[df["table_name"] == tbl_name]
                table_comment = tbl_df["table_comment"].iloc[0] if not tbl_df["table_comment"].isna().all() else None

                columns = []
                for _, row in tbl_df.iterrows():
                    if row["column_name"]:
                        columns.append(
                            ColumnInfo(
                                name=row["column_name"],
                                data_type="",  # Not fetched here
                                comment=row["column_comment"] if not pd.isna(row["column_comment"]) else None,
                                ordinal_position=int(row["col_num"]) if row["col_num"] else 0,
                            )
                        )

                tables[tbl_name] = TableInfo(
                    name=tbl_name, schema_name=pg_schema, comment=table_comment, columns=columns
                )

            return tables

        except Exception as e:
            logger.warning(f"Could not retrieve table comments: {e}")
            return {}

    def get_nested_schema(self, table_path: str) -> Dict[str, Any]:
        """Get full schema including nested types by querying with LIMIT 0.

        For complex types (ARRAY, STRUCT/ROW), the metadata methods don't
        return nested field information. This method executes a LIMIT 0
        query to get the full Arrow schema with nested structure.

        Args:
            table_path: Full table path (e.g., '"schema".table').

        Returns:
            Dictionary with column names and their Arrow type info.
        """
        sql = f"SELECT * FROM {table_path} LIMIT 0"

        try:
            result = self._execute_query(sql)
            schema_info = {}

            for field in result.schema:
                type_str = str(field.type)
                field_info = {
                    "name": field.name,
                    "arrow_type": type_str,
                    "nullable": field.nullable,
                }

                # Parse nested structure for struct types
                if type_str.startswith("struct<"):
                    field_info["nested_fields"] = []
                    for nested in field.type:
                        field_info["nested_fields"].append(
                            {"name": nested.name, "arrow_type": str(nested.type), "nullable": nested.nullable}
                        )

                # Parse list element type
                if hasattr(field.type, "value_type"):
                    field_info["element_type"] = str(field.type.value_type)

                schema_info[field.name] = field_info

            return schema_info

        except Exception as e:
            logger.warning(f"Could not get nested schema for {table_path}: {e}")
            return {}

    def induce_schema_view(self, include_foreign_keys: bool = True) -> SchemaView:
        """Induce a schema view from Dremio table structures.

        Args:
            include_foreign_keys: If True, attempt to retrieve FK info from
                                 PostgreSQL sources and add relationships.

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

        # Add foreign key relationships
        if include_foreign_keys:
            schema_to_use = path or default_schema
            if schema_to_use:
                fks = self.get_foreign_keys(schema_to_use)
                for fk in fks:
                    # Get or derive class names
                    src_class = fk.source_table
                    tgt_class = fk.target_table

                    # Skip if classes don't exist
                    if src_class not in sb.schema.classes or tgt_class not in sb.schema.classes:
                        continue

                    # For single-column FKs, update the slot's range to point to target class
                    if len(fk.source_columns) == 1:
                        src_col = fk.source_columns[0]
                        if src_col in sb.schema.classes[src_class].attributes:
                            slot = sb.schema.classes[src_class].attributes[src_col]
                            # Set range to target class, indicating a relationship
                            slot.range = tgt_class
                            slot.description = f"Foreign key to {tgt_class}"
                            logger.debug(f"Added FK relationship: {src_class}.{src_col} -> {tgt_class}")

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
