"""Dremio REST API database adapter.

This module provides a Database implementation for connecting to Dremio
data lakehouse using the REST API v3. This is useful when the Arrow Flight
SQL port (32010) is not accessible, such as behind Cloudflare or firewalls.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd
import requests
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

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


# Mapping from Dremio SQL type names to LinkML types
DREMIO_SQL_TO_LINKML = {
    "VARCHAR": "string",
    "CHAR": "string",
    "BIGINT": "integer",
    "INTEGER": "integer",
    "INT": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "BOOLEAN": "boolean",
    "DOUBLE": "float",
    "FLOAT": "float",
    "DECIMAL": "float",
    "DATE": "date",
    "TIMESTAMP": "datetime",
    "TIME": "string",
    "BINARY": "string",
    "VARBINARY": "string",
    "LIST": "string",
    "STRUCT": "string",
    "MAP": "string",
}


class DremioRestDatabase(Database):
    """
    An adapter for Dremio data lakehouse using the REST API v3.

    This adapter connects to Dremio using the standard REST API, which is
    useful when the Arrow Flight SQL port is not accessible.

    Handle format:
        dremio-rest://[username:password@]host[:port][/path][?params]

    Examples:
        - dremio-rest://localhost
        - dremio-rest://user:pass@lakehouse.example.com
        - dremio-rest://lakehouse.example.com?schema=gold.study
        - dremio-rest://lakehouse.example.com?cf_token_env=CF_AUTHORIZATION

    Parameters (query string):
        - schema: Default schema/space to use for unqualified table names
        - verify_ssl: Whether to verify SSL certificates (default: true)
        - cf_token_env: Environment variable name for Cloudflare Access token
        - username_env: Environment variable for username (default: DREMIO_USER)
        - password_env: Environment variable for password (default: DREMIO_PASSWORD)

    Environment variables:
        - DREMIO_USER: Default username
        - DREMIO_PASSWORD: Default password
        - CF_AUTHORIZATION: Cloudflare Access token (if behind Cloudflare)
    """

    _auth_token: Optional[str] = None
    _connection_info: Optional[Dict[str, Any]] = None
    _session: Optional[requests.Session] = None
    collection_class = DremioRestCollection

    def __init__(
        self,
        handle: Optional[str] = None,
        recreate_if_exists: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """Initialize a Dremio REST database connection.

        Args:
            handle: Connection string in format dremio-rest://host
            recreate_if_exists: Not applicable for Dremio (ignored)
            username: Optional username (overrides env var)
            password: Optional password (overrides env var)
            **kwargs: Additional arguments passed to parent
        """
        if handle is None:
            handle = "dremio-rest://localhost"

        self._connection_info = self._parse_handle(handle)

        # Override with explicit credentials if provided
        if username:
            self._connection_info["username"] = username
        if password:
            self._connection_info["password"] = password

        super().__init__(handle=handle, **kwargs)

    def _parse_handle(self, handle: str) -> Dict[str, Any]:
        """Parse a Dremio REST connection handle.

        Args:
            handle: Connection string like dremio-rest://user:pass@host:port/path?params

        Returns:
            Dictionary with connection parameters.
        """
        # Ensure scheme is present
        if not handle.startswith("dremio-rest://"):
            handle = f"dremio-rest://{handle}"

        parsed = urlparse(handle)

        # Parse query parameters
        params = parse_qs(parsed.query)

        # Extract parameters with defaults
        verify_ssl = params.get("verify_ssl", ["true"])[0].lower() == "true"
        default_schema = params.get("schema", [None])[0]
        cf_token_env = params.get("cf_token_env", ["CF_AUTHORIZATION"])[0]
        username_env = params.get("username_env", ["DREMIO_USER"])[0]
        password_env = params.get("password_env", ["DREMIO_PASSWORD"])[0]

        # Get credentials from URL or environment
        username = parsed.username or os.environ.get(username_env)
        password = parsed.password or os.environ.get(password_env)
        cf_token = os.environ.get(cf_token_env)

        # Determine port (default to 443 for HTTPS)
        port = parsed.port or 443

        return {
            "host": parsed.hostname or "localhost",
            "port": port,
            "username": username,
            "password": password,
            "path": parsed.path.lstrip("/") if parsed.path else None,
            "default_schema": default_schema,
            "verify_ssl": verify_ssl,
            "cf_token": cf_token,
        }

    @property
    def session(self) -> requests.Session:
        """Get or create the requests session."""
        if self._session is None:
            self._session = requests.Session()
            if not self._connection_info["verify_ssl"]:
                self._session.verify = False
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return self._session

    @property
    def base_url(self) -> str:
        """Get the base URL for API requests."""
        info = self._connection_info
        host = info["host"]
        port = info["port"]
        if port == 443:
            return f"https://{host}"
        else:
            return f"https://{host}:{port}"

    def _get_cookies(self) -> Dict[str, str]:
        """Get cookies for requests (e.g., Cloudflare Access token)."""
        cookies = {}
        cf_token = self._connection_info.get("cf_token")
        if cf_token:
            cookies["CF_Authorization"] = cf_token
        return cookies

    def _authenticate(self) -> str:
        """Authenticate with Dremio and get auth token.

        Returns:
            Authentication token for subsequent requests.

        Raises:
            ConnectionError: If authentication fails.
        """
        if self._auth_token:
            return self._auth_token

        info = self._connection_info
        username = info.get("username")
        password = info.get("password")

        if not username or not password:
            raise ConnectionError(
                "Dremio credentials required. Set DREMIO_USER and DREMIO_PASSWORD "
                "environment variables or provide in connection string."
            )

        url = f"{self.base_url}/apiv2/login"
        cookies = self._get_cookies()

        logger.info(f"Authenticating to Dremio at {self.base_url}")

        response = self.session.post(
            url,
            json={"userName": username, "password": password},
            cookies=cookies,
        )

        if not response.ok:
            raise ConnectionError(
                f"Dremio authentication failed: {response.status_code} - {response.text[:200]}"
            )

        token = response.json().get("token")
        if not token:
            raise ConnectionError("No token in authentication response")

        self._auth_token = f"_dremio{token}"
        logger.info("Dremio authentication successful")
        return self._auth_token

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        token = self._authenticate()
        return {"Authorization": token, "Content-Type": "application/json"}

    def _execute_query(self, sql: str, timeout: int = 300) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame.

        Args:
            sql: SQL query string.
            timeout: Maximum time to wait for query completion in seconds.

        Returns:
            Pandas DataFrame with query results.

        Raises:
            RuntimeError: If query fails or times out.
        """
        headers = self._get_headers()
        cookies = self._get_cookies()

        # Submit query
        url = f"{self.base_url}/api/v3/sql"
        logger.debug(f"Executing SQL: {sql}")

        response = self.session.post(
            url,
            headers=headers,
            json={"sql": sql},
            cookies=cookies,
        )

        if not response.ok:
            raise RuntimeError(f"Query submission failed: {response.status_code} - {response.text[:200]}")

        job_id = response.json().get("id")
        if not job_id:
            raise RuntimeError("No job ID in query response")

        logger.debug(f"Query job ID: {job_id}")

        # Wait for completion
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Query timed out after {timeout} seconds")

            status_url = f"{self.base_url}/api/v3/job/{job_id}"
            status_response = self.session.get(status_url, headers=headers, cookies=cookies)
            status = status_response.json()

            job_state = status.get("jobState")
            if job_state == "COMPLETED":
                break
            elif job_state in ("FAILED", "CANCELED"):
                error_msg = status.get("errorMessage", "Unknown error")
                raise RuntimeError(f"Query {job_state}: {error_msg}")

            time.sleep(0.5)

        # Fetch results with pagination
        row_count = status.get("rowCount", 0)
        logger.debug(f"Query completed with {row_count} rows")

        all_rows = []
        offset = 0
        limit = 500  # Dremio max per request

        while offset < row_count:
            results_url = f"{self.base_url}/api/v3/job/{job_id}/results"
            results_response = self.session.get(
                results_url,
                headers=headers,
                cookies=cookies,
                params={"offset": offset, "limit": limit},
            )

            if not results_response.ok:
                raise RuntimeError(f"Failed to fetch results: {results_response.status_code}")

            results = results_response.json()
            rows = results.get("rows", [])
            if not rows:
                break

            all_rows.extend(rows)
            offset += limit

        return pd.DataFrame(all_rows)

    def _execute_update(self, sql: str) -> int:
        """Execute a SQL update/insert/delete statement.

        Args:
            sql: SQL statement.

        Returns:
            Number of affected rows (-1 if unknown).
        """
        # For DML, we just execute and check for success
        self._execute_query(sql)
        return -1

    def commit(self, **kwargs):
        """Commit pending changes (no-op for Dremio REST)."""
        pass

    def close(self, **kwargs):
        """Close the Dremio connection."""
        if self._session:
            self._session.close()
            self._session = None
        self._auth_token = None

    def drop(self, missing_ok=True, **kwargs):
        """Drop the database.

        Note: This is not supported for Dremio as it's typically a read/query layer.
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

        collection = self.get_collection(from_table, create_if_not_exists=False)
        if collection:
            return collection.query(query, **kwargs)
        else:
            return QueryResult(query=query, num_rows=0, rows=[])

    @property
    def supports_sql(self) -> bool:
        """Return True - Dremio REST supports raw SQL queries."""
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
        Execute a raw SQL query against Dremio via REST API.

        If a default schema is configured in the connection URL, unqualified
        table names in FROM/JOIN clauses will be automatically qualified.

        :param sql: SQL query string
        :param kwargs: Additional arguments
        :return: QueryResult containing the results
        """
        sql = self._qualify_table_names(sql)
        logger.debug(f"Qualified SQL: {sql}")
        df = self._execute_query(sql)
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
            Full table path for SQL queries.
        """
        default_schema = self._connection_info.get("default_schema")
        path = self._connection_info.get("path")

        # If already has dots/quotes, assume it's qualified
        if "." in table_name or '"' in table_name:
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
        full_path = self._get_table_path(table_name)
        sql = f"SELECT * FROM {full_path} LIMIT 0"
        try:
            self._execute_query(sql)
            return True
        except Exception:
            return False

    def _list_table_names(self) -> List[str]:
        """List all table names in the database."""
        try:
            path = self._connection_info.get("path")
            default_schema = self._connection_info.get("default_schema")

            sql = """
                SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                FROM INFORMATION_SCHEMA."TABLES"
                WHERE TABLE_TYPE IN ('TABLE', 'VIEW')
            """

            if path:
                sql += f" AND TABLE_SCHEMA = '{path}'"
            elif default_schema:
                sql += f" AND TABLE_SCHEMA = '{default_schema}'"

            df = self._execute_query(sql)
            return df["TABLE_NAME"].tolist() if not df.empty else []
        except Exception as e:
            logger.warning(f"Could not list tables: {e}")
            return []

    def init_collections(self):
        """Initialize collections dict.

        Note: Unlike other adapters, we don't scan INFORMATION_SCHEMA here
        because it can be very slow on large Dremio instances. Collections
        are created on-demand when get_collection() is called with a table path.

        Use discover_collections() to explicitly scan for available tables.
        """
        if self._collections is None:
            self._collections = {}

    def discover_collections(self):
        """Discover and register all available tables from Dremio.

        This queries INFORMATION_SCHEMA to find all tables. This can be slow
        on large Dremio instances - use only when you need to list all tables.
        """
        if self._collections is None:
            self._collections = {}

        path = self._connection_info.get("path")
        default_schema = self._connection_info.get("default_schema")

        sql = """
            SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
            FROM INFORMATION_SCHEMA."TABLES"
            WHERE TABLE_TYPE IN ('TABLE', 'VIEW')
        """

        if path:
            sql += f" AND TABLE_SCHEMA = '{path}'"
        elif default_schema:
            sql += f" AND TABLE_SCHEMA = '{default_schema}'"

        df = self._execute_query(sql)

        for _, row in df.iterrows():
            table_name = row["TABLE_NAME"]
            collection_name = table_name

            if collection_name not in self._collections:
                collection = DremioRestCollection(name=collection_name, parent=self)
                collection.metadata.is_prepopulated = True
                self._collections[collection_name] = collection

        logger.info(f"Discovered {len(self._collections)} tables in Dremio")

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
            fk_df = self._execute_query(fk_sql)
            col_df = self._execute_query(col_sql)

            # Build column lookup: (table_oid, col_num) -> col_name
            col_lookup = {}
            for _, row in col_df.iterrows():
                key = (row["table_oid"], row["col_num"])
                col_lookup[key] = row["col_name"]

            # Build FK info list
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
            df = self._execute_query(sql)

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

        For complex types (ARRAY, STRUCT/ROW), the INFORMATION_SCHEMA doesn't
        return nested field information. This method uses the REST catalog API
        to get the full schema with nested structure.

        Args:
            table_path: Full table path (e.g., '"schema".table').

        Returns:
            Dictionary with column names and their type info.
        """
        # First try the REST catalog API for detailed type info
        path = self._connection_info.get("path")
        default_schema = self._connection_info.get("default_schema")

        # Parse table path to get catalog path
        table_path_clean = table_path.replace('"', '')
        parts = table_path_clean.split(".")

        # Try to find table via catalog API
        try:
            headers = self._get_headers()
            cookies = self._get_cookies()

            # Build catalog path
            if len(parts) >= 2:
                catalog_path = ".".join(parts[:-1])
                table_name = parts[-1]
            else:
                catalog_path = path or default_schema or ""
                table_name = parts[0] if parts else table_path_clean

            # URL encode the path
            encoded_path = quote(f"{catalog_path}.{table_name}" if catalog_path else table_name, safe="")
            url = f"{self.base_url}/api/v3/catalog/by-path/{encoded_path}"

            response = self.session.get(url, headers=headers, cookies=cookies)

            if response.ok:
                catalog_data = response.json()
                fields = catalog_data.get("fields", [])

                schema_info = {}
                for field_data in fields:
                    field_name = field_data.get("name")
                    field_type = field_data.get("type", {})

                    type_name = field_type.get("name", "UNKNOWN")
                    field_info = {
                        "name": field_name,
                        "dremio_type": type_name,
                        "nullable": True,  # Not always available in REST API
                    }

                    # Handle complex types with subSchema
                    sub_schema = field_type.get("subSchema")
                    if sub_schema:
                        field_info["nested_fields"] = []
                        for sub_field in sub_schema:
                            sub_name = sub_field.get("name")
                            sub_type = sub_field.get("type", {})
                            field_info["nested_fields"].append({
                                "name": sub_name,
                                "dremio_type": sub_type.get("name", "UNKNOWN"),
                            })

                    # Handle list element types
                    if type_name == "LIST":
                        sub_schema = field_type.get("subSchema", [])
                        if sub_schema:
                            field_info["element_type"] = sub_schema[0].get("type", {}).get("name", "UNKNOWN")

                    schema_info[field_name] = field_info

                return schema_info

        except Exception as e:
            logger.debug(f"Could not get schema via catalog API: {e}")

        # Fall back to LIMIT 0 query for column info (without nested structure)
        try:
            sql = f"SELECT * FROM {table_path} LIMIT 0"
            df = self._execute_query(sql)

            schema_info = {}
            for col in df.columns:
                schema_info[col] = {
                    "name": col,
                    "dremio_type": str(df[col].dtype),
                    "nullable": True,
                }

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

        if not self._collections:
            self.discover_collections()

        path = self._connection_info.get("path")
        default_schema = self._connection_info.get("default_schema")

        try:
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

            df = self._execute_query(sql)

            current_table = None
            for _, row in df.iterrows():
                schema_name = row["TABLE_SCHEMA"]
                table_name = row["TABLE_NAME"]
                column_name = row["COLUMN_NAME"]
                data_type = row["DATA_TYPE"]
                is_nullable = row["IS_NULLABLE"]

                # Get class name
                if schema_name in (path, default_schema):
                    class_name = table_name
                else:
                    class_name = f"{schema_name}_{table_name}"

                if class_name != current_table:
                    sb.add_class(class_name)
                    current_table = class_name

                # Map Dremio type to LinkML type
                base_type = re.split(r"[\(\[]", str(data_type))[0].upper()
                linkml_type = DREMIO_SQL_TO_LINKML.get(base_type, "string")

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
