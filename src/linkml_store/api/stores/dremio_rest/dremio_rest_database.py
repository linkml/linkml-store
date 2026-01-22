"""Dremio REST API database adapter.

This module provides a Database implementation for connecting to Dremio
data lakehouse using the REST API v3. This is useful when the Arrow Flight
SQL port (32010) is not accessible, such as behind Cloudflare or firewalls.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection

logger = logging.getLogger(__name__)

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
            return f"{default_schema}.{table_name}"
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
                # Extract schema part if it contains dots
                schema_part = default_schema.split(".")[0] if "." in default_schema else default_schema
                sql += f" AND TABLE_SCHEMA LIKE '{schema_part}%'"

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
            schema_part = default_schema.split(".")[0] if "." in default_schema else default_schema
            sql += f" AND TABLE_SCHEMA LIKE '{schema_part}%'"

        df = self._execute_query(sql)

        for _, row in df.iterrows():
            schema_name = row["TABLE_SCHEMA"]
            table_name = row["TABLE_NAME"]

            # Use simple name if in default schema
            if schema_name in (path, default_schema):
                collection_name = table_name
            else:
                collection_name = f"{schema_name}.{table_name}"

            if collection_name not in self._collections:
                collection = DremioRestCollection(name=collection_name, parent=self)
                collection.metadata.is_prepopulated = True
                self._collections[collection_name] = collection

        logger.info(f"Discovered {len(self._collections)} tables in Dremio")

    def induce_schema_view(self) -> SchemaView:
        """Induce a schema view from Dremio table structures.

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
                schema_part = default_schema.split(".")[0] if "." in default_schema else default_schema
                sql += f" WHERE TABLE_SCHEMA LIKE '{schema_part}%'"

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

        sb.add_defaults()
        return SchemaView(sb.schema)
