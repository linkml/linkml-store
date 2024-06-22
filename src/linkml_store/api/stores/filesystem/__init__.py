"""
Adapter for DuckDB embedded database.

Handles have the form:

 - ``duckdb:///<path>`` for a file-based database
 - ``duckdb:///:memory:`` for an in-memory database
"""

from linkml_store.api.stores.duckdb.duckdb_collection import DuckDBCollection
from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase

__all__ = [
    "DuckDBCollection",
    "DuckDBDatabase",
]
