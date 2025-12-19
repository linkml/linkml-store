"""Dremio database adapter for linkml-store.

This module provides a Dremio adapter that uses Arrow Flight SQL for high-performance
data access to Dremio data lakehouse.
"""

from linkml_store.api.stores.dremio.dremio_collection import DremioCollection
from linkml_store.api.stores.dremio.dremio_database import DremioDatabase

__all__ = ["DremioDatabase", "DremioCollection"]
