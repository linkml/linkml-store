"""Dremio REST API adapter for linkml-store.

This module provides a Dremio adapter that uses the REST API v3 for
connectivity to Dremio data lakehouse instances that don't expose
the Arrow Flight SQL port (e.g., behind Cloudflare or firewalls).
"""

from linkml_store.api.stores.dremio_rest.dremio_rest_collection import DremioRestCollection
from linkml_store.api.stores.dremio_rest.dremio_rest_database import DremioRestDatabase

__all__ = ["DremioRestDatabase", "DremioRestCollection"]
