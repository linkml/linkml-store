"""
Adapter for FileSystem wrapper

Handles have the form:

 - ``file:<path>`` for a local file
"""

from linkml_store.api.stores.filesystem.filesystem_collection import FileSystemCollection
from linkml_store.api.stores.filesystem.filesystem_database import FileSystemDatabase

__all__ = [
    "FileSystemCollection",
    "FileSystemDatabase",
]
