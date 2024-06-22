import logging
from typing import Optional

from linkml_store.api import Collection, Database
from linkml_store.api.config import CollectionConfig
from linkml_store.api.stores.duckdb import DuckDBDatabase
from linkml_store.api.stores.filesystem.filesystem_collection import FileSystemCollection

logger = logging.getLogger(__name__)


class FileSystemDatabase(Database):
    collection_class = FileSystemCollection
    wrapped_database: Database = None

    def __init__(self, handle: Optional[str] = None, recreate_if_exists: bool = False, **kwargs):
        self.wrapped_database = DuckDBDatabase("duckdb:///:memory:")
        super().__init__(handle=handle, **kwargs)

    def commit(self, **kwargs):
        # TODO: sync
        pass

    def close(self, **kwargs):
        self.wrapped_database.close()

    def create_collection(
        self,
        name: str,
        alias: Optional[str] = None,
        metadata: Optional[CollectionConfig] = None,
        recreate_if_exists=False,
        **kwargs,
    ) -> Collection:
        wd = self.wrapped_database
        wd.create_collection()
