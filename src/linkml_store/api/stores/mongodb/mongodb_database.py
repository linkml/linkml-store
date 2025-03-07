# mongodb_database.py

import logging
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from pymongo import MongoClient
from pymongo.database import Database as NativeDatabase

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.mongodb.mongodb_collection import MongoDBCollection
from linkml_store.utils.file_utils import safe_remove_directory
from linkml_store.utils.format_utils import Format
from linkml_store.utils.mongodb_utils import import_mongodb

logger = logging.getLogger(__name__)


class MongoDBDatabase(Database):
    """
    An adapter for MongoDB databases.

    The LinkML-Store Database abstraction combines mongodb Client and Database.
    """

    _native_client: MongoClient = None
    _native_db = None
    collection_class = MongoDBCollection

    def __init__(self, handle: Optional[str] = None, **kwargs):
        if handle is None:
            handle = "mongodb://localhost:27017/test"
        if handle == "mongodb":
            handle = "mongodb://localhost:27017/temporary"
        super().__init__(handle=handle, **kwargs)

    @property
    def _db_name(self) -> str:
        if self.handle:
            parsed_url = urlparse(self.handle)
            path_parts = parsed_url.path.lstrip("/").split("?")[0].split("/")
            db_name = path_parts[0] if path_parts else "default"
            if not db_name:
                db_name = self.alias
        else:
            db_name = "default"
        return db_name

    @property
    def native_client(self) -> MongoClient:
        if self._native_client is None:
            self._native_client = MongoClient(self.handle)
        return self._native_client

    @property
    def native_db(self) -> NativeDatabase:
        if self._native_db is None:
            alias = self.metadata.alias
            if not alias:
                alias = "default"
            self._native_db = self.native_client[self._db_name]
        return self._native_db

    def commit(self, **kwargs):
        pass

    def close(self, **kwargs):
        if self._native_client:
            self._native_client.close()

    def drop(self, **kwargs):
        self.native_client.drop_database(self.native_db.name)

    def query(self, query: Query, **kwargs) -> QueryResult:
        if query.from_table:
            collection = self.get_collection(query.from_table)
            return collection.query(query, **kwargs)
        else:
            raise NotImplementedError(f"Querying without a table is not supported in {self.__class__.__name__}")

    def init_collections(self):
        if self._collections is None:
            self._collections = {}

        for collection_name in self.native_db.list_collection_names():
            if collection_name not in self._collections:
                collection = MongoDBCollection(name=collection_name, parent=self)
                self._collections[collection_name] = collection

    def export_database(self, location: str, target_format: Optional[Union[str, Format]] = None, **kwargs):
        if target_format == Format.DUMP_MONGODB.value or target_format == Format.DUMP_MONGODB:
            path = Path(location)
            if path.exists():
                safe_remove_directory(path, no_backup=True)
            from linkml_store.utils.mongodb_utils import export_mongodb

            export_mongodb(self.handle, location)
        else:
            super().export_database(location, target_format=target_format, **kwargs)

    def import_database(self, location: str, source_format: Optional[str] = None, **kwargs):
        """
        Import a database from a file or location.

        :param location: location of the file
        :param source_format: source format
        :param kwargs: additional arguments
        """
        if source_format == Format.DUMP_MONGODB.value or source_format == Format.DUMP_MONGODB:
            import_mongodb(self.handle, location, drop=True)
        else:
            super().import_database(location, source_format=source_format, **kwargs)
