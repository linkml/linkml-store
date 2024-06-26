# mongodb_database.py

import logging
from typing import Optional

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from pymongo import MongoClient
from pymongo.database import Database as NativeDatabase

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.mongodb.mongodb_collection import MongoDBCollection

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
        super().__init__(handle=handle, **kwargs)

    @property
    def _db_name(self) -> str:
        if self.handle:
            db = self.handle.split("/")[-1]
        else:
            db = "default"
        return db

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
        self.native_client.drop_database(self.metadata.alias)

    def query(self, query: Query, **kwargs) -> QueryResult:
        # TODO: DRY
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

    def induce_schema_view(self) -> SchemaView:
        logger.info(f"Inducing schema view for {self.handle}")
        sb = SchemaBuilder()
        schema = sb.schema

        for collection_name in self.native_db.list_collection_names():
            sb.add_class(collection_name)
            mongo_collection = self.native_db[collection_name]
            sample_doc = mongo_collection.find_one()
            if sample_doc:
                for field, value in sample_doc.items():
                    if field == "_id":
                        continue
                    sd = SlotDefinition(field)
                    if isinstance(value, list):
                        sd.multivalued = True
                    if isinstance(value, dict):
                        sd.inlined = True
                    sb.schema.classes[collection_name].attributes[sd.name] = sd

        sb.add_defaults()
        for cls_name in schema.classes:
            if cls_name in self.metadata.collections:
                collection_metadata = self.metadata.collections[cls_name]
                if collection_metadata.attributes:
                    del schema.classes[cls_name]
                    cls = ClassDefinition(name=collection_metadata.type, attributes=collection_metadata.attributes)
                    schema.classes[cls.name] = cls

        return SchemaView(schema)
