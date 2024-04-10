from dataclasses import dataclass
from typing import Optional

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from pymongo import MongoClient

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.mongodb.mongodb_collection import MongoDBCollection


@dataclass
class MongoDBDatabase(Database):
    """
    A wrapper around a MongoDB database
    """

    _client: MongoClient = None
    _database = None

    def __post_init__(self):
        if not self.handle:
            self.handle = "mongodb://localhost:27017"

    @property
    def client(self) -> MongoClient:
        if not self._client:
            self._client = MongoClient(self.handle)
        return self._client

    @property
    def database(self):
        if not self._database:
            db_name = self.handle.split("/")[-1]
            self._database = self.client[db_name]
        return self._database

    def commit(self, **kwargs):
        pass

    def close(self, **kwargs):
        self.client.close()

    def query(self, query: Query, **kwargs) -> QueryResult:
        collection = self.database[query.from_table]
        where_clause = query.where_clause or {}
        cursor = collection.find(where_clause)
        if query.limit:
            cursor = cursor.limit(query.limit)
        if query.offset:
            cursor = cursor.skip(query.offset)
        if query.sort_by:
            sort_key = [(col, 1) for col in query.sort_by]
            cursor = cursor.sort(sort_key)
        rows = list(cursor)
        num_rows = len(rows)
        qr = QueryResult(query=query, num_rows=num_rows, rows=rows)
        return qr

    def init_collections(self):
        if self._collections is None:
            self._collections = {}
        for collection_name in self.database.list_collection_names():
            if collection_name not in self._collections:
                collection = MongoDBCollection(name=collection_name, parent=self)
                self._collections[collection_name] = collection

    def create_collection(self, name: str, alias: Optional[str] = None, **kwargs) -> MongoDBCollection:
        collection = MongoDBCollection(name=name, parent=self)
        if not self._collections:
            self._collections = {}
        if not alias:
            alias = name
        self._collections[alias] = collection
        return collection

    def induce_schema_view(self) -> SchemaView:
        sb = SchemaBuilder()
        schema = sb.schema
        collection_names = self.database.list_collection_names()
        for collection_name in collection_names:
            sb.add_class(collection_name)
            collection = self.database[collection_name]
            sample_doc = collection.find_one()
            if sample_doc:
                for key, value in sample_doc.items():
                    if key == "_id":
                        continue
                    if isinstance(value, list):
                        multivalued = True
                        if value:
                            value = value[0]
                        else:
                            value = None
                    else:
                        multivalued = False
                    if isinstance(value, str):
                        rng = "string"
                    elif isinstance(value, int):
                        rng = "integer"
                    elif isinstance(value, float):
                        rng = "float"
                    elif isinstance(value, bool):
                        rng = "boolean"
                    else:
                        rng = "string"
                    sd = SlotDefinition(key, range=rng, multivalued=multivalued)
                    sb.schema.classes[collection_name].attributes[sd.name] = sd
        sb.add_defaults()
        return SchemaView(schema)
