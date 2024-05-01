# chromadb_database.py

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.chromadb.chromadb_collection import ChromaDBCollection

logger = logging.getLogger(__name__)


class ChromaDBDatabase(Database):
    _client: chromadb.Client = None
    collection_class = ChromaDBCollection

    def __init__(self, handle: Optional[str] = None, **kwargs):
        if handle is None:
            handle = ".chromadb"
        super().__init__(handle=handle, **kwargs)

    @property
    def client(self) -> chromadb.Client:
        if self._client is None:
            self._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.handle,
                )
            )
        return self._client

    def commit(self, **kwargs):
        pass

    def close(self, **kwargs):
        if self._client:
            self._client.close()

    def query(self, query: Query, **kwargs) -> QueryResult:
        if query.from_table:
            collection = self.get_collection(query.from_table)
            return collection.query(query, **kwargs)

    def init_collections(self):
        if self._collections is None:
            self._collections = {}

        for collection_name in self.client.list_collections():
            if collection_name not in self._collections:
                collection = ChromaDBCollection(name=collection_name, parent=self)
                self._collections[collection_name] = collection

    def induce_schema_view(self) -> SchemaView:
        logger.info(f"Inducing schema view for {self.handle}")
        sb = SchemaBuilder()
        schema = sb.schema

        for collection_name in self.client.list_collections():
            sb.add_class(collection_name)
            chroma_collection = self.client.get_collection(collection_name)
            sample_doc = chroma_collection.peek(1)
            if sample_doc:
                for field, value in sample_doc[0].items():
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
