# hdf5_database.py

import logging
from typing import Optional

import h5py
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.hdf5.hdf5_collection import HDF5Collection

logger = logging.getLogger(__name__)


class HDF5Database(Database):
    _file: h5py.File = None
    collection_class = HDF5Collection

    def __init__(self, handle: Optional[str] = None, **kwargs):
        if handle is None:
            handle = "linkml_store.h5"
        super().__init__(handle=handle, **kwargs)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.handle, "a")
        return self._file

    def commit(self, **kwargs):
        self.file.flush()

    def close(self, **kwargs):
        if self._file:
            self._file.close()

    def query(self, query: Query, **kwargs) -> QueryResult:
        if query.from_table:
            collection = self.get_collection(query.from_table)
            return collection.query(query, **kwargs)

    def init_collections(self):
        if self._collections is None:
            self._collections = {}

        for collection_name in self.file:
            if collection_name not in self._collections:
                collection = HDF5Collection(name=collection_name, parent=self)
                self._collections[collection_name] = collection

    def induce_schema_view(self) -> SchemaView:
        logger.info(f"Inducing schema view for {self.handle}")
        sb = SchemaBuilder()
        schema = sb.schema

        for collection_name in self.file:
            sb.add_class(collection_name)
            hdf5_group = self.file[collection_name]
            for field in hdf5_group:
                if field == "_id":
                    continue
                sd = SlotDefinition(field)
                if isinstance(hdf5_group[field][()], list):
                    sd.multivalued = True
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
