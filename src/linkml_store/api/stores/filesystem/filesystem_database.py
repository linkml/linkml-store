import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from sqlalchemy import NullPool, text

from linkml_store.api import Database, Collection
from linkml_store.api.config import CollectionConfig
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.duckdb import DuckDBDatabase
from linkml_store.api.stores.filesystem.filesystem_collection import FileSystemCollection
from linkml_store.utils.sql_utils import introspect_schema, query_to_sql


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
