import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy
from duckdb import DuckDBPyConnection
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from sqlalchemy import NullPool, text

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.duckdb.duckdb_collection import DuckDBCollection
from linkml_store.utils.sql_utils import introspect_schema, query_to_sql

TYPE_MAP = {
    "VARCHAR": "string",
    "BIGINT": "integer",
    "BOOLEAN": "boolean",
    "DATE": "date",
    "DOUBLE": "float",
    "INTEGER": "integer",
    "JSON": "Any",
}


logger = logging.getLogger(__name__)


class DuckDBDatabase(Database):
    _connection: DuckDBPyConnection = None
    _engine: sqlalchemy.Engine = None
    collection_class = DuckDBCollection

    def __init__(self, handle: Optional[str] = None, recreate_if_exists: bool = False, **kwargs):
        if handle is None:
            handle = "duckdb:///:memory:"
        if recreate_if_exists:
            path = Path(handle.replace("duckdb:///", ""))
            if path.exists():
                path.unlink()
        super().__init__(handle=handle, **kwargs)

    @property
    def engine(self) -> sqlalchemy.Engine:
        if not self._engine:
            handle = self.handle
            if not handle.startswith("duckdb://") and not handle.startswith(":"):
                handle = f"duckdb:///{handle}"
            if ":memory:" not in handle:
                # TODO: investigate this; duckdb appears to be prematurely caching
                self._engine = sqlalchemy.create_engine(handle, poolclass=NullPool)
            else:
                self._engine = sqlalchemy.create_engine(handle)
        return self._engine

    def commit(self, **kwargs):
        with self.engine.connect() as conn:
            conn.commit()

    def close(self, **kwargs):
        self.engine.dispose()

    def query(self, query: Query, **kwargs) -> QueryResult:
        json_encoded_cols = []
        if query.from_table:
            if not query.from_table.startswith("information_schema"):
                meta_query = Query(
                    from_table="information_schema.tables", where_clause={"table_name": query.from_table}
                )
                qr = self.query(meta_query)
                if qr.num_rows == 0:
                    logger.debug(f"Table {query.from_table} not created yet")
                    return QueryResult(query=query, num_rows=0, rows=[])
            if not query.from_table.startswith("information_schema"):
                sv = self.schema_view
            else:
                sv = None
            if sv:
                cd = None
                for c in self._collections.values():
                    if c.name == query.from_table or c.metadata.alias == query.from_table:
                        cd = c.class_definition()
                        break
                if cd:
                    for att in sv.class_induced_slots(cd.name):
                        if att.inlined or att.inlined_as_list:
                            json_encoded_cols.append(att.name)
        with self.engine.connect() as conn:
            count_query_str = text(query_to_sql(query, count=True))
            num_rows = list(conn.execute(count_query_str))[0][0]
            logger.debug(f"num_rows: {num_rows}")
            query_str = query_to_sql(query, **kwargs)  # include offset, limit
            logger.debug(f"query_str: {query_str}")
            rows = list(conn.execute(text(query_str)).mappings())
            qr = QueryResult(query=query, num_rows=num_rows, rows=rows)
            if json_encoded_cols:
                for row in qr.rows:
                    for col in json_encoded_cols:
                        if row[col]:
                            if isinstance(row[col], list):
                                for i in range(len(row[col])):
                                    row[col][i] = json.loads(row[col][i])
                            else:
                                row[col] = json.loads(row[col])
            qr.set_rows(pd.DataFrame(rows))
            facet_columns = query.facet_slots
            if query.include_facet_counts and not facet_columns:
                raise ValueError("Facet counts requested but no facet columns specified")
            if facet_columns:
                raise NotImplementedError
            return qr

    def init_collections(self):
        # TODO: unify schema introspection
        if not self.schema_view:
            schema = introspect_schema(self.engine)
        else:
            schema = self.schema_view.schema
        table_names = schema.classes.keys()
        if self._collections is None:
            self._collections = {}
        for table_name in table_names:
            if table_name not in self._collections:
                collection = DuckDBCollection(name=table_name, parent=self)
                self._collections[table_name] = collection

    def induce_schema_view(self) -> SchemaView:
        # TODO: unify schema introspection
        # TODO: handle case where schema is provided in advance
        logger.info(f"Inducing schema view for {self.metadata.handle} // {self}")
        sb = SchemaBuilder()
        schema = sb.schema
        query = Query(from_table="information_schema.tables", where_clause={"table_type": "BASE TABLE"})
        qr = self.query(query)
        logger.info(f"Found {qr.num_rows} information_schema.tables // {qr.rows}")
        if qr.num_rows:
            table_names = [row["table_name"] for row in qr.rows]
            for tbl in table_names:
                sb.add_class(tbl)
        query = Query(from_table="information_schema.columns", sort_by=["ordinal_position"])
        for row in self.query(query, limit=-1).rows:
            tbl_name = row["table_name"]
            if tbl_name not in sb.schema.classes:
                continue
            dt = row["data_type"]
            if dt.endswith("[]"):
                dt = dt[0:-2]
                multivalued = True
            else:
                multivalued = False
            rng = TYPE_MAP.get(dt, "string")
            sd = SlotDefinition(
                row["column_name"], required=row["is_nullable"] == "NO", multivalued=multivalued, range=rng
            )
            if dt == "JSON":
                sd.inlined_as_list = True
            sb.schema.classes[tbl_name].attributes[sd.name] = sd
            logger.info(f"Introspected slot: {tbl_name}.{sd.name}: {sd.range} FROM {dt}")
        sb.add_defaults()
        for cls_name in schema.classes:
            if cls_name in self.metadata.collections:
                collection_metadata = self.metadata.collections[cls_name]
                if collection_metadata.attributes:
                    del schema.classes[cls_name]
                    cls = ClassDefinition(name=collection_metadata.type, attributes=collection_metadata.attributes)
                    schema.classes[cls.name] = cls
        return SchemaView(schema)
