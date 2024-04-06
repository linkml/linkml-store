import logging
from dataclasses import dataclass
from typing import Sequence, Union, List

import duckdb
import pandas as pd
import sqlalchemy
from duckdb import DuckDBPyConnection
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from sqlalchemy import text
import sqlalchemy as sqla

from linkml_store.api import Database
from linkml_store.api.collection import OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.duckdb.duckdb_collection import DuckDBCollection


TYPE_MAP = {
    "VARCHAR": "string",
    "BIGINT": "integer",
    "BOOLEAN": "boolean",
}


logger = logging.getLogger(__name__)

def run_query(con: DuckDBPyConnection, query: Query, **kwargs):
    """
    Run a query and return the result.

    >>> import duckdb
    >>> con = duckdb.connect("db/mgi.db")
    >>> query = Query(from_table="gaf_association", limit=5)
    >>> result = run_query(con, query)
    >>> print(result.num_rows)
    532233

    :param con:
    :param query:
    :return:
    """
    count_query_str = query.sql(count=True)
    num_rows = con.execute(count_query_str).fetchall()[0][0]
    logger.debug(f"num_rows: {num_rows}")
    query_str = query.sql(**kwargs)
    logger.debug(f"query_str: {query_str}")
    rows = con.execute(query_str).fetchdf()
    qr = QueryResult(query=query, num_rows=num_rows)
    qr.set_rows(rows)
    return qr


@dataclass
class DuckDBDatabase(Database):

    _connection: DuckDBPyConnection = None
    _engine: sqlalchemy.Engine = None

    def __post_init__(self):
        if not self.handle:
            self.handle = "duckdb:///:memory:"

    @property
    def engine(self) -> sqlalchemy.Engine:
        if not self._engine:
            handle = self.handle
            if not handle.startswith("duckdb://") and not handle.startswith(":"):
                handle = f"duckdb://{handle}"
            self._engine = sqlalchemy.create_engine(handle)
        con = self._engine.connect()
        return self._engine

    def query(self, query: Query, **kwargs) -> QueryResult:
        with self.engine.connect() as conn:
            count_query_str = text(query.sql(count=True))
            num_rows = list(conn.execute(count_query_str))[0][0]
            logger.debug(f"num_rows: {num_rows}")
            query_str = query.sql(**kwargs)  # include offset, limit
            logger.debug(f"query_str: {query_str}")
            rows = list(conn.execute(text(query_str)).mappings())
            qr = QueryResult(query=query, num_rows=num_rows, rows=rows)
            qr.set_rows(pd.DataFrame(rows))
            return qr

    def init_collections(self):
        with self.engine.connect() as conn:
            query = Query(from_table="information_schema.tables", where_clause={"table_type": "BASE TABLE"})
            qr = self.query(query)
            table_names = [row["table_name"] for row in qr.rows]
        if self._collections is None:
            self._collections = {}
        for table_name in table_names:
            if table_name not in self._collections:
                collection = DuckDBCollection(name=table_name, parent=self)
                self._collections[table_name] = collection

    def create_collection(self, name: str, **kwargs) -> DuckDBCollection:
        collection = DuckDBCollection(name=name, parent=self)
        if not self._collections:
            self._collections = {}
        self._collections[name] = collection
        return collection

    def induce_schema_view(self) -> SchemaView:
        sb = SchemaBuilder()
        schema = sb.schema
        query = Query(from_table="information_schema.tables", where_clause={"table_type": "BASE TABLE"})
        qr = self.query(query)
        if qr.num_rows:
            table_names = [row["table_name"] for row in qr.rows]
            for tbl in table_names:
                sb.add_class(tbl)
            query = Query(from_table="information_schema.columns", sort_by=["ordinal_position"])
            for row in self.query(query).rows:
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
                sd = SlotDefinition(row["column_name"],
                                    required=row["is_nullable"] == "NO",
                                    multivalued=multivalued,
                                    range=rng)
                sb.schema.classes[tbl_name].attributes[sd.name] = sd
        sb.add_defaults()
        return SchemaView(schema)
