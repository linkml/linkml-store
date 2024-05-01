import logging
from typing import Any, Dict, List, Optional, Union

import sqlalchemy as sqla
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from sqlalchemy import Column, Table, delete, insert, inspect, text
from sqlalchemy.sql.ddl import CreateTable

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query
from linkml_store.api.stores.duckdb.mappings import TMAP
from linkml_store.utils.sql_utils import facet_count_sql

logger = logging.getLogger(__name__)


class DuckDBCollection(Collection):
    _table_created: bool = None

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return
        cd = self.class_definition()
        if not cd:
            cd = self.induce_class_definition_from_objects(objs)
        self._create_table(cd)
        table = self._sqla_table(cd)
        logger.info(f"Inserting into: {self._alias} // T={table.name}")
        engine = self.parent.engine
        col_names = [c.name for c in table.columns]
        objs = [{k: obj.get(k, None) for k in col_names} for obj in objs]
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(insert(table), objs)
            conn.commit()

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        if not isinstance(objs, list):
            objs = [objs]
        cd = self.class_definition()
        if not cd:
            cd = self.induce_class_definition_from_objects(objs)
        table = self._sqla_table(cd)
        engine = self.parent.engine
        with engine.connect() as conn:
            for obj in objs:
                conditions = [table.c[k] == v for k, v in obj.items() if k in cd.attributes]
                stmt = delete(table).where(*conditions)
                stmt = stmt.compile(engine)
                conn.execute(stmt)
                conn.commit()
        return len(objs)

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> int:
        logger.info(f"Deleting from {self._target_class_name} where: {where}")
        if where is None:
            where = {}
        cd = self.class_definition()
        if not cd:
            logger.info(f"No class definition found for {self._target_class_name}, assuming not prepopulated")
            return 0
        table = self._sqla_table(cd)
        engine = self.parent.engine
        inspector = inspect(engine)
        table_exists = table.name in inspector.get_table_names()
        if not table_exists:
            logger.info(f"Table {table.name} does not exist, assuming no data")
            return 0
        with engine.connect() as conn:
            conditions = [table.c[k] == v for k, v in where.items()]
            stmt = delete(table).where(*conditions)
            stmt = stmt.compile(engine)
            result = conn.execute(stmt)
            deleted_rows_count = result.rowcount
            if deleted_rows_count == 0 and not missing_ok:
                raise ValueError(f"No rows found for {where}")
            conn.commit()
            return deleted_rows_count

    def query_facets(
        self, where: Dict = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[str, Dict[str, int]]:
        results = {}
        cd = self.class_definition()
        with self.parent.engine.connect() as conn:
            if not facet_columns:
                facet_columns = list(self.class_definition().attributes.keys())
            for col in facet_columns:
                logger.debug(f"Faceting on {col}")
                if isinstance(col, tuple):
                    sd = SlotDefinition(name="PLACEHOLDER")
                else:
                    sd = cd.attributes[col]
                facet_query = self._create_query(where_clause=where)
                facet_query_str = facet_count_sql(facet_query, col, multivalued=sd.multivalued)
                logger.debug(f"Facet query: {facet_query_str}")
                rows = list(conn.execute(text(facet_query_str)))
                results[col] = rows
            return results

    def _sqla_table(self, cd: ClassDefinition) -> Table:
        schema_view = self.parent.schema_view
        metadata_obj = sqla.MetaData()
        cols = []
        for att in schema_view.class_induced_slots(cd.name):
            typ = TMAP.get(att.range, sqla.String)
            if att.inlined:
                typ = sqla.JSON
            if att.multivalued:
                typ = sqla.ARRAY(typ, dimensions=1)
            if att.array:
                typ = sqla.ARRAY(typ, dimensions=1)
            col = Column(att.name, typ)
            cols.append(col)
        t = Table(self._alias, metadata_obj, *cols)
        return t

    def _create_table(self, cd: ClassDefinition):
        if self._table_created or self.metadata.is_prepopulated:
            logger.info(f"Already have table for: {cd.name}")
            return
        query = Query(
            from_table="information_schema.tables", where_clause={"table_type": "BASE TABLE", "table_name": self._alias}
        )
        qr = self.parent.query(query)
        if qr.num_rows > 0:
            logger.info(f"Table already exists for {cd.name}")
            self._table_created = True
            self.metadata.is_prepopulated = True
            return
        logger.info(f"Creating table for {cd.name}")
        t = self._sqla_table(cd)
        ct = CreateTable(t)
        ddl = str(ct.compile(self.parent.engine))
        with self.parent.engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()
        self._table_created = True
        self.metadata.is_prepopulated = True
