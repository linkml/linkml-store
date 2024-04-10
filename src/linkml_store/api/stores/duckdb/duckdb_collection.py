from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import sqlalchemy as sqla
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from sqlalchemy import Column, Table, delete, insert, text
from sqlalchemy.sql.ddl import CreateTable

from linkml_store.api import Collection
from linkml_store.api.collection import OBJECT
from linkml_store.api.stores.duckdb.mappings import TMAP
from linkml_store.utils.sql_utils import facet_count_sql


@dataclass
class DuckDBCollection(Collection):
    _table_created: bool = None

    def add(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return
        cd = self.class_definition()
        if not cd:
            cd = self.induce_class_definition_from_objects(objs)
        self._create_table(cd)
        table = self._sqla_table(cd)
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

    def delete_where(self, where: Optional[Dict[str, Any]] = None, **kwargs) -> int:
        cd = self.class_definition()
        table = self._sqla_table(cd)
        engine = self.parent.engine
        with engine.connect() as conn:
            conditions = [table.c[k] == v for k, v in where.items()]
            stmt = delete(table).where(*conditions)
            stmt = stmt.compile(engine)
            conn.execute(stmt)
            conn.commit()
        return 0

    def query_facets(self, where: Dict = None, facet_columns: List[str] = None) -> Dict[str, Dict[str, int]]:
        results = {}
        cd = self.class_definition()
        with self.parent.engine.connect() as conn:
            if not facet_columns:
                facet_columns = list(self.class_definition().attributes.keys())
            for col in facet_columns:
                if isinstance(col, tuple):
                    sd = SlotDefinition(name="PLACEHOLDER")
                else:
                    sd = cd.attributes[col]
                facet_query = self._create_query(where_clause=where)
                facet_query_str = facet_count_sql(facet_query, col, multivalued=sd.multivalued)
                rows = list(conn.execute(text(facet_query_str)))
                results[col] = rows
            return results

    def _sqla_table(self, cd: ClassDefinition) -> Table:
        metadata_obj = sqla.MetaData()
        cols = []
        for att in cd.attributes.values():
            typ = TMAP.get(att.range, sqla.String)
            if att.inlined:
                typ = sqla.JSON
            if att.multivalued:
                typ = sqla.ARRAY(typ, dimensions=1)
            if att.array:
                typ = sqla.ARRAY(typ, dimensions=1)
            col = Column(att.name, typ)
            cols.append(col)
        t = Table(self.name, metadata_obj, *cols)
        return t

    def _create_table(self, cd: ClassDefinition):
        if self._table_created:
            return
        t = self._sqla_table(cd)
        ct = CreateTable(t)
        ddl = str(ct.compile(self.parent.engine))
        with self.parent.engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()
        if not self._table_created:
            self._table_created = True
