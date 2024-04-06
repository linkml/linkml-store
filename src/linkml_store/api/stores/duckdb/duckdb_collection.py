from dataclasses import dataclass
from typing import Dict, List, Union

import sqlalchemy as sqla
from linkml_runtime.linkml_model import ClassDefinition
from sqlalchemy import Column, Table, insert, text
from sqlalchemy.sql.ddl import CreateTable

from linkml_store.api import Collection
from linkml_store.api.collection import OBJECT
from linkml_store.api.stores.duckdb.mappings import TMAP


@dataclass
class DuckDBCollection(Collection):

    _table_created: bool = None

    def add(self, objs: Union[OBJECT, List[OBJECT]]):
        if not isinstance(objs, list):
            objs = [objs]
        cd = self.class_definition()
        if not cd:
            cd = self.induce_class_definition_from_objects(objs)
        self._create_table(cd)
        table = self._sqla_table(cd)
        stmt = insert(table).values(objs)
        engine = self.parent.engine
        with engine.connect() as conn:
            result = conn.execute(stmt)
            conn.commit()

    def query_facets(self, where: Dict, facet_columns: List[str]) -> Dict[str, Dict[str, int]]:
        results = {}
        conn = self.parent.engine.connect()
        for col in facet_columns:
            facet_query = self._create_query(where=where)
            facet_query_str = facet_query.facet_count_sql(col)
            rows = list(conn.execute(text(facet_query_str)))
            results[col] = rows
        return results

    def _sqla_table(self, cd: ClassDefinition) -> Table:
        metadata_obj = sqla.MetaData()
        cols = []
        for att in cd.attributes.values():
            col = Column(att.name, TMAP.get(att.range, sqla.String))
            cols.append(col)
        t = Table(self.name, metadata_obj, *cols)
        return t

    def _create_table(self, cd: ClassDefinition):
        if self._table_created:
            return
        t = self._sqla_table(cd)
        ddl = str(CreateTable(t))
        with self.parent.engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()
        if not self._table_created:
            self._table_created = True
