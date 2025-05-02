import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import sqlalchemy as sqla
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from sqlalchemy import Column, Table, delete, insert, inspect, text
from sqlalchemy.sql.ddl import CreateTable

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.duckdb.mappings import TMAP
from linkml_store.utils.sql_utils import facet_count_sql

logger = logging.getLogger(__name__)


class DuckDBCollection(Collection):
    _table_created: bool = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        logger.debug(f"Inserting {len(objs)}")
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return
        cd = self.class_definition()
        if not cd:
            logger.debug(f"No class definition defined for {self.alias} {self.target_class_name}; will induce")
            cd = self.induce_class_definition_from_objects(objs)
        self._create_table(cd)
        table = self._sqla_table(cd)
        logger.info(f"Inserting into: {self.alias} // T={table.name}")
        engine = self.parent.engine
        col_names = [c.name for c in table.columns]
        bad_objs = [obj for obj in objs if not isinstance(obj, dict)]
        if bad_objs:
            logger.error(f"Bad objects: {bad_objs}")
        objs = [{k: obj.get(k, None) for k in col_names} for obj in objs]
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(insert(table), objs)
            conn.commit()
        self._post_insert_hook(objs)

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> Optional[int]:
        if not isinstance(objs, list):
            objs = [objs]
        cd = self.class_definition()
        if not cd or not cd.attributes:
            cd = self.induce_class_definition_from_objects(objs)
        assert cd.attributes
        table = self._sqla_table(cd)
        engine = self.parent.engine
        with engine.connect() as conn:
            for obj in objs:
                conditions = [table.c[k] == v for k, v in obj.items() if k in cd.attributes]
                stmt = delete(table).where(*conditions)
                stmt = stmt.compile(engine)
                conn.execute(stmt)
                conn.commit()
        self._post_delete_hook()
        return None

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> Optional[int]:
        logger.info(f"Deleting from {self.target_class_name} where: {where}")
        if where is None:
            where = {}
        cd = self.class_definition()
        if not cd:
            logger.info(f"No class definition found for {self.target_class_name}, assuming not prepopulated")
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
            self._post_delete_hook()
            return deleted_rows_count if deleted_rows_count > -1 else None

    def query_facets(
        self, where: Dict = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[Union[str, Tuple[str, ...]], List[Tuple[Any, int]]]:
        if facet_limit is None:
            facet_limit = DEFAULT_FACET_LIMIT
        results = {}
        cd = self.class_definition()
        with self.parent.engine.connect() as conn:
            if not facet_columns:
                if not cd:
                    raise ValueError(f"No class definition found for {self.target_class_name}")
                facet_columns = list(cd.attributes.keys())
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
                results[col] = [tuple(row) for row in rows]
            return results

    def _sqla_table(self, cd: ClassDefinition) -> Table:
        schema_view = self.parent.schema_view
        metadata_obj = sqla.MetaData()
        cols = []
        for att in schema_view.class_induced_slots(cd.name):
            typ = TMAP.get(att.range, sqla.String)
            if att.inlined or att.inlined_as_list:
                typ = sqla.JSON
            if att.multivalued:
                typ = sqla.ARRAY(typ, dimensions=1)
            if att.array:
                typ = sqla.ARRAY(typ, dimensions=1)
            col = Column(att.name, typ)
            cols.append(col)
        t = Table(self.alias, metadata_obj, *cols)
        return t

    def _check_if_initialized(self) -> bool:
        # if self._initialized:
        #    return True
        query = Query(
            from_table="information_schema.tables", where_clause={"table_type": "BASE TABLE", "table_name": self.alias}
        )
        qr = self.parent.query(query)
        if qr.num_rows > 0:
            return True
        return False

    def group_by(
        self,
        group_by_fields: List[str],
        inlined_field="objects",
        agg_map: Optional[Dict[str, str]] = None,
        where: Optional[Dict] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Group objects in the collection by specified fields using SQLAlchemy.

        This implementation leverages DuckDB's SQL capabilities for more efficient grouping.

        :param group_by_fields: List of fields to group by
        :param inlined_field: Field name to store aggregated objects
        :param agg_map: Dictionary mapping aggregation types to fields
        :param where: Filter conditions
        :param kwargs: Additional arguments
        :return: Query result containing grouped data
        """
        if isinstance(group_by_fields, str):
            group_by_fields = [group_by_fields]

        cd = self.class_definition()
        if not cd:
            logger.debug(f"No class definition defined for {self.alias} {self.target_class_name}")
            return super().group_by(group_by_fields, inlined_field, agg_map, where, **kwargs)

        # Check if the table exists
        if not self.parent._table_exists(self.alias):
            logger.debug(f"Table {self.alias} doesn't exist, falling back to parent implementation")
            return super().group_by(group_by_fields, inlined_field, agg_map, where, **kwargs)

        # Get table definition
        table = self._sqla_table(cd)
        engine = self.parent.engine

        # Create a SQLAlchemy select statement for groups
        from sqlalchemy import select

        group_cols = [table.c[field] for field in group_by_fields if field in table.columns.keys()]

        if not group_cols:
            logger.warning(f"None of the group_by fields {group_by_fields} found in table columns")
            return super().group_by(group_by_fields, inlined_field, agg_map, where, **kwargs)

        stmt = select(*group_cols).distinct()

        # Add where conditions if specified
        if where:
            conditions = []
            for k, v in where.items():
                if k in table.columns.keys():
                    # Handle different operator types (dict values for operators)
                    if isinstance(v, dict):
                        for op, val in v.items():
                            if op == "$gt":
                                conditions.append(table.c[k] > val)
                            elif op == "$gte":
                                conditions.append(table.c[k] >= val)
                            elif op == "$lt":
                                conditions.append(table.c[k] < val)
                            elif op == "$lte":
                                conditions.append(table.c[k] <= val)
                            elif op == "$ne":
                                conditions.append(table.c[k] != val)
                            elif op == "$in":
                                conditions.append(table.c[k].in_(val))
                            else:
                                # Default to equality for unknown operators
                                logger.warning(f"Unknown operator {op}, using equality")
                                conditions.append(table.c[k] == val)
                    else:
                        # Direct equality comparison
                        conditions.append(table.c[k] == v)

            if conditions:
                for condition in conditions:
                    stmt = stmt.where(condition)

        results = []
        try:
            with engine.connect() as conn:
                # Get all distinct groups
                group_result = conn.execute(stmt)
                group_rows = list(group_result)

                # For each group, get all objects
                for group_row in group_rows:
                    # Build conditions for this group
                    group_conditions = []
                    group_dict = {}

                    for i, field in enumerate(group_by_fields):
                        if field in table.columns.keys():
                            value = group_row[i]
                            group_dict[field] = value
                            if value is None:
                                group_conditions.append(table.c[field].is_(None))
                            else:
                                group_conditions.append(table.c[field] == value)

                    # Get all rows for this group
                    row_stmt = select(*table.columns)
                    for condition in group_conditions:
                        row_stmt = row_stmt.where(condition)

                    # Add original where conditions
                    if where:
                        for k, v in where.items():
                            if k in table.columns.keys():
                                # Handle different operator types for the row query as well
                                if isinstance(v, dict):
                                    for op, val in v.items():
                                        if op == "$gt":
                                            row_stmt = row_stmt.where(table.c[k] > val)
                                        elif op == "$gte":
                                            row_stmt = row_stmt.where(table.c[k] >= val)
                                        elif op == "$lt":
                                            row_stmt = row_stmt.where(table.c[k] < val)
                                        elif op == "$lte":
                                            row_stmt = row_stmt.where(table.c[k] <= val)
                                        elif op == "$ne":
                                            row_stmt = row_stmt.where(table.c[k] != val)
                                        elif op == "$in":
                                            row_stmt = row_stmt.where(table.c[k].in_(val))
                                        else:
                                            # Default to equality for unknown operators
                                            row_stmt = row_stmt.where(table.c[k] == val)
                                else:
                                    # Direct equality comparison
                                    row_stmt = row_stmt.where(table.c[k] == v)

                    row_result = conn.execute(row_stmt)
                    rows = list(row_result)

                    # Convert rows to dictionaries
                    objects = []
                    for row in rows:
                        obj = {}
                        for i, col in enumerate(row._fields):
                            obj[col] = row[i]
                        objects.append(obj)

                    # Apply agg_map to filter fields if specified
                    if agg_map and "list" in agg_map:
                        list_fields = agg_map["list"]
                        if list_fields:
                            objects = [{k: obj.get(k) for k in list_fields if k in obj} for obj in objects]

                    # Create the result object
                    result_obj = group_dict.copy()
                    result_obj[inlined_field] = objects
                    results.append(result_obj)

                return QueryResult(num_rows=len(results), rows=results)
        except Exception as e:
            logger.warning(f"Error in DuckDB group_by: {e}")
            # Fall back to parent implementation
            return super().group_by(group_by_fields, inlined_field, agg_map, where, **kwargs)

    def _create_table(self, cd: ClassDefinition):
        if self._table_created or self.metadata.is_prepopulated:
            logger.info(f"Already have table for: {cd.name}")
            return
        if self.parent._table_exists(self.alias):
            logger.info(f"Table already exists for {cd.name}")
            self._table_created = True
            self._initialized = True
            self.metadata.is_prepopulated = True
            return
        # query = Query(
        #     from_table="information_schema.tables",
        #     where_clause={"table_type": "BASE TABLE", "table_name": self.alias}
        # )
        # qr = self.parent.query(query)
        # if qr.num_rows > 0:
        #     logger.info(f"Table already exists for {cd.name}")
        #     self._table_created = True
        #     self._initialized = True
        #     self.metadata.is_prepopulated = True
        #     return
        logger.info(f"Creating table for {cd.name}")
        t = self._sqla_table(cd)
        ct = CreateTable(t)
        ddl = str(ct.compile(self.parent.engine))
        with self.parent.engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()
        self._table_created = True
        self._initialized = True
        self.metadata.is_prepopulated = True
