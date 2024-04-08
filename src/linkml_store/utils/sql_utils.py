import logging
from typing import Optional, Tuple, Union, Type

import sqlalchemy
from linkml_runtime.linkml_model import SchemaDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from sqlalchemy import MetaData
import sqlalchemy.sql.sqltypes as sqlt


from linkml_store.api.queries import Query

logger = logging.getLogger(__name__)

TYPE_MAP = {
    sqlt.TEXT: 'string',
    sqlt.INTEGER: 'integer',
    sqlt.FLOAT: 'float',
}

def _map_type(typ: Type) -> str:
    for k, v in TYPE_MAP.items():
        if isinstance(typ, k):
            return v
    return 'string'

def where_clause_to_sql(query: Query) -> str:
    if not query.where_clause:
        return ""
    if isinstance(query.where_clause, str):
        where_clause_sql = query.where_clause
    elif isinstance(query.where_clause, list):
        where_clause_sql = " AND ".join(query.where_clause)
    elif isinstance(query.where_clause, dict):
        # TODO: bobby tables
        where_clause_sql = " AND ".join([f"{k} = '{v}'" for k, v in query.where_clause.items()])
    else:
        raise ValueError(f"Invalid where_clause type: {type(query.where_clause)}")
    return "WHERE " + where_clause_sql


def query_to_sql(query: Query, count=False, limit=None, offset: Optional[int] = None):
    select_cols = query.select_cols if query.select_cols else ["*"]
    if count:
        sql_str = ["SELECT COUNT(*)"]
    else:
        sql_str = [f"SELECT {', '.join(select_cols)}"]
    sql_str.append(f"FROM {query.from_table}")
    sql_str.append(where_clause_to_sql(query))
    if not count:
        if query.sort_by:
            sql_str.append(f"ORDER BY {', '.join(query.sort_by)}")
    if not count:
        if limit is None:
            limit = query.limit
        if limit is None:
            limit = 100
        if limit:
            sql_str.append(f" LIMIT {limit}")
        offset = offset if offset else query.offset
        if offset:
            sql_str.append(f" OFFSET {offset}")
    sql_str = [line for line in sql_str if line]
    return "\n".join(sql_str)


def facet_count_sql(query: Query, facet_column: Union[str, Tuple[str, ...]], multivalued=False) -> str:
    # Create a modified WHERE clause that excludes conditions directly related to facet_column
    modified_where = None
    if query.where_clause:
        where_clause_sql = where_clause_to_sql(query)
        # Split the where clause into conditions and exclude those related to the facet_column
        conditions = [cond for cond in where_clause_sql.split(" AND ") if not cond.startswith(f"{facet_column} ")]
        modified_where = " AND ".join(conditions)

    if isinstance(facet_column, tuple):
        if multivalued:
            raise NotImplementedError("Multivalued facets are not supported for multiple columns")
        facet_column = ", ".join(facet_column)
    from_table = query.from_table
    if multivalued:
        from_table = f"(SELECT UNNEST({facet_column}) as {facet_column} FROM {query.from_table})"
    sql_str = [f"SELECT {facet_column}, COUNT(*) as count",
               f"FROM {from_table}"]
    if modified_where:
        sql_str.append(f"{modified_where}")
    sql_str.append(f"GROUP BY {facet_column}")
    sql_str.append("ORDER BY count DESC")  # Optional, order by count for convenience
    return "\n".join(sql_str)


def introspect_schema(engine: sqlalchemy.Engine) -> SchemaDefinition:
    metadata_obj = MetaData()
    logging.info(f"Reflecting using {engine}")
    metadata_obj.reflect(bind=engine)
    sb = SchemaBuilder()
    schema = sb.schema
    for table in metadata_obj.sorted_tables:
        logging.info(f"Importing {table.name}")
        sb.add_class(table.name)
        cls = schema.classes[table.name]
        pks = [column for column in table.columns if column.primary_key]
        if len(pks) == 1:
            pk = pks.pop().name
        else:
            pk = None
        for column in table.columns:
            slot = SlotDefinition(column.name)
            cls.attributes[slot.name] = slot
            if pk and pk == column.name:
                slot.identifier = True
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    [fk_table, fk_table_col] = str(fk.column).split('.')
                    slot.range = fk_table
            else:
                slot.range = _map_type(column.type)
    return schema