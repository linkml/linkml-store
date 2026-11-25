import logging
from typing import Any, Optional, Tuple, Type, Union

import sqlalchemy
import sqlalchemy.sql.sqltypes as sqlt
from linkml_runtime.linkml_model import SchemaDefinition, SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from sqlalchemy import MetaData, quoted_name

from linkml_store.api.queries import Query

logger = logging.getLogger(__name__)

TYPE_MAP = {
    sqlt.TEXT: "string",
    sqlt.INTEGER: "integer",
    sqlt.FLOAT: "float",
}

OP_MAP = {
    "eq": "=",
    "$in": "ARRAY_CONTAINS",  ## mongodb
    "$contains": "ARRAY_CONTAINS",  ## TODO: this is chromadb-specific
    "in": "ARRAY_CONTAINS",
}

# MongoDB-style comparison operators and their SQL equivalents
COMPARISON_OP_MAP = {
    "$eq": "=",
    "$ne": "<>",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def _map_type(typ: Type) -> str:
    for k, v in TYPE_MAP.items():
        if isinstance(typ, k):
            return v
    return "string"


def where_clause_to_sql(query: Query) -> str:
    """
    Translate the where clause of a query into a SQL WHERE clause.

    >>> where_clause_to_sql(Query(where_clause={"name": "O'Neil", "age": {"$gte": 21, "$lt": 65}}))
    "WHERE name = 'O''Neil' AND age >= 21 AND age < 65"
    >>> where_clause_to_sql(Query(where_clause={"name": None}))
    'WHERE name IS NULL'
    >>> where_clause_to_sql(Query(where_clause={}))
    ''

    :param query:
    :return: SQL WHERE clause, or an empty string if there are no conditions
    """
    where_clause = query.where_clause
    if not where_clause:
        return ""
    if isinstance(where_clause, str):
        conjs = [where_clause]
    elif isinstance(where_clause, list):
        conjs = where_clause
    elif isinstance(where_clause, dict):
        conjs = [conj for k, v in where_clause.items() for conj in col_val_constraints_to_conjs(k, v)]
    else:
        raise ValueError(f"Invalid where_clause type: {type(where_clause)}")
    if not conjs:
        return ""
    return "WHERE " + " AND ".join(conjs)


def col_val_constraints_to_conjs(col_name: str, val_constraints: Any) -> list:
    """
    Translate the constraints on a single column into a list of SQL conditions.

    A plain value is an equality test, and None tests for NULL:

    >>> col_val_constraints_to_conjs("name", "Alice")
    ["name = 'Alice'"]
    >>> col_val_constraints_to_conjs("name", None)
    ['name IS NULL']

    A dict holds MongoDB-style operators:

    >>> col_val_constraints_to_conjs("age", {"$gt": 21, "$ne": 30})
    ['age > 21', 'age <> 30']
    >>> col_val_constraints_to_conjs("id", {"$in": [1, 2]})
    ['id IN (1, 2)']
    >>> col_val_constraints_to_conjs("id", {"$nin": ["a", "b"]})
    ["id NOT IN ('a', 'b')"]
    >>> col_val_constraints_to_conjs("name", {"$ne": None})
    ['name IS NOT NULL']
    >>> col_val_constraints_to_conjs("age", {"$near": 21})
    Traceback (most recent call last):
    ...
    ValueError: Unsupported operator for age: $near

    :param col_name:
    :param val_constraints:
    :return: list of SQL conditions
    """

    def _quote(v: Any) -> str:
        if isinstance(v, str):
            # escape internal quotes
            v = v.replace("'", "''")
            return f"'{v}'"
        if v is None:
            return "NULL"
        return str(v)

    if val_constraints is None:
        return [f"{col_name} IS NULL"]
    if isinstance(val_constraints, dict):
        conjs = []
        for k, v in val_constraints.items():
            if k in ("$in", "$nin") and isinstance(v, (list, tuple)):
                sql_op = "IN" if k == "$in" else "NOT IN"
                t = f"{col_name} {sql_op} ({', '.join(_quote(v1) for v1 in v)})"
            elif k in COMPARISON_OP_MAP:
                if v is None and k in ("$eq", "$ne"):
                    t = f"{col_name} IS NULL" if k == "$eq" else f"{col_name} IS NOT NULL"
                else:
                    t = f"{col_name} {COMPARISON_OP_MAP[k]} {_quote(v)}"
            elif k in OP_MAP:
                t = f"{OP_MAP[k]}({col_name}, {_quote(v)})"
            elif k.startswith("$"):
                raise ValueError(f"Unsupported operator for {col_name}: {k}")
            else:
                # a native SQL operator, e.g. {"name": {"LIKE": "A%"}}
                t = f"{col_name} {k} {_quote(v)}"
            conjs.append(t)
        return conjs
    else:
        return [f"{col_name} = {_quote(val_constraints)}"]


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
        if limit < 0:
            limit = None
        if limit is not None:
            sql_str.append(f" LIMIT {limit}")
        offset = offset if offset else query.offset
        if offset:
            sql_str.append(f" OFFSET {offset}")
    sql_str = [line for line in sql_str if line]
    return "\n".join(sql_str)


def facet_count_sql(query: Query, facet_column: Union[str, Tuple[str, ...]], multivalued=False, limit=100) -> str:
    """
    Generate SQL to count the occurrences of each value of a column (or combination of columns).

    >>> q = Query(from_table="person", where_clause={"city": "Paris"})
    >>> print(facet_count_sql(q, "job", limit=5))
    SELECT job, COUNT(*) as count
    FROM person WHERE city = 'Paris'
    GROUP BY job
    ORDER BY count DESC
    LIMIT 5

    :param query: base query
    :param facet_column: column name, or a tuple of column names for a combined facet
    :param multivalued: whether the column holds lists, in which case each element is counted
    :param limit: maximum number of facet values to return
    :return: SQL string
    """
    # Create a modified WHERE clause that excludes conditions directly related to facet_column
    # TODO: the first condition is never excluded (it starts with "WHERE "); MongoDB and the
    # filesystem adapter exclude nothing. Decide on one behavior for all adapters.
    modified_where = None
    if query.where_clause:
        where_clause_sql = where_clause_to_sql(query)
        # Split the where clause into conditions and exclude those related to the facet_column
        conditions = [cond for cond in where_clause_sql.split(" AND ") if not cond.startswith(f"{facet_column} ")]
        modified_where = " AND ".join(conditions)

    def make_col_safe(col):
        return '"' + quoted_name(col, True) + '"' if " " in col else col

    if isinstance(facet_column, str):
        facet_column = make_col_safe(facet_column)
    if isinstance(facet_column, tuple):
        facet_column = [make_col_safe(col) for col in facet_column]
        if multivalued:
            raise NotImplementedError("Multivalued facets are not supported for multiple columns")
        facet_column = ", ".join(facet_column)
    from_table = query.from_table
    if multivalued:
        from_table = f"(SELECT UNNEST({facet_column}) as {facet_column} FROM {query.from_table}"
        from_table += f" {modified_where}" if modified_where else ""
        from_table += ")"
    else:
        from_table += f" {modified_where}" if modified_where else ""
    sql_str = [f"SELECT {facet_column}, COUNT(*) as count", f"FROM {from_table}"]
    # if modified_where:
    #    sql_str.append(f"{modified_where}")
    sql_str.append(f"GROUP BY {facet_column}")
    sql_str.append("ORDER BY count DESC")  # Optional, order by count for convenience
    if limit is not None:
        sql_str.append(f"LIMIT {limit}")
    return "\n".join(sql_str)


def introspect_schema(engine: sqlalchemy.Engine) -> SchemaDefinition:
    """
    Introspect a database schema and return a SchemaDefinition object

    :param engine:
    :return:
    """
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
                    [fk_table, fk_table_col] = str(fk.column).split(".")
                    slot.range = fk_table
            else:
                slot.range = _map_type(column.type)
    return schema
