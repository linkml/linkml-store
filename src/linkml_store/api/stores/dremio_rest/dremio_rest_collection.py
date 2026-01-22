"""Dremio REST API collection implementation.

This module provides the Collection implementation for Dremio REST API,
supporting query operations via the REST API v3.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult

logger = logging.getLogger(__name__)


class DremioRestCollection(Collection):
    """Collection implementation for Dremio data lakehouse via REST API.

    This collection connects to Dremio tables via the REST API v3
    and provides query capabilities. Write operations may be limited
    depending on the underlying data source configuration in Dremio.
    """

    _table_exists_checked: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_table_path(self) -> str:
        """Get the full qualified table path.

        Returns:
            Full table path for SQL queries.
        """
        return self.parent._get_table_path(self.alias)

    def _build_select_sql(
        self,
        select_cols: Optional[List[str]] = None,
        where_clause: Optional[Union[str, Dict[str, Any]]] = None,
        sort_by: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> str:
        """Build a SELECT SQL statement.

        Args:
            select_cols: Columns to select (None for all).
            where_clause: WHERE conditions.
            sort_by: ORDER BY columns.
            limit: Maximum rows to return.
            offset: Number of rows to skip.

        Returns:
            SQL SELECT statement.
        """
        table_path = self._get_table_path()

        # Build SELECT clause
        if select_cols:
            cols = ", ".join(f'"{c}"' for c in select_cols)
        else:
            cols = "*"

        sql = f"SELECT {cols} FROM {table_path}"

        # Build WHERE clause
        if where_clause:
            conditions = self._build_where_conditions(where_clause)
            if conditions:
                sql += f" WHERE {conditions}"

        # Build ORDER BY clause
        if sort_by:
            order_cols = ", ".join(f'"{c}"' for c in sort_by)
            sql += f" ORDER BY {order_cols}"

        # Build LIMIT/OFFSET
        if limit is not None and limit >= 0:
            sql += f" LIMIT {limit}"
        if offset is not None and offset > 0:
            sql += f" OFFSET {offset}"

        return sql

    def _build_where_conditions(self, where_clause: Union[str, Dict[str, Any]]) -> str:
        """Build WHERE clause conditions from a dict or string.

        Args:
            where_clause: WHERE conditions as dict or string.

        Returns:
            SQL WHERE clause (without WHERE keyword).
        """
        if isinstance(where_clause, str):
            return where_clause

        if not isinstance(where_clause, dict):
            return ""

        conditions = []
        for key, value in where_clause.items():
            condition = self._build_single_condition(key, value)
            if condition:
                conditions.append(condition)

        return " AND ".join(conditions)

    def _build_single_condition(self, key: str, value: Any) -> str:
        """Build a single WHERE condition.

        Supports MongoDB-style operators like $gt, $gte, $lt, $lte, $in, $ne.

        Args:
            key: Column name.
            value: Value or operator dict.

        Returns:
            SQL condition string.
        """
        col = f'"{key}"'

        if value is None:
            return f"{col} IS NULL"

        if isinstance(value, dict):
            # Handle operators
            sub_conditions = []
            for op, val in value.items():
                if op == "$gt":
                    sub_conditions.append(f"{col} > {self._sql_value(val)}")
                elif op == "$gte":
                    sub_conditions.append(f"{col} >= {self._sql_value(val)}")
                elif op == "$lt":
                    sub_conditions.append(f"{col} < {self._sql_value(val)}")
                elif op == "$lte":
                    sub_conditions.append(f"{col} <= {self._sql_value(val)}")
                elif op == "$ne":
                    if val is None:
                        sub_conditions.append(f"{col} IS NOT NULL")
                    else:
                        sub_conditions.append(f"{col} != {self._sql_value(val)}")
                elif op == "$in":
                    if isinstance(val, (list, tuple)):
                        vals = ", ".join(self._sql_value(v) for v in val)
                        sub_conditions.append(f"{col} IN ({vals})")
                elif op == "$nin":
                    if isinstance(val, (list, tuple)):
                        vals = ", ".join(self._sql_value(v) for v in val)
                        sub_conditions.append(f"{col} NOT IN ({vals})")
                elif op == "$like":
                    sub_conditions.append(f"{col} LIKE {self._sql_value(val)}")
                elif op == "$ilike":
                    sub_conditions.append(f"LOWER({col}) LIKE LOWER({self._sql_value(val)})")
                elif op == "$regex":
                    # Dremio uses REGEXP_LIKE
                    sub_conditions.append(f"REGEXP_LIKE({col}, {self._sql_value(val)})")
                else:
                    logger.warning(f"Unknown operator: {op}")

            return " AND ".join(sub_conditions) if sub_conditions else ""
        else:
            return f"{col} = {self._sql_value(value)}"

    def _sql_value(self, value: Any) -> str:
        """Convert a Python value to SQL literal.

        Args:
            value: Python value.

        Returns:
            SQL literal string.
        """
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, (list, dict)):
            # Convert to JSON string
            escaped = json.dumps(value).replace("'", "''")
            return f"'{escaped}'"
        else:
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """Insert objects into the collection.

        Note: Write operations in Dremio depend on the underlying data source.
        Some sources (like Iceberg, Delta Lake) support writes, while others
        (like file-based sources) may not.

        Args:
            objs: Object(s) to insert.
            **kwargs: Additional arguments.
        """
        if not isinstance(objs, list):
            objs = [objs]

        if not objs:
            return

        logger.debug(f"Inserting {len(objs)} objects into {self.alias}")

        cd = self.class_definition()
        if not cd:
            logger.debug(f"No class definition for {self.alias}; inducing from objects")
            cd = self.induce_class_definition_from_objects(objs)

        table_path = self._get_table_path()

        if cd and cd.attributes:
            columns = list(cd.attributes.keys())
        else:
            columns = list(objs[0].keys())

        col_list = ", ".join(f'"{c}"' for c in columns)

        batch_size = 100
        for i in range(0, len(objs), batch_size):
            batch = objs[i : i + batch_size]

            values_list = []
            for obj in batch:
                values = []
                for col in columns:
                    val = obj.get(col)
                    values.append(self._sql_value(val))
                values_list.append(f"({', '.join(values)})")

            values_sql = ", ".join(values_list)
            sql = f"INSERT INTO {table_path} ({col_list}) VALUES {values_sql}"

            self.parent._execute_update(sql)

        self._post_insert_hook(objs)

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> Optional[int]:
        """Delete specific objects from the collection.

        Args:
            objs: Object(s) to delete.
            **kwargs: Additional arguments.

        Returns:
            Number of deleted rows, or None if unknown.
        """
        if not isinstance(objs, list):
            objs = [objs]

        if not objs:
            return 0

        table_path = self._get_table_path()
        total_deleted = 0

        for obj in objs:
            conditions = []
            for key, value in obj.items():
                if key.startswith("_"):
                    continue
                condition = self._build_single_condition(key, value)
                if condition:
                    conditions.append(condition)

            if not conditions:
                continue

            sql = f"DELETE FROM {table_path} WHERE {' AND '.join(conditions)}"
            result = self.parent._execute_update(sql)
            if result > 0:
                total_deleted += result

        self._post_delete_hook()
        return total_deleted if total_deleted > 0 else None

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> Optional[int]:
        """Delete objects matching a condition.

        Args:
            where: WHERE conditions (empty dict means delete all).
            missing_ok: If True, don't raise error if no rows deleted.
            **kwargs: Additional arguments.

        Returns:
            Number of deleted rows, or None if unknown.
        """
        if where is None:
            where = {}

        table_path = self._get_table_path()

        if where:
            conditions = self._build_where_conditions(where)
            sql = f"DELETE FROM {table_path} WHERE {conditions}"
        else:
            sql = f"DELETE FROM {table_path}"

        result = self.parent._execute_update(sql)
        if result == 0 and not missing_ok:
            raise ValueError(f"No rows found for {where}")
        self._post_delete_hook()
        return result if result >= 0 else None

    def update(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """Update objects in the collection.

        Note: Requires a primary key field to identify rows.

        Args:
            objs: Object(s) to update.
            **kwargs: Additional arguments.
        """
        if not isinstance(objs, list):
            objs = [objs]

        if not objs:
            return

        table_path = self._get_table_path()
        pk = self.identifier_attribute_name

        if not pk:
            raise ValueError("Cannot update without an identifier attribute")

        for obj in objs:
            if pk not in obj:
                raise ValueError(f"Object missing primary key field: {pk}")

            pk_value = obj[pk]

            set_parts = []
            for key, value in obj.items():
                if key == pk or key.startswith("_"):
                    continue
                set_parts.append(f'"{key}" = {self._sql_value(value)}')

            if not set_parts:
                continue

            set_clause = ", ".join(set_parts)
            sql = f'UPDATE {table_path} SET {set_clause} WHERE "{pk}" = {self._sql_value(pk_value)}'
            self.parent._execute_update(sql)

    def query(self, query: Query, **kwargs) -> QueryResult:
        """Execute a query against the collection.

        Args:
            query: Query specification.
            **kwargs: Additional arguments.

        Returns:
            QueryResult with matching rows.
        """
        self._pre_query_hook(query)

        limit = query.limit
        if limit == -1:
            limit = None

        sql = self._build_select_sql(
            select_cols=query.select_cols,
            where_clause=query.where_clause,
            sort_by=query.sort_by,
            limit=limit,
            offset=query.offset,
        )

        df = self.parent._execute_query(sql)

        # Convert DataFrame to list of dicts
        row_list = df.to_dict("records") if not df.empty else []

        # Get total count for pagination
        if query.offset or (limit is not None and len(row_list) == limit):
            count_sql = self._build_count_sql(query.where_clause)
            try:
                count_df = self.parent._execute_query(count_sql)
                total_rows = int(count_df.iloc[0, 0]) if not count_df.empty else len(row_list)
            except Exception:
                total_rows = len(row_list)
        else:
            total_rows = len(row_list)

        qr = QueryResult(query=query, num_rows=total_rows, rows=row_list, offset=query.offset or 0)

        if query.include_facet_counts and query.facet_slots:
            qr.facet_counts = self.query_facets(where=query.where_clause, facet_columns=query.facet_slots)

        return qr

    def _build_count_sql(self, where_clause: Optional[Union[str, Dict[str, Any]]] = None) -> str:
        """Build a COUNT SQL statement.

        Args:
            where_clause: WHERE conditions.

        Returns:
            SQL COUNT statement.
        """
        table_path = self._get_table_path()
        sql = f"SELECT COUNT(*) FROM {table_path}"

        if where_clause:
            conditions = self._build_where_conditions(where_clause)
            if conditions:
                sql += f" WHERE {conditions}"

        return sql

    def query_facets(
        self,
        where: Optional[Dict] = None,
        facet_columns: Optional[List[str]] = None,
        facet_limit: int = DEFAULT_FACET_LIMIT,
        **kwargs,
    ) -> Dict[Union[str, Tuple[str, ...]], List[Tuple[Any, int]]]:
        """Get facet counts for columns.

        Args:
            where: Filter conditions.
            facet_columns: Columns to get facets for.
            facet_limit: Maximum facet values per column.
            **kwargs: Additional arguments.

        Returns:
            Dictionary mapping column names to list of (value, count) tuples.
        """
        if facet_limit is None:
            facet_limit = DEFAULT_FACET_LIMIT

        results = {}
        cd = self.class_definition()
        table_path = self._get_table_path()

        if not facet_columns:
            if cd and cd.attributes:
                facet_columns = list(cd.attributes.keys())
            else:
                return results

        for col in facet_columns:
            if isinstance(col, tuple):
                col_list = ", ".join(f'"{c}"' for c in col)
                col_name = col
            else:
                col_list = f'"{col}"'
                col_name = col

            sql = f"SELECT {col_list}, COUNT(*) as cnt FROM {table_path}"

            if where:
                conditions = self._build_where_conditions(where)
                if conditions:
                    sql += f" WHERE {conditions}"

            sql += f" GROUP BY {col_list} ORDER BY cnt DESC"

            if facet_limit > 0:
                sql += f" LIMIT {facet_limit}"

            try:
                df = self.parent._execute_query(sql)

                facets = []
                for _, row in df.iterrows():
                    if isinstance(col, tuple):
                        value = tuple(row[c] for c in col)
                    else:
                        value = row[col]
                    count = int(row["cnt"])
                    facets.append((value, count))

                results[col_name] = facets

            except Exception as e:
                logger.warning(f"Facet query failed for {col}: {e}")
                results[col_name] = []

        return results

    def _check_if_initialized(self) -> bool:
        """Check if the collection's table exists.

        Returns:
            True if table exists.
        """
        if self._table_exists_checked:
            return True

        result = self.parent._table_exists(self.alias)
        if result:
            self._table_exists_checked = True
        return result
