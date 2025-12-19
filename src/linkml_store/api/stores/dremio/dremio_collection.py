"""Dremio collection implementation.

This module provides the Collection implementation for Dremio,
supporting CRUD operations and queries via Arrow Flight SQL.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from linkml_runtime.linkml_model import ClassDefinition

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult

logger = logging.getLogger(__name__)


class DremioCollection(Collection):
    """Collection implementation for Dremio data lakehouse.

    This collection connects to Dremio tables via Arrow Flight SQL
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

        # Get or induce class definition
        cd = self.class_definition()
        if not cd:
            logger.debug(f"No class definition for {self.alias}; inducing from objects")
            cd = self.induce_class_definition_from_objects(objs)

        table_path = self._get_table_path()

        # Get column names from class definition or first object
        if cd and cd.attributes:
            columns = list(cd.attributes.keys())
        else:
            columns = list(objs[0].keys())

        # Build INSERT statement
        col_list = ", ".join(f'"{c}"' for c in columns)

        # Insert objects in batches
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

            try:
                self.parent._execute_update(sql)
            except Exception as e:
                logger.error(f"Insert failed: {e}")
                raise

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
            # Build WHERE clause from object fields
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

            try:
                result = self.parent._execute_update(sql)
                if result > 0:
                    total_deleted += result
            except Exception as e:
                logger.error(f"Delete failed: {e}")
                raise

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
            # Delete all
            sql = f"DELETE FROM {table_path}"

        try:
            result = self.parent._execute_update(sql)
            if result == 0 and not missing_ok:
                raise ValueError(f"No rows found for {where}")
            self._post_delete_hook()
            return result if result >= 0 else None
        except Exception as e:
            if "does not exist" in str(e).lower():
                if missing_ok:
                    return 0
            raise

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

            # Build SET clause (exclude primary key)
            set_parts = []
            for key, value in obj.items():
                if key == pk or key.startswith("_"):
                    continue
                set_parts.append(f'"{key}" = {self._sql_value(value)}')

            if not set_parts:
                continue

            set_clause = ", ".join(set_parts)
            sql = f'UPDATE {table_path} SET {set_clause} WHERE "{pk}" = {self._sql_value(pk_value)}'

            try:
                self.parent._execute_update(sql)
            except Exception as e:
                logger.error(f"Update failed: {e}")
                raise

    def query(self, query: Query, **kwargs) -> QueryResult:
        """Execute a query against the collection.

        Args:
            query: Query specification.
            **kwargs: Additional arguments.

        Returns:
            QueryResult with matching rows.
        """
        self._pre_query_hook(query)

        # Handle limit=-1 as "no limit"
        limit = query.limit
        if limit == -1:
            limit = None

        # Build and execute SQL
        sql = self._build_select_sql(
            select_cols=query.select_cols,
            where_clause=query.where_clause,
            sort_by=query.sort_by,
            limit=limit,
            offset=query.offset,
        )

        try:
            result_table = self.parent._execute_query(sql)

            # Convert Arrow table to list of dicts
            rows = result_table.to_pydict()
            num_result_rows = result_table.num_rows

            # Restructure from column-oriented to row-oriented
            if rows and num_result_rows > 0:
                row_list = []
                columns = list(rows.keys())
                for i in range(num_result_rows):
                    row = {col: rows[col][i] for col in columns}
                    row_list.append(row)
            else:
                row_list = []

            # Get total count (for pagination)
            if query.offset or (limit is not None and len(row_list) == limit):
                # Need to get actual count
                count_sql = self._build_count_sql(query.where_clause)
                try:
                    count_result = self.parent._execute_query(count_sql)
                    total_rows = count_result.column(0)[0].as_py()
                except Exception:
                    total_rows = len(row_list)
            else:
                total_rows = len(row_list)

            qr = QueryResult(query=query, num_rows=total_rows, rows=row_list, offset=query.offset or 0)

            # Handle facets if requested
            if query.include_facet_counts and query.facet_slots:
                qr.facet_counts = self.query_facets(where=query.where_clause, facet_columns=query.facet_slots)

            return qr

        except Exception as e:
            logger.error(f"Query failed: {e}")
            # Return empty result on error
            return QueryResult(query=query, num_rows=0, rows=[])

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
                # Multi-column facet
                col_list = ", ".join(f'"{c}"' for c in col)
                col_name = col
            else:
                col_list = f'"{col}"'
                col_name = col

            # Build facet query
            sql = f"SELECT {col_list}, COUNT(*) as cnt FROM {table_path}"

            if where:
                conditions = self._build_where_conditions(where)
                if conditions:
                    sql += f" WHERE {conditions}"

            sql += f" GROUP BY {col_list} ORDER BY cnt DESC"

            if facet_limit > 0:
                sql += f" LIMIT {facet_limit}"

            try:
                result = self.parent._execute_query(sql)

                facets = []
                for i in range(result.num_rows):
                    if isinstance(col, tuple):
                        value = tuple(result.column(c)[i].as_py() for c in col)
                    else:
                        value = result.column(col)[i].as_py()
                    count = result.column("cnt")[i].as_py()
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

        try:
            result = self.parent._table_exists(self.alias)
            if result:
                self._table_exists_checked = True
            return result
        except Exception:
            return False
