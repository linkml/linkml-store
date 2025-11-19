"""Ibis collection adapter for linkml-store."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult

logger = logging.getLogger(__name__)


class IbisCollection(Collection):
    """
    Collection implementation using Ibis tables.

    This adapter maps LinkML collections to Ibis tables, providing a unified
    interface across multiple database backends through Ibis.
    """

    _table_created: bool = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """Insert objects into the collection."""
        logger.debug(f"Inserting {len(objs) if isinstance(objs, list) else 1} objects")
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return

        cd = self.class_definition()
        if not cd:
            logger.debug(f"No class definition for {self.alias}; inducing from objects")
            cd = self.induce_class_definition_from_objects(objs)

        self._create_table(cd)

        # Convert objects to DataFrame for efficient insertion
        df = pd.DataFrame(objs)

        # Get the Ibis connection and table
        conn = self.parent.connection
        table_name = self.alias or self.target_class_name

        try:
            # Insert using Ibis
            # For most backends, we can use insert or create_table with data
            if table_name in conn.list_tables():
                # Table exists, insert into it
                table = conn.table(table_name)
                # Convert DataFrame to records and insert
                # Note: Ibis insert semantics vary by backend
                try:
                    # Try using insert (if supported)
                    conn.insert(table_name, df)
                except (AttributeError, NotImplementedError):
                    # Fallback: use backend-specific methods
                    # For DuckDB and similar, we can use raw SQL
                    try:
                        # Create a temp table and insert from it
                        temp_name = f"_temp_{table_name}"
                        conn.create_table(temp_name, df, overwrite=True)
                        sql = f"INSERT INTO {table_name} SELECT * FROM {temp_name}"
                        conn.raw_sql(sql)
                        conn.drop_table(temp_name)
                    except Exception as e:
                        logger.error(f"Error inserting data: {e}")
                        # Last resort: use pandas to_sql if available
                        if hasattr(conn, "con"):
                            # Some Ibis backends expose the underlying connection
                            df.to_sql(table_name, conn.con, if_exists="append", index=False)
                        else:
                            raise
            else:
                # Table doesn't exist, create it with data
                conn.create_table(table_name, df)

            logger.info(f"Inserted {len(objs)} objects into {table_name}")
        except Exception as e:
            logger.error(f"Error inserting into {table_name}: {e}")
            raise

        self._post_insert_hook(objs)

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> Optional[int]:
        """Delete specific objects from the collection."""
        if not isinstance(objs, list):
            objs = [objs]

        cd = self.class_definition()
        if not cd or not cd.attributes:
            cd = self.induce_class_definition_from_objects(objs)

        conn = self.parent.connection
        table_name = self.alias or self.target_class_name

        if table_name not in conn.list_tables():
            logger.warning(f"Table {table_name} does not exist")
            return 0

        # For Ibis, deletion is backend-specific
        # We'll use raw SQL for broader compatibility
        deleted_count = 0
        for obj in objs:
            conditions = []
            for k, v in obj.items():
                if k in cd.attributes:
                    if isinstance(v, str):
                        conditions.append(f"{k} = '{v}'")
                    else:
                        conditions.append(f"{k} = {v}")

            if conditions:
                where_clause = " AND ".join(conditions)
                sql = f"DELETE FROM {table_name} WHERE {where_clause}"
                try:
                    conn.raw_sql(sql)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting object: {e}")

        self._post_delete_hook()
        return deleted_count

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> Optional[int]:
        """Delete objects matching a where clause."""
        logger.info(f"Deleting from {self.target_class_name} where: {where}")
        if where is None:
            where = {}

        cd = self.class_definition()
        if not cd:
            logger.info(f"No class definition found for {self.target_class_name}")
            return 0

        conn = self.parent.connection
        table_name = self.alias or self.target_class_name

        if table_name not in conn.list_tables():
            logger.info(f"Table {table_name} does not exist")
            return 0

        # Build where clause
        conditions = []
        for k, v in where.items():
            if isinstance(v, str):
                conditions.append(f"{k} = '{v}'")
            else:
                conditions.append(f"{k} = {v}")

        if conditions:
            where_clause = " AND ".join(conditions)
            sql = f"DELETE FROM {table_name} WHERE {where_clause}"
        else:
            sql = f"DELETE FROM {table_name}"

        try:
            result = conn.raw_sql(sql)
            # Note: Getting rowcount from raw SQL varies by backend
            # For now, return None to indicate success without count
            self._post_delete_hook()
            return None
        except Exception as e:
            if not missing_ok:
                raise
            logger.warning(f"Error deleting: {e}")
            return 0

    def query(self, query: Query = None, **kwargs) -> QueryResult:
        """Execute a query against the collection."""
        if query is None:
            query = Query()

        conn = self.parent.connection
        table_name = self.alias or self.target_class_name

        if table_name not in conn.list_tables():
            logger.warning(f"Table {table_name} does not exist")
            return QueryResult(num_rows=0, rows=[])

        # Get the Ibis table
        table = conn.table(table_name)

        # Apply filters
        if query.where_clause:
            table = self._apply_where(table, query.where_clause)

        # Apply column selection
        if query.select_cols:
            table = table.select(query.select_cols)

        # Apply sorting
        if query.sort_by:
            # Convert sort specs to Ibis sort expressions
            sort_exprs = []
            for sort_spec in query.sort_by:
                if sort_spec.startswith("-"):
                    # Descending
                    col_name = sort_spec[1:]
                    sort_exprs.append(table[col_name].desc())
                else:
                    # Ascending
                    sort_exprs.append(table[sort_spec].asc())
            table = table.order_by(sort_exprs)

        # Apply limit and offset
        if query.offset:
            table = table.limit(None, offset=query.offset)
        if query.limit:
            table = table.limit(query.limit)

        # Execute query and convert to pandas
        try:
            df = table.to_pandas()
            rows = df.to_dict("records")

            result = QueryResult(
                query=query,
                num_rows=len(rows),
                offset=query.offset,
                rows=rows,
                rows_dataframe=df,
            )

            # Handle facets if requested
            if query.include_facet_counts and query.facet_slots:
                result.facet_counts = self._compute_facets(table_name, query.where_clause, query.facet_slots)

            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def _apply_where(self, table, where_clause):
        """Apply where clause filters to an Ibis table."""
        if isinstance(where_clause, dict):
            # Simple equality filters
            for k, v in where_clause.items():
                table = table.filter(table[k] == v)
        elif isinstance(where_clause, list):
            # Multiple conditions (AND)
            for condition in where_clause:
                if isinstance(condition, dict):
                    for k, v in condition.items():
                        table = table.filter(table[k] == v)
                else:
                    # String condition - use SQL
                    logger.warning(f"String where clauses not fully supported in Ibis: {condition}")
        elif isinstance(where_clause, str):
            # SQL string - limited support
            logger.warning(f"String where clauses require SQL mode: {where_clause}")

        return table

    def _compute_facets(
        self, table_name: str, where_clause, facet_columns: List[str]
    ) -> Dict[str, List[Tuple[Any, int]]]:
        """Compute facet counts for specified columns."""
        conn = self.parent.connection
        table = conn.table(table_name)

        if where_clause:
            table = self._apply_where(table, where_clause)

        facets = {}
        for col in facet_columns:
            try:
                # Group by and count
                grouped = table.group_by(col).aggregate(count=table.count())
                df = grouped.to_pandas()
                # Convert to list of tuples
                facets[col] = list(zip(df[col], df["count"]))
            except Exception as e:
                logger.warning(f"Error computing facets for {col}: {e}")
                facets[col] = []

        return facets

    def _create_table(self, cd: ClassDefinition):
        """Create the table if it doesn't exist."""
        if self._table_created:
            return

        conn = self.parent.connection
        table_name = self.alias or self.target_class_name

        if table_name in conn.list_tables():
            self._table_created = True
            return

        # Create an empty table with the schema
        # Build a sample DataFrame with correct types
        columns = {}
        for attr_name, slot in cd.attributes.items():
            # Map LinkML types to Python types for DataFrame
            slot_range = slot.range or "string"
            if slot_range == "integer":
                columns[attr_name] = pd.Series([], dtype="Int64")
            elif slot_range == "float":
                columns[attr_name] = pd.Series([], dtype="float64")
            elif slot_range == "boolean":
                columns[attr_name] = pd.Series([], dtype="boolean")
            elif slot_range == "date":
                columns[attr_name] = pd.Series([], dtype="object")
            elif slot_range == "datetime":
                columns[attr_name] = pd.Series([], dtype="datetime64[ns]")
            else:
                columns[attr_name] = pd.Series([], dtype="string")

        # Create empty DataFrame with schema
        df = pd.DataFrame(columns)

        try:
            # Create table using Ibis
            conn.create_table(table_name, df)
            self._table_created = True
            logger.info(f"Created table {table_name}")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise

    def find(self, where: Optional[Dict[str, Any]] = None, **kwargs) -> List[OBJECT]:
        """Find objects matching the where clause."""
        query = Query(where_clause=where, limit=kwargs.get("limit"), offset=kwargs.get("offset"))
        result = self.query(query)
        return result.rows or []

    def peek(self, limit=5) -> List[OBJECT]:
        """Get a few sample objects from the collection."""
        query = Query(limit=limit)
        result = self.query(query)
        return result.rows or []
