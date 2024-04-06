from typing import Optional, List, Union, Dict, Any

import pandas as pd
from pydantic import BaseModel


class Query(BaseModel):
    from_table: Optional[str]
    select_cols: Optional[List[str]] = None
    where_clause: Optional[Union[str, List[str], Dict[str, str]]] = None
    sort_by: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None

    def _where_sql(self) -> str:
        if not self.where_clause:
            return ""
        where_clause_sql = None
        if isinstance(self.where_clause, str):
            where_clause_sql = self.where_clause
        elif isinstance(self.where_clause, list):
            where_clause_sql = " AND ".join(self.where_clause)
        elif isinstance(self.where_clause, dict):
            # TODO: bobby tables
            where_clause_sql = " AND ".join([f"{k} = '{v}'" for k, v in self.where_clause.items()])
        else:
            raise ValueError(f"Invalid where_clause type: {type(self.where_clause)}")
        return "WHERE " + where_clause_sql

    def sql(self, count=False, limit=None, offset: Optional[int] = None):
        select_cols = self.select_cols if self.select_cols else ["*"]
        if count:
            query = [f"SELECT COUNT(*)"]
        else:
            query = [f"SELECT {', '.join(select_cols)}"]
        query.append(f"FROM {self.from_table}")
        query.append(self._where_sql())
        if not count:
            if self.sort_by:
                query.append(f"ORDER BY {', '.join(self.sort_by)}")
        if not count:
            if limit is None:
                limit = self.limit
            if limit is None:
                limit = 100
            if limit:
                query.append(f" LIMIT {limit}")
            offset = offset if offset else self.offset
            if offset:
                query.append(f" OFFSET {offset}")
        query = [line for line in query if line]
        return "\n".join(query)

    def facet_count_sql(self, facet_column: str):
        # Create a modified WHERE clause that excludes conditions directly related to facet_column
        modified_where = None
        if self.where_clause:
            # Split the where clause into conditions and exclude those related to the facet_column
            conditions = [cond for cond in self.where_clause.split(" AND ") if not cond.startswith(f"{facet_column} ")]
            modified_where = " AND ".join(conditions)

        query = [
            f"SELECT {facet_column}, COUNT(*) as count",
            f"FROM {self.from_table}"
        ]
        if modified_where:
            query.append(f"WHERE {modified_where}")
        query.append(f"GROUP BY {facet_column}")
        query.append("ORDER BY count DESC")  # Optional, order by count for convenience
        return "\n".join(query)


class QueryResult(BaseModel):
    """
    A query result
    """
    query: Query
    num_rows: int
    offset: Optional[int] = 0
    rows: Optional[List[Dict[str, Any]]] = None
    _rows_dataframe: Optional[pd.DataFrame] = None

    @property
    def rows_dataframe(self) -> pd.DataFrame:
        if self._rows_dataframe is None and self.rows:
            self._rows_dataframe = pd.DataFrame(self.rows)
        return self._rows_dataframe

    def set_rows(self, rows: pd.DataFrame):
        self._rows_dataframe = rows

    class Config:
        arbitrary_types_allowed = True
