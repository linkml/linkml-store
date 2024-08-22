from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel

# defined a named tuple called between with two values (start, end):
# This is used in the Query class to represent a range of values
# This is used in the Query class to represent a range of values
Between = namedtuple("Between", "min max")

FACET_GROUP_ATOM = Union[str, int, float, Between]
FACET_GROUP = Union[FACET_GROUP_ATOM, Tuple[FACET_GROUP_ATOM, ...]]


class Query(BaseModel):
    """
    A query object.

    - In SQL this would be a SQL query string
    """

    from_table: Optional[str]
    select_cols: Optional[List[str]] = None
    where_clause: Optional[Union[str, List[str], Dict[str, Any]]] = None
    sort_by: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    include_facet_counts: bool = False
    facet_slots: Optional[List[str]] = None


class FacetCountResult(BaseModel):
    """
    A facet count result
    """

    as_dict: Dict[FACET_GROUP, List[Tuple[FACET_GROUP, int]]]


class QueryResult(BaseModel):
    """
    A query result.

    TODO: make this a subclass of Collection
    """

    query: Optional[Query] = None
    search_term: Optional[str] = None
    num_rows: int
    offset: Optional[int] = 0
    rows: Optional[List[Dict[str, Any]]] = None
    ranked_rows: Optional[List[Tuple[float, Dict[str, Any]]]] = None
    _rows_dataframe: Optional[pd.DataFrame] = None
    facet_counts: Optional[Dict[str, List[Tuple[FACET_GROUP, int]]]] = None

    @property
    def rows_dataframe(self) -> pd.DataFrame:
        if self.ranked_rows is not None:
            self._rows_dataframe = pd.DataFrame([{"score": score, **row} for score, row in self.ranked_rows])
        if self._rows_dataframe is None and self.rows:
            self._rows_dataframe = pd.DataFrame(self.rows)
        return self._rows_dataframe

    def set_rows(self, rows: pd.DataFrame):
        self._rows_dataframe = rows

    class Config:
        arbitrary_types_allowed = True
