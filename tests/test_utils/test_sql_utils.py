"""Tests for translation of queries to SQL."""

import pytest

from linkml_store.api.queries import Query
from linkml_store.utils.sql_utils import col_val_constraints_to_conjs, facet_count_sql, where_clause_to_sql


@pytest.mark.parametrize(
    "where,expected",
    [
        (None, ""),
        ({}, ""),
        ({"name": "Alice"}, "WHERE name = 'Alice'"),
        ({"name": "O'Neil"}, "WHERE name = 'O''Neil'"),
        ({"age": 5, "name": "Alice"}, "WHERE age = 5 AND name = 'Alice'"),
        ({"name": None}, "WHERE name IS NULL"),
        ({"id": {"$in": [1, 2]}}, "WHERE id IN (1, 2)"),
        ({"id": {"$in": ["a", "b"]}}, "WHERE id IN ('a', 'b')"),
        ({"id": {"$nin": [1, 2]}}, "WHERE id NOT IN (1, 2)"),
        ({"age": {"$gt": 1}}, "WHERE age > 1"),
        ({"age": {"$gte": 1, "$lt": 9}}, "WHERE age >= 1 AND age < 9"),
        ({"age": {"$lte": 1}}, "WHERE age <= 1"),
        ({"name": {"$ne": "Alice"}}, "WHERE name <> 'Alice'"),
        ({"name": {"$eq": "Alice"}}, "WHERE name = 'Alice'"),
        ({"name": {"$ne": None}}, "WHERE name IS NOT NULL"),
        ({"aliases": {"$contains": "Al"}}, "WHERE ARRAY_CONTAINS(aliases, 'Al')"),
        ({"name": {"LIKE": "A%"}}, "WHERE name LIKE 'A%'"),
        ("age > 5", "WHERE age > 5"),
        (["age > 5", "age < 9"], "WHERE age > 5 AND age < 9"),
    ],
)
def test_where_clause_to_sql(where, expected):
    assert where_clause_to_sql(Query(from_table="t", where_clause=where)) == expected


def test_unsupported_operator():
    with pytest.raises(ValueError, match="Unsupported operator"):
        col_val_constraints_to_conjs("age", {"$near": 5})


@pytest.mark.parametrize("limit,expected_last_line", [(5, "LIMIT 5"), (None, "ORDER BY count DESC")])
def test_facet_count_sql_limit(limit, expected_last_line):
    sql = facet_count_sql(Query(from_table="t"), "name", limit=limit)
    assert sql.splitlines()[-1] == expected_last_line
