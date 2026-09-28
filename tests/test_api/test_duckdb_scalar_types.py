"""DuckDB columns keep LinkML scalar values: 64-bit floats and integers, and real booleans."""

from linkml_runtime import SchemaView
from sqlalchemy import text

from linkml_store import Client

SCHEMA = """
id: https://example.org/scalar-types
name: scalar_types
prefixes: {linkml: https://w3id.org/linkml/}
imports: [linkml:types]
default_range: string
classes:
  Container:
    tree_root: true
    attributes:
      features: {range: Feature, multivalued: true, inlined_as_list: true}
  Feature:
    attributes:
      id: {identifier: true}
      score: {range: float}
      weight: {range: double}
      is_selected: {range: boolean}
      end: {range: integer}
"""

FEATURES = [
    {"id": "f1", "score": 239.1, "weight": 0.1, "is_selected": True, "end": 2**33},
    {"id": "f2", "score": 0.1, "weight": 1e-300, "is_selected": False, "end": 100},
]


def stored():
    db = Client().attach_database("duckdb", alias="scalar_types")
    db.set_schema_view(SchemaView(SCHEMA))
    db.store({"features": [dict(f) for f in FEATURES]})
    return db


def test_columns_use_64_bit_and_boolean_types():
    with stored().engine.connect() as conn:
        columns = {row[0]: row[1] for row in conn.execute(text("describe features"))}
    assert columns["score"] == "DOUBLE"
    assert columns["weight"] == "DOUBLE"
    assert columns["is_selected"] == "BOOLEAN"
    assert columns["end"] == "BIGINT"


def test_values_read_back_unchanged():
    rows = sorted(stored().get_collection("features").find({}).rows, key=lambda row: row["id"])
    assert rows == FEATURES
