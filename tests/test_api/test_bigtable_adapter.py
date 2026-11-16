"""
Tests for the Bigtable adapter.

Tests that talk to Bigtable run only against the emulator, and are skipped
unless ``BIGTABLE_EMULATOR_HOST`` is set and reachable:

.. code-block:: bash

    gcloud beta emulators bigtable start --host-port=localhost:8086
    export BIGTABLE_EMULATOR_HOST=localhost:8086
    pytest tests/test_api/test_bigtable_adapter.py
"""

import os
import socket

import pytest

pytest.importorskip("google.cloud.bigtable")

from google.cloud.bigtable import column_family  # noqa: E402
from linkml_runtime import SchemaView  # noqa: E402
from linkml_runtime.linkml_model import SlotDefinition  # noqa: E402
from linkml_runtime.utils.schema_builder import SchemaBuilder  # noqa: E402

from linkml_store import Client  # noqa: E402
from linkml_store.api.queries import Query  # noqa: E402
from linkml_store.api.stores.bigtable import BigtableCollection, BigtableDatabase  # noqa: E402

HANDLE = "bigtable://test-project/test-instance"

PEOPLE = [
    {
        "id": "P1",
        "name": "Alice",
        "age": 25,
        "occupation": "Architect",
        "foods": ["apple", "banana"],
        "meta": {"date": "2021-01-01", "notes": "likes fruit"},
    },
    {
        "id": "P2",
        "name": "Bob",
        "age": 30,
        "occupation": "Builder",
        "foods": ["carrot", "apple"],
        "meta": {"date": "2021-01-02", "notes": "likes veg"},
    },
    {"id": "P3", "name": "Carol", "age": 35, "occupation": "Builder", "foods": []},
]


def _emulator_available() -> bool:
    host = os.environ.get("BIGTABLE_EMULATOR_HOST")
    if not host:
        return False
    hostname, _, port = host.rpartition(":")
    try:
        with socket.create_connection((hostname or "localhost", int(port)), timeout=2):
            return True
    except (OSError, ValueError):
        return False


requires_emulator = pytest.mark.skipif(
    not _emulator_available(), reason="requires the Bigtable emulator (set BIGTABLE_EMULATOR_HOST)"
)


def person_schema() -> SchemaView:
    sb = SchemaBuilder()
    slots = [
        SlotDefinition("id", identifier=True),
        SlotDefinition("name"),
        SlotDefinition("age", range="integer"),
        SlotDefinition("occupation"),
        SlotDefinition("foods", multivalued=True),
        SlotDefinition("meta"),
    ]
    sb.add_class("Person", slots, use_attributes=True)
    return SchemaView(sb.schema)


def strip(rows):
    """Bigtable does not store nulls; normalize rows for comparison."""
    return sorted(({k: v for k, v in row.items() if v is not None} for row in rows), key=lambda r: r.get("id", ""))


@pytest.fixture
def db():
    client = Client()
    database = client.attach_database(HANDLE, alias="bt")
    database.drop()
    yield database
    database.drop()


@pytest.fixture
def people(db):
    """A collection with an identifier attribute, so ids are row keys."""
    db.set_schema_view(person_schema())
    collection = db.create_collection("Person", alias="people")
    collection.insert([dict(p) for p in PEOPLE])
    return collection


def test_handle_parsing():
    db = BigtableDatabase("bigtable://my-project/my-instance?column_family=cf&app_profile_id=batch")
    assert db.project_id == "my-project"
    assert db.instance_id == "my-instance"
    assert db.column_family == "cf"
    assert db.app_profile_id == "batch"
    db = BigtableDatabase("bigtable:///my-instance")
    assert db.project_id is None
    assert db.column_family == "d"
    db = BigtableDatabase()
    assert db.handle == HANDLE
    with pytest.raises(ValueError):
        BigtableDatabase("bigtable://my-project")


def test_client_scheme():
    client = Client()
    db = client.attach_database("bigtable://p/i", alias="x")
    assert isinstance(db, BigtableDatabase)
    db = client.attach_database("bigtable", alias="y")
    assert db.handle == HANDLE


@requires_emulator
def test_insert_and_query_without_schema(db):
    collection = db.create_collection("Thing", alias="things")
    assert isinstance(collection, BigtableCollection)
    collection.insert([dict(p) for p in PEOPLE])
    assert "things" in db.list_collection_names()

    qr = collection.find({})
    assert qr.num_rows == 3
    assert strip(qr.rows) == strip(PEOPLE)

    qr = collection.find({"occupation": "Builder"})
    assert qr.num_rows == 2
    assert {r["name"] for r in qr.rows} == {"Bob", "Carol"}

    qr = collection.find({"age": {"$gte": 30}, "foods": {"$in": ["apple"]}})
    assert [r["name"] for r in qr.rows] == ["Bob"]

    qr = collection.find({"meta.notes": "likes fruit"}, select_cols=["name", "meta.date"])
    assert qr.rows == [{"name": "Alice", "meta.date": "2021-01-01"}]

    # limit and offset; num_rows is the total
    qr = collection.find({}, limit=2)
    assert qr.num_rows == 3
    assert len(qr.rows) == 2
    qr2 = collection.find({}, limit=2, offset=2)
    assert len(qr2.rows) == 1
    assert {r["name"] for r in qr.rows + qr2.rows} == {"Alice", "Bob", "Carol"}
    qr = collection.find({"occupation": "Builder"}, limit=1, offset=1)
    assert qr.num_rows == 2
    assert len(qr.rows) == 1

    # objects with no non-null fields cannot be stored as Bigtable rows
    collection.insert([{}, {"name": None}])
    assert collection.find({}).num_rows == 3

    # without an identifier, row keys are generated, so inserting again duplicates
    collection.insert(dict(PEOPLE[0]))
    assert collection.find({"name": "Alice"}).num_rows == 2


@requires_emulator
def test_identifier_is_row_key(people, db):
    row = people.native_table.read_row(b"P2")
    assert row is not None
    # constraints on the identifier are answered by row key lookups, not scans
    assert people._identifier_field() == "id"
    assert people._row_keys_for_where({"id": {"$in": ["P1", "P2"]}}, "id") == [b"P1", b"P2"]
    assert people._row_keys_for_where({"name": "Bob"}, "id") is None
    assert people.get_one("P2")["name"] == "Bob"
    assert people.get_one("NOPE") is None
    qr = people.get(["P1", "P3", "NOPE"])
    assert {r["id"] for r in qr.rows} == {"P1", "P3"}
    # identifier lookups still apply other conditions
    assert people.find({"id": "P1", "name": "Bob"}).num_rows == 0

    # inserting an existing identifier replaces the whole row
    people.insert({"id": "P1", "name": "Alicia"})
    assert people.get_one("P1") == {"id": "P1", "name": "Alicia"}
    assert people.find({}).num_rows == 3

    with pytest.raises(ValueError):
        people.insert({"name": "No id"})


@requires_emulator
def test_delete(people):
    assert people.delete({"id": "P1"}) == 1
    assert people.delete({"id": "P1"}) == 0
    assert people.find({}).num_rows == 2
    assert people.delete_where({"occupation": "Builder", "age": {"$gt": 30}}) == 1
    assert [r["id"] for r in people.find({}).rows] == ["P2"]
    assert people.delete_where({}) == 1
    assert people.find({}).num_rows == 0
    with pytest.raises(ValueError):
        people.delete_where({"id": "P2"}, missing_ok=False)


@requires_emulator
def test_delete_by_value_without_identifier(db):
    collection = db.create_collection("Thing", alias="things")
    collection.insert([{"x": 1, "y": "a"}, {"x": 1, "y": "b"}, {"x": 2, "y": "a"}])
    assert collection.delete({"x": 1, "y": "a"}) == 1
    assert sorted((r["x"], r["y"]) for r in collection.find({}).rows) == [(1, "b"), (2, "a")]


@requires_emulator
def test_upsert(people):
    people.upsert({"id": "P2", "age": 31}, filter_fields=["id"], update_fields=["age"])
    bob = people.get_one("P2")
    assert bob["age"] == 31
    assert bob["occupation"] == "Builder"

    people.upsert({"id": "P4", "name": "Dan", "age": 40}, filter_fields=["id"])
    assert people.get_one("P4")["name"] == "Dan"
    assert people.find({}).num_rows == 4

    # match on a non-identifier field; None removes the field
    people.upsert({"name": "Carol", "occupation": None}, filter_fields=["name"])
    assert "occupation" not in people.get_one("P3")

    with pytest.raises(ValueError):
        people.upsert({"age": 1}, filter_fields=["id"])


@requires_emulator
def test_query_facets(people):
    facets = people.query_facets(facet_columns=["occupation", "foods", ("occupation", "age")])
    assert dict(facets["occupation"]) == {"Builder": 2, "Architect": 1}
    assert dict(facets["foods"]) == {"apple": 2, "banana": 1, "carrot": 1}
    assert dict(facets[("occupation", "age")]) == {("Architect", 25): 1, ("Builder", 30): 1, ("Builder", 35): 1}
    facets = people.query_facets(where={"age": {"$lt": 30}}, facet_columns=["occupation"])
    assert facets["occupation"] == [("Architect", 1)]
    facets = people.query_facets(facet_columns=["occupation"], facet_limit=1)
    assert facets["occupation"] == [("Builder", 2)]


@requires_emulator
def test_sort(people):
    qr = people.query(Query(from_table="people", sort_by=["-age"], limit=2))
    assert [r["name"] for r in qr.rows] == ["Carol", "Bob"]
    assert qr.num_rows == 3
    qr = people.query(Query(from_table="people", sort_by=["occupation", "name DESC"]))
    assert [r["name"] for r in qr.rows] == ["Alice", "Carol", "Bob"]


@requires_emulator
def test_find_iter(people):
    names = [r["name"] for r in people.find_iter({"occupation": "Builder"}, page_size=1)]
    assert sorted(names) == ["Bob", "Carol"]
    assert list(people.find_iter({"id": "P1"}, select_cols=["name"])) == [{"name": "Alice"}]


@requires_emulator
def test_reattach_and_drop(people):
    client = Client()
    db2 = client.attach_database(HANDLE, alias="bt2")
    assert "people" in db2.list_collection_names()
    assert db2.get_collection("people").find({}).num_rows == 3
    db2.drop()
    assert db2.list_collection_names() == []
    assert people.find({}).num_rows == 0


@requires_emulator
def test_recreate_if_exists(people, db):
    collection = db.create_collection("Person", alias="people", recreate_if_exists=True)
    assert collection.find({}).num_rows == 0


@requires_emulator
def test_missing_table(db):
    collection = db.get_collection("nothing_here")
    assert collection.find({}).num_rows == 0
    assert collection.delete_where({}) == 0
    assert list(collection.find_iter()) == []


@requires_emulator
def test_existing_table_with_other_families(db):
    """Tables written by other tools: non-JSON values and extra column families."""
    table = db.native_instance.table("legacy")
    table.create(column_families={"d": column_family.MaxVersionsGCRule(1), "stats": column_family.MaxVersionsGCRule(1)})
    row = table.direct_row(b"row1")
    row.set_cell("d", b"name", b"plain text")
    row.set_cell("d", b"count", b"3")
    row.set_cell("stats", b"views", b"10")
    row.commit()
    collection = db.get_collection("legacy")
    assert collection.find({}).rows == [{"name": "plain text", "count": 3, "stats": {"views": 10}}]
