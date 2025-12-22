import json
import logging
import shutil
import unittest
from pathlib import Path

import pystow
import pytest
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api.client import Client
from linkml_store.api.config import ClientConfig, CollectionConfig, DerivationConfiguration
from linkml_store.api.queries import Query
from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
from linkml_store.api.stores.mongodb import MongoDBDatabase
from linkml_store.api.stores.solr.solr_database import SolrDatabase
from linkml_store.constants import LINKML_STORE_MODULE
from linkml_store.index import get_indexer
from linkml_store.index.implementations.simple_indexer import SimpleIndexer
from linkml_store.utils.format_utils import load_objects
from linkml_store.utils.sql_utils import introspect_schema
from tests import COUNTRIES_DATA_JSONL, COUNTRIES_SCHEMA, INPUT_DIR, OUTPUT_DIR, PERSONINFO_SCHEMA

logger = logging.getLogger(__name__)

TEST_DB = INPUT_DIR / "integration" / "mgi.db"
TEMP_DB_PATH = OUTPUT_DIR / "temp.db"
TEMP_IBIS_PATH = OUTPUT_DIR / "temp_ibis.duckdb"

SCHEMES = [
    "duckdb",
    f"duckdb:///{TEMP_DB_PATH}",
    # "mongodb://localhost:27017/test_db",
]

# Ibis schemes - only included if ibis is installed
try:
    import ibis
    IBIS_SCHEMES = [
        "ibis+duckdb:///:memory:",
        f"ibis+duckdb:///{TEMP_IBIS_PATH}",
    ]
except ImportError:
    IBIS_SCHEMES = []

# SCHEMES_PLUS = SCHEMES + ["mongodb://localhost:27017/test_db"]
SCHEMES_PLUS = SCHEMES + ["mongodb://localhost:27017/test_db", f"file:{OUTPUT_DIR}/api_test_fs"] + IBIS_SCHEMES

DEFAULT_DB = "default"

PERSONS = [
    {
        "id": 1,
        "name": "n1",
        "history": [
            {"event": "birth", "date": "2021-01-01"},
            {"event": "death", "date": "2021-02-01"},
            {"event": "hired", "date": "2021-02-01", "organization": "Org1"},
        ],
    },
    {"id": 2, "name": "n2", "age_in_years": 30},
]

ORGANIZATIONS = [
    # {"id": "Org1", "name": "org1", "changes": {"editor": "joe", "date": "2021-01-01"}},
    {"id": "Org1", "name": "org1"},
    {"id": "Org2", "name": "org2", "found_date": "2021-01-01"},
]

EMPLOYED_AT = [
    {"person": 1, "organization": "Org1"},
    {"person": 2, "organization": "Org2"},
]


def is_persistent(handle: str) -> bool:
    # if "duckdb" in handle:
    #    # NOTE: in previous versions of duckdb, in-memory databases were not persistent
    #    return True
    if ":memory:" in handle:
        return False
    return ".db" in handle or "mongodb" in handle or "file:" in handle or ".duckdb" in handle


def remove_none(d: dict, additional_keys=None):
    """
    Remove keys with None values from a dictionary

    :param d:
    :param additional_keys:
    :return:
    """
    import math
    if additional_keys is None:
        additional_keys = []

    def is_null(v):
        if v is None:
            return True
        # Handle NaN values (returned by Ibis/DuckDB for missing columns)
        if isinstance(v, float) and math.isnan(v):
            return True
        return False

    return {k: v for k, v in d.items() if not is_null(v) and k not in additional_keys}


@pytest.fixture()
def schema_view() -> SchemaView:
    """
    Create a simple person schema view for testing

    :return:
    """
    sb = SchemaBuilder()
    id_slot = SlotDefinition("id", identifier=True)
    name_slot = SlotDefinition("name")
    age_in_years_slot = SlotDefinition("age_in_years", range="integer")
    occupation_slot = SlotDefinition("occupation")
    moon_slot = SlotDefinition("moon")
    sb.add_class("Person", [id_slot, name_slot, age_in_years_slot, occupation_slot, moon_slot], use_attributes=True)
    return SchemaView(sb.schema)


@pytest.fixture()
def countries_schema_view() -> SchemaView:
    """
    Use a simple predefined country schema view for testing

    :return:
    """
    return SchemaView(COUNTRIES_SCHEMA)


@pytest.fixture()
def personinfo_schema_view() -> SchemaView:
    """
    Nested personinfo schema

    :return:
    """
    return SchemaView(PERSONINFO_SCHEMA)


def create_client(handle: str, recreate_if_exists=True) -> Client:
    """
    Create a client with a database attached

    :param handle:
    :param recreate_if_exists:
    :return:
    """
    client = Client()
    if recreate_if_exists:
        client.drop_database(handle, missing_ok=True, no_backup=True)
    print(f"DBs={client.databases}")
    if handle.endswith(".db"):
        path = handle.replace("duckdb:///", "")
        if recreate_if_exists:
            print(f"UNLINKING: {path}")
            Path(path).unlink(missing_ok=True)
            assert not Path(path).exists()
    if handle.endswith(".duckdb") and "ibis" in handle:
        # Handle Ibis DuckDB file paths
        if "ibis+duckdb:///" in handle:
            path = handle.replace("ibis+duckdb:///", "")
        elif "ibis:///" in handle:
            path = handle.replace("ibis:///", "")
        else:
            path = None
        if path and recreate_if_exists:
            print(f"UNLINKING IBIS DB: {path}")
            path_obj = Path(path)
            # Also clean up WAL files that DuckDB may create
            wal_path = Path(str(path) + ".wal")
            if wal_path.exists():
                wal_path.unlink()
            path_obj.unlink(missing_ok=True)
            assert not path_obj.exists(), f"Failed to delete Ibis DB file: {path}"
    if handle.startswith("mongodb:"):
        # because the mongo instance is shared, we want to avoid destroying
        # existing databases
        pass
    print(f"DB2s={client.databases}")
    # client.attach_database(handle, alias=DEFAULT_DB)
    client.attach_database(handle, recreate_if_exists=recreate_if_exists)
    print(f"ATTACHED: {handle} // num={len(client.databases)} // {client.databases}")
    return client


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_store(handle):
    """
    Tests storing of objects in a database automatically creating collections.

    The store command is at the level of a database; it's assumed that
    each key in the stored object corresponds to a :class:`Collection`.

    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    obj = {
        "persons": [
            {"id": 1, "name": "n1", "employed_by": "Org1"},
            {"id": 2, "name": "n2", "age_in_years": 30},
        ],
        "organizations": [
            {"id": "Org1", "name": "org1"},
            {"id": "Org2", "name": "org2", "found_date": "2021-01-01"},
        ],
    }
    database.store(obj)
    persons_coll = database.get_collection("persons")
    qr = persons_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["persons"][0]
    qr = persons_coll.find({"id": 1})
    assert qr.num_rows == 1
    # qr = persons_coll.find({"id": [1, 2]})
    # assert qr.num_rows == 2
    orgs_coll = database.get_collection("organizations")
    qr = orgs_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["organizations"][0]


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_find_iter(handle):
    """
    Tests querying using iterator

    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    coll = database.create_collection("Person", recreate_if_exists=True)
    n = 200
    objs = [{"id": i, "name": f"n{i}", "score": i * 10} for i in range(n)]
    coll.insert(objs)
    retrieved_objs = list(coll.find_iter())
    assert len(retrieved_objs) == n
    for offset, limit in [(0, 1), (10, 10)]:
        qr = coll.find({}, offset=offset, limit=limit)
        assert qr.num_rows == len(objs)
        assert len(qr.rows) == limit
        assert qr.rows[0] == retrieved_objs[offset]


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_derivations(handle):
    """
    Tests querying using iterator

    :param handle:
    :return:
    """
    pytest.importorskip("linkml_map")
    client = create_client(handle)
    database = client.get_database()
    coll = database.create_collection("Person", recreate_if_exists=True)
    n = 100
    base_age_in_months = 120
    objs = [{"id": i, "name": f"n{i}", "age_in_months": base_age_in_months + i} for i in range(n)]
    age_by_id = {obj["id"]: obj["age_in_months"] for obj in objs}
    coll.insert(objs)
    coll.commit()
    tgt_coll = database.create_collection("Person2", recreate_if_exists=True, alias="derived_persons")
    deriv = DerivationConfiguration(
        database=database.alias,
        collection=coll.alias,
        mappings={
            "id": {"populated_from": "id"},
            "full_name": {"populated_from": "name"},
            "age": {"expr": "age_in_months / 12"},
        },
    )
    tgt_coll.metadata.derived_from = [deriv]
    qr = tgt_coll.find({})
    assert qr.num_rows > 0
    for obj in qr.rows:
        assert obj["age"] == pytest.approx(age_by_id[obj["id"]] / 12)
        assert obj["full_name"]


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
@pytest.mark.parametrize(
    "location,export_format",
    [
        (OUTPUT_DIR / "export.yaml", "yaml"),
        (OUTPUT_DIR / "export.json", "json"),
        (OUTPUT_DIR / "export.duckdb", "duckdb"),
        (OUTPUT_DIR / "export.mongodb", "mongodb"),
    ],
)
def test_export(handle, location, export_format):
    """
    Tests export and re-import

    :param handle: scheme/handle to test
    :param location: where to export to
    :param export_format: format to export to
    :return:
    """
    is_mongodb = handle.startswith("mongodb")
    path_to_mongodump = shutil.which("mongodump")
    if is_mongodb and path_to_mongodump is None:
        pytest.skip("mongodump not found in PATH")
    client = create_client(handle)
    database = client.get_database()
    obj = {
        "persons": [
            {"id": 1, "name": "n1", "employed_by": "Org1"},
            {"id": 2, "name": "n2", "age_in_years": 30},
        ],
        "organizations": [
            {"id": "Org1", "name": "org1"},
            {"id": "Org2", "name": "org2", "found_date": "2021-01-01"},
        ],
    }
    database.store(obj)
    if export_format == "mongodb" and path_to_mongodump is None:
        pytest.skip("mongodump not found in PATH")
    database.export_database(location, export_format)
    if is_mongodb:
        client2 = create_client("mongodb://localhost:27017/test_db_tmp")
        database = client2.get_database()
        # database = client.attach_database("mongodb://localhost:27017/test_db_tmp")
    else:
        database = client.attach_database("duckdb", recreate_if_exists=True)
    print(f"Importing from {location} to {database.handle}")
    database.import_database(location, export_format)
    persons_coll = database.get_collection("persons")
    qr = persons_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["persons"][0]
    qr = persons_coll.find({"id": 1})
    assert qr.num_rows == 1
    # qr = persons_coll.find({"id": [1, 2]})
    # assert qr.num_rows == 2
    orgs_coll = database.get_collection("organizations")
    qr = orgs_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["organizations"][0]


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_group_by(handle):
    client = create_client(handle)
    database = client.get_database()
    rows = [
        {"id": 1, "name": "n1", "address": "a1", "address_city": "c1"},
        {"id": 1, "name": "n1", "address": "a2", "address_city": "c2"},
        {"id": 2, "name": "n2", "address": "a3", "address_city": "c3"},
    ]
    c = database.create_collection("PersonAddress", recreate_if_exists=True)
    c.insert(rows)
    qr = c.find()
    assert qr.num_rows == 3
    rs = c.group_by(["id"], inlined_field="addresses", agg_map={"name": "first"})
    c2 = database.create_collection("Person", recreate_if_exists=True)
    c2.insert(rs.rows)
    qr = c2.find()
    assert qr.num_rows == 2
    for row in qr.rows:
        if row["id"] == 1:
            assert len(row["addresses"]) == 2
        elif row["id"] == 2:
            assert len(row["addresses"]) == 1
        else:
            assert False, f"Unexpected id: {row['id']}"


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_group_by_advanced(handle):
    """
    Test more advanced group_by features for specific store implementations.

    Tests various features:
    1. Multi-field grouping
    2. Filtering with where clause
    3. Aggregation of specific fields
    4. Different inlined field name
    """
    client = create_client(handle)
    database = client.get_database()

    # Create a more complex dataset with multiple grouping possibilities
    rows = [
        {"id": 1, "category": "A", "name": "Item1", "price": 10.0, "qty": 5, "tags": ["red", "small"]},
        {"id": 2, "category": "A", "name": "Item2", "price": 20.0, "qty": 3, "tags": ["blue", "medium"]},
        {"id": 3, "category": "B", "name": "Item3", "price": 15.0, "qty": 7, "tags": ["red", "large"]},
        {"id": 4, "category": "B", "name": "Item4", "price": 25.0, "qty": 2, "tags": ["green", "small"]},
        {"id": 5, "category": "A", "name": "Item5", "price": 30.0, "qty": 1, "tags": ["blue", "large"]},
    ]

    collection = database.create_collection("Products", recreate_if_exists=True)
    collection.insert(rows)

    # Test 1: Group by a single field
    result = collection.group_by(["category"])
    assert result.num_rows == 2

    # Verify correct grouping
    for group in result.rows:
        if group["category"] == "A":
            assert len(group["objects"]) == 3
        elif group["category"] == "B":
            assert len(group["objects"]) == 2
        else:
            assert False, f"Unexpected category: {group['category']}"

    # Test 2: Group by multiple scalar fields (avoid using array fields in multi-field grouping)
    result = collection.group_by(["category", "name"])
    # Just check that it doesn't error - the exact results will depend on implementation

    # Test 3: Group with a where clause - use exact match for compatibility
    # Filter for category "A" items only
    result = collection.group_by(["category"], where={"category": "A"})
    assert result.num_rows == 1
    assert result.rows[0]["category"] == "A"
    assert len(result.rows[0]["objects"]) == 3

    # For MongoDB specific test, if this is MongoDB handle
    if "mongodb" in handle:
        # Uses MongoDB's query operators
        result = collection.group_by(["category"], where={"price": {"$gt": 15.0}})

        # Find the group with category "A"
        a_group = next((g for g in result.rows if g["category"] == "A"), None)
        if a_group is not None:
            # Should only include items with price > 15.0
            for item in a_group["objects"]:
                assert item["price"] > 15.0

    # Test 4: Custom inlined field name
    result = collection.group_by(["category"], inlined_field="items")
    for group in result.rows:
        assert "items" in group
        assert "objects" not in group

    # Test 5: Test with agg_map for field selection (skip for file adapter which doesn't fully support agg_map)
    if "file:" not in handle:
        result = collection.group_by(["category"], agg_map={"first": ["category"], "list": ["name", "price"]})

        # Verify that only specified fields are included
        for group in result.rows:
            for item in group["objects"]:
                assert "name" in item
                assert "price" in item
                assert "qty" not in item  # This field should be excluded
                assert "tags" not in item  # This field should be excluded


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_collections_of_same_type(handle):
    """
    Tests storing of objects in collections of the same type

    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    persons_a = [
        {"id": 1, "name": "n1", "employed_by": "Org1"},
        {"id": 2, "name": "n2", "age_in_years": 30},
    ]
    persons_b = [
        {"id": 3, "name": "n3", "employed_by": "Org1"},
        {"id": 4, "name": "n4", "age_in_years": 30},
        {"id": 5, "name": "n5", "age_in_years": 33},
    ]
    collection_a = database.create_collection("Person", alias="persons_a", recreate_if_exists=True)
    collection_b = database.create_collection("Person", alias="persons_b", recreate_if_exists=True)
    qr_a = collection_a.find()
    assert qr_a.num_rows == 0
    qr_b = collection_b.find()
    assert qr_b.num_rows == 0
    collection_a.insert(persons_a)
    collection_b.insert(persons_b)
    qr_a = collection_a.find()
    assert qr_a.num_rows == 2
    qr_b = collection_b.find()
    assert qr_b.num_rows == 3


# TODO: mongo works locally but fails on github actions
# @pytest.mark.parametrize("handle", SCHEMES_PLUS)
@pytest.mark.parametrize("handle", SCHEMES)
def test_patch(handle):
    """
    Tests patches
    """
    client = create_client(handle)
    database = client.get_database()
    persons_a = [
        {"id": "P1", "name": "n1", "employed_by": "Org1"},
        {"id": "P2", "name": "n2", "age_in_years": 30},
        {"id": "P3", "name": "n3", "employed_by": "Org3"},
    ]
    persons_b = [
        {"id": "P1", "name": "n1", "employed_by": "Org1b"},  # changes org
        {"id": "P2", "name": "n2", "age_in_years": 30},  # no change
        {"id": "P3", "name": "n3", "employed_by": "Org3", "age_in_years": 33},  # same org and adds age
        {"id": "P4", "name": "n4", "employed_by": "Org4"},
    ]
    collection_a = database.create_collection("Person", alias="persons_a", recreate_if_exists=True)
    collection_b = database.create_collection("Person", alias="persons_b", recreate_if_exists=True)
    # p3 = collection_a.find({"id": "P3"}).rows[0]
    # assert p3["name"] == "n3"
    # assert "age" not in p3 - TODO; https://github.com/orgs/linkml/discussions/1975
    collection_a.insert(persons_a)
    collection_b.insert(persons_b)
    collection_a.set_identifier_attribute_name("id")
    collection_b.set_identifier_attribute_name("id")
    patches = collection_a.diff(collection_b)
    assert isinstance(patches, list)
    rev_patches = collection_b.diff(collection_a)
    print("## PATCHES")
    for p in patches:
        print(p)
    # {'op': 'add', 'path': '/P4', 'value': {'id': 'P4', 'name': 'n4', 'employed_by': 'Org4'}}
    # {'op': 'add', 'path': '/P3/age_in_years', 'value': 33}
    # {'op': 'replace', 'path': '/P1/employed_by', 'value': 'Org1b'}
    ops = sorted([p["op"] for p in patches])
    assert ops == ["add", "add", "replace"]
    collection_a.apply_patches(patches)
    assert collection_a.diff(collection_b) == []
    assert collection_b.diff(collection_a) == []
    collection_b.apply_patches(rev_patches)
    case = unittest.TestCase()
    case.assertCountEqual(collection_a.diff(collection_b), rev_patches)


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_store_nested(handle):
    """
    Test storing of nested objects

    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    obj = {
        "persons": PERSONS,
        "organizations": ORGANIZATIONS,
        "employed_at": EMPLOYED_AT,
    }
    database.store(obj)
    database.commit()
    persons_coll = database.get_collection("persons")
    qr = persons_coll.find()
    assert qr.num_rows == 2
    p1 = qr.rows[0]
    p1events = p1["history"]
    assert all(isinstance(e, dict) for e in p1events)
    events = {e["event"] for e in p1events}
    assert events == {"birth", "death", "hired"}
    ignore = ["history"]
    assert remove_none(qr.rows[0], ignore) == remove_none(obj["persons"][0], ignore)
    orgs_coll = database.get_collection("organizations")
    if False:
        # check induced class definition
        sv = database.schema_view
        for cn in sv.all_classes():
            print(f"CLASS: {cn}")
        slots = sv.class_induced_slots("organizations")
        assert len(slots) == 4
        slot = sv.induced_slot("changes", "organizations")
        assert slot.inlined
    qr = orgs_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["organizations"][0]


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_load_from_source(handle):
    """
    Test syncing with sources

    :param handle:
    :return:
    """
    pytest.skip("TODO - in progress")
    client = create_client(handle)
    database = client.get_database()
    coll = database.create_collection("Country", metadata=CollectionConfig(source_location=str(COUNTRIES_DATA_JSONL)))
    assert coll.find({}).num_rows > 0


# @pytest.mark.parametrize("handle", SCHEMES)  # TODO - mongodb
@pytest.mark.parametrize("handle", SCHEMES_PLUS)
@pytest.mark.parametrize(
    "type_alias",
    [
        (
            "Person",
            None,
        ),
        ("Person", "persons"),
    ],
)
def test_induced_schema(handle, type_alias):
    """
    Test inference of schema from data

    1. Create a database and collection
    2. Insert data
    3. Check that schema is inferred

    TODO: test partial induction case where schema is implicit but class name is in config

    :param handle: duckdb, mongodb, etc
    :param type_alias: a tuple of (ClassName, alias)
    :return:
    """
    typ, alias = type_alias
    client = create_client(handle)
    assert len(client.databases) == 1, "expected single (default) database in fresh client"
    database = client.get_database()
    assert database is not None, "expected singleton to be retrieved without name"
    if not isinstance(database, MongoDBDatabase):
        assert len(database.list_collections()) == 0, "fresh database should have no collections"
    if alias:
        # test the use of alias to refer to a collection
        collection = database.create_collection(typ, alias=alias, recreate_if_exists=True)
        assert collection.alias == alias, "expected alias to propagate from create_collection"
        assert collection.target_class_name == typ, "expected target_class_name to be set"
    else:
        # no alias set - the alias is equivalent to the class name (e.g. Person)
        collection = database.create_collection(typ, recreate_if_exists=True)
        assert collection.alias == typ, "expected alias to default to class name"
        assert collection.target_class_name == typ, "expected target_class_name to be set"
    assert len(database.list_collections()) == 1, "expected single collection to be created"
    assert (
        collection.class_definition() is None or not collection.class_definition().attributes
    ), "no explicit schema and no data to induce a schema from"
    qr = collection.find()
    assert qr.num_rows == 0, "fresh database and fresh collection should be empty"
    # --
    # insert data (should allow induction of schema)
    # --
    objs = [
        {"id": 1, "name": "n1"},
        {"id": 2, "name": "n2", "age_in_years": 30},
        {"id": 3, "name": "n3", "aliases": ["a", "b"]},
        {"id": 4, "name": "n4", "metadata": {"created": "2021-01-01", "editor": "joe"}},
        {"id": 5, "name": "n5", "addresses": [{"street": "1 foo street"}]},
    ]
    collection.insert(objs)
    collection.commit()
    assert collection.find().num_rows == len(objs), "expected all objects to be added"
    # assert collection.class_definition().name == typ
    assert collection.parent.schema_view is not None, "expected schema view to be initialized from data"
    assert collection.parent.schema_view.schema is not None, "expected schema to be initialized from data"
    # print(yaml_dumper.dumps(collection.parent.schema_view.schema))
    assert collection.parent.schema_view.schema.classes, "expected single class to be initialized from data"
    assert (
        len(collection.parent.schema_view.schema.classes.keys()) == 1
    ), "expected single class to be initialized from data"
    # print(typ, alias, collection.target_class_name, collection.parent.schema_view.schema.classes.keys())
    assert typ in collection.parent.schema_view.schema.classes.keys(), "name of class is collection name by default"
    assert (
        collection.parent.schema_view.schema.classes[typ].name == collection.target_class_name
    ), "name of class is collection name by default"
    assert collection.parent.schema_view.get_class(typ), "schema view should work"
    assert collection.class_definition() is not None, "expected class definition to be created"
    cd = collection.class_definition()
    assert cd.attributes
    assert sorted(cd.attributes.keys()) == ["addresses", "age_in_years", "aliases", "id", "metadata", "name"]
    assert len(database.list_collections()) == 1, "collections should be unmodified"
    assert collection.find().num_rows == len(objs), "expected no change in data"
    assert len(database.list_collections()) == 1, "collections should be unmodified"
    qr = collection.peek()
    assert remove_none(qr.rows[0]) == objs[0], "expected first object to be first"
    assert len(database.list_collections()) == 1, "collections should be unmodified"
    dummy_collection = database.get_collection("dummy", create_if_not_exists=True)
    assert dummy_collection.class_definition() is None, "new collection has no schema"
    with pytest.raises(KeyError):
        database.get_collection("dummy2", create_if_not_exists=False)
    # if alias:
    #    collection = database.get_collection(alias, create_if_not_exists=True)
    # else:
    #    collection = database.get_collection(name, create_if_not_exists=True)
    sv = database.schema_view
    cd = sv.get_class(typ)
    assert cd is not None, "class should be named using name (even if alias is set)"
    assert cd.name == typ
    assert len(cd.attributes) == 6, "expected 6 attributes induced from data"
    assert cd.attributes["id"].range == "integer", "expected id to be induced as integer"
    assert cd.attributes["name"].range == "string", "expected name to be induced as string"
    assert cd.attributes["age_in_years"].range == "integer", "expected age_in_years to be induced as integer"
    assert cd.attributes["addresses"].multivalued, "expected addresses to be multivalued"
    assert cd.attributes["addresses"].range != "string", "expected addresses range to be complex"
    assert cd.attributes["addresses"].inlined, "expected addresses to be inlined"
    assert cd.attributes["addresses"].inlined_as_list, "expected addresses to be inlined_as_list"
    assert cd.attributes["metadata"].inlined, "expected metadata to be inlined"
    # assert cd.attributes["metadata"].range != "string", "expected metadata range to be complex"
    induced_addresses = sv.induced_slot("addresses", typ)
    assert induced_addresses.multivalued, "expected addresses to be multivalued"
    collection.delete(objs[0])
    collection.delete(objs[2])
    collection.delete(objs[3])
    collection.delete(objs[4])
    qr = collection.find()
    assert qr.num_rows == 1, "expected 1 row after delete"
    assert remove_none(qr.rows[0]) == objs[1], "expected last object to be first after delete"
    assert not collection.delete_where({"age_in_years": 99}), "expected 0 rows to be deleted"
    qr = collection.find()
    assert qr.num_rows == 1, "delete with no matching conditions should have no effect"
    collection.delete_where({"age_in_years": 30})
    qr = collection.find()
    assert qr.num_rows == 0
    # --
    # new client object,
    # --
    client = create_client(handle, recreate_if_exists=False)
    assert len(client.databases) == 1
    database = client.get_database()
    collections = database.list_collections()
    if is_persistent(handle):
        assert len(collections) == 1, "expected collection to persist"
    else:
        assert len(collections) == 0, "expected collection to be recreated"
    new_objs = [{"id": 3, "name": "n3"}, {"id": 4, "name": "n4", "age_in_years": 44}]
    collection.insert(new_objs)


@pytest.mark.parametrize("handle", SCHEMES)
def test_induced_multivalued(handle):
    """
    Test induced schema and collection creation with multivalued slots

    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    collection = database.create_collection("foo", recreate_if_exists=True)
    objs = [
        {"id": 1, "name": "n1", "aliases": ["a", "b"]},
        {"id": 2, "name": "n2", "age_in_years": 30, "aliases": ["b", "c"]},
    ]
    collection.insert(objs)
    assert collection.parent.schema_view is not None
    assert collection.parent.schema_view.schema is not None
    assert collection.parent.schema_view.schema.classes
    assert collection.parent.schema_view.schema.classes["foo"]
    assert collection.parent.schema_view.schema.classes["foo"].name == collection.target_class_name
    assert collection.parent.schema_view.get_class("foo")
    assert collection.class_definition() is not None
    collection.query(collection._create_query())
    qr = collection.peek()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == objs[0]
    assert len(database.list_collections()) == 1
    sv = database.schema_view
    cd = sv.get_class("foo")
    assert cd is not None
    assert cd.name == "foo"
    for a in cd.attributes.values():
        print(a.name)
    assert len(cd.attributes) == 4
    assert cd.attributes["id"].range == "integer"
    assert cd.attributes["name"].range == "string"
    assert cd.attributes["age_in_years"].range == "integer"
    assert cd.attributes["aliases"].range == "string"
    assert cd.attributes["aliases"].multivalued
    # test facets with multivalued slots
    fcs = collection.query_facets(facet_columns=["aliases"])["aliases"]
    all_fcs = collection.query_facets()
    assert set(all_fcs["aliases"]) == set(fcs)
    assert set(fcs) == {("a", 1), ("b", 2), ("c", 1)}
    # check ordering
    assert fcs[0] == ("b", 2)
    for fc_key, fc_counts in all_fcs.items():
        for fc_val, num in fc_counts:
            # print(fc_key, fc_val, num)
            if fc_val is None:
                # Currently there is no distinction between a missing key and
                # explicit null value from that key
                continue
            if fc_val is None:
                w = None
            else:
                w = {fc_key: fc_val}
                if fc_key == "aliases":
                    # w = {fc_key: ("ARRAY_CONTAINS", fc_val)}
                    w = {fc_key: {"$in": fc_val}}
            fc_results = collection.find(w)
            assert fc_results.num_rows == num, f"unexpected diff for fc_key={fc_key}, fc_val={fc_val} // {w}, num={num}"
            facet_subq = collection.query_facets(w, facet_columns=[fc_key])
            if fc_key == "aliases":
                assert (fc_val, num) in facet_subq[fc_key]
            else:
                assert facet_subq[fc_key] == [(fc_val, num)], f"expected single facet result for {fc_key}={fc_val}"
    collection.delete(objs[0])
    qr = collection.find()
    assert qr.num_rows == 1, "expected 1 row after delete"
    assert remove_none(qr.rows[0]) == objs[1]
    collection.delete_where(qr.rows[0])
    qr = collection.find()
    assert qr.num_rows == 0, "expected 0 rows after deleting final object"


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_predefined_schema(schema_view, handle):
    """
    Test working with a predefined schema

    :param schema_view:
    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(schema_view)
    cd = schema_view.get_class("Person")
    assert cd.attributes["id"].identifier
    collection = database.create_collection("Person", recreate_if_exists=True)
    cd = collection.class_definition()
    assert cd is not None
    assert cd.name == "Person"
    assert len(cd.attributes) == 5
    assert cd.attributes["id"].identifier
    obj = {"id": "p1", "name": "n1"}
    collection.insert([obj])
    collection.query(collection._create_query())
    qr = collection.peek()
    assert qr.num_rows == 1
    ret_obj = qr.rows[0]
    ret_obj = {k: v for k, v in ret_obj.items() if v is not None}
    assert ret_obj == obj
    assert len(database.list_collections()) == 1
    qr = collection.get(["p1"])
    assert qr.num_rows == 1
    assert remove_none(qr.rows[0]) == obj
    obj2 = {"id": "p2", "name": "n2"}
    collection.insert([obj2])
    qr = collection.get(["p1"])
    assert qr.num_rows == 1
    assert remove_none(qr.rows[0]) == obj
    qr = collection.get(["p1", "p2"])
    assert qr.num_rows == 2


@pytest.mark.parametrize("handle", SCHEMES)
@pytest.mark.parametrize("index_class", [SimpleIndexer])
def test_facets(schema_view, handle, index_class):
    """
    Test faceted querying and counts, as well as search

    :param schema_view:
    :param handle:
    :param index_class:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(schema_view)
    collection = database.create_collection("Person", recreate_if_exists=True)
    objs = [
        {"id": "P1", "name": "n1", "occupation": "Welder", "moon": "Io"},
        {"id": "P2", "name": "n2", "occupation": "Welder", "moon": "Europa"},
        {"id": "P3", "name": "n3", "occupation": "Bricklayer", "moon": "Io"},
    ]
    collection.insert(objs)
    ix = index_class(name="test")
    collection.attach_indexer(ix)
    cases = [
        ("Bricklayer", "P3"),
        ("bricks", "P3"),
        ("welder on Io", "P1"),
        ("welding on europa", "P2"),
    ]
    for q, expected_top in cases:
        results = collection.search(q).ranked_rows
        top_result = results[0][1]
        assert top_result["id"] == expected_top
    r = collection.query_facets(facet_columns=["occupation"])
    assert r == {"occupation": [("Welder", 2), ("Bricklayer", 1)]}
    cases = [
        (["occupation"], {"occupation": "Welder"}, {"P1", "P2"}, {"occupation": [("Welder", 2)]}),
        (["occupation"], None, {"P1", "P2", "P3"}, {"occupation": [("Welder", 2), ("Bricklayer", 1)]}),
        (["moon"], None, {"P1", "P2", "P3"}, {"moon": [("Io", 2), ("Europa", 1)]}),
        (
            ["occupation", "moon"],
            None,
            {"P1", "P2", "P3"},
            {"occupation": [("Welder", 2), ("Bricklayer", 1)], "moon": [("Io", 2), ("Europa", 1)]},
        ),
        (
            [("occupation", "moon")],
            None,
            {"P1", "P2", "P3"},
            {("occupation", "moon"): [("Welder", "Io", 1), ("Bricklayer", "Io", 1), ("Welder", "Europa", 1)]},
        ),
    ]
    for facet_slots, where, expected, expected_facets in cases:
        qr = database.query(Query(from_table="Person", where_clause=where))
        ids = {row["id"] for row in qr.rows}
        assert ids == expected
        qr = collection.find(where)
        ids = {row["id"] for row in qr.rows}
        assert ids == expected
        r = collection.query_facets(where=where, facet_columns=facet_slots)
        for k, v in r.items():
            assert set(r[k]) == set(expected_facets[k])
            del expected_facets[k]
        assert expected_facets == {}
    assert database.list_collection_names() == ["Person"]
    assert len(database.list_collection_names(include_internal=True)) > 1


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_validation(countries_schema_view, handle):
    """
    Test validation of objects against a schema

    :param countries_schema_view:
    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(countries_schema_view)
    objects = load_objects(COUNTRIES_DATA_JSONL)
    collection = database.create_collection("Country", recreate_if_exists=True)
    collection.insert(objects)
    vrs = list(collection.iter_validate_collection())
    assert vrs == []
    collection.insert(
        {"name": "Foo", "code": "SPACES NOT VALID", "capital": "Zog", "continent": "Europe", "languages": ["English"]}
    )
    vrs = list(collection.iter_validate_collection())
    for vr in vrs:
        print(yaml_dumper.dumps(vr))
    assert len(vrs) == 1


@pytest.mark.parametrize("handle", SCHEMES)
def test_validate_referential_integrity(personinfo_schema_view, handle):
    """
    Test validating where one collection references another

    :param handle:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    database.metadata.ensure_referential_integrity = True
    obj = {
        "persons": PERSONS,
        "organizations": ORGANIZATIONS,
        "employed_at": EMPLOYED_AT
        + [
            {"person": 3, "organization": "Org1"},
            {"person": 1, "organization": "Org3"},
        ],
    }
    database.store(obj)
    database.commit()
    database.set_schema_view(personinfo_schema_view)
    objs = database.get_collection("persons").find().rows
    for obj in objs:
        if obj["history"]:
            assert isinstance(obj["history"][0], dict)
    results = list(database.iter_validate_database())
    for vr in results:
        print(yaml_dumper.dumps(vr))
    assert len(results) == 2


@pytest.mark.parametrize("auto_index", [True, False])
@pytest.mark.parametrize("indexer_type", ["simple"])
@pytest.mark.parametrize("handle", SCHEMES)
def test_search(handle, countries_schema_view, auto_index, indexer_type):
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(countries_schema_view)
    objects = load_objects(COUNTRIES_DATA_JSONL)
    collection = database.create_collection("Country", recreate_if_exists=True)
    collection.insert(objects)
    indexer = get_indexer(indexer_type)
    collection.attach_indexer(indexer, auto_index=auto_index)
    ix_collection_name = collection.get_index_collection_name(indexer)
    ix_collection = database.get_collection(ix_collection_name)
    ix_collection_size = ix_collection.size()
    if auto_index:
        assert ix_collection_size == collection.size(), "expected index to be pre-populated"
    else:
        assert ix_collection_size == 0, "expected delayed population of index"
    for row in collection.find(limit=5).rows:
        search_term = str(row)
        results = collection.search(search_term, limit=5)
        scores = [score for score, _ in results.ranked_rows]
        assert scores, "expected scores for search"
        # ensure for any consecutive pair of scores, the first is greater than the second
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "results should be ranked"
        result_objs = [r for _, r in results.ranked_rows]
        assert any(r for r in result_objs if r["code"] == row["code"]), "query should be in results"
        assert result_objs == results.rows, "results should be same regardless of method"
    # ensure index updates on delete
    num_objects = len(objects)
    assert num_objects
    first_half = objects[: (num_objects // 2)]
    second_half = objects[(num_objects // 2) :]
    assert len(first_half) + len(second_half) == num_objects
    collection.delete(first_half)
    # check deletion worked
    assert collection.find(limit=-1).num_rows == len(second_half)
    assert collection.size() == len(second_half)
    for row in first_half:
        search_term = str(row)
        results = collection.search(search_term, limit=5)
        result_objs = [r for _, r in results.ranked_rows]
        assert not any(r for r in result_objs if r["code"] == row["code"]), "deleted object should not be results"
    for row in second_half:
        search_term = str(row)
        results = collection.search(search_term, limit=5)
        result_objs = [r for _, r in results.ranked_rows]
        assert result_objs
        assert any(r for r in result_objs if r["code"] == row["code"]), "undeleted object should be in results"


@pytest.mark.parametrize("auto_index", [True, False])
@pytest.mark.parametrize("indexer_type", ["simple"])
@pytest.mark.parametrize("handle", SCHEMES)
def test_search_multiple_indexes(handle, countries_schema_view, auto_index, indexer_type):
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(countries_schema_view)
    objects = load_objects(COUNTRIES_DATA_JSONL)
    collection = database.create_collection("Country", recreate_if_exists=True)
    collection.insert(objects)
    default_indexer = get_indexer(indexer_type, name="default")
    name_indexer = get_indexer(indexer_type, name="name_indexer", index_attributes=["name"])
    fstring_indexer = get_indexer(indexer_type, name="fstring_indexer", text_template="{name} :: {code} :: {capital}")
    indexers = {ixr.name: ixr for ixr in [default_indexer, name_indexer, fstring_indexer]}
    ix_collection_name_to_indexer_name = {}
    for indexer_name, indexer in indexers.items():
        collection.attach_indexer(indexer, auto_index=auto_index)
        ix_collection_name = collection.get_index_collection_name(indexer)
        ix_collection = database.get_collection(ix_collection_name)
        ix_collection_size = ix_collection.size()
        ix_collection_name_to_indexer_name[ix_collection_name] = indexer_name
        if auto_index:
            assert ix_collection_size == collection.size(), "expected index to be pre-populated"
        else:
            assert ix_collection_size == 0, "expected delayed population of index"
    assert len(ix_collection_name_to_indexer_name) == len(indexers), "expected 1-1 mapping"
    for row in collection.find(limit=5).rows:
        for indexer in indexers.values():
            search_term = indexer.object_to_text(row)
            results = collection.search(search_term, index_name=indexer.name, limit=5)
            scores = [score for score, _ in results.ranked_rows]
            assert scores, "expected scores for search"
            # ensure for any consecutive pair of scores, the first is greater than the second
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], "results should be ranked"
            result_objs = [r for _, r in results.ranked_rows]
            assert any(r for r in result_objs if r["code"] == row["code"]), "query should be in results"
            assert result_objs == results.rows, "results should be same regardless of method"


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
def test_from_config_object(handle):
    """
    Test creating a client from a configuration

    :return:
    """
    config = ClientConfig(
        databases={
            "test1": {
                "handle": handle,
                "collections": {
                    "persons": {"type": "Person"},
                },
            }
        }
    )
    assert config.databases["test1"].handle == handle
    client = Client().from_config(config)
    db1 = client.get_database("test1")
    assert db1 is not None
    collection1 = db1.get_collection("persons")
    assert collection1 is not None


@pytest.mark.parametrize(
    "name,inserts",
    [
        (
            "conf1",
            [
                (
                    "personnel",
                    "persons",
                    [
                        {"id": "p1", "name": "n1", "employed_by": "org1"},
                        {"id": "p2", "name": "n2", "employed_by": "org1"},
                    ],
                ),
                ("personnel", "organizations", [{"id": "org1", "name": "o1", "category": "non-profit"}]),
            ],
        ),
        (
            "conf1",
            [
                (
                    "clinical",
                    "samples",
                    [
                        {"id": "s1", "name": "s1", "employed_by": "org1"},
                        {"id": "s2", "name": "s2", "employed_by": "org1"},
                    ],
                ),
            ],
        ),
        (
            "conf1",
            [
                (
                    "phenopackets_duckdb",
                    "test",
                    [
                        {
                            "id": "s1",
                            "phenotypicFeatures": [
                                {"type": {"id": "HP:0001263", "label": "Global developmental delay"}}
                            ],
                        },
                    ],
                ),
            ],
        ),
        (
            "conf1",
            [
                (
                    "phenopackets_fs",
                    "test",
                    [
                        {
                            "id": "s1",
                            "phenotypicFeatures": [
                                {"type": {"id": "HP:0001263", "label": "Global developmental delay"}}
                            ],
                        },
                    ],
                ),
            ],
        ),
    ],
)
def test_from_config_file(name, inserts):
    """
    Test creating a client from a configuration file

    :param name: configuration name
    :param inserts: list of (db_alias, collection_alias, rows) tuples
    :return:
    """
    source_dir = INPUT_DIR / "configurations" / name
    target_dir = OUTPUT_DIR / "configurations" / name
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    schema_dir = INPUT_DIR / "schemas"
    shutil.copytree(schema_dir, target_dir, dirs_exist_ok=True)

    path = target_dir / "config.yaml"
    client = Client().from_config(path, base_dir=path.parent)
    config = client.metadata
    # index = SimpleIndexer(name="test")

    for db_alias in config.databases:
        db = client.get_database(db_alias)
        sv = db.schema_view
        for coll in db.list_collections():
            logger.info(f"Looking up coll: {coll.alias} in {config.databases[db_alias].collections.keys()}")
            coll_config = config.databases[db_alias].collections[coll.alias]
            if coll_config.attributes:
                logger.info(f"Checking CD; expected as schema has {sv.schema.classes.keys()}")
                cd = coll.class_definition()
                assert cd is not None, f"expected a class definitions from {coll_config}"
                assert cd.attributes.keys() == coll_config.attributes.keys(), f"columns do not match for {coll_config}"

    for insert in inserts:
        db_alias, coll_alias, objs = insert
        db = client.get_database(db_alias)
        collection = db.get_collection(coll_alias)
        assert collection is not None
        assert collection.alias == coll_alias
        collection.insert(objs)
        logger.info(f"Searching in {coll_alias}; TC={collection.target_class_name}, ALIAS={collection.alias}")
        qr = collection.find()
        assert qr.num_rows == len(objs), f"expected {len(objs)} for n={coll_alias} I= {insert}"
        collection.commit()
        # check works with a fresh connection
        db2 = client.get_database(db_alias)
        collection2 = db2.get_collection(coll_alias)
        qr2 = collection2.find()
        assert qr2.num_rows == len(objs), f"expected {len(objs)} for n={coll_alias} I= {insert}"

    for db_alias in config.databases:
        db = client.get_database(db_alias)
        for coll in db.list_collections():
            index = get_indexer("simple")
            coll.attach_indexer(index, auto_index=True)
            _results = coll.search("org")
            # assert len(results.ranked_rows) > 0, f"did not find search results for {coll.alias} using {index}"


@pytest.mark.parametrize("handle", SCHEMES_PLUS)
@pytest.mark.parametrize(
    "data_path, schema_location", [("Phenopacket-001.json", "phenopackets_linkml/phenopackets.yaml")]
)
def test_domain_objects(data_path, schema_location, handle):
    """
    Test loading domain objects

    :param handle:
    :param data_path:
    :param schema_location:
    :return:
    """
    client = create_client(handle)
    database = client.get_database()
    schema_dir = INPUT_DIR / "schemas" / schema_location
    data_dir = INPUT_DIR
    schema_view = SchemaView(schema_dir)
    database.set_schema_view(schema_view)
    collection = database.create_collection("Phenopacket", alias="test", recreate_if_exists=True)
    data_path = data_dir / data_path
    objs = json.load(open(data_path))
    collection.insert(objs)
    qr = collection.find()
    rows = [remove_none(r) for r in qr.rows]
    if isinstance(objs, list):
        assert qr.num_rows == len(objs)
        for obj in objs:
            assert obj in rows
    else:
        assert qr.num_rows == 1
        assert rows[0] == objs
    facets = collection.query_facets()
    print(facets)


@pytest.mark.integration
def test_integration():
    abs_db_path = str(TEST_DB.resolve())

    # Now, construct the connection string
    handle = f"duckdb:///{abs_db_path}"
    database = DuckDBDatabase(handle)
    collection = database.get_collection("gaf_association")
    qr = collection.find()
    print(type(qr.rows))
    print(type(qr.rows[0]))
    row = qr.rows[0]
    print(type(row))
    print(row)
    print(row.keys())
    # print(pd.DataFrame(qr.rows))
    # print(qr.rows_dataframe)


MONARCH_KG_DB = "https://data.monarchinitiative.org/monarch-kg/latest/monarch-kg.db.gz"


@pytest.mark.integration
def test_integration_kg():
    pytest.skip("TODO - in progress")
    path = LINKML_STORE_MODULE.ensure_gunzip(url=MONARCH_KG_DB, autoclean=True)
    print(path)
    handle = f"duckdb:///{path}"
    database = DuckDBDatabase(handle)
    _schema = introspect_schema(database.engine)
    collection = database.get_collection("denormalized_edges")
    qr = collection.find()
    print(qr.num_rows)
    print(qr.rows_dataframe)


@pytest.mark.integration
def test_integration_store():
    path = pystow.ensure("tmp", "eccode.json", url="https://w3id.org/biopragmatics/resources/eccode/eccode.json")
    graphdoc = json.load(open(path))
    graph = graphdoc["graphs"][0]
    client = Client()
    db = client.attach_database("duckdb")
    # db.store(graph)
    # coll = db.create_collection("Edge", "edges")
    # coll.induce_class_definition_from_objects(graph["edges"])
    # cd = coll.class_definition()
    # assert cd is not None
    # assert len(cd.attributes) > 2
    # coll.add(graph["edges"])
    coll = db.create_collection("Node", "nodes")
    coll.insert(graph["nodes"])
    index = SimpleIndexer(name="test")
    coll.attach_indexer(index)
    cases = ["lyase", "lysine", "abc"]
    for case in cases:
        results = coll.search(case).ranked_rows
        print(f"Results for {case}")
        for score, r in results[0:3]:
            print(f"  * {score} :: {r['id']} :: {r['lbl']}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "collection_name,where,has_data",
    [
        ("annotation", None, True),
        ("annotation", {"isa_partof_closure": "GO:0005634"}, True),
        ("annotation", {"taxon_closure": "NCBITaxon:83333"}, True),
        ("annotation", {"taxon_closure": "NCBITaxon:9606"}, True),
        ("annotation", {"taxon_closure": "NCBITaxon:9606", "isa_partof_closure": "GO:0005634"}, True),
        ("annotation", {"taxon_closure": "NCBITaxon:83333", "isa_partof_closure": "GO:0060174"}, False),
        ("bioentity", None, True),
        ("ontology_class", None, True),
        ("no_such_collection", None, False),
    ],
)
def test_integration_solr(collection_name, where, has_data):
    handle = "https://golr.geneontology.org/solr"
    database = SolrDatabase(handle)
    database.metadata.collection_type_slot = "document_category"
    collection = database.get_collection(collection_name)
    assert collection.parent.metadata.collection_type_slot == "document_category"
    qr = collection.find(where)
    print(qr.num_rows)
    for row in qr.rows:
        print(row)
    if has_data:
        assert qr.num_rows > 0, f"expected data in {collection_name}"
    else:
        assert qr.num_rows == 0, f"expected no data in {collection_name}"
    qr = collection.query_facets(where, facet_columns=["taxon_closure"])


def test_sql_utils():
    import duckdb

    # con = duckdb.connect("file.db")
    con = duckdb.connect()
    ddl = """
    CREATE TABLE foo (
        a JSON
    );
    """
    con.sql(ddl)
    con.sql("INSERT INTO foo VALUES ([1, {a: 2}::JSON])")
    r = con.sql("SELECT * FROM foo").fetchall()
    assert isinstance(r[0][0], str)
    r = con.sql("SELECT json_extract(a, '/1/a') AS e FROM foo").fetchall()
    assert r[0][0] == "2"
