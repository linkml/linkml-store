import pandas as pd
import pytest
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api.client import Client
from linkml_store.api.queries import Query
from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
from tests import INPUT_DIR

TEST_DB = INPUT_DIR / "integration" / "mgi.db"

SCHEMES = [
    "duckdb",
]

DEFAULT_DB = "default"

def remove_none(d: dict):
    return {k: v for k, v in d.items() if v is not None}

@pytest.fixture()
def schema_view() -> SchemaView:
    sb = SchemaBuilder()
    id_slot = SlotDefinition("id", identifier=True)
    name_slot = SlotDefinition("name")
    age_in_years_slot = SlotDefinition("age_in_years", range="integer")
    occupation_slot = SlotDefinition("occupation")
    sb.add_class("Person", [id_slot, name_slot, age_in_years_slot, occupation_slot], use_attributes=True)
    return SchemaView(sb.schema)


def create_client(handle: str) -> Client:
    client = Client()
    client.attach_database(handle, alias=DEFAULT_DB)
    return client


@pytest.mark.parametrize("handle", SCHEMES)
def test_induced(handle):
    client = create_client(handle)
    database = client.get_database()
    # database = database_class()
    assert len(database.list_collections()) == 0
    collection = database.create_collection("foo")
    assert collection.class_definition() is None
    objs = [{"id": 1, "name": "n1"},
            {"id": 2, "name": "n2", "age_in_years": 30}]
    collection.add(objs)
    collection.query(collection._create_query())
    qr = collection.peek()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == objs[0]
    assert len(database.list_collections()) == 1
    sv = database.schema_view
    cd = sv.get_class("foo")
    assert cd is not None
    assert cd.name == "foo"
    assert len(cd.attributes) == 3
    assert cd.attributes["id"].range == "integer"
    assert cd.attributes["name"].range == "string"
    # TODO:
    # assert cd.attributes["age_in_years"].range == "integer"


@pytest.mark.parametrize("handle", SCHEMES)
def test_schema(schema_view, handle):
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(schema_view)
    cd = schema_view.get_class("Person")
    assert cd.attributes["id"].identifier
    collection = database.create_collection("Person")
    cd = collection.class_definition()
    assert cd is not None
    assert cd.name == "Person"
    assert len(cd.attributes) == 4
    assert cd.attributes["id"].identifier
    obj = {"id": "p1", "name": "n1"}
    collection.add([obj])
    collection.query(collection._create_query())
    qr = collection.peek()
    assert qr.num_rows == 1
    ret_obj = qr.rows[0]
    ret_obj = {k: v for k, v in ret_obj.items() if v is not None}
    assert ret_obj == obj
    assert len(database.list_collections()) == 1


@pytest.mark.parametrize("handle", SCHEMES)
def test_facets(schema_view, handle):
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(schema_view)
    collection = database.create_collection("Person")
    objs = [
        {"id": "P1", "name": "n1", "occupation": "Welder"},
        {"id": "P2", "name": "n2", "occupation": "Welder"},
        {"id": "P3", "name": "n3", "occupation": "Bricklayer"},
    ]
    collection.add(objs)
    for where, expected in [({"occupation": "Welder"}, {"P1", "P2"})]:
        qr = database.query(Query(from_table="Person", where_clause=where))
        ids = {row["id"] for row in qr.rows}
        assert ids == expected
        qr = collection.find(where)
        ids = {row["id"] for row in qr.rows}
        assert ids == expected
    r = collection.query_facets({}, ["occupation"])
    print(r)


def test_integration():
    abs_db_path = str(TEST_DB.resolve())

    # Now, construct the connection string
    handle = f"duckdb:///{abs_db_path}"  # Notice the three slashes (duckdb:///absolute/path/to/db)
    database = DuckDBDatabase(handle)
    collection = database.get_collection("gaf_association")
    qr = collection.find()
    print(type(qr.rows))
    print(type(qr.rows[0]))
    row = qr.rows[0]
    print(type(row))
    print(row)
    print(row.keys())
    print(pd.DataFrame(qr.rows))
    # print(qr.rows_dataframe)

