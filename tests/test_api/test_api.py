import json
import shutil

import pystow
import pytest
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_store.api.client import Client
from linkml_store.api.config import ClientConfig
from linkml_store.api.queries import Query
from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
from linkml_store.constants import LINKML_STORE_MODULE
from linkml_store.index.implementations.simple_index import SimpleIndex
from linkml_store.utils.sql_utils import introspect_schema

from tests import INPUT_DIR, OUTPUT_DIR

TEST_DB = INPUT_DIR / "integration" / "mgi.db"

SCHEMES = [
    "duckdb",
]

DEFAULT_DB = "default"


def remove_none(d: dict, additional_keys=None):
    if additional_keys is None:
        additional_keys = []
    return {k: v for k, v in d.items() if v is not None and k not in additional_keys}


@pytest.fixture()
def schema_view() -> SchemaView:
    sb = SchemaBuilder()
    id_slot = SlotDefinition("id", identifier=True)
    name_slot = SlotDefinition("name")
    age_in_years_slot = SlotDefinition("age_in_years", range="integer")
    occupation_slot = SlotDefinition("occupation")
    moon_slot = SlotDefinition("moon")
    sb.add_class("Person", [id_slot, name_slot, age_in_years_slot, occupation_slot, moon_slot], use_attributes=True)
    return SchemaView(sb.schema)


def create_client(handle: str) -> Client:
    client = Client()
    client.attach_database(handle, alias=DEFAULT_DB)
    return client


@pytest.mark.parametrize("handle", SCHEMES)
@pytest.mark.parametrize(
    "name_alias",
    [
        (
            "Person",
            None,
        ),
        ("Person", "persons"),
    ],
)
def test_induced(handle, name_alias):
    name, alias = name_alias
    client = create_client(handle)
    database = client.get_database()
    # database = database_class()
    assert len(database.list_collections()) == 0
    if alias:
        collection = database.create_collection(name, alias=alias)
    else:
        collection = database.create_collection(name)
    assert len(database.list_collections()) == 1
    assert collection.class_definition() is None
    # check is empty
    qr = collection.find()
    assert qr.num_rows == 0
    objs = [{"id": 1, "name": "n1"}, {"id": 2, "name": "n2", "age_in_years": 30}]
    collection.add(objs)
    assert collection.parent.schema_view is not None
    assert collection.parent.schema_view.schema is not None
    assert collection.parent.schema_view.schema.classes
    assert collection.parent.schema_view.schema.classes[name]
    assert collection.parent.schema_view.schema.classes[name].name == collection.name
    assert collection.parent.schema_view.get_class(name)
    assert collection.class_definition() is not None
    assert len(database.list_collections()) == 1, "xxxx"
    collection.query(collection._create_query())
    assert len(database.list_collections()) == 1, "FOOOZ"
    qr = collection.peek()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == objs[0]
    for coll in database.list_collections():
        print(coll.name)
    assert len(database.list_collections()) == 1, "BAAAZ"
    dummy_collection = database.get_collection("dummy", create_if_not_exists=True)
    assert dummy_collection.class_definition() is None
    with pytest.raises(KeyError):
        database.get_collection("dummy2", create_if_not_exists=False)
    if alias:
        collection = database.get_collection(alias, create_if_not_exists=True)
    else:
        collection = database.get_collection(name, create_if_not_exists=True)
    sv = database.schema_view
    cd = sv.get_class(name)
    assert cd is not None
    assert cd.name == name
    assert len(cd.attributes) == 3
    assert cd.attributes["id"].range == "integer"
    assert cd.attributes["name"].range == "string"
    assert cd.attributes["age_in_years"].range == "integer"
    collection.delete(objs[0])
    qr = collection.find()
    assert qr.num_rows == 1
    assert remove_none(qr.rows[0]) == objs[1]
    collection.delete_where({"age_in_years": 99})
    qr = collection.find()
    assert qr.num_rows == 1
    collection.delete_where({"age_in_years": 30})
    qr = collection.find()
    assert qr.num_rows == 0


@pytest.mark.parametrize("handle", SCHEMES)
def test_store(handle):
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
    orgs_coll = database.get_collection("organizations")
    qr = orgs_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["organizations"][0]


@pytest.mark.parametrize("handle", SCHEMES)
def test_store_nested(handle):
    client = create_client(handle)
    database = client.get_database()
    obj = {
        "persons": [
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
        ],
        "organizations": [
            {"id": "Org1", "name": "org1"},
            {"id": "Org2", "name": "org2", "found_date": "2021-01-01"},
        ],
    }
    database.store(obj)
    database.commit()
    persons_coll = database.get_collection("persons")
    qr = persons_coll.find()
    assert qr.num_rows == 2
    p1 = qr.rows[0]
    p1events = p1["history"]
    assert all(isinstance(e, dict) for e in p1events)
    ignore = ["history"]
    assert remove_none(qr.rows[0], ignore) == remove_none(obj["persons"][0], ignore)
    orgs_coll = database.get_collection("organizations")
    qr = orgs_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["organizations"][0]


@pytest.mark.parametrize("handle", SCHEMES)
def test_induced_multivalued(handle):
    client = create_client(handle)
    database = client.get_database()
    collection = database.create_collection("foo")
    objs = [
        {"id": 1, "name": "n1", "aliases": ["a", "b"]},
        {"id": 2, "name": "n2", "age_in_years": 30, "aliases": ["b", "c"]},
    ]
    collection.add(objs)
    assert collection.parent.schema_view is not None
    assert collection.parent.schema_view.schema is not None
    assert collection.parent.schema_view.schema.classes
    assert collection.parent.schema_view.schema.classes["foo"]
    assert collection.parent.schema_view.schema.classes["foo"].name == collection.name
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
                    #w = {fc_key: ("ARRAY_CONTAINS", fc_val)}
                    w = {fc_key: {"$contains": fc_val}}
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
    assert len(cd.attributes) == 5
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
@pytest.mark.parametrize("index_class", [SimpleIndex])
def test_facets(schema_view, handle, index_class):
    client = create_client(handle)
    database = client.get_database()
    database.set_schema_view(schema_view)
    collection = database.create_collection("Person")
    objs = [
        {"id": "P1", "name": "n1", "occupation": "Welder", "moon": "Io"},
        {"id": "P2", "name": "n2", "occupation": "Welder", "moon": "Europa"},
        {"id": "P3", "name": "n3", "occupation": "Bricklayer", "moon": "Io"},
    ]
    collection.add(objs)
    ix = index_class(name="test")
    collection.attach_index(ix)
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


def test_from_config():
    config = ClientConfig(
        databases={
            "test1": {
                "handle": "duckdb",
                "collections": {
                    "persons": {"category": "Person"},
                }
            }
        })
    assert config.databases["test1"].handle == "duckdb"
    client = Client().from_config(config)
    db1 = client.get_database("test1")
    assert db1 is not None
    collection1 = db1.get_collection("persons")
    assert collection1 is not None


@pytest.mark.parametrize("name,inserts", [
    ("conf1", [
        ("personnel", "persons", [
            {"id": "p1", "name": "n1", "employed_by": "org1"},
            {"id": "p2", "name": "n2", "employed_by": "org1"},
                                  ]),
        ("personnel", "organizations", [{"id": "org1", "name": "o1", "category": "non-profit"}]),
        ]
     ),
     ("conf1", [
        ("clinical", "samples", [
            {"id": "s1", "name": "s1", "employed_by": "org1"},
            {"id": "s2", "name": "s2", "employed_by": "org1"},
                                  ]),
        ]
     ),
])
def test_from_config_file(name, inserts):
    source_dir = INPUT_DIR / "configurations" / name
    target_dir = OUTPUT_DIR / "configurations" / name
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    path = target_dir / "config.yaml"
    client = Client().from_config(path)
    config = client.config
    index = SimpleIndex(name="test")

    for db_name in config.databases:
        print(db_name)
        db = client.get_database(db_name)
        for coll in db.list_collections():
            print(coll.name)
            coll_config = config.databases[db_name].collections[coll.name]
            if coll_config.attributes:
                cd = coll.class_definition()
                assert cd is not None
                assert cd.attributes.keys() == coll_config.attributes.keys()

    for insert in inserts:
        db_name, coll_name, objs = insert
        db = client.get_database(db_name)
        collection = db.get_collection(coll_name)
        collection.add(objs)
        qr = collection.find()
        assert qr.num_rows == len(objs)

    for db_name in config.databases:
        db = client.get_database(db_name)
        for coll in db.list_collections():
            coll.attach_index(index)
            _results = coll.search("e")



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
    coll.add(graph["nodes"])
    index = SimpleIndex(name="test")
    coll.attach_index(index)
    cases = ["lyase", "lysine", "abc"]
    for case in cases:
        results = coll.search(case).ranked_rows
        print(f"Results for {case}")
        for score, r in results[0:3]:
            print(f"  * {score} :: {r['id']} :: {r['lbl']}")


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
