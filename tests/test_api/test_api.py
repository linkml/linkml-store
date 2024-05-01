import json
import shutil
from pathlib import Path

import pystow
import pytest
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.linkml_model import SlotDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_store.api.client import Client
from linkml_store.api.config import ClientConfig
from linkml_store.api.queries import Query
from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
from linkml_store.api.stores.solr.solr_database import SolrDatabase
from linkml_store.constants import LINKML_STORE_MODULE
from linkml_store.index.implementations.simple_indexer import SimpleIndexer
from linkml_store.utils.format_utils import load_objects
from linkml_store.utils.sql_utils import introspect_schema

from tests import COUNTRIES_DATA_JSONL, COUNTRIES_SCHEMA, INPUT_DIR, OUTPUT_DIR

TEST_DB = INPUT_DIR / "integration" / "mgi.db"
TEMP_DB_PATH = OUTPUT_DIR / "temp.db"

SCHEMES = [
    "duckdb",
    f"duckdb:///{TEMP_DB_PATH}",
#    "mongodb://localhost:27017/test_db",
]

DEFAULT_DB = "default"


def is_persistent(handle: str) -> bool:
    return ".db" in handle


def remove_none(d: dict, additional_keys=None):
    """
    Remove keys with None values from a dictionary

    :param d:
    :param additional_keys:
    :return:
    """
    if additional_keys is None:
        additional_keys = []
    return {k: v for k, v in d.items() if v is not None and k not in additional_keys}


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


def create_client(handle: str, recreate_if_exists=True) -> Client:
    """
    Create a client with a database attached

    :param handle:
    :param recreate_if_exists:
    :return:
    """
    client = Client()
    if handle.endswith(".db"):
        path = handle.replace("duckdb:///", "")
        if recreate_if_exists:
            print(f"UNLINKING: {path}")
            Path(path).unlink(missing_ok=True)
            assert not Path(path).exists()
    if handle.startswith("mongodb:"):
        client.drop_all_databases()
    client.attach_database(handle, alias=DEFAULT_DB)
    print(f"ATTACHED: {handle}")
    return client


@pytest.mark.parametrize("handle", SCHEMES)
def test_store(handle):
    """
    Tests storing of objects in a database automatically creating collections

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
    orgs_coll = database.get_collection("organizations")
    qr = orgs_coll.find()
    assert qr.num_rows == 2
    assert remove_none(qr.rows[0]) == obj["organizations"][0]


@pytest.mark.parametrize("handle", SCHEMES)
def test_store_nested(handle):
    """
    Test storing of nested objects

    :param handle:
    :return:
    """
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
def test_induced_schema(handle, name_alias):
    """
    Test induced schema and collection creation

    :param handle:
    :param name_alias:
    :return:
    """
    name, alias = name_alias
    client = create_client(handle)
    assert len(client.databases) == 1
    database = client.get_database()
    assert len(database.list_collections()) == 0, "fresh database should have no collections"
    if alias:
        collection = database.create_collection(name, alias=alias)
    else:
        collection = database.create_collection(name)
    assert len(database.list_collections()) == 1, "expected collection to be created"
    assert collection.class_definition() is None, "no explicit schema and no data to induce from"
    # check is empty
    qr = collection.find()
    assert qr.num_rows == 0, "database should be empty"
    objs = [{"id": 1, "name": "n1"}, {"id": 2, "name": "n2", "age_in_years": 30}]
    collection.insert(objs)
    assert collection.find().num_rows == len(objs), "expected all objects to be added"
    assert collection.parent.schema_view is not None, "expected schema view to be initialized from data"
    assert collection.parent.schema_view.schema is not None, "expected schema to be initialized from data"
    assert collection.parent.schema_view.schema.classes, "expected single class to be initialized from data"
    assert len(collection.parent.schema_view.schema.classes) == 1, "expected single class to be initialized from data"
    assert collection.parent.schema_view.schema.classes[name], "name of class is collection name by default"
    assert (
        collection.parent.schema_view.schema.classes[name].name == collection.name
    ), "name of class is collection name by default"
    assert collection.parent.schema_view.get_class(name), "schema view should work"
    assert collection.class_definition() is not None, "expected class definition to be created"
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
    cd = sv.get_class(name)
    assert cd is not None, "class should be named using name (even if alias is set)"
    assert cd.name == name
    assert len(cd.attributes) == 3, "expected 3 attributes induced from data"
    assert cd.attributes["id"].range == "integer", "expected id to be induced as integer"
    assert cd.attributes["name"].range == "string", "expected name to be induced as string"
    assert cd.attributes["age_in_years"].range == "integer", "expected age_in_years to be induced as integer"
    collection.delete(objs[0])
    qr = collection.find()
    assert qr.num_rows == 1, "expected 1 row after delete"
    assert remove_none(qr.rows[0]) == objs[1], "expected second object to be first after delete"
    assert collection.delete_where({"age_in_years": 99}) <= 0, "expected 0 rows to be deleted"
    qr = collection.find()
    assert qr.num_rows == 1, "delete with no matching conditions should have no effect"
    collection.delete_where({"age_in_years": 30})
    qr = collection.find()
    assert qr.num_rows == 0
    # recreate
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
    collection = database.create_collection("foo")
    objs = [
        {"id": 1, "name": "n1", "aliases": ["a", "b"]},
        {"id": 2, "name": "n2", "age_in_years": 30, "aliases": ["b", "c"]},
    ]
    collection.insert(objs)
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
                    # w = {fc_key: ("ARRAY_CONTAINS", fc_val)}
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
    collection.delete_where(qr.rows[0])
    qr = collection.find()
    assert qr.num_rows == 0, "expected 0 rows after deleting final object"


@pytest.mark.parametrize("handle", SCHEMES)
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
    collection = database.create_collection("Person")
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
    collection = database.create_collection("Person")
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


@pytest.mark.parametrize("handle", SCHEMES)
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
    collection = database.create_collection("Country")
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


def test_from_config_object():
    """
    Test creating a client from a configuration

    :return:
    """
    config = ClientConfig(
        databases={
            "test1": {
                "handle": "duckdb",
                "collections": {
                    "persons": {"category": "Person"},
                },
            }
        }
    )
    assert config.databases["test1"].handle == "duckdb"
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
    ],
)
def test_from_config_file(name, inserts):
    """
    Test creating a client from a configuration file

    :param name:
    :param inserts:
    :return:
    """
    source_dir = INPUT_DIR / "configurations" / name
    target_dir = OUTPUT_DIR / "configurations" / name
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    path = target_dir / "config.yaml"
    client = Client().from_config(path)
    config = client.metadata
    index = SimpleIndexer(name="test")

    for db_name in config.databases:
        print(f"DB: {db_name}")
        db = client.get_database(db_name)
        sv = db.schema_view
        print(f"SV: {sv.schema.classes.keys()}")
        for coll in db.list_collections():
            print(f"Looking up coll: {coll.name} in {config.databases[db_name].collections.keys()}")
            coll_config = config.databases[db_name].collections[coll.name]
            if coll_config.attributes:
                print(f"Checking CD; expected as schema has {sv.schema.classes.keys()}")
                cd = coll.class_definition()
                assert cd is not None
                assert cd.attributes.keys() == coll_config.attributes.keys()

    for insert in inserts:
        db_name, coll_name, objs = insert
        db = client.get_database(db_name)
        collection = db.get_collection(coll_name)
        assert collection is not None
        assert collection.name == coll_name
        collection.insert(objs)
        print(f"Searching in {coll_name}; TC={collection._target_class_name}, ALIAS={collection._alias}")
        qr = collection.find()
        assert qr.num_rows == len(objs), f"expected {len(objs)} for n={coll_name} I= {insert}"

    for db_name in config.databases:
        db = client.get_database(db_name)
        for coll in db.list_collections():
            coll.attach_indexer(index)
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
