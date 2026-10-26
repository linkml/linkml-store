"""Tests for the TypeDB adapter. Tests marked ``typedb`` need a TypeDB 3.12+ server."""

from pathlib import Path

import pytest
from linkml_runtime import SchemaView

from linkml_store.api.client import Client
from linkml_store.api.stores.typedb.typeql_insert import (
    DanglingReferenceError,
    TypeQLInserter,
    plan_inserts,
    typeql_literal,
)
from tests import INPUT_DIR

SCHEMA = INPUT_DIR / "schemas" / "typedb_mini.yaml"

STUDY = {
    "id": "mini:sty-1",
    "name": "Study One",
    "type": "mini:Study",
    "credit_associations": [
        {
            "type": "prov:Association",
            "applied_role": "PI",
            "applies_to_person": {"orcid": "0000-1", "name": "Ada"},
        }
    ],
    "part_of": "mini:sty-0",
}


@pytest.fixture
def schema_view() -> SchemaView:
    return SchemaView(str(SCHEMA))


@pytest.fixture
def inserter(schema_view) -> TypeQLInserter:
    return TypeQLInserter(schema_view)


def test_handle_registered():
    """The ``typedb`` scheme resolves without a server or driver present."""
    from linkml_store.api.client import HANDLE_MAP

    assert HANDLE_MAP["typedb"].endswith("TypeDBDatabase")


@pytest.mark.parametrize(
    "handle,address,user,password,database,tls,connection_name",
    [
        # local, gRPC-style port
        ("typedb://localhost:1729/nmdc", "localhost:1729", "admin", "password", "nmdc", False, None),
        ("typedb://u:p@example.org:1730/db", "example.org:1730", "u", "p", "db", True, None),
        ("typedb://localhost/db", "localhost:1729", "admin", "password", "db", False, None),
        ("localhost:1729/db", "localhost:1729", "admin", "password", "db", False, None),
        # Studio / Cloud form: the address has its own scheme; "?name=" is a display name
        (
            "typedb://admin:pw@https://x.cluster.typedb.com:80/?name=engineers-dev",
            "x.cluster.typedb.com:80",
            "admin",
            "pw",
            "linkml",
            True,
            "engineers-dev",
        ),
        (
            "typedb://admin:pw@https://x.cluster.typedb.com:80/nmdc?name=engineers-dev",
            "x.cluster.typedb.com:80",
            "admin",
            "pw",
            "nmdc",
            True,
            "engineers-dev",
        ),
        # http:// in the address turns TLS off; percent-encoded credentials are decoded
        (
            "typedb://admin:p%40ss@http://host:8000/db",
            "host:8000",
            "admin",
            "p@ss",
            "db",
            False,
            None,
        ),
        # explicit override wins over the scheme
        ("typedb://a:b@https://host:8000/db?tls=false", "host:8000", "a", "b", "db", False, None),
    ],
)
def test_handle_parsing(handle, address, user, password, database, tls, connection_name):
    """Handles parse into address, credentials, database, TLS and connection name."""
    from linkml_store.api.stores.typedb.typedb_database import TypeDBDatabase

    parsed = TypeDBDatabase(handle=handle)._parse_handle()
    assert parsed.address == address
    assert parsed.user == user
    assert parsed.password == password
    assert parsed.database == database
    assert parsed.tls is tls
    assert parsed.connection_name == connection_name


def test_handle_repr_redacts_password():
    """A handle must not leak its password through logs or tracebacks."""
    from linkml_store.api.stores.typedb.typedb_database import TypeDBDatabase

    parsed = TypeDBDatabase(handle="typedb://admin:s3cr3t@https://host:80/db")._parse_handle()
    assert "s3cr3t" not in repr(parsed)
    assert parsed.password == "s3cr3t"


def test_multi_node_address_rejected():
    """A comma-separated cluster address is refused rather than silently mangled."""
    from linkml_store.api.stores.typedb.typedb_database import TypeDBDatabase

    db = TypeDBDatabase(handle="typedb://a:b@https://n1:8000,https://n2:8000/db")
    with pytest.raises(NotImplementedError, match="Multi-node"):
        db._parse_handle()


@pytest.mark.parametrize(
    "value,value_type,expected",
    [
        ("hi", "string", '"hi"'),
        ('say "hi"', "string", '"say \\"hi\\""'),
        (3, "integer", "3"),
        (1.5, "double", "1.5"),
        ("0.1", "decimal", "0.1dec"),
        (True, "boolean", "true"),
        (False, "boolean", "false"),
        ("true", "boolean", "true"),
        ("2024-01-02T03:04:05", "datetime", "2024-01-02T03:04:05"),
        # 'datetime' drops the offset
        ("2024-01-02T03:04:05+00:00", "datetime", "2024-01-02T03:04:05"),
        ("2024-01-02T03:04:05Z", "datetime", "2024-01-02T03:04:05"),
        ("2024-01-02T03:04:05+05:30", "datetime", "2024-01-01T21:34:05"),
        ("2024-01-02T03:04:05-08:00", "datetime", "2024-01-02T11:04:05"),
        ("not-a-date", "datetime", "not-a-date"),
        # 'datetime-tz' keeps the offset verbatim
        ("2024-01-02T03:04:05+05:30", "datetime-tz", "2024-01-02T03:04:05+05:30"),
    ],
)
def test_typeql_literal(value, value_type, expected):
    """Literals render per TypeDB value type; only strings are quoted."""
    assert typeql_literal(value, value_type) == expected


@pytest.mark.parametrize(
    "value,expect_loss",
    [
        ("2024-01-02T03:04:05", False),  # naive: nothing to lose
        ("2024-01-02T03:04:05+00:00", False),  # +00:00 carries nothing beyond the instant
        ("2024-01-02T03:04:05Z", False),
        ("2024-01-02T03:04:05+05:30", True),  # a real loss
        ("2024-01-02T03:04:05-08:00", True),
    ],
)
def test_only_non_zero_offsets_are_recorded_as_losses(value, expect_loss):
    """Dropping ``+00:00`` is lossless, so only non-zero offsets are reported."""
    losses = []
    typeql_literal(value, "datetime", losses, "created")
    assert bool(losses) is expect_loss
    if expect_loss:
        label, original, offset = losses[0]
        assert (label, original) == ("created", value)
        assert offset in ("+05:30", "-08:00")


def test_offset_loss_warns_once_per_load(schema_view, caplog):
    """Several discarded offsets produce one summary warning, not one per value."""
    objs = [
        {"id": "mini:a", "name": "A", "created": "2020-03-24T09:00:00+05:30"},
        {"id": "mini:b", "name": "B", "created": "2020-03-25T09:00:00-08:00"},
        {"id": "mini:c", "name": "C", "created": "2020-03-26T09:00:00+02:00"},
    ]
    with caplog.at_level("WARNING"):
        plan_inserts(schema_view, "Dated", objs)
    warnings_seen = [r for r in caplog.records if "non-zero UTC offset" in r.getMessage()]
    assert len(warnings_seen) == 1, f"expected one summary warning, got {len(warnings_seen)}"
    assert "3 datetime value(s)" in warnings_seen[0].getMessage()


def test_utc_offsets_do_not_warn(schema_view, caplog):
    """A load whose offsets are all ``+00:00`` loses nothing and says nothing."""
    objs = [{"id": "mini:a", "name": "A", "created": "2020-03-24T09:00:00+00:00"}]
    with caplog.at_level("WARNING"):
        plan_inserts(schema_view, "Dated", objs)
    assert not [r for r in caplog.records if "non-zero UTC offset" in r.getMessage()]


def test_reserved_slot_name_is_suffixed(inserter):
    """``type`` is a TypeDB keyword, so the attribute label gains ``_attr``."""
    assert inserter.attr_names["type"] == "type_attr"


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("prov:Association", "CreditAssociation"),
        ("CreditAssociation", "CreditAssociation"),
        ("http://www.w3.org/ns/prov#Association", "CreditAssociation"),
        ("mini:Study", "Study"),
        (None, "NamedThing"),
        ("nonsense:Nope", "NamedThing"),
    ],
)
def test_resolve_class_from_type_field(inserter, declared, expected):
    """A ``type`` field holding a class URI or CURIE resolves to the LinkML class."""
    assert inserter.resolve_class(declared, "NamedThing") == expected


def test_inlined_object_becomes_its_own_entity(schema_view):
    """An inlined object with no identifier is created and linked by variable."""
    typeql = plan_inserts(schema_view, "Study", [STUDY])[0].to_typeql()
    assert 'isa Person, has orcid "0000-1"' in typeql
    assert "isa CreditAssociation" in typeql
    assert "(applies_to_person: $e1, Person: $e2) isa applies_to_person;" in typeql


def test_narrowed_slot_uses_sub_relation(schema_view):
    """``Study.part_of`` narrows ``NamedThing`` to ``Study``, so the sub-relation is used."""
    typeql = plan_inserts(schema_view, "Study", [STUDY])[0].to_typeql()
    assert "isa part_of_Study;" in typeql
    assert "Study_NamedThing: " in typeql


def test_reference_is_matched_not_inserted(schema_view):
    """A reference by identifier becomes a ``match`` clause."""
    query = plan_inserts(schema_view, "Study", [STUDY])[0]
    assert any('isa Study, has id "mini:sty-0"' in line for line in query.match)


def test_dangling_reference_errors_by_default(schema_view):
    """The default policy refuses, since a failed match silently inserts nothing."""
    with pytest.raises(DanglingReferenceError, match="mini:sty-0"):
        plan_inserts(schema_view, "Study", [STUDY], known_ids=set())


def test_dangling_reference_error_reports_every_violation(schema_view):
    """The error is raised once and carries every violation, not just the first."""
    objs = [
        {"id": "mini:a", "name": "A", "part_of": "mini:missing-1"},
        {"id": "mini:b", "name": "B", "part_of": "mini:missing-2"},
        {"id": "mini:c", "name": "C", "part_of": "mini:missing-1"},
    ]
    with pytest.raises(DanglingReferenceError) as exc:
        plan_inserts(schema_view, "Study", objs, known_ids=set())

    violations = exc.value.violations
    assert len(violations) == 3, "expected all three, not just the first"
    assert {v.identifier for v in violations} == {"mini:missing-1", "mini:missing-2"}
    assert {v.source_id for v in violations} == {"mini:a", "mini:b", "mini:c"}
    # Study narrows part_of to Study via slot_usage, so the induced range is Study
    assert all(v.slot == "part_of" and v.target_class == "Study" for v in violations)

    message = str(exc.value)
    assert "3 unresolvable reference(s)" in message
    assert "2 missing object(s)" in message
    assert "Nothing was written" in message


def test_dangling_reference_skip(schema_view):
    """The ``skip`` policy drops the link and records it."""
    query = plan_inserts(schema_view, "Study", [STUDY], on_dangling_reference="skip", known_ids=set())[0]
    assert query.skipped_references == [("part_of", "mini:sty-0")]
    assert "part_of_Study" not in query.to_typeql()


def test_dangling_reference_stub(schema_view):
    """The ``stub`` policy creates a placeholder carrying only the identifier."""
    query = plan_inserts(schema_view, "Study", [STUDY], on_dangling_reference="stub", known_ids=set())[0]
    typeql = query.to_typeql()
    assert 'isa Study, has id "mini:sty-0";' in typeql
    assert "isa part_of_Study;" in typeql


def test_skipped_references_warn_once_per_load(schema_view, caplog):
    """Many skipped references produce one summary, not one line each."""
    objs = [{"id": f"mini:s{i}", "name": "S", "part_of": f"mini:missing-{i % 2}"} for i in range(6)]
    with caplog.at_level("WARNING"):
        plan_inserts(schema_view, "Study", objs, on_dangling_reference="skip", known_ids=set())
    seen = [r for r in caplog.records if "dangling reference" in r.getMessage()]
    assert len(seen) == 1, f"expected one summary, got {len(seen)}"
    msg = seen[0].getMessage()
    assert "6 dangling reference(s)" in msg
    assert "2 missing object(s)" in msg


def test_stubbed_references_warn_once_per_load(schema_view, caplog):
    """Placeholder creation is also summarised once per load."""
    objs = [{"id": f"mini:d{i}", "name": "D", "part_of": "mini:absent"} for i in range(4)]
    with caplog.at_level("WARNING"):
        plan_inserts(schema_view, "Dataset", objs, on_dangling_reference="stub", known_ids=set())
    seen = [r for r in caplog.records if "placeholder" in r.getMessage()]
    assert len(seen) == 1, f"expected one summary, got {len(seen)}"
    assert "1 missing object(s)" in seen[0].getMessage()


def test_reference_within_batch_resolves(schema_view):
    """Identifiers of the batch count as known, so intra-batch references resolve."""
    batch = [{"id": "mini:sty-0", "name": "Zero"}, STUDY]
    queries = plan_inserts(schema_view, "Study", batch, known_ids=set())
    assert queries[1].match


def test_unknown_dangling_policy_rejected(schema_view):
    """An unrecognised policy is rejected up front."""
    with pytest.raises(ValueError, match="Unknown dangling-reference policy"):
        TypeQLInserter(schema_view, on_dangling_reference="whatever")


@pytest.fixture
def offline_collection(schema_view):
    """A collection bound to the mini schema, with no server connection."""
    from linkml_store.api.stores.typedb.typedb_database import TypeDBDatabase

    db = TypeDBDatabase(handle="typedb://localhost:1729/test")
    db.set_schema_view(schema_view)
    return db.create_collection("Study", alias="Study")


def test_match_block_without_where(offline_collection):
    """An empty where clause matches the whole class, with a stable sort key bound."""
    block = offline_collection._match_block(None)
    assert "$x isa Study;" in block
    assert "$x has id $sort_key;" in block


def test_match_block_equality(offline_collection):
    """A scalar constraint becomes a `has ... ==` pattern on the renamed attribute."""
    block = offline_collection._match_block({"name": "Study One"})
    assert '$x has name == "Study One";' in block


def test_match_block_renames_reserved_slot(offline_collection):
    """A where clause on `type` targets the renamed `type_attr` attribute."""
    block = offline_collection._match_block({"type": "mini:Study"})
    assert '$x has type_attr == "mini:Study";' in block


def test_match_block_in_operator(offline_collection):
    """`$in` becomes a disjunction over the bound value variable."""
    block = offline_collection._match_block({"id": {"$in": ["mini:a", "mini:b"]}})
    assert "$x has id $v_id;" in block
    assert '{ $v_id == "mini:a"; } or { $v_id == "mini:b"; };' in block


@pytest.mark.parametrize(
    "constraint,expected",
    [
        ({"$gt": "a"}, '$v_name > "a";'),
        ({"$gte": "a"}, '$v_name >= "a";'),
        ({"$lt": "z"}, '$v_name < "z";'),
        ({"$lte": "z"}, '$v_name <= "z";'),
        ({"$contains": "x"}, '$v_name == "x";'),
        ({"$like": "a%"}, '$v_name like "^a.*$";'),
    ],
)
def test_match_block_operators(offline_collection, constraint, expected):
    """The operators duckdb accepts translate to TypeQL comparators."""
    block = offline_collection._match_block({"name": constraint})
    assert "$x has name $v_name;" in block
    assert expected in block


def test_match_block_ne_matches_missing(offline_collection):
    """`$ne` is a negation, so objects lacking the slot still match, as in duckdb."""
    block = offline_collection._match_block({"name": {"$ne": "x"}})
    assert 'not { $x has name == "x"; };' in block
    assert "$v_name" not in block


def test_match_block_combines_operators(offline_collection):
    """Several operators on one slot combine conjunctively, as in duckdb."""
    block = offline_collection._match_block({"name": {"$gte": "a", "$lt": "m"}})
    assert '$v_name >= "a";' in block
    assert '$v_name < "m";' in block


def test_match_block_rejects_unsupported_operator(offline_collection):
    """An operator the adapter cannot translate fails loudly rather than silently."""
    with pytest.raises(NotImplementedError, match=r"\$exists"):
        offline_collection._match_block({"name": {"$exists": True}})


def test_match_block_rejects_raw_where_clause(offline_collection):
    """Raw SQL text has no TypeQL equivalent, so it is refused explicitly."""
    with pytest.raises(NotImplementedError, match="raw where clause"):
        offline_collection._match_block("name = 'x'")


def test_fetch_block_shapes(offline_collection):
    """Attributes come back wholesale; references by id, inlined objects nested."""
    block = offline_collection._fetch_block("Study")
    assert '"_attributes": { $x.* }' in block
    # part_of ranges over NamedThing, which has an identifier -> return the id
    assert '"part_of"' in block and '"id": $y_part_of.id' in block
    # credit_associations ranges over CreditAssociation, which has none -> nest it
    assert '"credit_associations"' in block
    assert "fetch { $y_credit_associations.* };" in block


def test_row_to_object_maps_labels_back(offline_collection):
    """TypeDB labels map back to LinkML slot names, undoing the reserved-word rename."""
    row = {
        "_attributes": {"id": "mini:a", "name": "A", "type_attr": "mini:Study"},
        "part_of": [{"id": "mini:parent"}],
        "credit_associations": [{"type_attr": "prov:Association", "applied_role": "PI"}],
    }
    obj = offline_collection._row_to_object(row, "Study")
    assert obj["type"] == "mini:Study"
    assert "type_attr" not in obj
    assert obj["part_of"] == "mini:parent"
    assert obj["credit_associations"] == [{"type": "prov:Association", "applied_role": "PI"}]


def test_update_queries_single_valued(offline_collection):
    """A single-valued scalar is replaced through the `update` stage."""
    queries = offline_collection._update_queries({"id": "mini:a", "name": "New"}, {"id": "mini:a"}, None)
    assert len(queries) == 1
    assert '$x has id == "mini:a";' in queries[0]
    assert 'update $x has name "New";' in queries[0]


def test_update_queries_multivalued_replaces_set(offline_collection):
    """A multivalued scalar is replaced wholesale: delete the old, insert the new."""
    queries = offline_collection._update_queries({"id": "mini:a", "alias": ["x", "y"]}, {"id": "mini:a"}, None)
    joined = "\n---\n".join(queries)
    assert "delete $old of $x;" in joined, joined
    assert '$x has alias_attr "x";' in joined
    assert '$x has alias_attr "y";' in joined
    assert "update" not in joined, "update rejects multivalued attributes"


def test_update_queries_skips_relations(offline_collection):
    """Object-valued slots are insert-only, so upsert never touches them."""
    queries = offline_collection._update_queries(
        {"id": "mini:a", "name": "N", "part_of": "mini:b"}, {"id": "mini:a"}, None
    )
    assert all("part_of" not in q for q in queries)


def test_update_queries_honours_update_fields(offline_collection):
    """Only the named fields are written when update_fields is given."""
    obj = {"id": "mini:a", "name": "N", "study_category": "c"}
    queries = offline_collection._update_queries(obj, {"id": "mini:a"}, ["name"])
    joined = " ".join(queries)
    assert "has name" in joined
    assert "study_category" not in joined


def test_upsert_requires_a_filter_field(offline_collection):
    """An object with none of the filter fields is refused rather than guessed at."""
    with pytest.raises(ValueError, match="missing every filter field"):
        offline_collection.upsert([{"name": "no id here"}])


def test_vendored_naming_matches_upstream():
    """The vendored naming rules must be byte-identical to linkml's copy."""
    upstream = pytest.importorskip(
        "linkml.generators.typedb.naming",
        reason="requires linkml, which owns the upstream naming rules",
    )
    from linkml_store.api.stores.typedb import naming as vendored

    marker = "# >>> BEGIN VENDORED CONTENT >>>\n"
    vendored_text = Path(vendored.__file__).read_text()
    assert marker in vendored_text, "vendored file lost its provenance marker"
    body = vendored_text.split(marker, 1)[1]

    assert body == Path(upstream.__file__).read_text(), (
        "Vendored TypeDB naming rules are stale. Run: just sync-typedb-naming"
    )


def test_labels_agree_with_generated_schema(schema_view):
    """Every label the planner emits must be defined by ``gen-typedb``."""
    linkml_generators = pytest.importorskip("linkml.generators.typedbgen", reason="requires the typedb extra (linkml)")
    schema = linkml_generators.TypeDBGenerator(str(SCHEMA)).serialize()

    queries = plan_inserts(schema_view, "Study", [STUDY], on_dangling_reference="stub", known_ids=set())
    typeql = queries[0].to_typeql()

    relations = {line.split("isa ")[-1].rstrip(";") for line in typeql.splitlines() if ") isa " in line}
    assert relations, "expected the plan to use at least one relation"
    for relation in relations:
        assert f"relation {relation}" in schema, f"{relation} is not defined in the generated schema"

    roles = set()
    for line in typeql.splitlines():
        if ") isa " in line:
            for part in line.split("(")[1].split(")")[0].split(","):
                roles.add(part.split(":")[0].strip())
    for role in roles:
        assert f"relates {role}" in schema, f"role {role} is not declared in the generated schema"

    for attr_label in ("id", "name", "type_attr", "orcid", "applied_role"):
        assert f"attribute {attr_label}" in schema


@pytest.fixture(scope="function")
def typedb_db():
    """A live TypeDB database with the mini schema defined."""
    pytest.importorskip("typedb.driver", reason="requires the typedb extra")
    typedbgen = pytest.importorskip(
        "linkml.generators.typedbgen",
        reason="requires linkml to generate the schema to load",
    )
    from typedb.driver import TransactionType

    client = Client()
    db = client.attach_database("typedb://localhost:1729/test_linkml_store", alias="test_typedb")
    db.set_schema_view(str(SCHEMA))
    try:
        db.driver.databases.all()
    except Exception as e:
        pytest.skip(f"no TypeDB server reachable at localhost:1729 ({type(e).__name__})")
    db.drop()
    db.driver.databases.create(db._db_name)
    with db.driver.transaction(db._db_name, TransactionType.SCHEMA) as tx:
        tx.query(typedbgen.TypeDBGenerator(str(SCHEMA)).serialize()).resolve()
        tx.commit()
    yield db
    db.close()


@pytest.mark.typedb
def test_insert_against_live_server(typedb_db):
    """A batch inserts and commits, with intra-batch references resolved."""
    collection = typedb_db.create_collection("Study", alias="Study")
    collection.insert([{"id": "mini:sty-0", "name": "Zero"}, STUDY])

    rows = collection._read("match $x isa Study; reduce $n = count;")
    assert rows[0]["n"] == 2

    rows = collection._read("match $r isa part_of_Study; reduce $n = count;")
    assert rows[0]["n"] == 1

    rows = collection._read("match $p isa Person; reduce $n = count;")
    assert rows[0]["n"] == 1


@pytest.mark.typedb
def test_read_back_against_live_server(typedb_db):
    """Objects round-trip: attributes, id references and nested inlined objects."""
    collection = typedb_db.create_collection("Study", alias="Study")
    collection.insert([{"id": "mini:sty-0", "name": "Zero", "study_category": "x"}, STUDY])

    qr = collection.find({})
    assert qr.num_rows == 2

    by_id = {row["id"]: row for row in qr.rows}
    one = by_id["mini:sty-1"]
    assert one["name"] == "Study One"
    assert one["type"] == "mini:Study"  # type_attr mapped back
    assert one["part_of"] == "mini:sty-0"  # reference returned as an identifier
    # CreditAssociation has no identifier, so it comes back nested
    assert one["credit_associations"][0]["applied_role"] == "PI"

    # equality, $in and paging all go through the same path
    assert collection.find({"id": "mini:sty-1"}).num_rows == 1
    assert collection.get(["mini:sty-0", "mini:sty-1"]).num_rows == 2
    assert collection.get_one("mini:sty-1")["name"] == "Study One"
    assert len({o["id"] for o in collection.find_iter(page_size=1)}) == 2


def test_no_batch_open_by_default(offline_collection):
    """Outside a batch, inserts own their transaction."""
    assert offline_collection.parent._active_tx is None


@pytest.mark.typedb
def test_batch_write_spans_collections(typedb_db):
    """A cyclic two-phase load works when one transaction spans every collection."""
    from linkml_store.api.stores.typedb.typeql_insert import MODE_ENTITIES, MODE_REFERENCES

    studies = [
        {"id": "mini:sty-a", "name": "A", "study_category": "x", "part_of": "mini:sty-b"},
        {"id": "mini:sty-b", "name": "B", "study_category": "x", "part_of": "mini:sty-a"},
    ]
    collection = typedb_db.create_collection("Study", alias="Study")

    with typedb_db.batch_write():
        collection.insert(studies, mode=MODE_ENTITIES)
        collection.insert(studies, mode=MODE_REFERENCES)

    assert collection.find({}).num_rows == 2
    rows = {r["id"]: r for r in collection.find({}).rows}
    assert rows["mini:sty-a"]["part_of"] == "mini:sty-b"
    assert rows["mini:sty-b"]["part_of"] == "mini:sty-a"


@pytest.mark.typedb
def test_batch_write_rolls_back_on_error(typedb_db):
    """If the block raises, nothing the batch wrote is committed."""
    collection = typedb_db.create_collection("Study", alias="Study")
    before = collection.find({}).num_rows

    with pytest.raises(RuntimeError, match="deliberate"):
        with typedb_db.batch_write():
            collection.insert([{"id": "mini:sty-z", "name": "Z", "study_category": "x"}])
            raise RuntimeError("deliberate failure inside the batch")

    assert collection.find({}).num_rows == before
    assert typedb_db._active_tx is None, "the batch must be cleared even on failure"


@pytest.mark.typedb
def test_batch_write_refuses_to_nest(typedb_db):
    """Two open batches on one database would have ambiguous commit semantics."""
    with typedb_db.batch_write():
        with pytest.raises(RuntimeError, match="already open"):
            with typedb_db.batch_write():
                pass
