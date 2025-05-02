"""
Note that neo4j must be running for these tests to pass.

Example:

    .. code-block:: bash

       docker run \
          --name myneo4j \
          --publish 7474:7474 --publish 7687:7687 \
          --volume $HOME/neo4j/data/:/data \
          --volume $HOME/neo4j/logs/:/logs \
          -e NEO4J_AUTH=none \
          neo4j

"""

import pytest
from linkml_runtime import SchemaView
from linkml_runtime.utils.schema_builder import SchemaBuilder

from linkml_store.api.client import Client
from linkml_store.api.stores.neo4j.neo4j_collection import DeletePolicy
from linkml_store.graphs.graph_map import EdgeProjection, NodeProjection


@pytest.mark.neo4j
@pytest.fixture(scope="function")
def neo4j_db():
    client = Client()
    db = client.attach_database("neo4j", alias="test_neo4j")
    # cleanup before
    db.drop()
    db = client.attach_database("neo4j", alias="test_neo4j")
    yield db
    # Clean up after the test
    # NOTE: we leave in place for now as it makes for easier testing
    # db.drop()


@pytest.mark.neo4j
@pytest.mark.parametrize(
    "edge_projection,node_projection",
    [
        (EdgeProjection(), None),
        (EdgeProjection(), NodeProjection()),
        (EdgeProjection(subject_attribute="source", predicate_attribute="relation", object_attribute="target"), None),
        (
            EdgeProjection(identifier_attribute="uid"),
            NodeProjection(identifier_attribute="uid", category_labels_attribute="category"),
        ),
    ],
)
def test_neo4j_adapter(neo4j_db, edge_projection, node_projection):
    # Create a collection
    collection = neo4j_db.create_collection("Node", recreate_if_exists=True)
    if node_projection:
        collection.metadata.graph_projection = node_projection
    else:
        node_projection = NodeProjection()
    ec = neo4j_db.create_collection("Edge", recreate_if_exists=True)
    ec.metadata.graph_projection = edge_projection
    ia = node_projection.identifier_attribute
    ca = node_projection.category_labels_attribute
    sa = edge_projection.subject_attribute
    pa = edge_projection.predicate_attribute
    oa = edge_projection.object_attribute
    assert collection.identifier_attribute == ia

    persons = [
        {ia: "P:1", "name": "Alice", "age": 30, ca: "Manager"},
        {ia: "P:2", "name": "Bob", "age": 35, ca: "Employee"},
    ]
    collection.insert(persons)
    ec.insert({sa: "P:1", oa: "P:2", pa: "employs", "role": "coder"})

    result = collection.find({})
    assert result.num_rows == 2

    cases = [({}, 1), ({sa: "P:1"}, 1), ({sa: "..."}, 0)]
    for query, num_rows in cases:
        result = ec.find(query)
        assert len(result.rows) == num_rows
        assert result.num_rows == num_rows

    result = collection.find({ia: "P:1"})
    assert result.num_rows == 1
    r = result.rows[0]
    assert r[ia] == "P:1"
    assert r["name"] == "Alice"
    assert r["age"] == 30
    assert r[ca] == "Manager"

    # Query data
    result = collection.find({ca: "Manager"})
    assert result.num_rows == 1
    assert result.rows[0]["name"] == "Alice"

    # Update data
    collection.update({ia: "P:1", "category": "Manager", "age": 31})
    result = collection.find({ia: "P:1"})
    assert result.rows[0]["age"] == 31
    assert result.rows[0]["category"] == "Manager"

    # Delete data
    collection.delete({ia: "P:2"})
    result = collection.find({})
    assert result.num_rows == 1

    # Test facets

    collection.insert(persons[1])
    # add edge back
    ec.insert({sa: "P:1", oa: "P:2", pa: "employs", "role": "coder"})
    facets = collection.query_facets(facet_columns=["age"])
    assert len(facets["age"]) == 2
    assert sorted(facets["age"]) == [(31, 1), (35, 1)]

    facets = collection.query_facets(facet_columns=[ca])
    assert len(facets[ca]) == 2
    assert sorted(facets[ca]) == [("Employee", 1), ("Manager", 1)]

    # add an edge with a dangling reference
    ec.delete_policy = DeletePolicy.STUB
    ec.insert({sa: "P:1", oa: "P:3", pa: "employs", "role": "PM"})
    result = ec.find({})
    assert len(result.rows) == 2
    assert result.num_rows == 2
    result = ec.find({pa: "employs"})
    assert len(result.rows) == 2
    assert result.num_rows == 2
    result = ec.find({pa: "fake"})
    assert len(result.rows) == 0
    assert result.num_rows == 0
    cases = [
        ({}, 2),
        ({sa: "P:1"}, 2),
        ({oa: "P:2"}, 1),
        ({sa: "..."}, 0),
        ({pa: "employs"}, 2),
        ({pa: "fake"}, 0),
    ]
    for query, num_rows in cases:
        result = ec.find(query)
        assert len(result.rows) == num_rows
        assert result.num_rows == num_rows
    sb = SchemaBuilder()
    sb.add_slot(ia, range="string", identifier=True)
    sb.add_slot("name", range="string")
    sb.add_slot(ca, range="string", designates_type=True)
    sb.add_slot("age", range="integer")
    sb.add_slot("role", range="string")
    sb.add_slot(sa, range="Node")
    sb.add_slot(pa, range="string", designates_type=True)
    sb.add_slot(oa, range="Node")
    sb.add_class("Node", slots=[ia, "name", ca])
    sb.add_class("Person", is_a="Node", slots=["name", "age"])
    sb.add_class("Employee", is_a="Person")
    sb.add_class("Manager", is_a="Employee")
    sb.add_class("Edge", slots=[sa, oa, pa])
    sb.add_class("employs", is_a="Edge", slots=["role"])
    sb.add_class("related_to", is_a="Edge")
    sb.add_defaults()
    schema = sb.schema
    neo4j_db.set_schema_view(SchemaView(schema))
    assert ca in neo4j_db.schema_view.get_class("Node").slots
    att = neo4j_db.schema_view.induced_slot(ca, "Manager")
    assert att.designates_type
    assert collection.target_class_name == "Node"
    errs = list(neo4j_db.iter_validate_database())
    assert len(errs) == 0
    collection.metadata.validate_modifications = True
    invalid_objs = [
        {ia: "P:4", "name": "Frida", "age": "40 years", ca: "Manager"},
        {ia: "P:4", "name": "Frida", "age": 40, ca: "Manager", "made_up": "extra"},
        {ia: "P:4", "name": ["Frida"], "age": 40, ca: "Manager"},
    ]
    for obj in invalid_objs:
        with pytest.raises(ValueError):
            collection.insert(obj)
    ec.metadata.validate_modifications = True
    ec.delete_policy = DeletePolicy.ERROR
    invalid_edges = [
        {sa: "P:1", oa: "P:2", pa: "employs", "role": "coder", "made_up": "extra"},
        # no such class for type designator:
        {sa: "P:1", pa: "other_rel", oa: "P:2"},
        # edge property is specific to employs
        {sa: "P:1", pa: "related_to", oa: "P:2", "role": "coder"},
        # incomplete:
        {sa: "P:1", oa: "P:2"},
        {sa: "P:1", pa: "employs"},
        {pa: "employs", oa: "P:2"},
        # dangling:
        {sa: "P:1", oa: "P:99999", pa: "employs", "role": "coder"},
        {sa: "P:99999", oa: "P:1", pa: "employs", "role": "coder"},
    ]
    for edge in invalid_edges:
        with pytest.raises(ValueError):
            ec.insert(edge)
    collection.metadata.validate_modifications = False
    collection.insert(invalid_objs)
