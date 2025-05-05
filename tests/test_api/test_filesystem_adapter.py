import logging
import shutil

import pytest

from linkml_store.api.stores.filesystem.filesystem_database import FileSystemDatabase
from linkml_store.index import get_indexer
from tests import OUTPUT_DIR

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("fmt", ["json", "yaml", "tsv", "csv", "jsonl", "parquet"])
def test_insert_and_query(fmt):
    tmpdir = OUTPUT_DIR / "fs_test" / "tmp"
    # Remove the directory and its contents if it exists
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    db = FileSystemDatabase(handle=f"file:{tmpdir}")
    assert db.list_collection_names() == []

    # Create a collection
    collection = db.create_collection("persons", recreate_if_exists=True)
    collection.file_format = fmt

    # Insert a few documents
    documents = [
        {"name": "Alice", "age": 25, "occupation": "Architect"},
        {"name": "Bob", "age": 30, "occupation": "Builder"},
        {"name": "Charlie", "age": 35, "occupation": "Lawyer"},
        {"name": "Jie", "age": 27, "occupation": "Architect"},
    ]
    collection.insert(documents)
    cd = collection.class_definition()
    assert cd is not None
    assert cd.name in ["Person", "persons"]
    assert sorted(cd.attributes.keys()) == ["age", "name", "occupation"]
    assert db.list_collection_names() == ["persons"]

    query_result = collection.find()
    assert query_result.num_rows == len(documents)

    db.commit()

    # Query the collection
    query_result = collection.find({"occupation": "Architect"})

    # Assert the query results
    assert query_result.num_rows == 2
    assert len(query_result.rows) == 2
    assert query_result.rows[0]["name"] == "Alice"
    assert query_result.rows[1]["name"] == "Jie"
    assert set(query_result.rows[0].keys()) == {"name", "age", "occupation"}
    query_result = collection.find({"age": {"$gte": 30}})

    # Assert the query results
    assert query_result.num_rows == 2
    assert len(query_result.rows) == 2
    assert query_result.rows[0]["name"] == "Bob"
    assert query_result.rows[1]["name"] == "Charlie"

    cases = [
        ({}, "occupation", {("Architect", 2), ("Lawyer", 1), ("Builder", 1)}),
        ({"occupation": "Architect"}, "occupation", {("Architect", 2)}),
    ]
    for where, fc, expected in cases:
        fr = collection.query_facets(where, facet_columns=[fc])
        print(fr)
        results = set(fr[fc])
        assert results == expected

    db2 = FileSystemDatabase(handle=f"file:{tmpdir}")
    for coll in db2.list_collection_names():
        print(coll)
        print(db2.get_collection(coll).find().rows)
    assert db2.list_collection_names() == db.list_collection_names()
    collection2 = db2.get_collection("persons")
    collection.set_identifier_attribute_name("name")
    collection2.set_identifier_attribute_name("name")
    if fmt not in ["csv", "tsv"]:
        assert collection.diff(collection2) == []
    else:
        logger.info("TODO: Fix number loading for CSV and TSV")
    ixe = get_indexer("simple")
    collection.attach_indexer(ixe, auto_index=True)
    qr = collection.search("arch")
    top2 = qr.ranked_rows[:2]
    assert len(top2) == 2
    top2_names = sorted([r["name"] for _, r in top2])
    assert top2_names == ["Alice", "Jie"]
