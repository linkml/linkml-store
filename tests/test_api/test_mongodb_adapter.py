# test_mongodb_adapter.py

from copy import deepcopy
from unittest.mock import Mock, PropertyMock, patch

import pytest
import yaml
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, ConnectionFailure, OperationFailure

from linkml_store.api.stores.mongodb.mongodb_collection import MongoDBCollection
from linkml_store.api.stores.mongodb.mongodb_database import MongoDBDatabase


@pytest.fixture(scope="module")
def mongodb_client():
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")  # Check MongoDB connectivity
    except ConnectionFailure:
        client.close()
        pytest.skip("Skipping tests: MongoDB is not available.")
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="function")
def mongodb_database(mongodb_client):
    db = mongodb_client["test_db"]
    yield db
    db.drop_collection("test_collection")


@pytest.fixture(scope="function")
def mongodb_collection(mongodb_client):
    """Fixture to provide a MongoDB test collection, ensuring database and collection creation if necessary."""
    if mongodb_client is None:
        pytest.skip("Skipping tests: MongoDB client is not available.")

    db_name = "test_db"
    collection_name = "test_collection"

    # Ensure database exists by creating a temporary collection
    existing_dbs = mongodb_client.list_database_names()
    if db_name not in existing_dbs:
        temp_db = mongodb_client[db_name]
        temp_db.create_collection("temp_init_collection")
        temp_db.drop_collection("temp_init_collection")  # Clean up temp collection

    # Now attach to the database and collection
    db = mongodb_client[db_name]

    # Ensure the test collection exists
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    parent = MongoDBDatabase(handle=f"mongodb://localhost:27017/{db_name}")
    parent._native_db = db
    collection = MongoDBCollection(name=collection_name, parent=parent)

    yield collection

    # Cleanup: Drop the test collection after each test
    db.drop_collection(collection_name)


def test_upsert_insert(mongodb_collection):
    """
    Test that the upsert method inserts a new document if it does not exist.
    """
    obj = {"_id": "1", "name": "Alice", "age": 25, "occupation": "Engineer"}

    # Upsert operation: should insert because no document with _id=1 exists
    mongodb_collection.upsert(obj, filter_fields=["_id"])

    # Check if the document exists in the collection
    result = mongodb_collection.mongo_collection.find_one({"_id": "1"})
    assert result is not None
    assert result["name"] == "Alice"
    assert result["age"] == 25
    assert result["occupation"] == "Engineer"


def test_upsert_update(mongodb_collection):
    """
    Test that the upsert method updates an existing document while preserving unchanged fields.
    """

    initial_obj = {"_id": "2", "name": "Bob", "age": 30, "occupation": "Builder"}
    mongodb_collection.mongo_collection.insert_one(initial_obj)

    updated_obj = {"_id": "2", "age": 35}
    mongodb_collection.upsert(updated_obj, filter_fields=["_id"], update_fields=["age"])
    result = mongodb_collection.mongo_collection.find_one({"_id": "2"})
    assert result is not None
    assert result["_id"] == "2"
    assert result["age"] == 35  # Should be updated
    assert result["name"] == "Bob"  # Should remain unchanged
    assert result["occupation"] == "Builder"  # Should remain unchanged


@pytest.mark.parametrize("handle", ["mongodb://localhost:27017/test_db", None, "mongodb"])
@pytest.mark.integration
def test_insert_and_query(handle, mongodb_client):
    """
    Test inserting and querying documents in MongoDB.
    """
    db = MongoDBDatabase(handle=handle)
    collection = db.create_collection("test_collection", recreate_if_exists=True)
    documents = [
        {
            "name": "Alice",
            "age": 25,
            "occupation": "Architect",
            "foods": ["apple", "banana"],
            "relationships": [
                {"person": "Bob", "relation": "friend"},
                {"person": "Charlie", "relation": "brother"},
            ],
            "meta": {"date": "2021-01-01", "notes": "likes fruit"},
        },
        {
            "name": "Bob",
            "age": 30,
            "occupation": "Builder",
            "foods": ["carrot", "date"],
            "relationships": [{"person": "Alice", "relation": "friend"}],
            "meta": {"date": "2021-01-01"},
        },
        {
            "name": "Charlie",
            "age": 35,
            "occupation": "Lawyer",
            "foods": ["eggplant", "fig", "banana"],
            "meta": {"date": "2021-01-03", "notes": "likes fruit", "curator": "Ziggy"},
        },
        {
            "name": "Jie",
            "age": 27,
            "occupation": "Architect",
            "foods": ["grape", "honey", "apple"],
            "meta": {"date": "2021-01-03"},
        },
    ]
    collection.insert(documents)

    query_result = collection.find()
    assert query_result.num_rows == len(documents)
    query_result = collection.find({"age": {"$gte": 30}})

    assert query_result.num_rows == 2
    assert len(query_result.rows) == 2
    assert query_result.rows[0]["name"] == "Bob"
    assert query_result.rows[1]["name"] == "Charlie"
    assert set(query_result.rows[0].keys()) == {"name", "age", "occupation", "foods", "meta", "relationships"}
    cases = [
        ({}, "occupation", {("Architect", 2), ("Lawyer", 1), ("Builder", 1)}),
        ({"occupation": "Architect"}, "occupation", {("Architect", 2)}),
        # test unwinding multivalued
        (
            {},
            "foods",
            {
                ("fig", 1),
                ("banana", 2),
                ("eggplant", 1),
                ("apple", 2),
                ("carrot", 1),
                ("date", 1),
                ("grape", 1),
                ("honey", 1),
            },
        ),
        ({}, "relationships.relation", {("friend", 2), ("brother", 1)}),
        ({}, "meta.date", {("2021-01-01", 2), ("2021-01-03", 2)}),
        (
            {},
            "meta",
            {
                ("curator: Ziggy\ndate: '2021-01-03'\nnotes: likes fruit\n", 1),
                ("date: '2021-01-01'\nnotes: likes fruit\n", 1),
                ("date: '2021-01-01'\n", 1),
                ("date: '2021-01-03'\n", 1),
            },
        ),
        (
            {},
            ("occupation", "foods"),
            {
                (("Architect", "apple"), 2),
                (("Architect", "banana"), 1),
                (("Architect", "grape"), 1),
                (("Architect", "honey"), 1),
                (("Builder", "carrot"), 1),
                (("Builder", "date"), 1),
                (("Lawyer", "banana"), 1),
                (("Lawyer", "eggplant"), 1),
                (("Lawyer", "fig"), 1),
            },
        ),
    ]
    for where, fc, expected in cases:
        fr = collection.query_facets(where, facet_columns=[fc])
        val_count_tuples = fr[fc]
        if any(isinstance(v, dict) for v, _ in val_count_tuples):
            val_count_tuples = {(yaml.dump(v), c) for v, c in val_count_tuples}
        results = set(val_count_tuples)
        assert results == expected
        # test re-querying with facets
        for v, c in fr[fc]:
            if isinstance(fc, tuple):
                where = {fc[i]: v[i] for i in range(len(fc))}
            else:
                where = {fc: v}
            results = collection.find(where, limit=-1)
            assert results.num_rows == c, f"where {where} failed to find expected"


@pytest.mark.parametrize("unique_flag", [False, True])
def test_index_creation(mongodb_collection, unique_flag):
    """Test the index creation method in MongoDBCollection with and without unique constraint."""

    index_field = "test_field"
    index_name = f"test_index_{'unique' if unique_flag else 'non_unique'}"
    mongodb_collection.mongo_collection.drop_indexes()

    # Ensure the collection is empty before creating a unique index
    mongodb_collection.mongo_collection.delete_many({})

    # Insert **unique, non-null** values for test_field to avoid duplicate key error
    mongodb_collection.mongo_collection.insert_many(
        [{"_id": 1, "test_field": "value1"}, {"_id": 2, "test_field": "value2"}, {"_id": 3, "test_field": "value3"}]
    )

    # Create the index using the method with the unique flag
    mongodb_collection.index(index_field, index_name=index_name, replace=True, unique=unique_flag)

    # Retrieve indexes after creation
    created_indexes = mongodb_collection.mongo_collection.index_information()

    # Verify that the new index exists
    assert index_name in created_indexes, f"Index {index_name} was not created"

    # Check if the index is unique if requested
    if unique_flag:
        assert created_indexes[index_name]["unique"], f"Index {index_name} should be unique"
    else:
        assert (
            "unique" not in created_indexes[index_name] or not created_indexes[index_name]["unique"]
        ), f"Index {index_name} should not be unique"


@pytest.mark.integration
def test_find_iter_no_count_calls(mongodb_client):
    """find_iter must never call count_documents or estimated_document_count.

    Regression test for https://github.com/linkml/linkml-store/issues/69.
    Without the fix, a 54M-row collection at page_size=1000 paid ~25s per page
    (~15 days total) in count_documents calls.
    """
    from linkml_store.api.stores.mongodb.mongodb_database import MongoDBDatabase

    db = MongoDBDatabase(handle="mongodb://localhost:27017/test_find_iter_db")
    try:
        collection = db.create_collection("count_test", recreate_if_exists=True)
        docs = [{"id": i, "val": f"v{i}"} for i in range(10)]
        collection.insert(docs)

        with patch.object(
            collection.mongo_collection,
            "count_documents",
            wraps=collection.mongo_collection.count_documents,
        ) as mock_count, patch.object(
            collection.mongo_collection,
            "estimated_document_count",
            wraps=collection.mongo_collection.estimated_document_count,
        ) as mock_est:
            rows = list(collection.find_iter(page_size=3))

        assert len(rows) == 10
        assert mock_count.call_count == 0, (
            f"count_documents called {mock_count.call_count} times; expected 0"
        )
        assert mock_est.call_count == 0, (
            f"estimated_document_count called {mock_est.call_count} times; expected 0"
        )
    finally:
        db.drop()


@pytest.fixture
def duplicate_batch(mongodb_collection):
    native = mongodb_collection.mongo_collection
    native.delete_many({})
    native.create_index("id", unique=True)
    native.insert_one({"id": "existing"})
    batch = [{"id": "before"}, {"id": "existing"}, {"id": "after"}]
    assert native.index_information()["id_1"]["unique"]
    assert [native.count_documents(obj) for obj in batch] == [0, 1, 0]
    return mongodb_collection, batch


@pytest.mark.parametrize("ordered", [True, False])
def test_insert_duplicate_raises(duplicate_batch, ordered):
    collection, batch = duplicate_batch
    kwargs = {} if ordered else {"ordered": False}
    with pytest.raises(BulkWriteError) as exc:
        collection.insert(batch, **kwargs)
    assert all("_id" not in obj for obj in batch)
    assert [error["code"] for error in exc.value.details["writeErrors"]] == [11000]
    assert exc.value.details["nInserted"] == (1 if ordered else 2)
    assert collection.mongo_collection.count_documents({"id": "before"}) == 1
    assert collection.mongo_collection.count_documents({"id": "after"}) == (0 if ordered else 1)


@pytest.mark.parametrize("ordered", [True, False])
def test_insert_ignore_duplicates(duplicate_batch, ordered):
    collection, batch = duplicate_batch
    result = collection.insert(batch, ordered=ordered, ignore_duplicates=True)
    assert result == {"inserted": 1 if ordered else 2, "skipped": 1}
    assert collection.mongo_collection.count_documents({}) == 1 + result["inserted"]
    assert collection.mongo_collection.count_documents({"id": "after"}) == (0 if ordered else 1)
    assert all("_id" not in obj for obj in batch)


@pytest.fixture
def mock_insert_collection():
    collection = MongoDBCollection(name="test_collection")
    native = Mock()
    with patch.object(
        MongoDBCollection, "mongo_collection", new_callable=PropertyMock, return_value=native
    ), patch.object(MongoDBCollection, "_post_insert_hook") as hook:
        yield collection, native, hook


@pytest.mark.parametrize("ignore_duplicates", [False, True])
def test_insert_success(mock_insert_collection, ignore_duplicates):
    collection, native, hook = mock_insert_collection
    obj = {"id": "new"}

    def insert_many(objs, ordered):
        assert objs == [obj]
        objs[0]["_id"] = "generated"
        return Mock(inserted_ids=["generated"])

    native.insert_many.side_effect = insert_many
    result = collection.insert(obj, ignore_duplicates=ignore_duplicates)
    native.insert_many.assert_called_once_with([obj], ordered=True)
    assert result == ({"inserted": 1, "skipped": 0} if ignore_duplicates else None)
    assert obj == {"id": "new"}
    hook.assert_called_once_with([obj])


@pytest.mark.parametrize("ordered", [True, False])
def test_insert_duplicate_counts_and_hook(mock_insert_collection, ordered):
    collection, native, hook = mock_insert_collection
    batch = [{"id": i} for i in range(4)]
    errors = [{"code": 11000, "index": 1}]
    inserted = 1 if ordered else 3
    failure = BulkWriteError({"nInserted": inserted, "writeErrors": errors, "writeConcernErrors": []})
    assert failure.details["writeErrors"][0]["code"] == 11000
    if ordered:
        assert inserted != len(batch) - len(errors)

    def insert_many(objs, ordered):
        for obj in objs:
            obj["_id"] = ObjectId()
        raise failure

    native.insert_many.side_effect = insert_many
    def check_hook(objs):
        assert all("_id" not in obj for obj in objs)

    hook.side_effect = check_hook
    result = collection.insert(batch, ordered=ordered, ignore_duplicates=True)
    native.insert_many.assert_called_once_with(batch, ordered=ordered)
    assert result == {"inserted": inserted, "skipped": 1}
    hook.assert_called_once_with(batch[:1] if ordered else [batch[0], batch[2], batch[3]])
    assert all("_id" not in obj for obj in batch)


@pytest.mark.parametrize(
    "errors,concerns",
    [
        ([{"code": 121, "index": 0}], []),
        ([{"code": 11000, "index": 0}, {"code": 121, "index": 1}], []),
        ([], [{"code": 64, "errmsg": "waiting for replication timed out"}]),
        ([{"code": 11000, "index": 0}], [{"code": 64, "errmsg": "write concern failed"}]),
        ([], []),
    ],
)
def test_insert_rejects_other_failures(mock_insert_collection, errors, concerns):
    collection, native, hook = mock_insert_collection
    failure = BulkWriteError({"nInserted": 0, "writeErrors": errors, "writeConcernErrors": concerns})
    assert concerns or not errors or any(error["code"] != 11000 for error in errors)
    native.insert_many.side_effect = failure
    with pytest.raises(BulkWriteError) as exc:
        collection.insert([{"id": 0}, {"id": 1}], ordered=False, ignore_duplicates=True)
    assert exc.value is failure
    native.insert_many.assert_called_once()
    hook.assert_not_called()


def test_insert_duplicates_raise_by_default(mock_insert_collection):
    collection, native, hook = mock_insert_collection
    failure = BulkWriteError({"nInserted": 0, "writeErrors": [{"code": 11000, "index": 0}]})
    assert failure.details["writeErrors"][0]["code"] == 11000
    native.insert_many.side_effect = failure
    with pytest.raises(BulkWriteError) as exc:
        collection.insert({"id": 0})
    assert exc.value is failure
    native.insert_many.assert_called_once_with([{"id": 0}], ordered=True)
    hook.assert_not_called()


def test_insert_all_duplicates(mock_insert_collection):
    collection, native, hook = mock_insert_collection
    batch = [{"id": 0}, {"id": 1}]
    errors = [{"code": 11000, "index": i} for i in range(len(batch))]
    assert len(errors) == len(batch)
    assert all(error["code"] == 11000 for error in errors)
    def insert_many(objs, ordered):
        for obj in objs:
            obj["_id"] = ObjectId()
        raise BulkWriteError({"nInserted": 0, "writeErrors": errors, "writeConcernErrors": []})

    native.insert_many.side_effect = insert_many
    assert collection.insert(batch, ordered=False, ignore_duplicates=True) == {"inserted": 0, "skipped": 2}
    native.insert_many.assert_called_once_with(batch, ordered=False)
    hook.assert_not_called()
    assert all("_id" not in obj for obj in batch)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("error_kind", ["duplicate", "validation", "write_concern", "operation"])
def test_insert_raising_restores_documents(mock_insert_collection, ordered, error_kind):
    collection, native, hook = mock_insert_collection
    batch = [{"id": "before"}, {"id": "existing"}, {"id": "after", "_id": "caller-id"}]
    original = deepcopy(batch)
    if error_kind == "operation":
        failure = OperationFailure("not authorized", code=13)
    else:
        errors = [] if error_kind == "write_concern" else [
            {"code": 11000 if error_kind == "duplicate" else 121, "index": 1}
        ]
        concerns = [{"code": 64, "errmsg": "write concern failed"}] if error_kind == "write_concern" else []
        failure = BulkWriteError({"nInserted": 1, "writeErrors": errors, "writeConcernErrors": concerns})

    def insert_many(objs, ordered):
        for obj in objs:
            obj.setdefault("_id", ObjectId())
        raise failure

    native.insert_many.side_effect = insert_many
    kwargs = {} if ordered else {"ordered": False}
    if error_kind != "duplicate":
        kwargs["ignore_duplicates"] = True
    with pytest.raises(type(failure)) as exc:
        collection.insert(batch, **kwargs)
    assert exc.value is failure
    hook.assert_not_called()
    assert all("_id" not in obj for obj in batch[:2])
    assert batch == original
    native.insert_many.assert_called_once_with(batch, ordered=ordered)


@pytest.mark.parametrize("ignore_duplicates", [False, True])
@pytest.mark.parametrize("original_id", [None, "caller-id"])
def test_insert_preserves_existing_id(mock_insert_collection, ignore_duplicates, original_id):
    collection, native, hook = mock_insert_collection
    obj = {"id": "new", "_id": original_id}
    original = deepcopy(obj)
    native.insert_many.return_value = Mock(inserted_ids=[original_id])
    collection.insert(obj, ignore_duplicates=ignore_duplicates)
    assert obj == original
    hook.assert_called_once_with([obj])
