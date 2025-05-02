"""A structure for representing collections of similar objects."""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    TextIO,
    Tuple,
    Type,
    Union,
)

import numpy as np
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.linkml_model.meta import ArrayExpression
from pydantic import BaseModel

from linkml_store.api.types import DatabaseType
from linkml_store.index import get_indexer
from linkml_store.utils.format_utils import load_objects, load_objects_from_url
from linkml_store.utils.object_utils import clean_empties
from linkml_store.utils.patch_utils import PatchDict, apply_patches_to_list, patches_from_objects_lists

try:
    from linkml.validator.report import ValidationResult
except ImportError:
    ValidationResult = None

from linkml_store.api.config import CollectionConfig
from linkml_store.api.queries import Query, QueryResult
from linkml_store.index.indexer import Indexer

if TYPE_CHECKING:
    from linkml_store.api.database import Database

logger = logging.getLogger(__name__)

OBJECT = Union[Dict[str, Any], BaseModel, Type]

DEFAULT_FACET_LIMIT = 100
IDENTIFIER = str
FIELD_NAME = str


class Collection(Generic[DatabaseType]):
    """
    A collection is an organized set of objects of the same or similar type.

    - For relational databases, a collection is typically a table
    - For document databases such as MongoDB, a collection is the native type
    - For a file system, a collection could be a single tabular file such as Parquet or CSV.

    Collection objects are typically not created directly - instead they are generated
    from a parent :class:`.Database` object:

    >>> from linkml_store import Client
    >>> client = Client()
    >>> db = client.attach_database("duckdb", alias="test")
    >>> collection = db.create_collection("Person")
    >>> objs = [{"id": "P1", "name": "John", "age_in_years": 30}, {"id": "P2", "name": "Alice", "age_in_years": 25}]
    >>> collection.insert(objs)
    """

    # name: str
    parent: Optional[DatabaseType] = None
    _indexers: Optional[Dict[str, Indexer]] = None
    _initialized: Optional[bool] = None
    # hidden: Optional[bool] = False

    metadata: Optional[CollectionConfig] = None
    default_index_name: ClassVar[str] = "simple"

    def __init__(
        self, name: str, parent: Optional["Database"] = None, metadata: Optional[CollectionConfig] = None, **kwargs
    ):
        self.parent = parent
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = CollectionConfig(type=name, **kwargs)
            if not self.metadata.alias:
                self.metadata.alias = name
            if not self.metadata.type:
                self.metadata.type = name
        # if name is not None and self.metadata.name is not None and name != self.metadata.name:
        #    raise ValueError(f"Name mismatch: {name} != {self.metadata.name}")

    @property
    def hidden(self) -> bool:
        """
        True if the collection is hidden.

        An example of a hidden collection is a collection that indexes another
        collection

        :return: True if the collection is hidden
        """
        # return self.metadata.hidden

    @property
    def target_class_name(self):
        """
        Return the name of the class that this collection represents

        This MUST be a LinkML class name

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person", alias="persons")
        >>> collection.target_class_name
        'Person'

        >>> collection = db.create_collection("Organization")
        >>> collection.target_class_name
        'Organization'
        >>> collection.alias
        'Organization'

        :return: name of the class which members of this collection instantiate
        """
        # TODO: this is a shim layer until we can normalize on this
        if self.metadata.type:
            return self.metadata.type
        return self.alias

    @property
    def alias(self):
        """
        Return the primary name/alias used for the collection.

        This MAY be the name of the LinkML class, but it may be desirable
        to have an alias, for example "persons" which collects all instances
        of class Person.

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person", alias="persons")
        >>> collection.alias
        'persons'

        If no explicit alias is provided, then the target class name is used:

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> collection.alias
        'Person'

        The alias SHOULD be used for Table names in SQL.

        For nested data, the alias SHOULD be used as the key; e.g

        .. code-block:: json

           { "persons": [ { "name": "Alice" }, { "name": "Bob" } ] }

        :return:
        """
        # TODO: this is a shim layer until we can normalize on this
        if self.metadata.alias:
            return self.metadata.alias
        return self.target_class_name

    def replace(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Replace entire collection with objects.

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> objs = [{"id": "P1", "name": "John", "age_in_years": 30}, {"id": "P2", "name": "Alice", "age_in_years": 25}]
        >>> collection.insert(objs)

        :param objs:
        :param kwargs:
        :return:
        """
        self.delete_where({})
        self.insert(objs, **kwargs)

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Add one or more objects to the collection.

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> objs = [{"id": "P1", "name": "John", "age_in_years": 30}, {"id": "P2", "name": "Alice", "age_in_years": 25}]
        >>> collection.insert(objs)

        :param objs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def index(
        self,
        objs: Union[OBJECT, List[OBJECT]],
        index_name: Optional[str] = None,
        replace: bool = False,
        unique: bool = False,
        **kwargs,
    ) -> None:
        """
        Index objects in the collection.

        :param objs:
        :param index_name:
        :param replace: replace the index, or not
        :param unique: boolean used to declare the index unique or not
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def upsert(
        self,
        objs: Union[OBJECT, List[OBJECT]],
        filter_fields: List[str],
        update_fields: Union[List[str], None] = None,
        **kwargs,
    ):
        """
        Add one or more objects to the collection.

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("mongodb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> objs = [{"id": "P1", "name": "John", "age_in_years": 30}, {"id": "P2", "name": "Alice", "age_in_years": 25}]
        >>> collection.upsert(objs)

        :param objs:
        :param filter_fields: List of field names to use as the filter for matching existing collections.
        :param update_fields: List of field names to include in the update. If None, all fields are updated.
        :param kwargs:

        :return:
        """
        raise NotImplementedError

    def _pre_query_hook(self, query: Optional[Query] = None, **kwargs):
        """
        Pre-query hook.

        This is called before a query is executed. It is used to materialize derivations and indexes.
        :param query:
        :param kwargs:
        :return:
        """
        logger.debug(f"Pre-query hook (state: {self._initialized}; Q= {query}")  # if logging.info, this is very noisy.
        if not self._initialized:
            self._materialize_derivations()
            self._initialized = True

    def _pre_insert_hook(self, objs: List[OBJECT], **kwargs):
        if self.metadata.validate_modifications:
            errors = list(self.iter_validate_collection(objs))
            if errors:
                raise ValueError(f"Validation errors: {errors}")

    def _post_insert_hook(self, objs: List[OBJECT], **kwargs):
        self._initialized = True
        patches = [{"op": "add", "path": "/0", "value": obj} for obj in objs]
        self._broadcast(patches, **kwargs)
        self._post_modification_hook(**kwargs)

    def _post_delete_hook(self, **kwargs):
        self._post_modification_hook(**kwargs)

    def _post_modification_hook(self, **kwargs):
        for indexer in self.indexers.values():
            ix_collection_name = self.get_index_collection_name(indexer)
            ix_collection = self.parent.get_collection(ix_collection_name)
            # Currently updating the source triggers complete reindexing
            # TODO: make this more efficient by only deleting modified
            ix_collection.delete_where({})

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> Optional[int]:
        """
        Delete one or more objects from the collection.

        First let's set up a collection:

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> objs = [{"id": "P1", "name": "John", "age_in_years": 30}, {"id": "P2", "name": "Alice", "age_in_years": 25}]
        >>> collection.insert(objs)
        >>> collection.find({}).num_rows
        2

        Now let's delete an object:

        >>> collection.delete(objs[0])
        >>> collection.find({}).num_rows
        1

        Deleting the same object again should have no effect:

        >>> collection.delete(objs[0])
        >>> collection.find({}).num_rows
        1

        :param objs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> Optional[int]:
        """
        Delete objects that match a query.

        First let's set up a collection:

        >>> from linkml_store import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> objs = [{"id": "P1", "name": "John", "age_in_years": 30}, {"id": "P2", "name": "Alice", "age_in_years": 25}]
        >>> collection.insert(objs)

        Now let's delete an object:

        >>> collection.delete_where({"id": "P1"})
        >>> collection.find({}).num_rows
        1

        Match everything:

        >>> collection.delete_where({})
        >>> collection.find({}).num_rows
        0

        :param where: where conditions
        :param missing_ok: if True, do not raise an error if the collection does not exist
        :param kwargs:
        :return: number of objects deleted (or -1 if unsupported)
        """
        raise NotImplementedError

    def update(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Update one or more objects in the collection.

        :param objs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def _create_query(self, **kwargs) -> Query:
        return Query(from_table=self.alias, **kwargs)

    def query(self, query: Query, **kwargs) -> QueryResult:
        """
        Run a query against the collection.

        First let's load a collection:

        >>> from linkml_store import Client
        >>> from linkml_store.utils.format_utils import load_objects
        >>> client = Client()
        >>> db = client.attach_database("duckdb")
        >>> collection = db.create_collection("Country")
        >>> objs = load_objects("tests/input/countries/countries.jsonl")
        >>> collection.insert(objs)

        Now let's run a query:

        TODO

        :param query:
        :param kwargs:
        :return:
        """
        self._pre_query_hook()
        return self.parent.query(query, **kwargs)

    def query_facets(
        self, where: Optional[Dict] = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[str, List[Tuple[Any, int]]]:
        """
        Run a query to get facet counts for one or more columns.

        This function takes a database connection, a Query object, and a list of column names.
        It generates and executes a facet count query for each specified column and returns
        the results as a dictionary where the keys are the column names and the values are
        pandas DataFrames containing the facet counts.

        The facet count query is generated by modifying the original query's WHERE clause
        to exclude conditions directly related to the facet column. This allows for counting
        the occurrences of each unique value in the facet column while still applying the
        other filtering conditions.

        :param con: A DuckDB database connection.
        :param query: A Query object representing the base query.
        :param facet_columns: A list of column names to get facet counts for.
        :param facet_limit:
        :return: A dictionary where keys are column names and values are tuples
                 containing the facet counts for each unique value in the respective column.
        """
        raise NotImplementedError

    def get(self, ids: Optional[List[IDENTIFIER]], **kwargs) -> QueryResult:
        """
        Get one or more objects by ID.

        :param ids:
        :param kwargs:
        :return:
        """
        id_field = self.identifier_attribute_name
        if not id_field:
            raise ValueError(f"No identifier for {self.name}")
        if len(ids) == 1:
            return self.find({id_field: ids[0]})
        else:
            return self.find({id_field: {"$in": ids}})

    def get_one(self, id: IDENTIFIER, **kwargs) -> Optional[OBJECT]:
        """
        Get one object by ID.

        :param id:
        :param kwargs:
        :return:
        """
        if not id:
            raise ValueError("Must pass an ID")
        id_field = self.identifier_attribute_name
        if not id_field:
            raise ValueError(f"No identifier for {self.name}")
        w = {id_field: id}
        qr = self.find(w)
        if qr.num_rows == 1:
            return qr.rows[0]
        return None

    def find(
        self,
        where: Optional[Any] = None,
        select_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Find objects in the collection using a where query.

        As an example, first load a collection:

        >>> from linkml_store import Client
        >>> from linkml_store.utils.format_utils import load_objects
        >>> client = Client()
        >>> db = client.attach_database("duckdb")
        >>> collection = db.create_collection("Country")
        >>> objs = load_objects("tests/input/countries/countries.jsonl")
        >>> collection.insert(objs)

        Now let's find all objects:

        >>> qr = collection.find({})
        >>> qr.num_rows
        20

        We can do a more restrictive query:

        >>> qr = collection.find({"code": "FR"})
        >>> qr.num_rows
        1
        >>> qr.rows[0]["name"]
        'France'


        :param where:
        :param select_cols:
        :param kwargs:
        :return:
        """
        query = self._create_query(
            where_clause=where,
            select_cols=select_cols,
        )
        self._pre_query_hook(query)
        return self.query(query, **kwargs)

    def find_iter(self, where: Optional[Any] = None, page_size=100, **kwargs) -> Iterator[OBJECT]:
        """
        Find objects in the collection using a where query.

        :param where:
        :param kwargs:
        :return:
        """
        total_rows = None
        offset = 0
        if page_size < 1:
            raise ValueError(f"Invalid page size: {page_size}")
        while True:
            qr = self.find(where=where, offset=offset, limit=page_size, **kwargs)
            if total_rows is None:
                total_rows = qr.num_rows
            if not qr.rows:
                return
            for row in qr.rows:
                yield row
            offset += page_size
            if offset >= total_rows:
                break
        return

    def search(
        self,
        query: str,
        where: Optional[Any] = None,
        index_name: Optional[str] = None,
        limit: Optional[int] = None,
        select_cols: Optional[List[str]] = None,
        mmr_relevance_factor: Optional[float] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Search the collection using a text-based index index.

        Example:

        >>> from linkml_store import Client
        >>> from linkml_store.utils.format_utils import load_objects
        >>> client = Client()
        >>> db = client.attach_database("duckdb")
        >>> collection = db.create_collection("Country")
        >>> objs = load_objects("tests/input/countries/countries.jsonl")
        >>> collection.insert(objs)

        Now let's index, using the simple trigram-based index

        >>> index = get_indexer("simple")
        >>> _ = collection.attach_indexer(index)

        Now let's find all objects:

        >>> qr = collection.search("France")
        >>> score, top_obj = qr.ranked_rows[0]
        >>> assert score > 0.1
        >>> top_obj["code"]
        'FR'

        :param query:
        :param where:
        :param index_name:
        :param limit:
        :param select_cols:
        :param kwargs:
        :return:
        """
        self._pre_query_hook()
        if index_name is None:
            if len(self.indexers) == 1:
                index_name = list(self.indexers.keys())[0]
            else:
                logger.warning("Multiple indexes found. Using default index.")
                index_name = self.default_index_name
        ix_coll = self.parent.get_collection(self._index_collection_name(index_name))
        if index_name not in self.indexers:
            logger.debug(f"Indexer not found: {index_name} -- creating")
            ix = get_indexer(index_name)
            if not self._indexers:
                self._indexers = {}
            self._indexers[index_name] = ix
        ix = self.indexers.get(index_name)
        if not ix:
            raise ValueError(f"No index named {index_name}")
        logger.debug(f"Using indexer {type(ix)} with name {index_name}")
        if ix_coll.size() == 0:
            logger.info(f"Index {index_name} is empty; indexing all objects")
            all_objs = self.find(limit=-1).rows
            if all_objs:
                # print(f"Index {index_name} is empty; indexing all objects {len(all_objs)}")
                self.index_objects(all_objs, index_name, replace=True, **kwargs)
                assert ix_coll.size() > 0
        qr = ix_coll.find(where=where, limit=-1, **kwargs)
        index_col = ix.index_field

        # TODO: optimize this for large indexes
        def row2array(row):
            v = row[index_col]
            if isinstance(v, str):
                # sqlite stores arrays as strings
                v = json.loads(v)
            return np.array(v, dtype=float)

        vector_pairs = [(row, row2array(row)) for row in qr.rows]
        results = ix.search(query, vector_pairs, limit=limit, mmr_relevance_factor=mmr_relevance_factor, **kwargs)
        for r in results:
            del r[1][index_col]
        if select_cols:
            new_results = []
            for r in results:
                new_results.append((r[0], {k: v for k, v in r[1].items() if k in select_cols}))
            results = new_results
        new_qr = QueryResult(num_rows=len(results))
        new_qr.ranked_rows = results
        new_qr.rows = [r[1] for r in results]
        return new_qr

    def group_by(
        self,
        group_by_fields: List[str],
        inlined_field="objects",
        agg_map: Optional[Dict[str, str]] = None,
        where: Optional[Dict] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Group objects in the collection by a column.

        :param group_by:
        :param where:
        :param kwargs:
        :return:
        """
        if isinstance(group_by_fields, str):
            group_by_fields = [group_by_fields]
        df = self.find(where=where, limit=-1).rows_dataframe

        # Handle the case where agg_map is None
        if agg_map is None:
            agg_map = {}

        pk_fields = agg_map.get("first", []) + group_by_fields
        list_fields = agg_map.get("list", [])
        if not list_fields:
            list_fields = [a for a in df.columns if a not in pk_fields]

        grouped_objs = defaultdict(list)
        for _, row in df.iterrows():
            pk = tuple(row[pk_fields])
            grouped_objs[pk].append({k: row[k] for k in list_fields})
        results = []
        for pk, objs in grouped_objs.items():
            top_obj = {k: v for k, v in zip(pk_fields, pk)}
            top_obj[inlined_field] = objs
            results.append(top_obj)
        r = QueryResult(num_rows=len(results), rows=results)
        return r

    @property
    def is_internal(self) -> bool:
        """
        Check if the collection is internal.

        Internal collections are hidden by default. Examples of internal collections
        include shadow "index" collections

        :return:
        """
        if not self.alias:
            raise ValueError(f"Collection has no alias: {self} // {self.metadata}")
        return self.alias.startswith("internal__")

    def exists(self) -> Optional[bool]:
        """
        Check if the collection exists.

        :return:
        """
        cd = self.class_definition()
        return cd is not None and cd.attributes

    def load_from_source(self, load_if_exists=False):
        """
        Load objects from the source location.

        :param load_if_exists:
        :return:
        """
        if not load_if_exists and self.exists():
            return
        metadata = self.metadata
        if metadata.source:
            source = metadata.source
            kwargs = source.arguments or {}
            if source.local_path:
                objects = load_objects(
                    metadata.source.local_path,
                    format=source.format,
                    expected_type=source.expected_type,
                    compression=source.compression,
                    select_query=source.select_query,
                    **kwargs,
                )
            elif metadata.source.url:
                objects = load_objects_from_url(
                    metadata.source.url,
                    format=source.format,
                    expected_type=source.expected_type,
                    compression=source.compression,
                    select_query=source.select_query,
                    **kwargs,
                )
            else:
                raise ValueError("No source local_path or url provided")
            self.insert(objects)

    def _check_if_initialized(self) -> bool:
        return self._initialized

    def _materialize_derivations(self, **kwargs):
        metadata = self.metadata
        if not metadata.derived_from:
            logger.info(f"No metadata for {self.alias}; no derivations")
            return
        if self._check_if_initialized():
            logger.info(f"Already initialized {self.alias}; no derivations")
            return
        parent_db = self.parent
        client = parent_db.parent
        # cd = self.class_definition()
        for derivation in metadata.derived_from:
            # TODO: optimize this; utilize underlying engine
            logger.info(f"Deriving from {derivation}")
            if derivation.database:
                db = client.get_database(derivation.database)
            else:
                db = parent_db
            if derivation.collection:
                coll = db.get_collection(derivation.collection)
            else:
                coll = self
            coll.class_definition()
            source_obj_iter = coll.find_iter(derivation.where or {})
            mappings = derivation.mappings
            if not mappings:
                raise ValueError(f"No mappings for {self.name}")
            target_class_name = self.target_class_name
            from linkml_map.session import Session

            session = Session()
            session.set_source_schema(db.schema_view.schema)
            session.set_object_transformer(
                {
                    "class_derivations": {
                        target_class_name: {
                            "populated_from": coll.target_class_name,
                            "slot_derivations": mappings,
                        },
                    }
                },
            )
            logger.debug(f"Session Spec: {session.object_transformer}")
            tr_objs = []
            for source_obj in source_obj_iter:
                tr_obj = session.transform(source_obj, source_type=coll.target_class_name)
                tr_objs.append(tr_obj)
            if not tr_objs:
                raise ValueError(f"No objects derived from {coll.name}")
            self.insert(tr_objs)
            self.commit()

    def size(self) -> int:
        """
        Return the number of objects in the collection.

        :return: The number of objects in the collection.
        """
        return self.find({}, limit=1).num_rows

    def rows_iter(self) -> Iterable[OBJECT]:
        """
        Return an iterator over the objects in the collection.

        :return:
        """
        yield from self.find({}, limit=-1).rows

    @property
    def rows(self) -> List[OBJECT]:
        """
        Return a list of objects in the collection.

        :return:
        """
        return list(self.rows_iter())

    def ranked_rows(self) -> List[Tuple[float, OBJECT]]:
        """
        Return a list of objects in the collection, with scores.
        """
        return [(n, obj) for n, obj in enumerate(self.rows_iter())]

    def attach_indexer(
        self, index: Union[Indexer, str], name: Optional[str] = None, auto_index=True, **kwargs
    ) -> Indexer:
        """
        Attach an index to the collection.

        As an example, first let's create a collection in a database:

        >>> from linkml_store import Client
        >>> from linkml_store.utils.format_utils import load_objects
        >>> client = Client()
        >>> db = client.attach_database("duckdb")
        >>> collection = db.create_collection("Country")
        >>> objs = load_objects("tests/input/countries/countries.jsonl")
        >>> collection.insert(objs)

        We will create two indexes - one that indexes the whole object
        (default behavior), the other one indexes the name only

        >>> full_index = get_indexer("simple")
        >>> full_index.name = "full"
        >>> name_index = get_indexer("simple", text_template="{name}")
        >>> name_index.name = "name"
        >>> _ = collection.attach_indexer(full_index)
        >>> _ = collection.attach_indexer(name_index)

        Now let's find objects using the full index, using the string "France".
        We expect the country France to be the top hit, but the score will
        be less than zero because we did not match all fields in the object.

        >>> qr = collection.search("France", index_name="full")
        >>> score, top_obj = qr.ranked_rows[0]
        >>> assert score > 0.1
        >>> assert score < 0.5
        >>> top_obj["code"]
        'FR'

        Now using the name index

        >>> qr = collection.search("France", index_name="name")
        >>> score, top_obj = qr.ranked_rows[0]
        >>> assert score > 0.99
        >>> top_obj["code"]
        'FR'

        :param index:
        :param name:
        :param auto_index: Automatically index all objects in the collection
        :param kwargs:
        :return:
        """
        if isinstance(index, str):
            index = get_indexer(index)
        if name:
            index.name = name
        if not index.name:
            index.name = type(index).__name__.lower()
        index_name = index.name
        if not index_name:
            raise ValueError("Index must have a name")
        if not self._indexers:
            self._indexers = {}
        self._indexers[index_name] = index
        if auto_index:
            all_objs = self.find(limit=-1).rows
            logger.info(f"Auto-indexing {len(all_objs)} objects")
            self.index_objects(all_objs, index_name, replace=True, **kwargs)
        return index

    def get_index_collection_name(self, indexer: Indexer) -> str:
        return self._index_collection_name(indexer.name)

    def _index_collection_name(self, index_name: str) -> str:
        """
        Create a name for a special collection that holds index data

        :param index_name:
        :param indexer:
        :return:
        """
        return f"internal__index__{self.alias}__{index_name}"

    def index_objects(self, objs: List[OBJECT], index_name: str, replace=False, **kwargs):
        """
        Index a list of objects using a specified index.

        By default, the indexed objects will be stored in a shadow
        collection in the same database, with additional fields for the index vector

        :param objs:
        :param index_name: e.g. simple, llm
        :param replace:
        :param kwargs:
        :return:
        """
        ix = self._indexers.get(index_name, None)
        if not ix:
            raise ValueError(f"No index named {index_name}")
        ix_coll_name = self._index_collection_name(index_name)
        ix_coll = self.parent.get_collection(ix_coll_name, create_if_not_exists=True)
        vectors = [list(float(e) for e in v) for v in ix.objects_to_vectors(objs)]
        objects_with_ix = []
        index_col = ix.index_field
        for obj, vector in zip(objs, vectors):
            # TODO: id field
            objects_with_ix.append({**obj, **{index_col: vector}})
        if replace:
            schema = self.parent.schema_view.schema
            logger.info(f"Checking if {ix_coll_name} is in {schema.classes.keys()}")
            if ix_coll_name in schema.classes:
                ix_coll.delete_where()

        ix_coll.insert(objects_with_ix, **kwargs)
        ix_coll.commit()

    def list_index_names(self) -> List[str]:
        """
        Return a list of index names

        :return:
        """
        return list(self._indexers.keys())

    @property
    def indexers(self) -> Dict[str, Indexer]:
        """
        Return a list of indexers

        :return:
        """
        return self._indexers if self._indexers else {}

    def peek(self, limit: Optional[int] = None) -> QueryResult:
        """
        Return the first N objects in the collection

        :param limit:
        :return:
        """
        q = self._create_query()
        return self.query(q, limit=limit)

    def class_definition(self) -> Optional[ClassDefinition]:
        """
        Return the class definition for the collection.

        If no schema has been explicitly set, and the native database does not
        have a schema, then a schema will be induced from the objects in the collection.

        :return:
        """
        sv: SchemaView = self.parent.schema_view
        if sv:
            cls = sv.get_class(self.target_class_name)
            # if not cls:
            #     logger.warning(f"{self.target_class_name} not in {sv.all_classes().keys()} ")
            # cls = sv.schema.classes[self.target_class_name]
            if cls and not cls.attributes:
                if not sv.class_induced_slots(cls.name):
                    for att in self._induce_attributes():
                        cls.attributes[att.name] = att
                    sv.set_modified()
            return cls
        return None

    def _induce_attributes(self) -> List[SlotDefinition]:
        result = self.find({}, limit=-1)
        cd = self.induce_class_definition_from_objects(result.rows, max_sample_size=None)
        return list(cd.attributes.values())

    @property
    def identifier_attribute_name(self) -> Optional[str]:
        """
        Return the name of the identifier attribute for the collection.

        AKA the primary key.

        :return: The name of the identifier attribute, if one exists.
        """
        cd = self.class_definition()
        if cd:
            for att in self.parent.schema_view.class_induced_slots(cd.name):
                if att.identifier:
                    return att.name
        return None

    def set_identifier_attribute_name(self, name: str):
        """
        Set the name of the identifier attribute for the collection.

        AKA the primary key.

        :param name: The name of the identifier attribute.
        """
        cd = self.class_definition()
        if not cd:
            raise ValueError(f"Cannot find class definition for {self.target_class_name}")
        id_att = None
        candidates = []
        sv: SchemaView = self.parent.schema_view
        cls = sv.get_class(cd.name)
        existing_id_slot = sv.get_identifier_slot(cls.name)
        if existing_id_slot:
            if existing_id_slot.name == name:
                return
            existing_id_slot.identifier = False
        for att in cls.attributes.values():
            candidates.append(att.name)
            if att.name == name:
                att.identifier = True
                id_att = att
            else:
                att.identifier = False
        if not id_att:
            raise ValueError(f"No attribute found with name {name} in {candidates}")
        sv.set_modified()

    def object_identifier(self, obj: OBJECT, auto=True) -> Optional[IDENTIFIER]:
        """
        Return the identifier for an object.

        :param obj:
        :param auto: If True, generate an identifier if one does not exist.
        :return:
        """
        pk = self.identifier_attribute_name
        if pk in obj:
            return obj[pk]
        elif auto:
            # TODO: use other unique keys if no primary key
            as_str = str(obj)
            md5 = hashlib.md5(as_str.encode()).hexdigest()
            return md5
        else:
            return None

    def induce_class_definition_from_objects(
        self, objs: List[OBJECT], max_sample_size: Optional[int] = None
    ) -> ClassDefinition:
        """
        Induce a class definition from a list of objects.

        This uses a heuristic procedure to infer the class definition from a list of objects.
        In general it is recommended you explicitly provide a schema.

        :param objs:
        :param max_sample_size:
        :return:
        """
        # TODO: use schemaview
        if max_sample_size is None:
            max_sample_size = 10
        if not self.target_class_name:
            raise ValueError(f"No target_class_name for {self.alias}")
        cd = ClassDefinition(self.target_class_name)
        keys = defaultdict(list)
        for obj in objs[0:max_sample_size]:
            if isinstance(obj, BaseModel):
                obj = obj.model_dump()
            if not isinstance(obj, dict):
                logger.warning(f"Skipping non-dict object: {obj}")
                continue
            for k, v in obj.items():
                keys[k].append(v)
        for k, vs in keys.items():
            if k == "_id":
                continue
            multivalueds = []
            inlineds = []
            rngs = []
            exact_dimensions_list = []
            for v in vs:
                if v is None:
                    continue
                if isinstance(v, np.ndarray):
                    rngs.append("float")
                    exact_dimensions_list.append(v.shape)
                    break
                if isinstance(v, list):
                    # sample first item. TODO: more robust strategy
                    v = v[0] if v else None
                    multivalueds.append(True)
                elif isinstance(v, dict):
                    pass
                    # TODO: check if this is a nested object or key-value list
                    # v = list(v.values())[0]
                    # multivalueds.append(True)
                else:
                    multivalueds.append(False)
                if not v:
                    continue
                if isinstance(v, str):
                    rng = "string"
                elif isinstance(v, bool):
                    rng = "boolean"
                elif isinstance(v, int):
                    rng = "integer"
                elif isinstance(v, float):
                    rng = "float"
                elif isinstance(v, dict):
                    rng = None
                    inlineds.append(True)
                else:
                    # raise ValueError(f"No mappings for {type(v)} // v={v}")
                    rng = None
                    inlineds.append(False)
                rngs.append(rng)
            multivalued = any(multivalueds)
            inlined = any(inlineds)
            if multivalued and False in multivalueds:
                logger.info(f"Mixed list non list: {vs} // inferred= {multivalueds}")
            # if not rngs:
            #    raise AssertionError(f"Empty rngs for {k} = {vs}")
            rng = rngs[0] if rngs else None
            for other_rng in rngs:
                coercions = {
                    ("integer", "float"): "float",
                }
                if rng != other_rng:
                    if (rng, other_rng) in coercions:
                        rng = coercions[(rng, other_rng)]
                    elif (other_rng, rng) in coercions:
                        rng = coercions[(other_rng, rng)]
                    else:
                        raise ValueError(f"Conflict: {rng} != {other_rng} for {vs}")
            logger.debug(f"Inducing {k} as {rng} {multivalued} {inlined}")
            inlined_as_list = inlined and multivalued
            cd.attributes[k] = SlotDefinition(
                k, range=rng, multivalued=multivalued, inlined=inlined, inlined_as_list=inlined_as_list
            )
            if exact_dimensions_list:
                array_expr = ArrayExpression(exact_number_dimensions=len(exact_dimensions_list[0]))
                cd.attributes[k].array = array_expr
        sv = self.parent.schema_view
        sv.schema.classes[self.target_class_name] = cd
        sv.set_modified()
        return cd

    def import_data(self, location: Union[Path, str, TextIO], **kwargs):
        """
        Import data from a file or stream

        :param location:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def export_data(self, location: Union[Path, str, TextIO], **kwargs):
        """
        Export data to a file or stream

        :param location:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def apply_patches(self, patches: List[PatchDict], **kwargs):
        """
        Apply a patch to the collection.

        Patches conform to the JSON Patch format.

        :param patches:
        :param kwargs:
        :return:
        """
        all_objs = self.find(limit=-1).rows
        primary_key = self.identifier_attribute_name
        if not primary_key:
            raise ValueError(f"No primary key for {self.target_class_name}")
        new_objs = apply_patches_to_list(all_objs, patches, primary_key=primary_key, **kwargs)
        self.replace(new_objs)

    def diff(self, other: "Collection", **kwargs) -> List[PatchDict]:
        """
        Diff two collections.

        :param other: The collection to diff against
        :param kwargs:
        :return:
        """
        src_objs = self.find(limit=-1).rows
        tgt_objs = other.find(limit=-1).rows
        primary_key = self.identifier_attribute_name
        if not primary_key:
            raise ValueError(f"No primary key for {self.target_class_name}")
        patches_from_objects_lists(src_objs, tgt_objs, primary_key=primary_key)
        return patches_from_objects_lists(src_objs, tgt_objs, primary_key=primary_key)

    def iter_validate_collection(
        self, objects: Optional[Iterable[OBJECT]] = None, **kwargs
    ) -> Iterator["ValidationResult"]:
        """
        Validate the contents of the collection

        :param kwargs:
        :param objects: objects to validate
        :return: iterator over validation results
        """
        from linkml.validator import JsonschemaValidationPlugin, Validator

        validation_plugins = [JsonschemaValidationPlugin(closed=True)]
        validator = Validator(self.parent.schema_view.schema, validation_plugins=validation_plugins)
        cd = self.class_definition()
        if not cd:
            raise ValueError(f"Cannot find class definition for {self.target_class_name}")
        type_designator = None
        for att in self.parent.schema_view.class_induced_slots(cd.name):
            if att.designates_type:
                type_designator = att.name
        class_name = cd.name
        if objects is None:
            objects = self.find_iter(**kwargs)
        for obj in objects:
            obj = clean_empties(obj)
            v_class_name = class_name
            if type_designator is not None:
                # TODO: move type designator logic to core linkml
                this_class_name = obj.get(type_designator)
                if this_class_name:
                    if ":" in this_class_name:
                        this_class_name = this_class_name.split(":")[-1]
                    v_class_name = this_class_name
            yield from validator.iter_results(obj, v_class_name)

    def commit(self):
        """
        Commit changes to the collection.

        :return:
        """
        pass

    def _broadcast(self, *args, **kwargs):
        self.parent.broadcast(self, *args, **kwargs)
