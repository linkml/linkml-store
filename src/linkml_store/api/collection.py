"""A structure for representing collections of similar objects."""

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterator, List, Optional, TextIO, Tuple, Type, Union

import numpy as np
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.linkml_model.meta import ArrayExpression
from pydantic import BaseModel

from linkml_store.api.types import DatabaseType
from linkml_store.index import get_indexer
from linkml_store.utils.format_utils import load_objects
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
    # hidden: Optional[bool] = False

    metadata: Optional[CollectionConfig] = None

    def __init__(
        self, name: str, parent: Optional["Database"] = None, metadata: Optional[CollectionConfig] = None, **kwargs
    ):
        self.parent = parent
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = CollectionConfig(name=name, **kwargs)
            if not self.metadata.alias:
                self.metadata.alias = name
            if not self.metadata.type:
                self.metadata.type = name
        # if name is not None and self.metadata.name is not None and name != self.metadata.name:
        #    raise ValueError(f"Name mismatch: {name} != {self.metadata.name}")

    @property
    def name(self) -> str:
        """
        Return the name of the collection.

        TODO: deprecate in favor of Type

        :return: name of the collection
        """
        return self.metadata.name

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

        :return: name of the class which members of this collection instantiate
        """
        # TODO: this is a shim layer until we can normalize on this
        if self.metadata.type:
            return self.metadata.type
        return self.name

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
        # TODO: this is a shim layer until we can normalize on this
        if self.metadata.alias:
            return self.metadata.alias
        return self.name

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

    def _post_insert_hook(self, objs: List[OBJECT], **kwargs):
        patches = [{"op": "add", "path": "/0", "value": obj} for obj in objs]
        self._broadcast(patches, **kwargs)

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
        # TODO
        id_field = self.identifier_attribute_name
        if not id_field:
            raise ValueError(f"No identifier for {self.name}")
        return self.find({id_field: ids})

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

    def find(self, where: Optional[Any] = None, **kwargs) -> QueryResult:
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
        :param kwargs:
        :return:
        """
        query = self._create_query(where_clause=where)
        return self.query(query, **kwargs)

    def find_iter(self, where: Optional[Any] = None, **kwargs) -> Iterator[OBJECT]:
        """
        Find objects in the collection using a where query.

        :param where:
        :param kwargs:
        :return:
        """
        qr = self.find(where=where, limit=-1, **kwargs)
        for row in qr.rows:
            yield row

    def search(
        self,
        query: str,
        where: Optional[Any] = None,
        index_name: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Search the collection using a full-text search index.

        :param query:
        :param where:
        :param index_name:
        :param limit:
        :param kwargs:
        :return:
        """
        if index_name is None:
            if len(self._indexers) == 1:
                index_name = list(self._indexers.keys())[0]
            else:
                raise ValueError("Multiple indexes found. Please specify an index name.")
        ix_coll = self.parent.get_collection(self._index_collection_name(index_name))
        ix = self._indexers.get(index_name)
        if not ix:
            raise ValueError(f"No index named {index_name}")
        qr = ix_coll.find(where=where, limit=-1, **kwargs)
        index_col = ix.index_field
        # TODO: optimize this for large indexes
        vector_pairs = [(row, np.array(row[index_col], dtype=float)) for row in qr.rows]
        results = ix.search(query, vector_pairs, limit=limit)
        for r in results:
            del r[1][index_col]
        new_qr = QueryResult(num_rows=len(results))
        new_qr.ranked_rows = results
        return new_qr

    @property
    def is_internal(self) -> bool:
        """
        Check if the collection is internal

        :return:
        """
        if not self.alias:
            raise ValueError(f"Collection has no alias: {self} // {self.metadata}")
        return self.alias.startswith("internal__")

    def load_from_source(self):
        objects = load_objects(self.metadata.source_location)
        self.insert(objects)

    def attach_indexer(self, index: Union[Indexer, str], name: Optional[str] = None, auto_index=True, **kwargs):
        """
        Attach an index to the collection.

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

    def _index_collection_name(self, index_name: str) -> str:
        """
        Create a name for a special collection that holds index data

        :param index_name:
        :param indexer:
        :return:
        """
        return f"internal__index__{self.name}__{index_name}"

    def index_objects(self, objs: List[OBJECT], index_name: str, replace=False, **kwargs):
        """
        Index a list of objects

        :param objs:
        :param index_name:
        :param replace:
        :param kwargs:
        :return:
        """
        ix = self._indexers.get(index_name)
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

        :return:
        """
        sv: SchemaView = self.parent.schema_view
        if sv:
            cls = sv.get_class(self.target_class_name)
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

    def induce_class_definition_from_objects(self, objs: List[OBJECT], max_sample_size=10) -> ClassDefinition:
        """
        Induce a class definition from a list of objects.

        This uses a heuristic procedure to infer the class definition from a list of objects.
        In general it is recommended you explicitly provide a schema.

        :param objs:
        :param max_sample_size:
        :return:
        """
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
                    v = v[0]
                    multivalueds.append(True)
                elif isinstance(v, dict):
                    v = list(v.values())[0]
                    multivalueds.append(True)
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
                raise ValueError(f"Mixed list non list: {vs} // inferred= {multivalueds}")
            # if not rngs:
            #    raise AssertionError(f"Empty rngs for {k} = {vs}")
            rng = rngs[0] if rngs else None
            for other_rng in rngs:
                if rng != other_rng:
                    raise ValueError(f"Conflict: {rng} != {other_rng} for {vs}")
            cd.attributes[k] = SlotDefinition(k, range=rng, multivalued=multivalued, inlined=inlined)
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

        Patches conform to the JSON Patch format,

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

    def diff(self, other: "Collection", **kwargs):
        """
        Diff two collections.

        :param other:
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

    def iter_validate_collection(self, **kwargs) -> Iterator["ValidationResult"]:
        """
        Validate the contents of the collection

        :param kwargs:
        :return: iterator over validation results
        """
        from linkml.validator import JsonschemaValidationPlugin, Validator

        validation_plugins = [JsonschemaValidationPlugin(closed=True)]
        validator = Validator(self.parent.schema_view.schema, validation_plugins=validation_plugins)
        cd = self.class_definition()
        if not cd:
            raise ValueError(f"Cannot find class definition for {self.target_class_name}")
        class_name = cd.name
        result = self.find(**kwargs)
        for obj in result.rows:
            obj = clean_empties(obj)
            yield from validator.iter_results(obj, class_name)

    def commit(self):
        """
        Commit changes to the collection.

        :return:
        """
        pass

    def _broadcast(self, *args, **kwargs):
        self.parent.broadcast(self, *args, **kwargs)
