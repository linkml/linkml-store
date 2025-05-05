import logging
from abc import ABC
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from linkml_store.api.types import CollectionType
from linkml_store.utils.format_utils import Format, load_objects, render_output
from linkml_store.utils.patch_utils import PatchDict

try:
    from linkml.validator.report import Severity, ValidationResult
except ImportError:
    ValidationResult = None

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SchemaDefinition

from linkml_store.api.collection import Collection
from linkml_store.api.config import CollectionConfig, DatabaseConfig
from linkml_store.api.queries import Query, QueryResult

if TYPE_CHECKING:
    from linkml_store.api.client import Client

logger = logging.getLogger(__name__)

LISTENER = Callable[[Collection, List[PatchDict]], None]


class Database(ABC, Generic[CollectionType]):
    """
    A Database provides access to named collections of data.

    A database object is owned by a :ref:`Client`. The database
    object uses a :ref:`handle` to know what kind of external
    dataase system to connect to (e.g. duckdb, mongodb). The handle
    is a string ``<DatabaseType>:<LocalLocator>``

    The
    database object may also have an :ref:`alias` that is mapped
    to the handle.

    Attaching a database
    --------------------
    >>> from linkml_store.api.client import Client
    >>> client = Client()
    >>> db = client.attach_database("duckdb:///:memory:", alias="test")

    We can check the value of the handle:

    >>> db.handle
    'duckdb:///:memory:'

    The alias can be used to retrieve the database object from the client

    >>> assert db == client.get_database("test")

    Creating a collection
    ---------------------

    >>> collection = db.create_collection("Person")
    >>> len(db.list_collections())
    1
    >>> db.get_collection("Person") == collection
    True
    >>> objs = [{"id": "P1", "name": "John", "age_in_years": 30}, {"id": "P2", "name": "Alice", "age_in_years": 25}]
    >>> collection.insert(objs)
    >>> qr = collection.find()
    >>> len(qr.rows)
    2
    >>> qr.rows[0]["id"]
    'P1'
    >>> qr.rows[1]["name"]
    'Alice'
    >>> qr = collection.find({"name": "John"})
    >>> len(qr.rows)
    1
    >>> qr.rows[0]["name"]
    'John'

    """

    _schema_view: Optional[SchemaView] = None
    """Schema for the database. May be transformed."""

    _original_schema_view: Optional[SchemaView] = None
    """If a schema must be transformed, then the original is stored here."""

    _collections: Optional[Dict[str, Collection]] = None
    parent: Optional["Client"] = None
    metadata: Optional[DatabaseConfig] = None
    collection_class: ClassVar[Optional[Type[Collection]]] = None

    listeners: Optional[List[LISTENER]] = None

    def __init__(self, handle: Optional[str] = None, metadata: Optional[DatabaseConfig] = None, **kwargs):
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = DatabaseConfig(handle=handle, **kwargs)
        if handle is not None and self.metadata.handle is not None and handle != self.metadata.handle:
            raise ValueError(f"Handle mismatch: {handle} != {self.metadata.handle}")
        self._initialize_schema()
        self._initialize_collections()

    def _initialize_schema(self, **kwargs):
        db_config = self.metadata
        if db_config.schema_location:
            schema_location = db_config.schema_location.format(base_dir=self.parent.metadata.base_dir)
            logger.info(f"Loading schema from: {schema_location}")
            self.load_schema_view(schema_location)
        if db_config.schema_dict:
            schema_dict = copy(db_config.schema_dict)
            if "id" not in schema_dict:
                schema_dict["id"] = "tmp"
            if "name" not in schema_dict:
                schema_dict["name"] = "tmp"
            self.set_schema_view(SchemaView(SchemaDefinition(**schema_dict)))

    def from_config(self, db_config: DatabaseConfig, **kwargs):
        """
        Initialize a database from a configuration.

        TODO: DEPRECATE

        :param db_config: database configuration
        :param kwargs: additional arguments
        """
        self.metadata = db_config
        self._initialize_schema()
        self._initialize_collections()
        return self

    def _initialize_collections(self):
        if not self.metadata.collections:
            return
        for k, collection_config in self.metadata.collections.items():
            if collection_config.alias:
                if collection_config.alias != k:
                    raise ValueError(f"Alias mismatch: {collection_config.alias} != {k}")
            alias = k
            typ = collection_config.type or alias
            _collection = self.create_collection(typ, alias=alias, metadata=collection_config)
            assert _collection.alias == alias
            assert _collection.target_class_name == typ
            if collection_config.attributes:
                # initialize schema
                sv = self.schema_view
                cd = ClassDefinition(typ, attributes=collection_config.attributes)
                sv.schema.classes[cd.name] = cd
                sv.set_modified()
                # assert collection.class_definition() is not None

    @property
    def recreate_if_exists(self) -> bool:
        """
        Return whether to recreate the database if it already exists.

        :return:
        """
        return self.metadata.recreate_if_exists

    @property
    def handle(self) -> str:
        """
        Return the database handle.

        Examples:

        - ``duckdb:///:memory:``
        - ``duckdb:///tmp/test.db``
        - ``mongodb://localhost:27017/``

        :return:
        """
        return self.metadata.handle

    @property
    def alias(self):
        return self.metadata.alias

    def store(self, obj: Dict[str, Any], **kwargs):
        """
        Store an object in the database.

        The object is assumed to be a Dictionary of Collections.

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> db.store({"persons": [{"id": "P1", "name": "John", "age_in_years": 30}]})
        >>> collection = db.get_collection("persons")
        >>> qr = collection.find()
        >>> qr.num_rows
        1

        :param obj: object to store
        :param kwargs: additional arguments
        """
        sv = self.schema_view
        roots = [c for c in sv.all_classes().values() if c.tree_root]
        root = roots[0] if roots else None
        for k, v in obj.items():
            logger.info(f"Storing collection {k}")
            if root:
                slot = sv.induced_slot(k, root.name)
                if not slot:
                    raise ValueError(f"Cannot determine type for {k}")
            else:
                slot = None
            if isinstance(v, dict):
                logger.debug(f"Coercing dict to list: {v}")
                v = [v]
            if not isinstance(v, list):
                continue
            if not v:
                continue
            if slot:
                logger.debug(f"Aligning to existing slot: {slot.name} range={slot.range}")
                collection = self.get_collection(slot.name, type=slot.range, create_if_not_exists=True)
            else:
                collection = self.get_collection(k, create_if_not_exists=True)
            logger.debug(f"Replacing using {collection.alias} {collection.target_class_name}")
            collection.replace(v)

    def commit(self, **kwargs):
        """
        Commit pending changes to the database.

        :param kwargs:
        :return:
        """
        for coll in self.list_collections():
            coll.commit()

    def close(self, **kwargs):
        """
        Close the database.

        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    @property
    def _collection_class(self) -> Type[Collection]:
        raise NotImplementedError()

    def create_collection(
        self,
        name: str,
        alias: Optional[str] = None,
        metadata: Optional[CollectionConfig] = None,
        recreate_if_exists=False,
        **kwargs,
    ) -> Collection:
        """
        Create a new collection in the current database.

        The collection must have a *Type*, and may have an *Alias*.

        Examples:

            >>> from linkml_store.api.client import Client
            >>> client = Client()
            >>> db = client.attach_database("duckdb", alias="test")
            >>> collection = db.create_collection("Person", alias="persons")
            >>> collection.alias
            'persons'

            >>> collection.target_class_name
            'Person'

        If alias is not provided, it defaults to the name of the type.

        >>> collection = db.create_collection("Organization")
        >>> collection.alias
        'Organization'

        :param name: name of the collection
        :param alias: alias for the collection
        :param metadata: metadata for the collection
        :param recreate_if_exists: recreate the collection if it already exists
        :param kwargs: additional arguments
        """
        if not name:
            raise ValueError(f"Collection name must be provided: alias: {alias} metadata: {metadata}")
        collection_cls = self.collection_class
        collection = collection_cls(name=name, parent=self, metadata=metadata)
        if alias:
            collection.metadata.alias = alias
        if metadata and metadata.source:
            collection.load_from_source()
        if metadata and metadata.attributes:
            sv = self.schema_view
            schema = sv.schema
            cd = ClassDefinition(name=metadata.type, attributes=metadata.attributes)
            schema.classes[cd.name] = cd
        if not self._collections:
            self._collections = {}
        if not alias:
            alias = name
        self._collections[alias] = collection
        if recreate_if_exists:
            logger.debug(f"Recreating collection {collection.alias}")
            collection.delete_where({}, missing_ok=True)
        return collection

    def list_collections(self, include_internal=False) -> Sequence[Collection]:
        """
        List all collections.

        Examples
        --------
        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> c1 = db.create_collection("Person")
        >>> c2 = db.create_collection("Product")
        >>> collections = db.list_collections()
        >>> len(collections)
        2
        >>> [c.target_class_name for c in collections]
        ['Person', 'Product']

        :param include_internal: include internal collections
        :return: list of collections
        """
        if not self._collections:
            self.init_collections()
        return [c for c in self._collections.values() if include_internal or not c.is_internal]

    def list_collection_names(self, **kwargs) -> Sequence[str]:
        """
        List all collection names.

        Examples
        --------
        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> c1 = db.create_collection("Person")
        >>> c2 = db.create_collection("Product")
        >>> collection_names = db.list_collection_names()
        >>> len(collection_names)
        2
        >>> collection_names
        ['Person', 'Product']

        """
        return [c.alias for c in self.list_collections(**kwargs)]

    def get_collection(
        self, name: str, type: Optional[str] = None, create_if_not_exists=True, **kwargs
    ) -> "Collection":
        """
        Get a named collection.

        Examples
        --------
        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> db.get_collection("Person") == collection
        True
        >>> db.get_collection("NonExistent", create_if_not_exists=False)
        Traceback (most recent call last):
            ...
        KeyError: 'Collection NonExistent does not exist'

        :param name: name of the collection
        :param type: target class name
        :param create_if_not_exists: create the collection if it does not exist

        """
        if not self._collections:
            logger.debug("Initializing collections")
            self.init_collections()
        if name not in self._collections.keys():
            if create_if_not_exists:
                if type is None:
                    type = name
                logger.debug(f"Creating new collection: {name} kwargs: {kwargs}")
                self._collections[name] = self.create_collection(type, alias=name, **kwargs)
            else:
                raise KeyError(f"Collection {name} does not exist")
        return self._collections[name]

    def init_collections(self):
        """
        Initialize collections.

        TODO: Not typically called directly: consider making this private
        :return:
        """
        raise NotImplementedError

    def query(self, query: Query, **kwargs) -> QueryResult:
        """
        Run a query against the database.

        Examples
        --------
        >>> from linkml_store.api.client import Client
        >>> from linkml_store.api.queries import Query
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test", recreate_if_exists=True)
        >>> collection = db.create_collection("Person")
        >>> collection.insert([{"id": "P1", "name": "John"}, {"id": "P2", "name": "Alice"}])
        >>> query = Query(from_table="Person", where_clause={"name": "John"})
        >>> result = db.query(query)
        >>> len(result.rows)
        1
        >>> result.rows[0]["id"]
        'P1'

        :param query:
        :param kwargs:
        :return:

        """
        if query.from_table:
            collection = self.get_collection(query.from_table)
            return collection.query(query, **kwargs)
        else:
            raise NotImplementedError(f"Querying without a table is not supported in {self.__class__.__name__}")

    @property
    def schema_view(self) -> SchemaView:
        """
        Return a schema view for the named collection.

        If no explicit schema is provided, this will generalize one

        Induced schema example:

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test", recreate_if_exists=True)
        >>> collection = db.create_collection("Person", alias="persons")
        >>> collection.insert([{"id": "P1", "name": "John", "age_in_years": 25}])
        >>> schema_view = db.schema_view
        >>> cd = schema_view.get_class("Person")
        >>> cd.attributes["id"].range
        'string'
        >>> cd.attributes["age_in_years"].range
        'integer'

        We can reuse the same class:

        >>> collection2 = db.create_collection("Person", alias="other_persons")
        >>> collection2.class_definition().attributes["age_in_years"].range
        'integer'
        """
        if not self._schema_view:
            self._initialize_schema()
        if not self._schema_view:
            logger.info("Inducing schema view")
            self._schema_view = self.induce_schema_view()
        return self._schema_view

    def set_schema_view(self, schema_view: Union[str, Path, SchemaView]):
        """
        Set the schema view for the database.

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> sv = SchemaView("tests/input/countries/countries.linkml.yaml")
        >>> db.set_schema_view(sv)
        >>> cd = db.schema_view.schema.classes["Country"]
        >>> sorted(cd.slots)
        ['capital', 'code', 'continent', 'languages', 'name']
        >>> induced_slots = {s.name: s for s in sv.class_induced_slots("Country")}
        >>> sorted(induced_slots.keys())
        ['capital', 'code', 'continent', 'languages', 'name']
        >>> induced_slots["code"].identifier
        True

        Creating a new collection will align with the schema view:

        >>> collection = db.create_collection("Country", "all_countries")
        >>> sorted(collection.class_definition().slots)
        ['capital', 'code', 'continent', 'languages', 'name']

        :param schema_view: can be either a path to the schema, or a SchemaView object
        :return:
        """
        if isinstance(schema_view, Path):
            schema_view = str(schema_view)
        if isinstance(schema_view, str):
            schema_view = SchemaView(schema_view)
        self._schema_view = schema_view
        logger.info(f"Setting schema view for {self.handle}")
        # self._schema_view = SchemaView(schema_view.materialize_derived_schema())
        if not self._collections:
            return

        # align with induced schema
        roots = [c for c in schema_view.all_classes().values() if c.tree_root]
        if len(roots) == 0:
            all_ranges = set()
            for cn in schema_view.all_classes():
                for slot in schema_view.class_induced_slots(cn):
                    if slot.range:
                        all_ranges.add(slot.range)
            roots = [
                c
                for c in schema_view.all_classes().values()
                if not all_ranges.intersection(schema_view.class_ancestors(c.name, reflexive=True))
            ]
        if len(roots) == 1:
            root = roots[0]
            for slot in schema_view.class_induced_slots(root.name):
                inlined = slot.inlined or slot.inlined_as_list
                if inlined and slot.range:
                    if slot.name in self._collections:
                        coll = self._collections[slot.name]
                        coll.metadata.type = slot.range

    def load_schema_view(self, path: Union[str, Path]):
        """
        Load a schema view from a file.

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> db.load_schema_view("tests/input/countries/countries.linkml.yaml")
        >>> sv = db.schema_view
        >>> cd = sv.schema.classes["Country"]
        >>> sorted(cd.slots)
        ['capital', 'code', 'continent', 'languages', 'name']
        >>> induced_slots = {s.name: s for s in sv.class_induced_slots("Country")}
        >>> sorted(induced_slots.keys())
        ['capital', 'code', 'continent', 'languages', 'name']
        >>> induced_slots["code"].identifier
        True

        Creating a new collection will align with the schema view:

        >>> collection = db.create_collection("Country", "all_countries")
        >>> sorted(collection.class_definition().slots)
        ['capital', 'code', 'continent', 'languages', 'name']

        :param path:
        :return:
        """
        if isinstance(path, Path):
            path = str(path)
        self.set_schema_view(SchemaView(path))

    def induce_schema_view(self) -> SchemaView:
        """
        Induce a schema view from a schema definition.

        >>> from linkml_store.api.client import Client
        >>> from linkml_store.api.queries import Query
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> collection.insert([{"id": "P1", "name": "John", "age_in_years": 25},
        ...                 {"id": "P2", "name": "Alice", "age_in_years": 25}])
        >>> schema_view = db.induce_schema_view()
        >>> cd = schema_view.get_class("Person")
        >>> cd.attributes["id"].range
        'string'
        >>> cd.attributes["age_in_years"].range
        'integer'

        :return: A schema view
        """
        logger.info(f"Inducing schema view for {self.handle}")
        from linkml_runtime.utils.schema_builder import SchemaBuilder

        sb = SchemaBuilder()

        for collection_name in self.list_collection_names():
            coll = self.get_collection(collection_name)
            sb.add_class(coll.target_class_name)
        return SchemaView(sb.schema)

    def validate_database(self, **kwargs) -> List["ValidationResult"]:
        """
        Validate the contents of the database.

        As `iter_validate_database`, but returns a list of validation results.

        :param kwargs:
        :return:
        """
        return list(self.iter_validate_database(**kwargs))

    def iter_validate_database(
        self, ensure_referential_integrity: bool = None, **kwargs
    ) -> Iterator["ValidationResult"]:
        """
        Validate the contents of the database.

        An example, let's create a database with a predefined schema
        from the countries.linkml.yaml file:

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> db.load_schema_view("tests/input/countries/countries.linkml.yaml")

        Let's introspect the schema to see what slots are applicable for the class "Country":

        >>> sv = db.schema_view
        >>> for slot in sv.class_induced_slots("Country"):
        ...     print(slot.name, slot.range, slot.required)
        name string True
        code string True
        capital string True
        continent string True
        languages Language None

        Next we'll create a collection, binding it to the target class "Country", and insert
        valid data:

        >>> collection = db.create_collection("Country", "all_countries")
        >>> obj = {"code": "US", "name": "United States", "continent": "North America", "capital": "Washington, D.C."}
        >>> collection.insert([obj])
        >>> list(db.iter_validate_database())
        []

        Now let's insert some invalid data (missing required fields)

        >>> collection.insert([{"code": "FR", "name": "France"}])
        >>> for r in db.iter_validate_database():
        ...    print(r.message[0:32])
        'capital' is a required property
        'continent' is a required proper

        :param ensure_referential_integrity: ensure referential integrity
        :param kwargs:
        :return: iterator over validation results
        """
        for collection in self.list_collections():
            yield from collection.iter_validate_collection(**kwargs)
        if self.metadata.ensure_referential_integrity or ensure_referential_integrity:
            logger.info(f"Validating referential integrity on {self.alias}")
            yield from self._validate_referential_integrity(**kwargs)

    def _validate_referential_integrity(self, **kwargs) -> Iterator["ValidationResult"]:
        """
        Validate referential integrity of the database.

        :param kwargs:
        :return: iterator over validation results
        """
        sv = self.schema_view
        cmap = defaultdict(list)
        for collection in self.list_collections():
            if not collection.target_class_name:
                raise ValueError(f"Collection {collection.name} has no target class")
            cmap[collection.target_class_name].append(collection)
        for collection in self.list_collections():
            cd = collection.class_definition()
            induced_slots = sv.class_induced_slots(cd.name)
            slot_map = {s.name: s for s in induced_slots}
            # rmap = {s.name: s.range for s in induced_slots}
            # map slot ranges to a collection where that range is stored
            sr_to_coll = {s.name: cmap.get(s.range, []) for s in induced_slots if s.range}
            logger.debug(f"Validating referential integrity for {collection.target_class_name} // {sr_to_coll}")
            for obj in collection.find_iter():
                for k, v in obj.items():
                    if k not in sr_to_coll:
                        continue
                    ref_colls = sr_to_coll[k]
                    if not ref_colls:
                        continue
                    if not isinstance(v, (str, int)):
                        continue
                    slot = slot_map[k]
                    found = False
                    for ref_coll in ref_colls:
                        ref_obj = ref_coll.get_one(v)
                        if ref_obj:
                            found = True
                            break
                    if not found:
                        yield ValidationResult(
                            type="ReferentialIntegrity",
                            severity=Severity.ERROR,
                            message=f"Referential integrity error: {slot.range} not found",
                            instantiates=slot.range,
                            instance=v,
                        )

    def drop(self, **kwargs):
        """
        Drop the database and all collections.

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> path = Path("/tmp/test.db")
        >>> path.parent.mkdir(exist_ok=True, parents=True)
        >>> db = client.attach_database(f"duckdb:///{path}")
        >>> db.store({"persons": [{"id": "P1", "name": "John", "age_in_years": 30}]})
        >>> coll = db.get_collection("persons")
        >>> coll.find({}).num_rows
        1
        >>> db.drop()
        >>> db = client.attach_database("duckdb:///tmp/test.db", alias="test")
        >>> coll = db.get_collection("persons")
        >>> coll.find({}).num_rows
        0

        :param kwargs: additional arguments
        """
        raise NotImplementedError()

    def import_database(
        self,
        location: str,
        source_format: Optional[Union[str, Format]] = None,
        collection_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Import a database from a file or location.

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test", recreate_if_exists=True)
        >>> db.import_database("tests/input/iris.csv", Format.CSV, collection_name="iris")
        >>> db.list_collection_names()
        ['iris']
        >>> collection = db.get_collection("iris")
        >>> collection.find({}).num_rows
        150

        :param location: location of the file
        :param source_format: source format
        :param collection_name: (Optional) name of the collection, for data that is flat
        :param kwargs: additional arguments
        """
        if isinstance(source_format, str):
            source_format = Format(source_format)
        if isinstance(source_format, Format):
            if source_format.is_dump_format() and source_format in [Format.SQLDUMP_DUCKDB, Format.DUMP_MONGODB]:
                # import into a test instance
                tmp_handle = source_format.value
                client = self.parent
                tmp_alias = "tmp"
                client.drop_database(tmp_alias, missing_ok=True)
                tmp_db = client.attach_database(tmp_handle, alias=tmp_alias, recreate_if_exists=True)
                # TODO: check for infinite recursion
                tmp_db.import_database(location, source_format=source_format)
                obj = {}
                for coll in tmp_db.list_collections():
                    qr = coll.find({}, limit=-1)
                    obj[coll.alias] = qr.rows
                self.store(obj)
                return
        objects = load_objects(location, format=source_format)
        if collection_name:
            collection = self.get_collection(collection_name, create_if_not_exists=True)
            collection.insert(objects)
        else:
            for obj in objects:
                self.store(obj)

    def export_database(self, location: str, target_format: Optional[Union[str, Format]] = None, **kwargs):
        """
        Export a database to a file or location.

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> db.import_database("tests/input/iris.csv", Format.CSV, collection_name="iris")
        >>> db.export_database("/tmp/iris.yaml", Format.YAML)

        :param location: location of the file
        :param target_format: target format
        :param kwargs: additional arguments
        """
        obj = {}
        if isinstance(target_format, str):
            target_format = Format(target_format)
        for coll in self.list_collections():
            qr = coll.find({}, limit=-1)
            obj[coll.alias] = qr.rows
        logger.info(f"Exporting object with {len(obj)} collections to {location} in {target_format} format")
        if isinstance(target_format, Format):
            if target_format.is_dump_format() and target_format in [Format.SQLDUMP_DUCKDB, Format.DUMP_MONGODB]:
                tmp_handle = target_format.value
                client = self.parent
                tmp_db = client.attach_database(tmp_handle, alias="tmp")
                tmp_db.store(obj)
                # TODO: check for infinite recursion
                tmp_db.export_database(location, target_format=target_format)
                return
        if Path(location).is_dir():
            raise ValueError(f"{location} is a directory; cannot write {target_format} to a dir")
        with open(location, "w", encoding="utf-8") as stream:
            stream.write(render_output(obj, format=target_format))

    def broadcast(self, source: Collection, patches: List[PatchDict]):
        if not self.listeners:
            return
        for listener in self.listeners:
            listener(source, patches)
