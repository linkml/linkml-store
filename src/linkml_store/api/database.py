import logging
from abc import ABC
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, Iterator, Optional, Sequence, Type, Union

try:
    from linkml.validator.report import ValidationResult
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


class Database(ABC):
    """
    A Database provides access to named collections of data.

    Examples
    --------
    >>> from linkml_store.api.client import Client
    >>> client = Client()
    >>> db = client.attach_database("duckdb", alias="test")
    >>> db.handle
    'duckdb:///:memory:'
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
    _collections: Optional[Dict[str, Collection]] = None
    parent: Optional["Client"] = None
    metadata: Optional[DatabaseConfig] = None
    collection_class: ClassVar[Optional[Type[Collection]]] = None

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
        for name, collection_config in self.metadata.collections.items():
            alias = collection_config.alias
            typ = collection_config.type
            # if typ and alias is None:
            #    alias = name
            # if typ is None:
            #    typ = name
            # collection = self.create_collection(
            #    typ, alias=alias, metadata=collection_config.metadata
            # )
            if False and typ is not None:
                if not alias:
                    alias = name
                name = typ
            if not collection_config.name:
                collection_config.name = name
            _collection = self.create_collection(name, alias=alias, metadata=collection_config)
            if collection_config.attributes:
                sv = self.schema_view
                cd = ClassDefinition(name, attributes=collection_config.attributes)
                sv.schema.classes[cd.name] = cd
                sv.set_modified()
                # assert collection.class_definition() is not None

    @property
    def recreate_if_exists(self) -> bool:
        return self.metadata.recreate_if_exists

    @property
    def handle(self) -> str:
        return self.metadata.handle

    def store(self, obj: Dict[str, str], **kwargs):
        """
        Store an object in the database

        :param obj: object to store
        :param kwargs: additional arguments
        """
        sv = self.schema_view
        roots = [c for c in sv.all_classes().values() if c.tree_root]
        root = roots[0] if roots else None
        for k, v in obj.items():
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
                collection = self.get_collection(slot.range, create_if_not_exists=True)
            else:
                collection = self.get_collection(k, create_if_not_exists=True)
            collection.replace(v)

    def commit(self, **kwargs):
        """
        Commit any pending changes to the database
        """
        raise NotImplementedError()

    def close(self, **kwargs):
        """
        Close the database and all connection objects
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
        Create a new collection

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> collection.name
        'Person'

        :param name: name of the collection
        :param alias: alias for the collection
        :param metadata: metadata for the collection
        :param recreate_if_exists: recreate the collection if it already exists
        :param kwargs: additional arguments
        """
        if not name:
            raise ValueError(f"Collection name must be provided: alias: {alias} metadata: {metadata}")
        # collection_cls = self._collection_class
        collection_cls = self.collection_class
        collection = collection_cls(name=name, alias=alias, parent=self, metadata=metadata)
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
        >>> [c.name for c in collections]
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
        return [c.name for c in self.list_collections(**kwargs)]

    def get_collection(self, name: str, create_if_not_exists=True, **kwargs) -> "Collection":
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
        :param create_if_not_exists: create the collection if it does not exist

        """
        if not self._collections:
            self.init_collections()
        if name not in self._collections.keys():
            if create_if_not_exists:
                self._collections[name] = self.create_collection(name)
            else:
                raise KeyError(f"Collection {name} does not exist")
        return self._collections[name]

    def init_collections(self):
        """
        Initialize collections.

        Not typically called directly: consider making hidden
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
        >>> db = client.attach_database("duckdb", alias="test")
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
        raise NotImplementedError

    @property
    def schema_view(self) -> SchemaView:
        """
        Return a schema view for the named collection
        """
        if not self._schema_view:
            self._initialize_schema()
        if not self._schema_view:
            self._schema_view = self.induce_schema_view()
        return self._schema_view

    def set_schema_view(self, schema_view: SchemaView):
        """
        Set the schema view for the database.

        :param schema_view:
        :return:
        """
        self._schema_view = schema_view

    def load_schema_view(self, path: Union[str, Path]):
        """
        Load a schema view from a file.

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> db.load_schema_view("tests/input/countries/countries.linkml.yaml")

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
        raise NotImplementedError()

    def iter_validate_database(self, **kwargs) -> Iterator["ValidationResult"]:
        """
        Validate the contents of the database.

        :param kwargs:
        :return: iterator over validation results
        """
        for collection in self.list_collections():
            yield from collection.iter_validate_collection(**kwargs)

    def drop(self, **kwargs):
        """
        Drop the database and all collections
        """
        raise NotImplementedError()
