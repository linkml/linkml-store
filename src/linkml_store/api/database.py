from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from linkml_runtime import SchemaView

from linkml_store.api.collection import Collection
from linkml_store.api.metadata import MetaData
from linkml_store.api.queries import Query, QueryResult


@dataclass
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
    >>> collection.add(objs)
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

    handle: Optional[str] = None
    _schema_view: Optional[SchemaView] = None
    _collections: Optional[Dict[str, Collection]] = None

    def create_collection(self, name: str, metadata: Optional[MetaData] = None, **kwargs) -> Collection:
        """
        Create a new collection

        >>> from linkml_store.api.client import Client
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> collection.name
        'Person'

        :param name: name of the collection
        """
        raise NotImplementedError()

    def list_collections(self) -> Sequence[Collection]:
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

        """
        if not self._collections:
            self.init_collections()
        return list(self._collections.values())

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
        ValueError: Collection NonExistent does not exist

        :param name: name of the collection
        :param create_if_not_exists: create the collection if it does not exist

        """
        if not self._collections:
            self.init_collections()
        if name not in self._collections:
            if create_if_not_exists:
                self._collections[name] = self.create_collection(name)
            else:
                raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def init_collections(self):
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
        >>> collection.add([{"id": "P1", "name": "John"}, {"id": "P2", "name": "Alice"}])
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
            self._schema_view = self.induce_schema_view()
        return self._schema_view

    def set_schema_view(self, schema_view: SchemaView):
        self._schema_view = schema_view

    def induce_schema_view(self) -> SchemaView:
        """
        Induce a schema view from a schema definition.

        >>> from linkml_store.api.client import Client
        >>> from linkml_store.api.queries import Query
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="test")
        >>> collection = db.create_collection("Person")
        >>> collection.add([{"id": "P1", "name": "John", "age_in_years": 25}, {"id": "P2", "name": "Alice", "age_in_years": 25}])
        >>> schema_view = db.induce_schema_view()
        >>> cd = schema_view.get_class("Person")
        >>> cd.attributes["id"].range
        'string'

        :return: A schema view
        """
        raise NotImplementedError()
