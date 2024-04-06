from abc import ABC
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, List

from linkml_runtime import SchemaView

from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.metadata import MetaData
from linkml_store.api.collection import Collection


@dataclass
class Database(ABC):
    """
    A database provides access to named collections.

    Modeled after chromadb.

    Attributes:
        handle (Optional[str]): The handle for the database connection.
        _schema_view (Optional[SchemaView]): The schema view associated with the database.
        _collections (Optional[Dict[str, Collection]]): A dictionary of collections in the database.

    Examples:
        >>> from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
        >>> db = DuckDBDatabase("duckdb:///test.db")
        >>> db.handle
        'duckdb:///test.db'
        >>> db._schema_view is None
        True
        >>> db._collections is None
        True
        >>> collection = db.create_collection("Person")
        >>> len(db.list_collections())
        1
        >>> db.get_collection("Person") == collection
        True
    """
    handle: Optional[str] = None
    _schema_view: Optional[SchemaView] = None
    _collections: Optional[Dict[str, Collection]] = None

    def create_collection(self, name: str, metadata: Optional[MetaData] = None, **kwargs) -> Collection:
        """
        Create a new collection.

        Args:
            name (str): The name of the collection.
            metadata (Optional[MetaData]): Metadata associated with the collection.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection: The newly created collection.

        Examples:
            >>> from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
            >>> db = DuckDBDatabase("duckdb:///test.db")
            >>> collection = db.create_collection("Person")
            >>> collection.name
            'Person'
        """
        raise NotImplementedError()

    def list_collections(self) -> Sequence[Collection]:
        """
        List all collections in the database.

        Returns:
            Sequence[Collection]: A sequence of collections.

        Examples:
            >>> from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
            >>> db = DuckDBDatabase("duckdb:///test.db")
            >>> db.create_collection("Person")
            >>> db.create_collection("Product")
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

        Args:
            name (str): The name of the collection.
            create_if_not_exists (bool): Whether to create the collection if it doesn't exist.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection: The requested collection.

        Raises:
            ValueError: If the requested collection doesn't exist and `create_if_not_exists` is False.

        Examples:
            >>> from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
            >>> db = DuckDBDatabase("duckdb:///test.db")
            >>> collection = db.create_collection("Person")
            >>> db.get_collection("Person") == collection
            True
            >>> db.get_collection("NonExistent", create_if_not_exists=False)
            Traceback (most recent call last):
                ...
            ValueError: Collection NonExistent does not exist
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
        """
        Initialize the collections in the database.

        This method should be implemented by subclasses to initialize the collections.
        """
        raise NotImplementedError()

    def query(self, query: Query, **kwargs) -> QueryResult:
        """
        Execute a query on the database.

        Args:
            query (Query): The query to execute.
            **kwargs: Additional keyword arguments.

        Returns:
            QueryResult: The result of the query.

        Examples:
            >>> from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
            >>> from linkml_store.api.queries import Query
            >>> db = DuckDBDatabase("duckdb:///test.db")
            >>> collection = db.create_collection("Person")
            >>> collection.add([{"id": "P1", "name": "John"}, {"id": "P2", "name": "Alice"}])
            >>> query = Query(from_table="Person", where_clause={"name": "John"})
            >>> result = db.query(query)
            >>> len(result.rows)
            1
            >>> result.rows[0]["id"]
            'P1'
        """
        raise NotImplementedError()

    @property
    def schema_view(self) -> SchemaView:
        """
        Return the schema view for the database.

        If no schema view is set, it induces a schema view from the database schema.

        Returns:
            SchemaView: The schema view for the database.

        Examples:
            >>> from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
            >>> db = DuckDBDatabase("duckdb:///test.db")
            >>> db._schema_view is None
            True
            >>> schema_view = db.schema_view
            >>> isinstance(schema_view, SchemaView)
            True
        """
        if not self._schema_view:
            self._schema_view = self.induce_schema_view()
        return self._schema_view

    def set_schema_view(self, schema_view: SchemaView):
        """
        Set the schema view for the database.

        Args:
            schema_view (SchemaView): The schema view to set.

        Examples:
            >>> from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
            >>> from linkml_runtime import SchemaView
            >>> db = DuckDBDatabase("duckdb:///test.db")
            >>> schema_view = SchemaView()
            >>> db.set_schema_view(schema_view)
            >>> db._schema_view == schema_view
            True
        """
        self._schema_view = schema_view

    def induce_schema_view(self) -> SchemaView:
        """
        Induce a schema view from the database schema.

        This method should be implemented by subclasses to induce a schema view.

        Returns:
            SchemaView: The induced schema view.
        """
        raise NotImplementedError()
