from dataclasses import dataclass
from typing import Dict, Optional

from linkml_runtime import SchemaView

from linkml_store.api import Database
from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase

HANDLE_MAP = {
    "duckdb": DuckDBDatabase,
}


@dataclass
class Client:
    """
    A client provides access to named collections.

    Examples
    --------
    >>> client = Client()
    >>> db = client.attach_database("duckdb", alias="test")
    >>> collection = db.create_collection("Person")
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
    _databases: Optional[Dict[str, Database]] = None

    def attach_database(
        self,
        handle: str,
        alias: Optional[str] = None,
        schema_view: Optional[SchemaView] = None,
        recreate_if_exists=False,
        **kwargs,
    ) -> Database:
        """
        Associate a database with a handle.

        Examples
        --------
        >>> client = Client()
        >>> db = client.attach_database("duckdb", alias="memory")
        >>> "memory" in client.databases
        True
        >>> db = client.attach_database("duckdb:///tmp/another.db", alias="disk")
        >>> len(client.databases)
        2
        >>> "disk" in client.databases
        True

        :param handle: handle for the database, e.g. duckdb:///foo.db
        :param alias: alias for the database, e.g foo
        :param schema_view: schema view to associate with the database
        :param kwargs:
        :return:

        """
        if ":" not in handle:
            scheme = handle
            handle = None
        else:
            scheme, _ = handle.split(":", 1)
        if scheme not in HANDLE_MAP:
            raise ValueError(f"Unknown scheme: {scheme}")
        cls = HANDLE_MAP[scheme]
        db = cls(handle=handle, recreate_if_exists=recreate_if_exists, **kwargs)
        if schema_view:
            db.set_schema_view(schema_view)
        if not alias:
            alias = handle
        if not self._databases:
            self._databases = {}
        self._databases[alias] = db
        return db

    def get_database(self, name: Optional[str] = None, create_if_not_exists=True, **kwargs) -> Database:
        """
        Get a named database.

        Examples
        --------
        >>> client = Client()
        >>> db = client.attach_database("duckdb:///test.db", alias="test")
        >>> retrieved_db = client.get_database("test")
        >>> db == retrieved_db
        True

        :param name:
        :param create_if_not_exists:
        :param kwargs:
        :return:

        """
        if not name:
            if not self._databases:
                raise ValueError("No databases attached and no name provided")
            if len(self._databases) > 1:
                raise ValueError("Ambiguous: No name provided and multiple databases attached")
            return list(self._databases.values())[0]
        if not self._databases:
            self._databases = {}
        if name not in self._databases:
            if create_if_not_exists:
                self.attach_database(name, **kwargs)
            else:
                raise ValueError(f"Database {name} does not exist")
        return self._databases[name]

    @property
    def databases(self) -> Dict[str, Database]:
        """
        Return all attached databases

        Examples
        --------
        >>> client = Client()
        >>> _ = client.attach_database("duckdb", alias="test1")
        >>> _ = client.attach_database("duckdb", alias="test2")
        >>> len(client.databases)
        2
        >>> "test1" in client.databases
        True
        >>> "test2" in client.databases
        True
        >>> client.databases["test1"].handle
        'duckdb:///:memory:'
        >>> client.databases["test2"].handle
        'duckdb:///:memory:'

        :return:

        """
        if not self._databases:
            self._databases = {}
        return self._databases
