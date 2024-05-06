from pathlib import Path
from typing import Dict, Optional, Union

import yaml
from linkml_runtime import SchemaView

from linkml_store.api import Database
from linkml_store.api.config import ClientConfig
from linkml_store.api.stores.chromadb.chromadb_database import ChromaDBDatabase
from linkml_store.api.stores.duckdb.duckdb_database import DuckDBDatabase
from linkml_store.api.stores.mongodb.mongodb_database import MongoDBDatabase
from linkml_store.api.stores.solr.solr_database import SolrDatabase

HANDLE_MAP = {
    "duckdb": DuckDBDatabase,
    "solr": SolrDatabase,
    "mongodb": MongoDBDatabase,
    "chromadb": ChromaDBDatabase,
}


class Client:
    """
    A client is the top-level object for interacting with databases.

    A client has access to one or more :class:`Database` objects.

    Each database consists of a number of :class:`.Collection` objects.

    Examples
    --------
    >>> client = Client()
    >>> db = client.attach_database("duckdb", alias="test")
    >>> collection = db.create_collection("Person")
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

    metadata: Optional[ClientConfig] = None
    _databases: Optional[Dict[str, Database]] = None

    def __init__(self, handle: Optional[str] = None, metadata: Optional[ClientConfig] = None):
        """
        Initialize a client.

        :param handle:
        :param metadata:
        """
        self.metadata = metadata
        if not self.metadata:
            self.metadata = ClientConfig()
        self.metadata.handle = handle

    @property
    def handle(self) -> Optional[str]:
        return self.metadata.handle

    @property
    def base_dir(self) -> Optional[str]:
        """
        Get the base directory for the client.

        Wraps metadata.base_dir.

        :return:
        """
        return self.metadata.base_dir

    def from_config(self, config: Union[ClientConfig, str, Path], base_dir=None, **kwargs):
        """
        Create a client from a configuration.

        Examples
        --------
        >>> from linkml_store.api.config import ClientConfig
        >>> client = Client().from_config(ClientConfig(databases={"test": {"handle": "duckdb:///:memory:"}}))
        >>> len(client.databases)
        1
        >>> "test" in client.databases
        True
        >>> client.databases["test"].handle
        'duckdb:///:memory:'

        :param config:
        :param kwargs:
        :return:

        """
        if isinstance(config, Path):
            config = str(config)
        if isinstance(config, str):
            if not base_dir:
                base_dir = Path(config).parent
            parsed_obj = yaml.safe_load(open(config))
            config = ClientConfig(**parsed_obj)
        self.metadata = config
        if base_dir:
            self.metadata.base_dir = base_dir
        self._initialize_databases(**kwargs)
        return self

    def _initialize_databases(self, **kwargs):
        for name, db_config in self.metadata.databases.items():
            handle = db_config.handle.format(base_dir=self.base_dir)
            db_config.handle = handle
            db = self.attach_database(handle, alias=name, **kwargs)
            db.from_config(db_config)

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
        db.parent = self
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

        :param name: if None, there must be a single database attached
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

    def drop_database(self, name: str, missing_ok=False, **kwargs):
        """
        Drop a database.

        :param name:
        :param missing_ok:
        :return:
        """
        if name in self._databases:
            db = self._databases[name]
            db.drop(**kwargs)
            del self._databases[name]
        else:
            if not missing_ok:
                raise ValueError(f"Database {name} not found")

    def drop_all_databases(self, **kwargs):
        """
        Drop all databases.

        :param missing_ok:
        :return:
        """
        for name in list(self._databases.keys()):
            self.drop_database(name, missing_ok=False, **kwargs)
        self._databases = {}
