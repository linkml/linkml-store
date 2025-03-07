import importlib
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import yaml
from linkml_runtime import SchemaView

from linkml_store.api import Database
from linkml_store.api.config import ClientConfig

logger = logging.getLogger(__name__)


HANDLE_MAP = {
    "duckdb": "linkml_store.api.stores.duckdb.duckdb_database.DuckDBDatabase",
    "sqlite": "linkml_store.api.stores.duckdb.duckdb_database.DuckDBDatabase",
    "solr": "linkml_store.api.stores.solr.solr_database.SolrDatabase",
    "mongodb": "linkml_store.api.stores.mongodb.mongodb_database.MongoDBDatabase",
    "chromadb": "linkml_store.api.stores.chromadb.chromadb_database.ChromaDBDatabase",
    "neo4j": "linkml_store.api.stores.neo4j.neo4j_database.Neo4jDatabase",
    "file": "linkml_store.api.stores.filesystem.filesystem_database.FileSystemDatabase",
}

SUFFIX_MAP = {
    "ddb": "duckdb:///{path}",
    "duckdb": "duckdb:///{path}",
    "db": "duckdb:///{path}",
}


class Client:
    """
    A client is the top-level object for interacting with databases.

    * A client has access to one or more :class:`.Database` objects.
    * Each database consists of a number of :class:`.Collection` objects.

    Creating a client
    -----------------
    >>> client = Client()

    Attaching a database
    --------------------
    >>> db = client.attach_database("duckdb", alias="test")

    Note that normally a handle would be specified by a locator such as ``duckdb:///<PATH>``, but
    for convenience, an in-memory duckdb object can be specified without a full locator

    We can check the actual handle:

    >>> db.handle
    'duckdb:///:memory:'

    Creating a new collection
    -------------------------
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

    def from_config(self, config: Union[ClientConfig, dict, str, Path], base_dir=None, auto_attach=False, **kwargs):
        """
        Create a client from a configuration.

        Examples
        --------
        >>> from linkml_store.api.config import ClientConfig
        >>> client = Client().from_config(ClientConfig(databases={"test": {"handle": "duckdb:///:memory:"}}))
        >>> len(client.databases)
        0
        >>> client = Client().from_config(ClientConfig(databases={"test": {"handle": "duckdb:///:memory:"}}),
        ...                                auto_attach=True)
        >>> len(client.databases)
        1
        >>> "test" in client.databases
        True
        >>> client.databases["test"].handle
        'duckdb:///:memory:'

        :param config:
        :param base_dir:
        :param auto_attach:
        :param kwargs:
        :return:

        """
        if isinstance(config, dict):
            config = ClientConfig(**config)
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
        self._initialize_databases(auto_attach=auto_attach, **kwargs)
        return self

    def _initialize_databases(self, auto_attach=False, **kwargs):
        for name, db_config in self.metadata.databases.items():
            base_dir = self.base_dir
            logger.info(f"Initializing database: {name}, base_dir: {base_dir}")
            if not base_dir:
                base_dir = Path.cwd()
                logger.info(f"Using current working directory: {base_dir}")
            handle = db_config.handle.format(base_dir=base_dir)
            db_config.handle = handle
            if db_config.schema_location:
                db_config.schema_location = db_config.schema_location.format(base_dir=base_dir)
            if auto_attach:
                db = self.attach_database(handle, alias=name, **kwargs)
                db.from_config(db_config)
            if db_config.source:
                db = self.get_database(name)
                db.store(db_config.source.data)

    def _set_database_config(self, db: Database):
        """
        Set the configuration for a database.

        :param name:
        :param config:
        :return:
        """
        if not self.metadata:
            return
        if db.alias in self.metadata.databases:
            db.from_config(self.metadata.databases[db.alias])

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
            if alias is None:
                alias = handle
            if "." in handle:
                suffix = handle.split(".")[-1]
                if suffix in SUFFIX_MAP:
                    handle = SUFFIX_MAP[suffix].format(path=handle)
        if ":" not in handle:
            scheme = handle
            handle = None
            if alias is None:
                alias = scheme
        else:
            scheme, _ = handle.split(":", 1)
        if scheme not in HANDLE_MAP:
            raise ValueError(f"Unknown scheme: {scheme}")
        module_path, class_name = HANDLE_MAP[scheme].rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"Failed to import {scheme} database. Make sure the correct extras are installed: {e}")

        # cls = HANDLE_MAP[scheme]
        db = cls(handle=handle, recreate_if_exists=recreate_if_exists, **kwargs)
        if schema_view:
            db.set_schema_view(schema_view)
        if not alias:
            alias = handle
        if not self._databases:
            logger.info("Initializing databases")
            self._databases = {}
        logger.info(f"Attaching {alias}")
        self._databases[alias] = db
        db.parent = self
        if db.alias:
            if db.alias != alias:
                raise AssertionError(f"Inconsistent alias: {db.alias} != {alias}")
        else:
            db.metadata.alias = alias
        self._set_database_config(db)
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
        if name not in self._databases and name in self.metadata.databases:
            db_config = self.metadata.databases[name]
            db = self.attach_database(db_config.handle, alias=name, **kwargs)
            self._databases[name] = db
        if name not in self._databases:
            if create_if_not_exists:
                logger.info(f"Creating/attaching database: {name}")
                db = self.attach_database(name, **kwargs)
                name = db.alias
            else:
                raise ValueError(f"Database {name} does not exist")
        db = self._databases[name]
        self._set_database_config(db)
        return db

    @property
    def databases(self) -> Dict[str, Database]:
        """
        Return all attached databases

        Examples

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

        Example (in-memory):

        >>> client = Client()
        >>> db1 = client.attach_database("duckdb", alias="test1")
        >>> db2 = client.attach_database("duckdb", alias="test2")
        >>> len(client.databases)
        2
        >>> client.drop_database("test1")
        >>> len(client.databases)
        1

        Databases that persist on disk:

        >>> client = Client()
        >>> path = Path("tmp/test.db")
        >>> path.parent.mkdir(parents=True, exist_ok=True)
        >>> db = client.attach_database(f"duckdb:///{path}", alias="test")
        >>> len(client.databases)
        1
        >>> db.store({"persons": [{"id": "P1", "name": "John"}]})
        >>> db.commit()
        >>> Path("tmp/test.db").exists()
        True
        >>> client.drop_database("test")
        >>> len(client.databases)
        0
        >>> Path("tmp/test.db").exists()
        False

        Dropping a non-existent database:

        >>> client = Client()
        >>> client.drop_database("duckdb:///tmp/made-up1", missing_ok=True)
        >>> client.drop_database("duckdb:///tmp/made-up2", missing_ok=False)
        Traceback (most recent call last):
        ...
        ValueError: Database duckdb:///tmp/made-up2 not found

        :param name:
        :param missing_ok:
        :return:
        """
        if self._databases:
            if name in self._databases:
                db = self._databases[name]
                db.drop(**kwargs)
                del self._databases[name]
            else:
                if not missing_ok:
                    raise ValueError(f"Database {name} not found")
        else:
            db = self.get_database(name, create_if_not_exists=True)
            db.drop(**kwargs)

    def drop_all_databases(self, **kwargs):
        """
        Drop all databases.

        Example (in-memory):

        >>> client = Client()
        >>> db1 = client.attach_database("duckdb", alias="test1")
        >>> assert "test1" in client.databases
        >>> db2 = client.attach_database("duckdb", alias="test2")
        >>> assert "test2" in client.databases
        >>> client.drop_all_databases()
        >>> len(client.databases)
        0


        :param missing_ok:
        :return:
        """
        if not self._databases:
            return
        for name in list(self._databases.keys()):
            self.drop_database(name, missing_ok=False, **kwargs)
        self._databases = {}
