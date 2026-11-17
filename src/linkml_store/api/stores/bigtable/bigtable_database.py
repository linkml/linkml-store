import logging
from typing import Optional, Sequence
from urllib.parse import parse_qs, urlparse

from google.cloud import bigtable
from google.cloud.bigtable.instance import Instance

from linkml_store.api import Collection, Database
from linkml_store.api.config import DatabaseConfig
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.bigtable.bigtable_collection import BigtableCollection

logger = logging.getLogger(__name__)

DEFAULT_HANDLE = "bigtable://test-project/test-instance"
DEFAULT_COLUMN_FAMILY = "d"


class BigtableDatabase(Database):
    """
    An adapter for Google Cloud Bigtable.

    A LinkML-Store Database corresponds to a Bigtable instance; each collection is a table.

    Handles have the form ``bigtable://<project>/<instance>``. Optional query parameters:

    - ``column_family``: name of the column family used to store object fields (default ``d``)
    - ``app_profile_id``: Bigtable app profile to use for data operations

    If the project is omitted (``bigtable:///<instance>``), it is taken from the environment
    (e.g. ``GOOGLE_CLOUD_PROJECT`` or application default credentials).

    To connect to the emulator, set ``BIGTABLE_EMULATOR_HOST`` in the environment.

    With ``recreate_if_exists=True``, every table in the instance is deleted on first connection.

    Existing tables are discovered as collections whose class is the table name; pass ``type``
    to :meth:`get_collection` to map a discovered table to a class.

    >>> db = BigtableDatabase("bigtable://my-project/my-instance?column_family=cf")
    >>> db.project_id, db.instance_id, db.column_family
    ('my-project', 'my-instance', 'cf')
    """

    _native_client: Optional[bigtable.Client] = None
    _native_instance: Optional[Instance] = None
    _tables_discovered: bool = False
    _recreated: bool = False
    collection_class = BigtableCollection

    def __init__(self, handle: Optional[str] = None, metadata: Optional[DatabaseConfig] = None, **kwargs):
        if handle is None and metadata is not None:
            handle = metadata.handle
        if handle is None or handle == "bigtable":
            handle = DEFAULT_HANDLE
        if metadata is not None:
            metadata.handle = handle
        # parse connection settings first: the base constructor may already create collections
        self._parse_handle(handle)
        super().__init__(handle=handle, metadata=metadata, **kwargs)

    def _parse_handle(self, handle: str):
        parsed = urlparse(handle)
        if parsed.scheme != "bigtable":
            raise ValueError(f"Invalid Bigtable handle: {handle}")
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) != 1:
            raise ValueError(f"Bigtable handle must have the form bigtable://<project>/<instance>; got {handle}")
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self.project_id: Optional[str] = parsed.netloc or None
        self.instance_id: str = path_parts[0]
        self.column_family: str = params.get("column_family", DEFAULT_COLUMN_FAMILY)
        self.app_profile_id: Optional[str] = params.get("app_profile_id")

    @property
    def native_client(self) -> bigtable.Client:
        if self._native_client is None:
            self._native_client = bigtable.Client(project=self.project_id, admin=True)
        return self._native_client

    @property
    def native_instance(self) -> Instance:
        if self._native_instance is None:
            self._native_instance = self.native_client.instance(self.instance_id)
            if self.recreate_if_exists and not self._recreated:
                self._recreated = True
                logger.info(f"recreate_if_exists: deleting all tables in Bigtable instance {self.instance_id}")
                # keep collections registered so far (e.g. from config); only their tables go
                self._delete_all_tables()
        return self._native_instance

    def commit(self, **kwargs):
        pass

    def close(self, **kwargs):
        if self._native_client is not None:
            self._native_client.close()
            self._native_client = None
            self._native_instance = None

    def drop(self, missing_ok=True, **kwargs):
        """
        Delete all tables in the instance.

        The instance itself is not deleted.
        """
        self._delete_all_tables()
        self._collections = {}
        self._tables_discovered = False

    def _delete_all_tables(self):
        for table in self.native_instance.list_tables():
            logger.info(f"Deleting Bigtable table {table.table_id}")
            table.delete()
        for collection in (self._collections or {}).values():
            collection._reset_table_state()

    def query(self, query: Query, **kwargs) -> QueryResult:
        if query.from_table:
            collection = self.get_collection(query.from_table)
            return collection.query(query, **kwargs)
        raise NotImplementedError(f"Querying without a table is not supported in {self.__class__.__name__}")

    def init_collections(self):
        """Register a collection for each table in the instance that is not already known."""
        if self._collections is None:
            self._collections = {}
        if self._tables_discovered:
            return
        for table in self.native_instance.list_tables():
            if table.table_id not in self._collections:
                collection = BigtableCollection(name=table.table_id, parent=self)
                collection._discovered = True
                self._collections[table.table_id] = collection
        self._tables_discovered = True

    def list_collections(self, include_internal=False) -> Sequence[Collection]:
        # the base class only discovers tables when no collections are registered yet
        self.init_collections()
        return super().list_collections(include_internal=include_internal)

    def get_collection(self, name: str, type: Optional[str] = None, create_if_not_exists=True, **kwargs) -> Collection:
        self.init_collections()
        collection = self._collections.get(name)
        if collection is not None and type and getattr(collection, "_discovered", False):
            # a discovered table defaults to a class named after the table; apply the requested class
            collection.metadata.type = type
        return super().get_collection(name, type=type, create_if_not_exists=create_if_not_exists, **kwargs)
