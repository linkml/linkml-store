import logging
from typing import Optional
from urllib.parse import parse_qs, urlparse

from google.cloud import bigtable
from google.cloud.bigtable.instance import Instance

from linkml_store.api import Database
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

    >>> db = BigtableDatabase("bigtable://my-project/my-instance?column_family=cf")
    >>> db.project_id, db.instance_id, db.column_family
    ('my-project', 'my-instance', 'cf')
    """

    _native_client: Optional[bigtable.Client] = None
    _native_instance: Optional[Instance] = None
    collection_class = BigtableCollection

    def __init__(self, handle: Optional[str] = None, **kwargs):
        if handle is None or handle == "bigtable":
            handle = DEFAULT_HANDLE
        super().__init__(handle=handle, **kwargs)
        parsed = urlparse(self.handle)
        if parsed.scheme != "bigtable":
            raise ValueError(f"Invalid Bigtable handle: {self.handle}")
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) != 1:
            raise ValueError(f"Bigtable handle must have the form bigtable://<project>/<instance>; got {self.handle}")
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
        for table in self.native_instance.list_tables():
            logger.info(f"Deleting Bigtable table {table.table_id}")
            table.delete()
        for collection in (self._collections or {}).values():
            collection._reset_table_state()
        self._collections = {}

    def query(self, query: Query, **kwargs) -> QueryResult:
        if query.from_table:
            collection = self.get_collection(query.from_table)
            return collection.query(query, **kwargs)
        raise NotImplementedError(f"Querying without a table is not supported in {self.__class__.__name__}")

    def init_collections(self):
        if self._collections is None:
            self._collections = {}
        for table in self.native_instance.list_tables():
            if table.table_id not in self._collections:
                self._collections[table.table_id] = BigtableCollection(name=table.table_id, parent=self)
