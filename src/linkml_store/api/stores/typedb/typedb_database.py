"""A linkml-store adapter for TypeDB 3.x, targeting schemas that ``gen-typedb`` generated."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional
from urllib.parse import parse_qs, unquote

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.typedb.typedb_collection import TypeDBCollection

if TYPE_CHECKING:  # pragma: no cover
    from typedb.driver import Driver, Transaction

logger = logging.getLogger(__name__)

DEFAULT_HANDLE = "typedb://localhost:1729/linkml"
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "password"
DEFAULT_PORT = 1729
DEFAULT_DATABASE = "linkml"


@dataclass(frozen=True)
class TypeDBHandle:
    """The parts of a parsed ``typedb://`` handle."""

    address: str
    user: str
    password: str
    database: str
    tls: bool
    connection_name: Optional[str] = None

    def __repr__(self) -> str:
        """Redact the password."""
        return (
            f"TypeDBHandle(address={self.address!r}, user={self.user!r}, "
            f"password='***', database={self.database!r}, tls={self.tls!r}, "
            f"connection_name={self.connection_name!r})"
        )


class TypeDBDatabase(Database):
    """An adapter for TypeDB 3.x databases."""

    _driver: Optional["Driver"] = None
    collection_class = TypeDBCollection

    #: the open :meth:`batch_write` transaction
    _active_tx: Optional["Transaction"] = None

    def __init__(self, handle: Optional[str] = None, **kwargs):
        if handle is None:
            handle = DEFAULT_HANDLE
        if not handle.startswith("typedb://"):
            handle = f"typedb://{handle}"
        super().__init__(handle=handle, **kwargs)

    def _parse_handle(self) -> "TypeDBHandle":
        """Parse a Studio-style handle: ``typedb://user:pass@[scheme://]host:port[/db][?name=..&tls=..]``."""
        rest = self.handle
        for prefix in ("typedb://", "typedb+cloud://", "typedb:"):
            if rest.startswith(prefix):
                rest = rest[len(prefix) :].lstrip("/")
                break

        rest, _, query_text = rest.partition("?")
        query = parse_qs(query_text)

        # Split on the last '@': passwords may contain '@', ':' or '/'.
        userinfo, sep, hostpath = rest.rpartition("@")
        if not sep:
            userinfo, hostpath = "", rest
        user, _, password = userinfo.partition(":")
        user = unquote(user) if user else DEFAULT_USER
        password = unquote(password) if password else DEFAULT_PASSWORD

        # Cloud addresses include an http(s) scheme, which decides TLS.
        scheme_tls = None
        for scheme, enabled in (("https://", True), ("http://", False)):
            if hostpath.startswith(scheme):
                hostpath = hostpath[len(scheme) :]
                scheme_tls = enabled
                break

        if "," in hostpath.partition("/")[0]:
            raise NotImplementedError(
                "Multi-node TypeDB addresses (comma-separated, optionally as "
                "'external;internal' pairs) are not supported by this adapter yet; "
                "pass a single address."
            )

        host_port, _, path = hostpath.partition("/")
        host, _, port_text = host_port.rpartition(":")
        if not host:
            host, port_text = host_port, ""
        host = host or "localhost"
        port = int(port_text) if port_text.isdigit() else DEFAULT_PORT

        connection_name = query.get("name", [""])[0] or None
        db_name = path.strip("/") or DEFAULT_DATABASE

        is_loopback = host in ("localhost", "127.0.0.1", "::1")
        tls = scheme_tls if scheme_tls is not None else not is_loopback
        if "tls" in query:
            tls = query["tls"][0].lower() not in ("false", "0", "no")

        return TypeDBHandle(
            address=f"{host}:{port}",
            user=user,
            password=password,
            database=db_name,
            tls=tls,
            connection_name=connection_name,
        )

    @property
    def _db_name(self) -> str:
        return self._parse_handle().database

    @property
    def driver(self) -> "Driver":
        """Return the TypeDB driver, connecting on first use."""
        if self._driver is None:
            try:
                from typedb.driver import Credentials, DriverOptions, DriverTlsConfig, TypeDB
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "The TypeDB adapter requires the typedb-driver package. "
                    "Install it with: pip install 'linkml-store[typedb]'"
                ) from e
            parsed = self._parse_handle()
            tls = DriverTlsConfig.enabled_with_native_root_ca() if parsed.tls else DriverTlsConfig.disabled()
            logger.info(
                "Connecting to TypeDB at %s as %s (tls=%s, database=%s)",
                parsed.address,
                parsed.user,
                parsed.tls,
                parsed.database,
            )
            self._driver = TypeDB.driver(parsed.address, Credentials(parsed.user, parsed.password), DriverOptions(tls))
        return self._driver

    def transaction(self, transaction_type=None):
        """Open a transaction (READ by default) on the configured database."""
        from typedb.driver import TransactionType

        if transaction_type is None:
            transaction_type = TransactionType.READ
        return self.driver.transaction(self._db_name, transaction_type)

    @contextmanager
    def batch_write(self) -> Iterator["Transaction"]:
        """Route every collection write and read through one transaction, committed on exit.

        TypeDB checks ``@card`` at commit, so cyclic data has to load in one transaction::

            with db.batch_write():
                for cls in order:
                    db.get_collection(cls).insert(objs[cls], mode=MODE_ENTITIES)
                for cls in order:
                    db.get_collection(cls).insert(objs[cls], mode=MODE_REFERENCES)
        """
        from typedb.driver import TransactionType

        if self._active_tx is not None:
            raise RuntimeError("A batch_write is already open on this database.")
        tx = self.driver.transaction(self._db_name, TransactionType.WRITE)
        self._active_tx = tx
        try:
            yield tx
        except BaseException:
            tx.close()
            raise
        else:
            tx.commit()
        finally:
            self._active_tx = None

    def commit(self, **kwargs):
        """No-op: each write commits its own transaction."""

    def close(self, **kwargs):
        """Close the driver connection, if one was opened."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def drop(self, missing_ok=True, **kwargs):
        """Delete the TypeDB database named in the handle."""
        databases = self.driver.databases
        if databases.contains(self._db_name):
            databases.get(self._db_name).delete()
        elif not missing_ok:
            raise ValueError(f"No such TypeDB database: {self._db_name}")

    def query(self, query: Query, **kwargs) -> QueryResult:
        """Delegate a query to the collection named by ``from_table``."""
        if query.from_table:
            return self.get_collection(query.from_table).query(query, **kwargs)
        raise NotImplementedError(f"Querying without a table is not supported in {type(self).__name__}")

    def init_collections(self):
        """Create one collection per class in the LinkML schema."""
        if self._collections is None:
            self._collections = {}
        for class_name in self.schema_view.all_classes():
            if class_name not in self._collections:
                self._collections[class_name] = TypeDBCollection(name=class_name, parent=self)
