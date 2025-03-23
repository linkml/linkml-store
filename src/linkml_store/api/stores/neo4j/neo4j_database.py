# neo4j_database.py

import logging
from pathlib import Path
from typing import Optional, Union

from neo4j import Driver, GraphDatabase, Session

from linkml_store.api import Database
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.neo4j.neo4j_collection import Neo4jCollection
from linkml_store.utils.format_utils import Format

logger = logging.getLogger(__name__)


class Neo4jDatabase(Database):
    """
    An adapter for Neo4j databases.
    """

    _driver: Driver = None
    collection_class = Neo4jCollection

    def __init__(self, handle: Optional[str] = None, **kwargs):
        # Note: in the community editing the database must be "neo4j"
        if handle is None:
            handle = "bolt://localhost:7687/neo4j"
        if handle.startswith("neo4j:"):
            handle = handle.replace("neo4j:", "bolt:", 1)
        super().__init__(handle=handle, **kwargs)

    @property
    def _db_name(self) -> str:
        if self.handle:
            db = self.handle.split("/")[-1]
        else:
            db = "default"
        return db

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            uri, user, password = self._parse_handle()
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        return self._driver

    def session(self) -> Session:
        return self.driver.session(database=self._db_name)

    def _parse_handle(self):
        parts = self.handle.split("://")
        protocol = parts[0]
        rest = parts[1]

        if "@" in rest:
            auth, host = rest.split("@")
            user, password = auth.split(":")
        else:
            host = rest
            user, password = "neo4j", "password"  # Default credentials

        uri = f"{protocol}://{host}"
        return uri, user, password

    def commit(self, **kwargs):
        # Neo4j uses auto-commit by default for each transaction
        pass

    def close(self, **kwargs):
        if self._driver:
            self._driver.close()

    def drop(self, **kwargs):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def query(self, query: Query, **kwargs) -> QueryResult:
        if query.from_table:
            collection = self.get_collection(query.from_table)
            return collection.query(query, **kwargs)
        else:
            raise NotImplementedError(f"Querying without a table is not supported in {self.__class__.__name__}")

    def init_collections(self):
        if self._collections is None:
            self._collections = {}

        # In Neo4j, we don't have a direct equivalent to collections
        # We'll use node labels as a proxy for collections
        with self.driver.session() as session:
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]

            for label in labels:
                if label not in self._collections:
                    collection = Neo4jCollection(name=label, parent=self)
                    self._collections[label] = collection

    def export_database(self, location: str, target_format: Optional[Union[str, Format]] = None, **kwargs):
        # Neo4j doesn't have a built-in export function, so we'll implement a basic JSON export
        if target_format == Format.JSON or target_format == "json":
            path = Path(location)
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN n")
                nodes = [dict(record["n"].items()) for record in result]

                result = session.run("MATCH ()-[r]->() RETURN r")
                relationships = [
                    {
                        "type": record["r"].type,
                        "start": record["r"].start_node.id,
                        "end": record["r"].end_node.id,
                        **dict(record["r"].items()),
                    }
                    for record in result
                ]

                data = {"nodes": nodes, "relationships": relationships}

                import json

                with open(path, "w") as f:
                    json.dump(data, f)
        else:
            super().export_database(location, target_format=target_format, **kwargs)

    def import_database(self, location: str, source_format: Optional[str] = None, **kwargs):
        if source_format == Format.JSON or source_format == "json":
            path = Path(location)
            with open(path, "r") as f:
                import json

                data = json.load(f)

            with self.driver.session() as session:
                for node in data["nodes"]:
                    labels = node.pop("labels", ["Node"])
                    props = ", ".join([f"{k}: ${k}" for k in node.keys()])
                    query = f"CREATE (n:{':'.join(labels)} {{{props}}})"
                    session.run(query, **node)

                for rel in data["relationships"]:
                    # rel_type = rel.pop("type")
                    start = rel.pop("start")
                    end = rel.pop("end")
                    # props = ", ".join([f"{k}: ${k}" for k in rel.keys()])
                    query = (
                        f"MATCH (a), (b) WHERE id(a) = {start} AND id(b) = {end} "
                        "CREATE (a)-[r:{rel_type} {{{props}}}]->(b)"
                    )
                    session.run(query, **rel)
        else:
            super().import_database(location, source_format=source_format, **kwargs)
