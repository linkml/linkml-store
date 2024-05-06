import logging
from typing import Optional

import requests

from linkml_store.api import Collection, Database
from linkml_store.api.config import CollectionConfig
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.solr.solr_collection import SolrCollection

logger = logging.getLogger(__name__)


class SolrDatabase(Database):
    base_url: str
    collection_class = SolrCollection
    use_cores: bool = False

    def __init__(self, handle: Optional[str] = None, **kwargs):
        if handle.startswith("solr:"):
            self.base_url = handle.replace("solr:", "")
        else:
            self.base_url = handle
        super().__init__(handle=handle, **kwargs)

    def get_collection(self, name: str, create_if_not_exists=True, **kwargs) -> "Collection":
        if not self._collections:
            self.init_collections()

        if name not in self._collections.keys():
            if create_if_not_exists:
                self._collections[name] = self.create_collection(name)
            else:
                raise KeyError(f"Collection {name} does not exist")

        return self._collections[name]

    def create_collection(
        self, name: str, alias: Optional[str] = None, metadata: Optional[CollectionConfig] = None, **kwargs
    ) -> Collection:
        if not name:
            raise ValueError(f"Collection name must be provided: alias: {alias} metadata: {metadata}")

        collection_cls = self.collection_class
        collection = collection_cls(name=name, alias=alias, parent=self, metadata=metadata)

        if not self._collections:
            self._collections = {}

        if not alias:
            alias = name

        self._collections[alias] = collection
        return collection

    def init_collections(self):
        if self._collections is None:
            self._collections = {}
        if self.metadata.collection_type_slot:
            response = requests.get(
                f"{self.base_url}/select",
                params={
                    "q": "*:*",
                    "wt": "json",
                    "rows": 0,
                    "facet": "true",
                    "facet.field": self.metadata.collection_type_slot,
                    "facet.limit": -1,
                },
            )
            response.raise_for_status()
            data = response.json()
            coll_names = data["facet_counts"]["facet_fields"][self.metadata.collection_type_slot]
            coll_names = coll_names[::2]
            for coll_name in coll_names:
                self.create_collection(coll_name)
        else:
            self.create_collection("default")

    def query(self, query: Query, **kwargs) -> QueryResult:
        collection_name = query.from_table
        collection = self.get_collection(collection_name)
        return collection.query(query, **kwargs)
