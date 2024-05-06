"""
ChromaDB Collection
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from chromadb.api.models.Collection import Collection as ChromaCollection
from linkml_runtime.linkml_model import SlotDefinition

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.index import Indexer

logger = logging.getLogger(__name__)


class ChromaDBCollection(Collection):
    """
    A wrapper for ChromaDB collections.
    """

    @property
    def native_collection(self) -> ChromaCollection:
        return self.parent.client.get_collection(self.name)

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]

        documents = []
        metadatas = []
        ids = []
        indexer = Indexer()

        for obj in objs:
            obj_id = self.object_identifier(obj)
            ids.append(obj_id)
            doc_text = indexer.object_to_text(obj)
            documents.append(doc_text)
            # TODO: handle nesting
            metadata = {k: v for k, v in obj.items()}
            metadatas.append(metadata)

        self.native_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        if not isinstance(objs, list):
            objs = [objs]
        ids = [obj["id"] for obj in objs]
        self.native_collection.delete(ids=ids)
        return len(ids)

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> int:
        logger.info(f"Deleting from {self.target_class_name} where: {where}")
        if where is None:
            where = {}
        results = self.native_collection.get(where=where)
        ids = [result["id"] for result in results]
        self.native_collection.delete(ids=ids)
        return len(ids)

    def query(self, query: Query, **kwargs) -> QueryResult:
        chroma_filter = self._build_chroma_filter(query.where_clause)
        if query.limit:
            results = self.native_collection.get(where=chroma_filter, limit=query.limit)
        else:
            results = self.native_collection.get(where=chroma_filter)

        count = len(results)
        return QueryResult(query=query, num_rows=count, rows=results)

    def query_facets(
        self, where: Dict = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[str, List[Tuple[Any, int]]]:
        results = {}
        cd = self.class_definition()
        if not facet_columns:
            facet_columns = list(self.class_definition().attributes.keys())

        for col in facet_columns:
            logger.debug(f"Faceting on {col}")
            if isinstance(col, tuple):
                sd = SlotDefinition(name="PLACEHOLDER")
            else:
                sd = cd.attributes[col]

            if sd.multivalued:
                facet_results = self.native_collection.aggregate(
                    aggregation=[
                        {"$match": where} if where else {"$match": {}},
                        {"$unwind": f"${col}"},
                        {"$group": {"_id": f"${col}", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": facet_limit},
                    ]
                )
            else:
                facet_results = self.native_collection.aggregate(
                    aggregation=[
                        {"$match": where} if where else {"$match": {}},
                        {"$group": {"_id": f"${col}", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": facet_limit},
                    ]
                )

            results[col] = [(result["_id"], result["count"]) for result in facet_results]

        return results

    def _build_chroma_filter(self, where_clause: Dict[str, Any]) -> Dict[str, Any]:
        chroma_filter = {}
        for field, value in where_clause.items():
            chroma_filter[field] = value
        return chroma_filter
