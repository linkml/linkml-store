import logging
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from linkml_runtime.linkml_model import SlotDefinition
from pymongo.collection import Collection as MongoCollection

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult

logger = logging.getLogger(__name__)


class MongoDBCollection(Collection):
    """
    Adapter for collections in a MongoDB database.

    .. note::

        You should not use or manipulate this class directly.
        Instead, use the general :class:`linkml_store.api.Collection`
    """

    @property
    def mongo_collection(self) -> MongoCollection:
        if not self.name:
            raise ValueError("Collection name not set")
        collection_name = self.alias or self.name
        return self.parent.native_db[collection_name]

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]
        self.mongo_collection.insert_many(objs)
        # TODO: allow mapping of _id to id for efficiency
        for obj in objs:
            del obj["_id"]
        self._post_insert_hook(objs)

    def query(self, query: Query, limit: Optional[int] = None, **kwargs) -> QueryResult:
        mongo_filter = self._build_mongo_filter(query.where_clause)
        limit = limit or query.limit
        if limit and limit >= 0:
            cursor = self.mongo_collection.find(mongo_filter).limit(limit)
        else:
            cursor = self.mongo_collection.find(mongo_filter)

        def _as_row(row: dict):
            row = copy(row)
            del row["_id"]
            return row

        rows = [_as_row(row) for row in cursor]
        count = self.mongo_collection.count_documents(mongo_filter)

        return QueryResult(query=query, num_rows=count, rows=rows)

    def _build_mongo_filter(self, where_clause: Dict[str, Any]) -> Dict[str, Any]:
        mongo_filter = {}
        if where_clause:
            for field, value in where_clause.items():
                mongo_filter[field] = value
        return mongo_filter

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
                if col in cd.attributes:
                    sd = cd.attributes[col]
                else:
                    logger.info(f"No schema metadata for {col}")
                    sd = SlotDefinition(name=col)
            group = {"$group": {"_id": f"${col}", "count": {"$sum": 1}}}
            if isinstance(col, tuple):
                q = {k.replace(".", ""): f"${k}" for k in col}
                group["$group"]["_id"] = q
            if sd and sd.multivalued:
                facet_pipeline = [
                    {"$match": where} if where else {"$match": {}},
                    {"$unwind": f"${col}"},
                    group,
                    {"$sort": {"count": -1}},
                    {"$limit": facet_limit},
                ]
            else:
                facet_pipeline = [
                    {"$match": where} if where else {"$match": {}},
                    group,
                    {"$sort": {"count": -1}},
                    {"$limit": facet_limit},
                ]
            logger.info(f"Facet pipeline: {facet_pipeline}")
            facet_results = list(self.mongo_collection.aggregate(facet_pipeline))
            results[col] = [(result["_id"], result["count"]) for result in facet_results]

        return results

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        if not isinstance(objs, list):
            objs = [objs]
        filter_conditions = []
        for obj in objs:
            filter_condition = {}
            for key, value in obj.items():
                filter_condition[key] = value
            filter_conditions.append(filter_condition)
        result = self.mongo_collection.delete_many({"$or": filter_conditions})
        return result.deleted_count

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> int:
        logger.info(f"Deleting from {self.target_class_name} where: {where}")
        if where is None:
            where = {}
        result = self.mongo_collection.delete_many(where)
        deleted_rows_count = result.deleted_count
        if deleted_rows_count == 0 and not missing_ok:
            raise ValueError(f"No rows found for {where}")
        return deleted_rows_count
