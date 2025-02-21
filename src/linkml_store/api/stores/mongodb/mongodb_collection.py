import logging
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

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
        # collection_name = self.alias or self.name
        collection_name = self.alias
        if not collection_name:
            raise ValueError("Collection name not set")
        return self.parent.native_db[collection_name]

    def _check_if_initialized(self) -> bool:
        return self.alias in self.parent.native_db.list_collection_names()

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]
        self.mongo_collection.insert_many(objs)
        # TODO: allow mapping of _id to id for efficiency
        for obj in objs:
            del obj["_id"]
        self._post_insert_hook(objs)


    def index(self,
              objs: Union[OBJECT, List[OBJECT]],
              index_name: Optional[str] = None,
              replace: bool = False,
              unique: bool = False,
              **kwargs):
        """
        Create indexes on the collection.

        :param objs: Field(s) to index.
        :param index_name: Optional name for the index.
        :param replace: If True, the index will be dropped and recreated.
        :param unique: If True, creates a unique index (default: False).
        """

        if not isinstance(objs, list):
            objs = [objs]

        existing_indexes = self.mongo_collection.index_information()

        for obj in objs:
            field_exists = False
            index_to_drop = None

            # Extract existing index details
            for index_name_existing, index_details in existing_indexes.items():
                indexed_fields = [field[0] for field in index_details.get("key", [])]  # Extract field names

                if obj in indexed_fields:  # If this field is already indexed
                    field_exists = True
                    index_to_drop = index_name_existing if replace else None

            # Drop the index if replace=True and index_to_drop is valid
            if index_to_drop:
                self.mongo_collection.drop_index(index_to_drop)
                logging.debug(f"Dropped existing index: {index_to_drop}")

            # Create the new index only if it doesn't exist or was dropped
            if not field_exists or replace:
                self.mongo_collection.create_index(obj, name=index_name, unique=unique)
                logging.debug(f"Created new index: {index_name} on field {obj}, unique={unique}")
            else:
                logging.debug(f"Index already exists for field {obj}, skipping creation.")

    def upsert(self,
               objs: Union[OBJECT, List[OBJECT]],
               filter_fields: List[str],
               update_fields: Optional[List[str]] = None,
               **kwargs):
        """
        Upsert one or more documents into the MongoDB collection.

        :param objs: The document(s) to insert or update.
        :param filter_fields: List of field names to use as the filter for matching existing documents.
        :param update_fields: List of field names to include in the update. If None, all fields are updated.
        """
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            # Ensure filter fields exist in the object
            filter_criteria = {field: obj[field] for field in filter_fields if field in obj}
            if not filter_criteria:
                raise ValueError("At least one valid filter field must be present in each object.")

            # Check if a document already exists
            existing_doc = self.mongo_collection.find_one(filter_criteria)

            if existing_doc:
                # Update only changed fields
                updates = {key: obj[key] for key in update_fields if key in obj and obj[key] != existing_doc.get(key)}

                if updates:
                    self.mongo_collection.update_one(filter_criteria, {"$set": updates})
                    logging.debug(f"Updated existing document: {filter_criteria} with {updates}")
                else:
                    logging.debug(f"No changes detected for document: {filter_criteria}. Skipping update.")
            else:
                # Insert a new document
                self.mongo_collection.insert_one(obj)
                logging.debug(f"Inserted new document: {obj}")

    def query(self, query: Query, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs) -> QueryResult:
        mongo_filter = self._build_mongo_filter(query.where_clause)
        limit = limit or query.limit
        cursor = self.mongo_collection.find(mongo_filter)
        if limit and limit >= 0:
            cursor = cursor.limit(limit)
        offset = offset or query.offset
        if offset and offset >= 0:
            cursor = cursor.skip(offset)

        select_cols = query.select_cols

        def _as_row(row: dict):
            row = copy(row)
            del row["_id"]
            if select_cols:
                row = {k: row[k] for k in select_cols if k in row}
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

    from typing import Any, Dict, List, Union

    def query_facets(
        self,
        where: Dict = None,
        facet_columns: List[Union[str, Tuple[str, ...]]] = None,
        facet_limit=DEFAULT_FACET_LIMIT,
        **kwargs,
    ) -> Dict[Union[str, Tuple[str, ...]], List[Tuple[Any, int]]]:
        results = {}
        if not facet_columns:
            facet_columns = list(self.class_definition().attributes.keys())

        for col in facet_columns:
            logger.debug(f"Faceting on {col}")

            # Handle tuple columns
            if isinstance(col, tuple):
                group_id = {k.replace(".", "_"): f"${k}" for k in col}
                all_fields = col
            else:
                group_id = f"${col}"
                all_fields = [col]

            # Initial pipeline without unwinding
            facet_pipeline = [
                {"$match": where} if where else {"$match": {}},
                {"$group": {"_id": group_id, "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": facet_limit},
            ]

            logger.info(f"Initial facet pipeline: {facet_pipeline}")
            initial_results = list(self.mongo_collection.aggregate(facet_pipeline))

            # Check if we need to unwind based on the results
            needs_unwinding = False
            if isinstance(col, tuple):
                needs_unwinding = any(
                    isinstance(result["_id"], dict) and any(isinstance(v, list) for v in result["_id"].values())
                    for result in initial_results
                )
            else:
                needs_unwinding = any(isinstance(result["_id"], list) for result in initial_results)

            if needs_unwinding:
                logger.info(f"Detected array values for {col}, unwinding...")
                facet_pipeline = [{"$match": where} if where else {"$match": {}}]

                # Unwind each field if needed
                for field in all_fields:
                    field_parts = field.split(".")
                    for i in range(len(field_parts)):
                        facet_pipeline.append({"$unwind": f"${'.'.join(field_parts[:i + 1])}"})

                facet_pipeline.extend(
                    [
                        {"$group": {"_id": group_id, "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": facet_limit},
                    ]
                )

                logger.info(f"Updated facet pipeline with unwinding: {facet_pipeline}")
                facet_results = list(self.mongo_collection.aggregate(facet_pipeline))
            else:
                facet_results = initial_results

            logger.info(f"Facet results: {facet_results}")

            # Process results
            if isinstance(col, tuple):
                results[col] = [
                    (tuple(result["_id"].values()), result["count"])
                    for result in facet_results
                    if result["_id"] is not None and all(v is not None for v in result["_id"].values())
                ]
            else:
                results[col] = [
                    (result["_id"], result["count"]) for result in facet_results if result["_id"] is not None
                ]

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
