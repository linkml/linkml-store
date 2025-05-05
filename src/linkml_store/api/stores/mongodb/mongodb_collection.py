import logging
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from pymongo.collection import Collection as MongoCollection

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.utils.object_utils import object_path_get

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

    def index(
        self,
        objs: Union[OBJECT, List[OBJECT]],
        index_name: Optional[str] = None,
        replace: bool = False,
        unique: bool = False,
        **kwargs,
    ):
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

    def upsert(
        self,
        objs: Union[OBJECT, List[OBJECT]],
        filter_fields: List[str],
        update_fields: Optional[List[str]] = None,
        **kwargs,
    ):
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
        
        # Build projection if select_cols are provided
        projection = None
        if query.select_cols:
            projection = {"_id": 0}
            for col in query.select_cols:
                projection[col] = 1
        
        cursor = self.mongo_collection.find(mongo_filter, projection)
        if limit and limit >= 0:
            cursor = cursor.limit(limit)
        offset = offset or query.offset
        if offset and offset >= 0:
            cursor = cursor.skip(offset)

        select_cols = query.select_cols

        def _as_row(row: dict):
            row = copy(row)
            if "_id" in row:
                del row["_id"]
                
            if select_cols:
                # For nested fields, ensure we handle them properly
                result = {}
                for col in select_cols:
                    # If it's a nested field (contains dots)
                    if "." in col or "[" in col:
                        result[col]  = object_path_get(row, col)
                    elif col in row:
                        result[col] = row[col]
                return result
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
        if facet_limit is None:
            facet_limit = DEFAULT_FACET_LIMIT
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

    def group_by(
        self,
        group_by_fields: List[str],
        inlined_field="objects",
        agg_map: Optional[Dict[str, str]] = None,
        where: Optional[Dict] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Group objects in the collection by specified fields using MongoDB's aggregation pipeline.

        This implementation leverages MongoDB's native aggregation capabilities for efficient grouping.

        :param group_by_fields: List of fields to group by
        :param inlined_field: Field name to store aggregated objects
        :param agg_map: Dictionary mapping aggregation types to fields
        :param where: Filter conditions
        :param kwargs: Additional arguments
        :return: Query result containing grouped data
        """
        if isinstance(group_by_fields, str):
            group_by_fields = [group_by_fields]

        # Build the group key for MongoDB
        if len(group_by_fields) == 1:
            # Single field grouping
            group_id = f"${group_by_fields[0]}"
        else:
            # Multi-field grouping
            group_id = {field: f"${field}" for field in group_by_fields}

        # Start building the pipeline
        pipeline = []

        # Add match stage if where clause is provided
        if where:
            pipeline.append({"$match": where})

        # Add the group stage
        group_stage = {"$group": {"_id": group_id, "objects": {"$push": "$$ROOT"}}}
        pipeline.append(group_stage)

        # Execute the aggregation
        logger.debug(f"MongoDB group_by pipeline: {pipeline}")
        aggregation_results = list(self.mongo_collection.aggregate(pipeline))

        # Transform the results to match the expected format
        results = []
        for result in aggregation_results:
            # Skip null groups if needed
            if result["_id"] is None and kwargs.get("skip_nulls", False):
                continue

            # Create the group object
            if isinstance(result["_id"], dict):
                # Multi-field grouping
                group_obj = result["_id"]
            else:
                # Single field grouping
                group_obj = {group_by_fields[0]: result["_id"]}

            # Add the grouped objects
            objects = result["objects"]

            # Remove MongoDB _id field from each object
            for obj in objects:
                if "_id" in obj:
                    del obj["_id"]

            # Apply any field selection or transformations based on agg_map
            if agg_map:
                # Get first fields (fields to keep as single values)
                first_fields = agg_map.get("first", [])
                if first_fields:
                    # These are already in the group_obj from the _id
                    pass

                # Get list fields (fields to aggregate as lists)
                list_fields = agg_map.get("list", [])
                if list_fields:
                    # Filter objects to only include specified fields
                    objects = [{k: obj.get(k) for k in list_fields if k in obj} for obj in objects]
                elif not list_fields and first_fields:
                    # If list_fields is empty but first_fields is specified,
                    # filter out first_fields from objects to avoid duplication
                    objects = [{k: v for k, v in obj.items() if k not in first_fields} for obj in objects]

            # Add the objects to the group
            group_obj[inlined_field] = objects
            results.append(group_obj)

        return QueryResult(num_rows=len(results), rows=results)
