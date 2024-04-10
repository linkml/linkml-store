from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Any

from linkml_store.api import Collection
from linkml_store.api.collection import OBJECT


@dataclass
class MongoDBCollection(Collection):
    def add(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return
        cd = self.class_definition()
        if not cd:
            cd = self.induce_class_definition_from_objects(objs)
        collection = self.parent.database[self.name]
        collection.insert_many(objs)

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        if not isinstance(objs, list):
            objs = [objs]
        cd = self.class_definition()
        if not cd:
            cd = self.induce_class_definition_from_objects(objs)
        collection = self.parent.database[self.name]
        deleted_count = 0
        for obj in objs:
            result = collection.delete_one(obj)
            deleted_count += result.deleted_count
        return deleted_count

    def delete_where(self, where: Optional[Dict[str, Any]] = None, **kwargs) -> int:
        collection = self.parent.database[self.name]
        result = collection.delete_many(where)
        return result.deleted_count

    def query_facets(self, where: Dict = None, facet_columns: List[str] = None) -> Dict[str, Dict[str, int]]:
        results = {}
        cd = self.class_definition()
        collection = self.parent.database[self.name]
        if not facet_columns:
            facet_columns = list(self.class_definition().attributes.keys())
        for col in facet_columns:
            facet_pipeline = [
                {"$match": where} if where else {"$match": {}},
                {"$group": {"_id": f"${col}", "count": {"$sum": 1}}}
            ]
            facet_results = list(collection.aggregate(facet_pipeline))
            results[col] = [(row["_id"], row["count"]) for row in facet_results]
        return results