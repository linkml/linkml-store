import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult

logger = logging.getLogger(__name__)


class HDF5Collection(Collection):

    @property
    def hdf5_group(self) -> h5py.Group:
        return self.parent.file[self.name]

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            if "id" not in obj:
                raise ValueError("Each object must have an 'id' field.")
            obj_id = str(obj["id"])
            for key, value in obj.items():
                if key == "id":
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                self.hdf5_group.create_dataset(f"{obj_id}/{key}", data=value)

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        if not isinstance(objs, list):
            objs = [objs]
        count = 0
        for obj in objs:
            if "id" not in obj:
                raise ValueError("Each object must have an 'id' field.")
            obj_id = str(obj["id"])
            if obj_id in self.hdf5_group:
                del self.hdf5_group[obj_id]
                count += 1
        return count

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> int:
        logger.info(f"Deleting from {self.target_class_name} where: {where}")
        if where is None:
            where = {}
        results = self.query(Query(where_clause=where)).rows
        count = self.delete(results)
        return count

    def query(self, query: Query, **kwargs) -> QueryResult:
        results = []
        for obj_id in self.hdf5_group:
            obj = {"id": obj_id}
            for key, value in self.hdf5_group[obj_id].items():
                try:
                    obj[key] = json.loads(value[()])
                except json.JSONDecodeError:
                    obj[key] = value[()]
            if self._match_where_clause(obj, query.where_clause):
                results.append(obj)

        count = len(results)
        if query.limit:
            results = results[: query.limit]
        return QueryResult(query=query, num_rows=count, rows=results)

    def query_facets(
        self, where: Dict = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[str, List[Tuple[Any, int]]]:
        results = {}
        if not facet_columns:
            facet_columns = list(self.class_definition().attributes.keys())

        for col in facet_columns:
            logger.debug(f"Faceting on {col}")
            facet_counts = {}
            for obj in self.query(Query(where_clause=where)).rows:
                if col in obj:
                    value = obj[col]
                    if isinstance(value, list):
                        for v in value:
                            facet_counts[v] = facet_counts.get(v, 0) + 1
                    else:
                        facet_counts[value] = facet_counts.get(value, 0) + 1
            facet_counts = sorted(facet_counts.items(), key=lambda x: x[1], reverse=True)[:facet_limit]
            results[col] = facet_counts

        return results

    def _match_where_clause(self, obj: Dict[str, Any], where_clause: Optional[Dict[str, Any]]) -> bool:
        if where_clause is None:
            return True
        for key, value in where_clause.items():
            if key not in obj:
                return False
            if obj[key] != value:
                return False
        return True
