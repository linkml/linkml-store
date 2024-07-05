import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.types import DatabaseType
from linkml_store.utils.query_utils import mongo_query_to_match_function

logger = logging.getLogger(__name__)


class FileSystemCollection(Collection[DatabaseType]):
    path: Optional[Path] = None
    file_format: Optional[str] = None
    encoding: Optional[str] = None
    _objects_list: List[OBJECT] = None
    _object_map: Dict[str, OBJECT] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        parent: DatabaseType = self.parent
        if not self.path:
            if self.parent:
                self.path = Path(parent.directory_path)
        self._objects_list = []
        self._object_map = {}
        if not self.file_format:
            self.file_format = "json"

    @property
    def path_to_file(self):
        return Path(self.parent.directory_path) / f"{self.alias}.{self.file_format}"

    @property
    def objects_as_list(self) -> List[OBJECT]:
        if self._object_map:
            return list(self._object_map.values())
        else:
            return self._objects_list

    def _set_objects(self, objs: List[OBJECT]):
        pk = self.identifier_attribute_name
        if pk:
            self._object_map = {obj[pk]: obj for obj in objs}
            self._objects_list = []
        else:
            self._objects_list = objs
            self._object_map = {}

    def commit(self):
        path = self.path_to_file
        if not path:
            raise ValueError("Path not set")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save(path)

    def _save(self, path: Path):
        encoding = self.encoding or "utf-8"
        fmt = self.file_format or "json"
        mode = "w"
        if fmt == "parquet":
            mode = "wb"
            encoding = None
        with open(path, mode, encoding=encoding) as stream:
            if fmt == "json":
                import json

                json.dump(self.objects_as_list, stream, indent=2)
            elif fmt == "jsonl":
                import jsonlines

                writer = jsonlines.Writer(stream)
                writer.write_all(self.objects_as_list)
            elif fmt == "yaml":
                import yaml

                yaml.dump_all(self.objects_as_list, stream)
            elif fmt == "parquet":
                import pandas as pd
                import pyarrow
                import pyarrow.parquet as pq

                df = pd.DataFrame(self.objects_as_list)
                table = pyarrow.Table.from_pandas(df)
                pq.write_table(table, stream)
            elif fmt in {"csv", "tsv"}:
                import csv

                delimiter = "\t" if fmt == "tsv" else ","
                fieldnames = list(self.objects_as_list[0].keys())
                for obj in self.objects_as_list[1:]:
                    fieldnames.extend([k for k in obj.keys() if k not in fieldnames])
                writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                for obj in self.objects_as_list:
                    writer.writerow(obj)
            else:
                raise ValueError(f"Unsupported file format: {fmt}")

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return
        pk = self.identifier_attribute_name
        if pk:
            for obj in objs:
                if pk not in obj:
                    raise ValueError(f"Primary key {pk} not found in object {obj}")
                pk_val = obj[pk]
                self._object_map[pk_val] = obj
        else:
            self._objects_list.extend(objs)

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> Optional[int]:
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return 0
        pk = self.identifier_attribute_name
        n = 0
        if pk:
            for obj in objs:
                pk_val = obj[pk]
                if pk_val in self._object_map:
                    del self._object_map[pk_val]
                    n += 1
        else:
            n = len(objs)
            self._objects_list = [o for o in self._objects_list if o not in objs]
            n = n - len(objs)
        return n

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> Optional[int]:
        logger.info(f"Deleting from {self.target_class_name} where: {where}")
        if where is None:
            where = {}

        def matches(obj: OBJECT):
            for k, v in where.items():
                if obj.get(k) != v:
                    return False
            return True

        print(type(self))
        print(self)
        print(vars(self))
        curr_objects = [o for o in self.objects_as_list if not matches(o)]
        self._set_objects(curr_objects)

    def query(self, query: Query, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs) -> QueryResult:
        limit = limit or query.limit
        offset = offset or query.offset
        if offset is None:
            offset = 0
        where = query.where_clause or {}
        match = mongo_query_to_match_function(where)
        rows = [o for o in self.objects_as_list if match(o)]
        count = len(rows)
        if limit is None or limit < 0:
            limit = count
        # TODO: avoid recalculating
        returned_row = rows[offset : offset + limit]
        return QueryResult(query=query, num_rows=count, rows=returned_row)

    def query_facets(
        self, where: Dict = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[str, Dict[str, int]]:
        match = mongo_query_to_match_function(where)
        rows = [o for o in self.objects_as_list if match(o)]
        if not facet_columns:
            facet_columns = self.class_definition().attributes.keys()
        facet_results = {c: {} for c in facet_columns}
        for row in rows:
            for fc in facet_columns:
                if fc in row:
                    v = row[fc]
                    if not isinstance(v, str):
                        v = str(v)
                    if v not in facet_results[fc]:
                        facet_results[fc][v] = 1
                    else:
                        facet_results[fc][v] += 1
        return {fc: list(facet_results[fc].items()) for fc in facet_results}
